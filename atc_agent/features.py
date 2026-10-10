"""
Observation design: an ego-centric, intent-aware *entity* observation.

The flat vector produced here is parsed back into tokens by the policy
(``atc_agent.policy``). One function, ``build_observations``, is used by BOTH the
batched training simulator and the official-env subclasses used for scoring,
so training and evaluation observations are identical by construction.

Layout (all in the ownship body frame: +x nose, +y starboard)::

    [ ego (EGO_DIM) | aircraft tokens (M_AC x AC_DIM) | aircraft mask (M_AC)
      | obstacle tokens (N_OBS x OB_DIM) | obstacle mask (N_OBS) ]

ego      goal geometry, own speed/autopilot state, sector + obstacle summary,
         and a 32-ray "lidar" (closeness to obstacle edges and to the sector
         boundary) that captures the true polygon shapes.
aircraft the M_AC most threatening other aircraft, ranked by predicted miss
         distance over a 300 s horizon. Each token carries relative state, CPA
         (tcpa / dcpa) features, a learner-vs-scripted flag and the aircraft's
         *intent*: the active route waypoint for scripted intruders, the goal
         for other learning agents.
obstacle per restricted area: nearest boundary point, signed distance, centroid,
         angular extent (left/right tangents) and whether it blocks the direct
         line to the goal.
"""

import numpy as np

from atc_agent.geometry import (
    NM, latlon_to_nm, wrap180, pad_polygon, points_in_polygons,
    nearest_on_polygons, raycast, segment_hits_polygons, kwikqdrdist,
)

# ----------------------------------------------------------------- layout
N_RAYS = 32
RAY_RANGE_NM = 60.0
M_AC = 10            # aircraft tokens (SA eval: 10 intruders; MA eval: 9 others)
N_OBS = 5            # obstacle tokens (competition: n_obstacles = 5)
V_MAX = 8            # obstacle vertices (generator: 3..8)
S_MAX = 40           # sector vertices (generator adds points until area is met)

EGO_BASE = 20
EGO_DIM = EGO_BASE + 2 * N_RAYS
AC_DIM = 23
OB_DIM = 15
AC_OFF = EGO_DIM
AC_MASK_OFF = AC_OFF + M_AC * AC_DIM
OB_OFF = AC_MASK_OFF + M_AC
OB_MASK_OFF = OB_OFF + N_OBS * OB_DIM
OBS_DIM = OB_MASK_OFF + N_OBS

SEP_NM = 5.0         # loss-of-separation threshold
HORIZON_S = 300.0    # threat ranking / conflict prediction horizon
SHORT_HORIZON_S = 120.0
FAR = 1.0e5          # NM, where padded (missing) obstacles are parked


# ------------------------------------------------------- scenario geometry
def scenario_geometry(scenario):
    """Static per-episode geometry in the local NM frame (padded arrays)."""
    clat, clon = scenario.center
    sector = np.array([latlon_to_nm(la, lo, clat, clon) for la, lo in scenario.sector])
    obst = np.full((N_OBS, V_MAX, 2), FAR)
    valid = np.zeros(N_OBS, dtype=bool)
    centroid = np.full((N_OBS, 2), FAR)
    radius = np.zeros(N_OBS)
    for i, ob in enumerate(scenario.obstacles[:N_OBS]):
        v = np.array([latlon_to_nm(la, lo, clat, clon) for la, lo in ob.vertices])
        obst[i] = pad_polygon(v, V_MAX)
        valid[i] = True
        c = v.mean(axis=0)
        centroid[i] = c
        radius[i] = np.max(np.hypot(v[:, 0] - c[0], v[:, 1] - c[1]))
    return {
        "center": np.array([clat, clon], dtype=np.float64),
        "sector": pad_polygon(sector, S_MAX),
        "obst": obst,
        "obst_valid": valid,
        "obst_centroid": centroid,
        "obst_radius": radius,
    }


# ------------------------------------------------------------- observation
def build_observations(s):
    """Vectorised observation builder.

    ``s`` is a dict of numpy arrays for B ownships:
      own_lat, own_lon, own_hdg, own_tas, own_cas, own_seltrk, own_selspd  (B,)
      goal_lat, goal_lon, mode_ma                                          (B,)
      center (B, 2)    sector (B, S_MAX, 2)
      obst (B, N_OBS, V_MAX, 2)  obst_valid (B, N_OBS)  obst_centroid (B, N_OBS, 2)
      obst_radius (B, N_OBS)
      oth_lat, oth_lon, oth_hdg, oth_tas, oth_seltrk                       (B, O)
      oth_valid, oth_is_agent, oth_has_intent                              (B, O) bool
      oth_intent_lat, oth_intent_lon                                       (B, O)

    Returns (obs float32 (B, OBS_DIM), aux dict of per-ownship scalars used by
    the training reward).
    """
    B = s["own_lat"].shape[0]
    clat, clon = s["center"][:, 0], s["center"][:, 1]
    hdg = s["own_hdg"]
    tas = s["own_tas"]
    hr = np.radians(hdg)
    ch, sh = np.cos(hr), np.sin(hr)

    ox, oy = latlon_to_nm(s["own_lat"], s["own_lon"], clat, clon)
    gx, gy = latlon_to_nm(s["goal_lat"], s["goal_lon"], clat, clon)

    def to_body(dx, dy):
        c = ch.reshape((B,) + (1,) * (dx.ndim - 1))
        sn = sh.reshape((B,) + (1,) * (dx.ndim - 1))
        return dx * c + dy * sn, -dx * sn + dy * c

    # ---------------- goal
    qdr_g, d_goal = kwikqdrdist(s["own_lat"], s["own_lon"], s["goal_lat"], s["goal_lon"])
    rel_goal = np.radians(wrap180(qdr_g - hdg))

    # ---------------- other aircraft
    valid = s["oth_valid"]
    O = valid.shape[1]
    tx, ty = latlon_to_nm(s["oth_lat"], s["oth_lon"], clat[:, None], clon[:, None])
    rx, ry = tx - ox[:, None], ty - oy[:, None]
    dist = np.hypot(rx, ry)
    v_own_x, v_own_y = tas * ch / NM, tas * sh / NM                 # NM/s, north/east
    th = np.radians(s["oth_hdg"])
    v_oth_x, v_oth_y = s["oth_tas"] * np.cos(th) / NM, s["oth_tas"] * np.sin(th) / NM
    dvx, dvy = v_oth_x - v_own_x[:, None], v_oth_y - v_own_y[:, None]
    dv2 = dvx * dvx + dvy * dvy
    with np.errstate(divide="ignore", invalid="ignore"):
        tcpa = np.where(dv2 > 1e-14, -(rx * dvx + ry * dvy) / dv2, 0.0)

    def miss(h):
        t = np.clip(tcpa, 0.0, h)
        return np.hypot(rx + t * dvx, ry + t * dvy)

    dcpa = miss(np.inf)
    dmin_long = miss(HORIZON_S)
    dmin_short = miss(SHORT_HORIZON_S)

    bx, by = to_body(rx, ry)
    bvx, bvy = to_body(dvx * NM, dvy * NM)                           # m/s
    rel_hdg = np.radians(wrap180(s["oth_hdg"] - hdg[:, None]))
    cmd_turn = np.radians(wrap180(s["oth_seltrk"] - s["oth_hdg"]))
    has_int = s["oth_has_intent"] & valid
    ix, iy = latlon_to_nm(s["oth_intent_lat"], s["oth_intent_lon"], clat[:, None], clon[:, None])
    ibx, iby = to_body(ix - ox[:, None], iy - oy[:, None])
    int_brg = np.degrees(np.arctan2(iy - ty, ix - tx))               # other -> its intent
    int_rel = np.radians(wrap180(int_brg - hdg[:, None]))

    tok = np.zeros((B, O, AC_DIM))
    tok[..., 0] = np.clip(bx / 20.0, -10, 10)
    tok[..., 1] = np.clip(by / 20.0, -10, 10)
    tok[..., 2] = np.clip(dist / 20.0, 0, 10)
    tok[..., 3] = np.clip(1.0 - dist / 20.0, 0, 1)
    tok[..., 4] = np.clip(bvx / 150.0, -3, 3)
    tok[..., 5] = np.clip(bvy / 150.0, -3, 3)
    tok[..., 6] = np.cos(rel_hdg)
    tok[..., 7] = np.sin(rel_hdg)
    tok[..., 8] = np.clip((s["oth_tas"] - 150.0) / 15.0, -5, 5)
    tok[..., 9] = np.clip(tcpa / HORIZON_S, -1, 3)
    tok[..., 10] = np.clip(dcpa / SEP_NM, 0, 6)
    tok[..., 11] = np.clip(dmin_long / SEP_NM, 0, 6)
    tok[..., 12] = dmin_long < SEP_NM
    tok[..., 13] = dmin_short < SEP_NM
    tok[..., 14] = dist < SEP_NM
    tok[..., 15] = s["oth_is_agent"]
    tok[..., 16] = has_int
    tok[..., 17] = np.where(has_int, np.clip(ibx / 50.0, -5, 5), 0.0)
    tok[..., 18] = np.where(has_int, np.clip(iby / 50.0, -5, 5), 0.0)
    tok[..., 19] = np.where(has_int, np.cos(int_rel), 0.0)
    tok[..., 20] = np.where(has_int, np.sin(int_rel), 0.0)
    tok[..., 21] = np.cos(cmd_turn)
    tok[..., 22] = np.sin(cmd_turn)

    key = np.where(valid, dmin_long + 0.01 * dist, np.inf)
    order = np.argsort(key, axis=1, kind="stable")[:, :M_AC]
    n_keep = order.shape[1]
    ac_tok = np.zeros((B, M_AC, AC_DIM))
    ac_mask = np.zeros((B, M_AC))
    ac_tok[:, :n_keep] = np.take_along_axis(tok, order[..., None], axis=1)
    ac_mask[:, :n_keep] = np.take_along_axis(valid, order, axis=1)
    ac_tok *= ac_mask[..., None]

    # ---------------- obstacles
    ovalid = s["obst_valid"]
    verts = s["obst"]
    inside_ob = points_in_polygons(ox, oy, verts) & ovalid
    d_ob, q_ob = nearest_on_polygons(ox, oy, verts)
    sd_ob = np.where(inside_ob, -d_ob, d_ob)
    qbx, qby = to_body(q_ob[..., 0] - ox[:, None], q_ob[..., 1] - oy[:, None])
    cen = s["obst_centroid"]
    cbx, cby = to_body(cen[..., 0] - ox[:, None], cen[..., 1] - oy[:, None])
    vb = np.degrees(np.arctan2(verts[..., 1] - oy[:, None, None], verts[..., 0] - ox[:, None, None]))
    cb = np.degrees(np.arctan2(cen[..., 1] - oy[:, None], cen[..., 0] - ox[:, None]))
    delta = wrap180(vb - cb[..., None])
    left = np.radians(cb + delta.min(axis=-1) - hdg[:, None])
    right = np.radians(cb + delta.max(axis=-1) - hdg[:, None])
    frac = segment_hits_polygons(ox, oy, gx, gy, verts)
    seg_len = np.hypot(gx - ox, gy - oy)
    blocks = (np.isfinite(frac) | inside_ob) & ovalid
    entry = np.where(np.isfinite(frac), frac * seg_len[:, None], 0.0)

    otok = np.zeros((B, N_OBS, OB_DIM))
    otok[..., 0] = np.clip(qbx / 20.0, -10, 10)
    otok[..., 1] = np.clip(qby / 20.0, -10, 10)
    otok[..., 2] = np.clip(sd_ob / 20.0, -2, 10)
    otok[..., 3] = np.clip(1.0 - np.maximum(sd_ob, 0.0) / 10.0, 0, 1)
    otok[..., 4] = inside_ob
    otok[..., 5] = np.clip(cbx / 50.0, -5, 5)
    otok[..., 6] = np.clip(cby / 50.0, -5, 5)
    otok[..., 7] = s["obst_radius"] / 30.0
    otok[..., 8] = np.where(inside_ob, 0.0, np.cos(left))
    otok[..., 9] = np.where(inside_ob, 0.0, np.sin(left))
    otok[..., 10] = np.where(inside_ob, 0.0, np.cos(right))
    otok[..., 11] = np.where(inside_ob, 0.0, np.sin(right))
    otok[..., 12] = blocks
    otok[..., 13] = np.clip(entry / 50.0, 0, 5)
    otok[..., 14] = np.clip(1.0 - np.maximum(sd_ob, 0.0) / 3.0, 0, 1)   # 3 NM buffer
    okey = np.where(ovalid, sd_ob, np.inf)
    oorder = np.argsort(okey, axis=1, kind="stable")
    ob_tok = np.take_along_axis(otok, oorder[..., None], axis=1)
    ob_mask = np.take_along_axis(ovalid, oorder, axis=1).astype(np.float64)
    ob_tok *= ob_mask[..., None]

    # ---------------- sector + lidar
    sector = s["sector"][:, None]
    inside_sec = points_in_polygons(ox, oy, sector)[:, 0]
    d_sec, q_sec = nearest_on_polygons(ox, oy, sector)
    d_sec, q_sec = d_sec[:, 0], q_sec[:, 0]
    sd_sec = np.where(inside_sec, d_sec, -d_sec)                      # + inside
    sqx, sqy = to_body(q_sec[:, 0] - ox, q_sec[:, 1] - oy)
    ray_brg = hdg[:, None] + np.arange(N_RAYS)[None, :] * (360.0 / N_RAYS)
    lid_ob = raycast(ox, oy, ray_brg, verts, RAY_RANGE_NM)
    lid_sec = raycast(ox, oy, ray_brg, sector, RAY_RANGE_NM)

    n_pred_long = np.sum((dmin_long < SEP_NM) & valid, axis=1)
    n_pred_short = np.sum((dmin_short < SEP_NM) & valid, axis=1)
    min_dmin = np.min(np.where(valid, dmin_long, np.inf), axis=1)

    ego = np.zeros((B, EGO_DIM))
    ego[:, 0] = np.clip(d_goal / 50.0, 0, 5)
    ego[:, 1] = np.clip(d_goal / 10.0, 0, 1)
    ego[:, 2] = np.cos(rel_goal)
    ego[:, 3] = np.sin(rel_goal)
    ego[:, 4] = np.clip((tas - 150.0) / 15.0, -5, 5)
    ego[:, 5] = np.clip((s["own_cas"] - 140.0) / 25.0, -5, 5)
    sel_turn = np.radians(wrap180(s["own_seltrk"] - hdg))
    ego[:, 6] = np.cos(sel_turn)
    ego[:, 7] = np.sin(sel_turn)
    ego[:, 8] = np.clip((s["own_selspd"] - s["own_cas"]) / 25.0, -5, 5)
    ego[:, 9] = inside_sec
    ego[:, 10] = inside_ob.any(axis=1)
    ego[:, 11] = s["mode_ma"]
    ego[:, 12] = valid.sum(axis=1) / 10.0
    ego[:, 13] = np.clip(sd_sec / 20.0, -5, 5)
    ego[:, 14] = np.clip(sqx / 20.0, -10, 10)
    ego[:, 15] = np.clip(sqy / 20.0, -10, 10)
    ego[:, 16] = blocks.any(axis=1)
    ego[:, 17] = np.clip(n_pred_long / 3.0, 0, 3)
    ego[:, 18] = np.clip(n_pred_short / 3.0, 0, 3)
    ego[:, 19] = np.clip(np.where(np.isfinite(min_dmin), min_dmin, 30.0) / SEP_NM, 0, 6)
    ego[:, EGO_BASE:EGO_BASE + N_RAYS] = 1.0 - lid_ob / RAY_RANGE_NM
    ego[:, EGO_BASE + N_RAYS:] = 1.0 - lid_sec / RAY_RANGE_NM

    obs = np.concatenate([
        ego,
        ac_tok.reshape(B, -1), ac_mask,
        ob_tok.reshape(B, -1), ob_mask,
    ], axis=1).astype(np.float32)

    aux = {
        "n_pred_short": n_pred_short,
        "sd_sector": sd_sec,
        "sd_obst_min": np.min(np.where(ovalid, sd_ob, np.inf), axis=1),
    }
    return obs, aux
