"""
Vectorised planar geometry used by the observation builder and the batched
training simulator.

All geometry runs in the same local flat-earth frame the official
``core.scenario`` generator uses: x = north [NM], y = east [NM] around the
scenario ``center``. That frame is an affine transform of (lat, lon), so
point-in-polygon results are identical to BlueSky's ``areafilter`` (which tests
in raw lat/lon space). Bearings follow aviation convention: degrees clockwise
from north, i.e. ``atan2(east, north)``.

Polygons are passed as padded vertex arrays ``(..., V, 2)``. Padding repeats the
last real vertex, which yields zero-length edges that contribute nothing to
crossing tests or ray casts and give the correct vertex distance otherwise.
"""

import numpy as np

NM = 1852.0          # metres per nautical mile
NM2KM = 1.852


def wrap180(a):
    """Wrap angle(s) in degrees to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def latlon_to_nm(lat, lon, center_lat, center_lon):
    """(lat, lon) degrees -> (north, east) NM, exact inverse of core.scenario._nm_to_latlon."""
    north = (lat - center_lat) * 60.0
    east = (lon - center_lon) * (60.0 * np.cos(np.radians(center_lat)))
    return north, east


def pad_polygon(verts_nm, n_max):
    """Pad an (n, 2) vertex list to (n_max, 2) by repeating the last vertex."""
    verts_nm = np.asarray(verts_nm, dtype=np.float64)
    n = len(verts_nm)
    if n > n_max:
        raise ValueError(f"polygon has {n} vertices > n_max={n_max}")
    out = np.empty((n_max, 2), dtype=np.float64)
    out[:n] = verts_nm
    out[n:] = verts_nm[-1]
    return out


def _edges(verts):
    return verts, np.roll(verts, -1, axis=-2)


def points_in_polygons(px, py, verts):
    """Crossing-number point-in-polygon.

    px, py : (B,) points; verts : (B, P, V, 2) polygons -> (B, P) bool.
    """
    a, b = _edges(verts)
    ax, ay, bx, by = a[..., 0], a[..., 1], b[..., 0], b[..., 1]
    X = px[:, None, None]
    Y = py[:, None, None]
    straddle = (ay > Y) != (by > Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = ax + (Y - ay) * (bx - ax) / (by - ay)
    cross = straddle & (X < xint)
    return (np.count_nonzero(cross, axis=-1) % 2) == 1


def nearest_on_polygons(px, py, verts):
    """Distance from points to polygon boundaries and the nearest boundary point.

    px, py : (B,); verts : (B, P, V, 2)
    returns dist (B, P) [NM], nearest point (B, P, 2)
    """
    a, b = _edges(verts)
    p = np.stack([px, py], axis=-1)[:, None, None, :]
    ab = b - a
    ap = p - a
    ab2 = np.sum(ab * ab, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(ab2 > 0, np.sum(ap * ab, axis=-1) / ab2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    q = a + t[..., None] * ab
    d = np.linalg.norm(p - q, axis=-1)                     # (B, P, V)
    k = np.argmin(d, axis=-1)                              # (B, P)
    dmin = np.take_along_axis(d, k[..., None], axis=-1)[..., 0]
    qmin = np.take_along_axis(q, k[..., None, None], axis=-2)[..., 0, :]
    return dmin, qmin


def raycast(px, py, bearings_deg, verts, max_range):
    """Distance along rays to the nearest polygon edge.

    px, py : (B,); bearings_deg : (B, R) aviation bearings; verts : (B, P, V, 2)
    polygons with invalid ones already removed (e.g. moved far away).
    returns (B, R) distance [NM], clipped to max_range.
    """
    B = px.shape[0]
    a, b = _edges(verts)
    a = a.reshape(B, 1, -1, 2)
    v = b.reshape(B, 1, -1, 2) - a
    th = np.radians(bearings_deg)
    ux = np.cos(th)[..., None]                             # (B, R, 1) north
    uy = np.sin(th)[..., None]                             # east
    wx = a[..., 0] - px[:, None, None]
    wy = a[..., 1] - py[:, None, None]
    vx, vy = v[..., 0], v[..., 1]
    denom = ux * vy - uy * vx
    ok = np.abs(denom) > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wx * vy - wy * vx) / denom
        s = (wx * uy - wy * ux) / denom
    hit = ok & (t >= 0.0) & (s >= 0.0) & (s <= 1.0)
    t = np.where(hit, t, np.inf)
    return np.minimum(np.min(t, axis=-1), max_range)


def segment_hits_polygons(px, py, gx, gy, verts):
    """First intersection of segment p->g with each polygon boundary.

    returns (B, P) fraction along the segment of the first crossing (inf if none).
    """
    a, b = _edges(verts)
    ux = (gx - px)[:, None, None]
    uy = (gy - py)[:, None, None]
    wx = a[..., 0] - px[:, None, None]
    wy = a[..., 1] - py[:, None, None]
    vx = b[..., 0] - a[..., 0]
    vy = b[..., 1] - a[..., 1]
    denom = ux * vy - uy * vx
    ok = np.abs(denom) > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wx * vy - wy * vx) / denom
        s = (wx * uy - wy * ux) / denom
    hit = ok & (t >= 0.0) & (t <= 1.0) & (s >= 0.0) & (s <= 1.0)
    return np.min(np.where(hit, t, np.inf), axis=-1)


# The official deterministic scoring helper; it is pure numpy and broadcasts, so
# the batched simulator uses the very same function for its metric distances.
from core.tools import kwikqdrdist  # noqa: E402,F401
