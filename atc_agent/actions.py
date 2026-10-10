"""
Goal-relative categorical action space.

``action = [heading_bin, speed_bin]`` (``MultiDiscrete([17, 6])``):

* heading: target true heading = bearing-to-goal + HEADING_OFFSETS[bin]. The
  centre bin flies direct to the goal, so the policy only has to learn the
  *deviation* needed for separation / obstacle avoidance. A categorical head
  represents "pass left OR pass right" as two modes instead of averaging them
  into a straight line through the conflict, which a unimodal Gaussian would do.
* speed: absolute CAS target [m/s]. 163 m/s is the A320 VMO BlueSky's OpenAP
  envelope allows at the competition altitude, i.e. the time-optimal choice.

Commands are issued through BlueSky's own HDG / SPD autopilot command
implementations (``ap.selhdgcmd`` / ``ap.selspdcmd``), exactly what the stack
commands of the default env call, just vectorised over aircraft.
"""

import numpy as np
import bluesky as bs
from gymnasium import spaces

from atc_agent.geometry import kwikqdrdist

HEADING_OFFSETS = np.array([-120, -90, -70, -50, -35, -22, -12, -5, 0,
                            5, 12, 22, 35, 50, 70, 90, 120], dtype=np.float64)
SPEED_TARGETS = np.array([110.0, 125.0, 140.0, 150.0, 157.0, 163.0])   # CAS, m/s
DIRECT_BIN = int(np.where(HEADING_OFFSETS == 0)[0][0])


def action_space():
    return spaces.MultiDiscrete([len(HEADING_OFFSETS), len(SPEED_TARGETS)])


def action_targets(actions, own_lat, own_lon, goal_lat, goal_lon):
    """actions (B, 2) ints -> (target heading deg, target CAS m/s), each (B,)."""
    actions = np.asarray(actions).reshape(-1, 2).astype(np.int64)
    qdr, _ = kwikqdrdist(own_lat, own_lon, goal_lat, goal_lon)
    hdg = (qdr + HEADING_OFFSETS[actions[:, 0]]) % 360.0
    cas = SPEED_TARGETS[actions[:, 1]]
    return hdg, cas


def apply_targets(traf_idx, hdg, cas):
    """Issue HDG / SPD autopilot commands for BlueSky traffic indices."""
    traf_idx = np.asarray(traf_idx, dtype=np.int64)
    if traf_idx.size == 0:
        return
    bs.traf.ap.selhdgcmd(traf_idx, np.asarray(hdg, dtype=np.float64))
    bs.traf.ap.selspdcmd(traf_idx, np.asarray(cas, dtype=np.float64))
