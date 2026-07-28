"""Topology definitions shared by the drivers and the ns-3 scenario.

Single source of truth for the asymmetric per-AP link-set tiling used by
both the Python side (observation link masks) and the experiment design.
Extracted from the original research trainer (train_feddrl.py), which is
not part of this release.
"""
from __future__ import annotations

ASYMMETRIC_LINK_SETS = [[0, 1], [0, 1], [1, 2], [2]]


def make_asymmetric_link_sets(n_ap: int) -> list:
    """Return an asymmetric link-allocation pattern with exactly n_ap entries.

    Tiles the canonical 4-AP pattern [[0,1],[0,1],[1,2],[2]] for backward
    compatibility (4 AP) and natural scaling (8/12/16 AP). For n_ap not a
    multiple of 4, the trailing remainder is filled by cycling the same
    pattern. The asymmetric 4-AP block keeps OBSS overlap heterogeneous
    so that the band-sharing graph, not geometry, decides who contends.
    """
    base = ASYMMETRIC_LINK_SETS
    return [list(base[i % len(base)]) for i in range(n_ap)]
