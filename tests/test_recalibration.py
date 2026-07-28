"""v2 airtime-consistent recalibration (cfg.reuse_groups) unit tests.

Verifies the recalibration is faithful to the paper's formal model and fixes the
diagnosed link-concentration blind spot, WITHOUT disturbing v1 physics:
  - v1 (reuse_groups=None) never touches the busy-time state (byte-for-byte path);
  - v2 (reuse_groups=1) penalises network-wide link concentration (v1 does not);
  - busy-time/TXOP occupancy gives a finite airtime-bounded capacity (violation
    rises monotonically with load) -- the coupling ns-3 has and v1 lacked;
  - the eq:uhr decided-denominator violation accounting is preserved in v2.

pytest 미설치 대비 standalone runner 포함:
    PYTHONDONTWRITEBYTECODE=1 python -B tests/test_recalibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.python.env.wlan_env import WLANConfig, WLANEnv  # noqa: E402
from sim.python.mac.edca import (  # noqa: E402
    DEFAULT_AIFSN,
    DEFAULT_CWMAX,
    DEFAULT_CWMIN,
    DEFAULT_TXOP_US,
)


def _fixed_actions(env: WLANEnv, concentrate: bool) -> dict:
    """Balanced (link = j % L) or concentrated (all STAs -> link 0) fixed policy."""
    n_ap, n_sta, n_links = env.cfg.n_ap, env.cfg.n_sta_per_ap, env.cfg.n_links
    acts: dict = {}
    for ap in range(n_ap):
        md = np.zeros((n_sta, n_links))
        la = np.zeros((n_sta, n_links), dtype=np.int32)
        for i in range(n_sta):
            j = ap * n_sta + i
            link = 0 if concentrate else (j % n_links)
            md[i, link] = 1.0
            la[i, link] = 1
        acts[ap] = {
            "mlo_dist": md,
            "link_active": la,
            "ampdu_len": np.full((n_sta, n_links), 32, dtype=np.int64),
            "map_mode": 0,
            "edca_cwmin": np.array(DEFAULT_CWMIN, dtype=np.int32),
            "edca_cwmax": np.array(DEFAULT_CWMAX, dtype=np.int32),
            "edca_aifsn": np.array(DEFAULT_AIFSN, dtype=np.int32),
            "edca_txop_us": np.array(DEFAULT_TXOP_US, dtype=np.float64),
        }
    return acts


def _violation(
    reuse_groups: int | None,
    concentrate: bool,
    load: float,
    n_ap: int = 4,
    horizon: int = 1500,
    seed: int = 0,
) -> float:
    cfg = WLANConfig(
        n_ap=n_ap, n_sta_per_ap=5, n_links=3, horizon=horizon,
        arrival_pps=load, seed=seed, reuse_groups=reuse_groups,
    )
    env = WLANEnv(cfg)
    env.reset()
    info = {"p99_violation_rate": float("nan")}
    for _ in range(horizon):
        _, _, dones, info = env.step(_fixed_actions(env, concentrate))
        if all(dones.values()):
            break
    return float(info["p99_violation_rate"])


# --- v1 preservation --------------------------------------------------------

def test_v1_never_touches_busy_time() -> None:
    """reuse_groups=None keeps the v1 per-AP path; busy-time state stays zero."""
    cfg = WLANConfig(
        n_ap=4, n_sta_per_ap=5, n_links=3, horizon=10_000,
        arrival_pps=3000.0, seed=0, reuse_groups=None,
    )
    env = WLANEnv(cfg)
    env.reset()
    for _ in range(300):
        env.step(_fixed_actions(env, concentrate=True))
    assert np.all(env._link_busy_until == 0), "v1 must not use the v2 busy-time"


def test_v1_blind_to_concentration() -> None:
    """v1 (perfect spatial reuse) barely distinguishes concentrated vs balanced."""
    load = 3000.0
    bal = _violation(None, concentrate=False, load=load)
    con = _violation(None, concentrate=True, load=load)
    assert abs(con - bal) < 0.02, f"v1 should not penalise concentration: {bal} vs {con}"


# --- v2 airtime-consistent fixes -------------------------------------------

def test_v2_penalizes_concentration() -> None:
    """The diagnosed fix: v2 makes concentrating STAs onto one link cost more."""
    load = 3000.0
    bal = _violation(1, concentrate=False, load=load)
    con = _violation(1, concentrate=True, load=load)
    assert con > bal + 0.003, f"v2 must penalise concentration: bal={bal} con={con}"


def test_v2_capacity_is_airtime_bounded() -> None:
    """busy-time gives a finite capacity: violation rises with offered load."""
    v_lo = _violation(1, concentrate=False, load=200.0)
    v_hi = _violation(1, concentrate=False, load=8000.0)
    assert v_hi > v_lo + 0.05, f"airtime-bounded capacity expected: {v_lo} -> {v_hi}"


def test_v2_busy_time_set_and_reset() -> None:
    cfg = WLANConfig(
        n_ap=4, n_sta_per_ap=5, n_links=3, horizon=10_000,
        arrival_pps=3000.0, seed=0, reuse_groups=1,
    )
    env = WLANEnv(cfg)
    env.reset()
    for _ in range(80):
        env.step(_fixed_actions(env, concentrate=False))
    assert env._link_busy_until.max() > 0, "busy-time must be set after transmissions"
    env.reset()
    assert np.all(env._link_busy_until == 0), "reset must clear busy-time"


def test_v2_uhr_decided_denominator_under_starvation() -> None:
    """eq:uhr: with no service, every decided (aged-out) packet is a violation,
    so the rate -> 1 (not survivor-biased) on the v2 path too."""
    cfg = WLANConfig(
        n_ap=4, n_sta_per_ap=5, n_links=3, horizon=3000,
        arrival_pps=2000.0, seed=0, reuse_groups=1,
    )
    env = WLANEnv(cfg)
    env.reset()
    info: dict = {}
    for _ in range(3000):
        acts = _fixed_actions(env, concentrate=False)
        for ap in acts:  # starve: no active links -> no service
            acts[ap]["link_active"][:] = 0
        _, _, _, info = env.step(acts)
    assert info["packets_served"] == 0, info["packets_served"]
    assert info["p99_violation_rate"] > 0.9, info["p99_violation_rate"]
    assert 0.0 <= info["p99_9_violation_rate"] <= 1.0 + 1e-9


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
