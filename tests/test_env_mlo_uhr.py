"""Env unit tests for P1 (shared-buffer MLO) + P2/P3 (undelivered-violation accounting).

Includes a standalone runner for environments without pytest installed:
    PYTHONDONTWRITEBYTECODE=1 python -B tests/test_env_mlo_uhr.py
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

SLOTS_10MS = 2400   # 21.6 ms >> L^(99.9)=10 ms (margin for seed variation)
SLOTS_DRAIN = 3000


def _make_env(arrival_pps: float = 2000.0, n_sta: int = 1) -> WLANEnv:
    cfg = WLANConfig(
        n_ap=1,
        n_sta_per_ap=n_sta,
        n_links=3,
        arrival_pps=arrival_pps,
        horizon=10_000_000,  # so that done does not stop the test
        seed=7,
    )
    env = WLANEnv(cfg)
    env.reset()
    return env


def _actions(env: WLANEnv, link_active: list[int], argmax_link: int) -> dict:
    """Common to all AP0 STAs: activate the given links + pin the mlo_dist argmax to one link."""
    n_sta = env.cfg.n_sta_per_ap
    n_links = env.cfg.n_links
    la = np.tile(np.asarray(link_active, dtype=np.int32), (n_sta, 1))
    md = np.full((n_sta, n_links), 0.0)
    md[:, argmax_link] = 1.0
    return {
        0: {
            "mlo_dist": md,
            "link_active": la,
            "ampdu_len": np.full((n_sta, n_links), 32, dtype=np.int64),
            "map_mode": 0,
            "edca_cwmin": np.array(DEFAULT_CWMIN, dtype=np.int32),
            "edca_cwmax": np.array(DEFAULT_CWMAX, dtype=np.int32),
            "edca_aifsn": np.array(DEFAULT_AIFSN, dtype=np.int32),
            "edca_txop_us": np.array(DEFAULT_TXOP_US, dtype=np.float64),
        }
    }


def _run(env: WLANEnv, acts: dict, n_slots: int) -> dict:
    info: dict = {}
    for _ in range(n_slots):
        _, _, _, info = env.step(acts)
    return info


# --- P1: shared buffer -> servable on any selected link ----------------

def test_p1_arrivals_servable_on_nonzero_link() -> None:
    env = _make_env()
    acts = _actions(env, link_active=[0, 0, 1], argmax_link=2)
    info = _run(env, acts, SLOTS_DRAIN)
    assert info["packets_served"] > 0, (
        f"must be served even when only link-2 is selected (P1), served={info['packets_served']}"
    )


def test_p1_link_choice_does_not_strand() -> None:
    """Under identical conditions, selecting link-0 and link-2 must give comparable throughput."""
    served = {}
    for k in (0, 2):
        env = _make_env()
        acts = _actions(env, link_active=[1, 1, 1], argmax_link=k)
        served[k] = _run(env, acts, SLOTS_DRAIN)["packets_served"]
    assert served[2] > 0.5 * served[0], served


# --- P2: stranded packets are counted as violations ---------------------

def test_p2_stranded_packets_counted_as_violations() -> None:
    env = _make_env()
    acts = _actions(env, link_active=[0, 0, 0], argmax_link=0)  # no service possible
    info = _run(env, acts, SLOTS_10MS)
    assert info["packets_served"] == 0
    assert info["p99_violation_rate"] > 0.9, info["p99_violation_rate"]
    assert info["p99_9_violation_rate"] > 0.5, info["p99_9_violation_rate"]


def test_p2_on_time_delivery_low_violation() -> None:
    env = _make_env(arrival_pps=200.0)  # feasible load
    acts = _actions(env, link_active=[1, 1, 1], argmax_link=0)
    info = _run(env, acts, SLOTS_DRAIN)
    assert info["packets_served"] > 0
    assert info["p99_violation_rate"] < 0.05, info["p99_violation_rate"]


def test_p2_rate_bounded_no_double_count() -> None:
    """Even if late service resumes after stranding, count once per packet (rate <= 1)."""
    env = _make_env()
    _run(env, _actions(env, [0, 0, 0], 0), SLOTS_10MS)       # stranding phase
    info = _run(env, _actions(env, [1, 1, 1], 0), SLOTS_DRAIN)  # late drain
    for key in ("p99_violation_rate", "p99_9_violation_rate"):
        assert 0.0 <= info[key] <= 1.0 + 1e-9, (key, info[key])
    assert info["packets_served"] > 0


# --- P3: stranding creates Z (virtual queue) pressure --------------------

def test_p3_stranding_pressurizes_z() -> None:
    env = _make_env()
    acts = _actions(env, link_active=[0, 0, 0], argmax_link=0)
    info = _run(env, acts, SLOTS_10MS)
    assert info["mean_Z_99"] > 0.0, info
    assert info["Z_per_ap_99"][0] > 0.0, info


def test_p3_feasible_no_z_pressure() -> None:
    env = _make_env(arrival_pps=200.0)
    acts = _actions(env, link_active=[1, 1, 1], argmax_link=0)
    info = _run(env, acts, SLOTS_DRAIN)
    assert info["mean_Z_99"] < 0.5, info["mean_Z_99"]


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
