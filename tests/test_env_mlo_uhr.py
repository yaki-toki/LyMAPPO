"""P1 (공유 버퍼 MLO) + P2/P3 (미전달 위반 회계) 환경 단위 테스트.

pytest 미설치 환경 대비 standalone runner 포함:
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

SLOTS_10MS = 2400   # 21.6 ms >> L^(99.9)=10 ms (seed 편차 여유 포함)
SLOTS_DRAIN = 3000


def _make_env(arrival_pps: float = 2000.0, n_sta: int = 1) -> WLANEnv:
    cfg = WLANConfig(
        n_ap=1,
        n_sta_per_ap=n_sta,
        n_links=3,
        arrival_pps=arrival_pps,
        horizon=10_000_000,  # done 이 테스트를 중단하지 않도록
        seed=7,
    )
    env = WLANEnv(cfg)
    env.reset()
    return env


def _actions(env: WLANEnv, link_active: list[int], argmax_link: int) -> dict:
    """AP0 전 STA 공통: 지정 링크 활성 + mlo_dist argmax 를 한 링크에 고정."""
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


# --- P1: 공유 버퍼 -> 임의 선택 링크에서 서비스 가능 -------------------

def test_p1_arrivals_servable_on_nonzero_link() -> None:
    env = _make_env()
    acts = _actions(env, link_active=[0, 0, 1], argmax_link=2)
    info = _run(env, acts, SLOTS_DRAIN)
    assert info["packets_served"] > 0, (
        f"link-2 만 선택해도 서비스돼야 함 (P1), served={info['packets_served']}"
    )


def test_p1_link_choice_does_not_strand() -> None:
    """같은 조건에서 link-0 선택과 link-2 선택의 처리량이 동차여야 함."""
    served = {}
    for k in (0, 2):
        env = _make_env()
        acts = _actions(env, link_active=[1, 1, 1], argmax_link=k)
        served[k] = _run(env, acts, SLOTS_DRAIN)["packets_served"]
    assert served[2] > 0.5 * served[0], served


# --- P2: 좌초 패킷이 위반으로 집계 --------------------------------------

def test_p2_stranded_packets_counted_as_violations() -> None:
    env = _make_env()
    acts = _actions(env, link_active=[0, 0, 0], argmax_link=0)  # 서비스 불가
    info = _run(env, acts, SLOTS_10MS)
    assert info["packets_served"] == 0
    assert info["p99_violation_rate"] > 0.9, info["p99_violation_rate"]
    assert info["p99_9_violation_rate"] > 0.5, info["p99_9_violation_rate"]


def test_p2_on_time_delivery_low_violation() -> None:
    env = _make_env(arrival_pps=200.0)  # feasible 부하
    acts = _actions(env, link_active=[1, 1, 1], argmax_link=0)
    info = _run(env, acts, SLOTS_DRAIN)
    assert info["packets_served"] > 0
    assert info["p99_violation_rate"] < 0.05, info["p99_violation_rate"]


def test_p2_rate_bounded_no_double_count() -> None:
    """좌초 후 늦은 서비스가 재개돼도 패킷당 1회만 집계 (rate <= 1)."""
    env = _make_env()
    _run(env, _actions(env, [0, 0, 0], 0), SLOTS_10MS)       # 좌초 단계
    info = _run(env, _actions(env, [1, 1, 1], 0), SLOTS_DRAIN)  # 늦은 배출
    for key in ("p99_violation_rate", "p99_9_violation_rate"):
        assert 0.0 <= info[key] <= 1.0 + 1e-9, (key, info[key])
    assert info["packets_served"] > 0


# --- P3: 좌초가 Z (가상 큐) 압력을 만든다 --------------------------------

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
