"""FL 비교군 aggregator (q-FFL / AFL) 단위 테스트 — R2 대응.

pytest 미설치 환경 대비 standalone runner 포함:
    PYTHONDONTWRITEBYTECODE=1 python -B tests/test_fl_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.mappo import (  # noqa: E402
    AFLAggregator,
    QFFLAggregator,
    compute_agg_measure,
    make_aggregator,
)

N_AP = 4


def test_qffl_upweights_high_cost_ap() -> None:
    agg = QFFLAggregator(N_AP)
    cost = np.array([1.0, 1.0, 5.0, 1.0], dtype=np.float32)
    w = agg.compute_weights(cost)
    assert abs(w.sum() - 1.0) < 1e-6
    assert w[2] == w.max()
    assert w[2] > 0.3, w
    assert w.min() > 0.0, "q-FFL 은 클라이언트를 0 으로 만들지 않는다"


def test_qffl_equal_costs_uniform() -> None:
    agg = QFFLAggregator(N_AP)
    w = agg.compute_weights(np.full(N_AP, 3.0, dtype=np.float32))
    assert np.allclose(w, 0.25), w


def test_afl_concentrates_on_worst_client() -> None:
    agg = AFLAggregator(N_AP)
    cost = np.array([0.0, 0.0, 4.0, 0.0], dtype=np.float32)
    w = None
    for _ in range(8):  # mirror ascent 반복 -> worst client 로 질량 집중
        w = agg.compute_weights(cost)
    assert abs(w.sum() - 1.0) < 1e-6
    assert w[2] == w.max()
    assert w[2] > 0.6, w


def test_afl_state_persists_and_uniform_start() -> None:
    agg = AFLAggregator(N_AP)
    w0 = agg.compute_weights(np.zeros(N_AP, dtype=np.float32))
    assert np.allclose(w0, 0.25), w0  # 균등 cost -> lambda 불변 (uniform)
    cost = np.array([0.0, 3.0, 0.0, 0.0], dtype=np.float32)
    w1 = agg.compute_weights(cost)
    w2 = agg.compute_weights(cost)
    assert w2[1] > w1[1] > 0.25, (w1, w2)  # 상태 누적 확인


def test_agg_measure_cost_modes() -> None:
    cost = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    for mode in ("qffl", "afl"):
        measure = compute_agg_measure(
            mode, None, {}, N_AP, 3, cost_per_ap=cost
        )
        assert measure.tolist() == cost.tolist(), (mode, measure)
    try:
        compute_agg_measure("qffl", None, {}, N_AP, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("cost 없는 qffl 은 ValueError 여야 함")


def test_make_aggregator_fl_modes() -> None:
    assert isinstance(make_aggregator("qffl", N_AP), QFFLAggregator)
    assert isinstance(make_aggregator("afl", N_AP), AFLAggregator)


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
