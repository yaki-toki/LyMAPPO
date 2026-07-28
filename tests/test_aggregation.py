"""Unit tests for the ZQ / hybrid aggregation measure (WCL revision gate 1).

Includes a standalone runner for environments without pytest installed:
    PYTHONDONTWRITEBYTECODE=1 python -B tests/test_aggregation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.mappo import (  # noqa: E402
    InterferenceWeightedAggregator,
    UniformAggregator,
    compute_agg_measure,
    compute_z_measure,
    make_aggregator,
)

N_AP = 4
N_LINKS = 3


def _fake_actions(links_per_ap: list[list[int]]) -> dict[int, dict]:
    """Per-AP active link set -> the last_actions form used by collect_rollout."""
    actions: dict[int, dict] = {}
    for ap_id, links in enumerate(links_per_ap):
        la = np.zeros((5, N_LINKS), dtype=np.int64)  # 5 STA per AP
        for k in links:
            la[:, k] = 1
        actions[ap_id] = {"link_active": la}
    return actions


# --- compute_z_measure -------------------------------------------------

def test_z_measure_uniform_when_feasible() -> None:
    info = {"Z_per_ap_99": [0.0] * N_AP, "Z_per_ap_99_9": [0.0] * N_AP}
    measure = compute_z_measure(info, N_AP)
    assert np.allclose(measure, 1.0), measure
    weights = InterferenceWeightedAggregator(N_AP).compute_weights(measure)
    assert np.allclose(weights, 0.25), weights


def test_z_measure_prioritizes_pressured_ap() -> None:
    info = {"Z_per_ap_99": [0.0, 0.0, 3.0, 0.0], "Z_per_ap_99_9": [0.0] * N_AP}
    measure = compute_z_measure(info, N_AP)
    assert measure[2] == 4.0 and measure[0] == 1.0, measure
    weights = InterferenceWeightedAggregator(N_AP).compute_weights(measure)
    assert weights[2] == weights.max()
    assert abs(weights.sum() - 1.0) < 1e-6
    # A4 bound: [0.5/N, 2/N]
    assert weights.min() >= 0.5 / N_AP - 1e-6
    assert weights.max() <= 2.0 / N_AP + 1e-6


def test_z_measure_missing_keys_falls_back_to_uniform() -> None:
    measure = compute_z_measure({}, N_AP)
    assert np.allclose(measure, 1.0), measure


# --- compute_agg_measure dispatch --------------------------------------

def test_agg_measure_iw_uses_link_overlap() -> None:
    actions = _fake_actions([[0, 1], [0, 1], [1, 2], [2]])
    measure = compute_agg_measure("iw", actions, {}, N_AP, N_LINKS)
    # topology (2,2,3,1): AP2 shares a link with all three neighbours
    assert measure.tolist() == [2.0, 2.0, 3.0, 1.0], measure


def test_agg_measure_zq_ignores_actions() -> None:
    info = {"Z_per_ap_99": [1.0, 0.0, 0.0, 0.0], "Z_per_ap_99_9": [0.0] * N_AP}
    measure = compute_agg_measure("zq", None, info, N_AP, N_LINKS)
    assert measure[0] == 2.0 and measure[1] == 1.0, measure


def test_agg_measure_hybrid_is_product() -> None:
    actions = _fake_actions([[0, 1], [0, 1], [1, 2], [2]])
    info = {"Z_per_ap_99": [0.0, 0.0, 2.0, 0.0], "Z_per_ap_99_9": [0.0] * N_AP}
    measure = compute_agg_measure("hybrid", actions, info, N_AP, N_LINKS)
    # obss * (1 + Z) = [2,2,3,1] * [1,1,3,1]
    assert measure.tolist() == [2.0, 2.0, 9.0, 1.0], measure


def test_agg_measure_uniform_and_unknown() -> None:
    measure = compute_agg_measure("uniform", None, {}, N_AP, N_LINKS)
    assert np.allclose(measure, 1.0)
    try:
        compute_agg_measure("bogus", None, {}, N_AP, N_LINKS)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown mode must raise ValueError")


# --- make_aggregator ----------------------------------------------------

def test_make_aggregator_modes() -> None:
    for mode in ("iw", "zq", "hybrid"):
        agg = make_aggregator(mode, N_AP)
        assert isinstance(agg, InterferenceWeightedAggregator)
        assert not isinstance(agg, UniformAggregator), mode
    assert isinstance(make_aggregator("uniform", N_AP), UniformAggregator)
    try:
        make_aggregator("bogus", N_AP)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown mode must raise ValueError")


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
