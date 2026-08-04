"""Policy plug-in API unit tests -- registry, dotted-path loading, validation.

    PYTHONDONTWRITEBYTECODE=1 python -B -m pytest tests/test_policy_api.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.policy_api import (  # noqa: E402
    Policy,
    PolicyAction,
    load_policy,
    register_policy,
    validate_action,
)

N_AP, N_STA, N_LINKS = 4, 5, 3


def _mk_obs(
    n_ap: int = N_AP,
    n_sta: int = N_STA,
    n_links: int = N_LINKS,
    csi: np.ndarray | None = None,
    cbr: np.ndarray | None = None,
    link_mask: np.ndarray | None = None,
) -> list:
    """Per-AP obs dicts with the same schema the ns-3 driver bridge produces."""
    out = []
    for _ap in range(n_ap):
        out.append({
            "csi": (np.full((n_sta, n_links), 0.5, dtype=np.float32)
                    if csi is None else csi.copy()),
            "queue": np.zeros((n_sta, n_links, 4), dtype=np.float32),
            "hol_age": np.zeros((n_sta, n_links), dtype=np.float32),
            "cbr": (np.zeros(n_links, dtype=np.float32)
                    if cbr is None else cbr.copy()),
            "Z_99": np.zeros(n_sta, dtype=np.float32),
            "Z_99_9": np.zeros(n_sta, dtype=np.float32),
            "link_mask": (np.ones(n_links, dtype=np.float32)
                          if link_mask is None else link_mask.copy()),
        })
    return out


def _masks(link_mask: np.ndarray | None = None) -> np.ndarray:
    row = np.ones(N_LINKS, dtype=np.float32) if link_mask is None else link_mask
    return np.tile(row, (N_AP, 1))


# ── base class ──────────────────────────────────────────────────────────────

def test_base_policy_act_raises() -> None:
    pol = Policy(n_ap=N_AP, n_sta_per_ap=N_STA, n_links=N_LINKS)
    with pytest.raises(NotImplementedError):
        pol.act(_mk_obs(), _masks())


def test_base_policy_stores_dims_and_seed_rng() -> None:
    a = Policy(n_ap=2, n_sta_per_ap=3, n_links=3, seed=7)
    b = Policy(n_ap=2, n_sta_per_ap=3, n_links=3, seed=7)
    assert (a.n_ap, a.n_sta_per_ap, a.n_links) == (2, 3, 3)
    assert a.rng.integers(0, 1000) == b.rng.integers(0, 1000)


# ── registry + loader ───────────────────────────────────────────────────────

def test_register_and_load_by_name() -> None:
    @register_policy("unit-test-dummy")
    class DummyPolicy(Policy):
        def act(self, obs, link_masks):
            links = np.zeros((self.n_ap, self.n_sta_per_ap), dtype=np.int64)
            return PolicyAction(selected_links=links, map_mode=0)

    pol = load_policy("unit-test-dummy", n_ap=2, n_sta_per_ap=3, n_links=3)
    assert isinstance(pol, DummyPolicy)
    assert pol.n_ap == 2


def test_load_by_dotted_path() -> None:
    pol = load_policy(
        "examples.policies.greedy_score.GreedyScorePolicy",
        n_ap=N_AP, n_sta_per_ap=N_STA, n_links=N_LINKS,
    )
    assert isinstance(pol, Policy)


def test_load_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="no-such-policy"):
        load_policy("no-such-policy", n_ap=2, n_sta_per_ap=2, n_links=3)


def test_load_dotted_path_not_a_policy_raises() -> None:
    with pytest.raises(ValueError, match="Policy"):
        load_policy("models.policy_api.PolicyAction",
                    n_ap=2, n_sta_per_ap=2, n_links=3)


def test_load_leading_dot_spec_raises_value_error() -> None:
    # A relative-import typo must surface as the documented ValueError,
    # not importlib's raw TypeError.
    with pytest.raises(ValueError, match="cannot import"):
        load_policy(".examples.policies.greedy_score.GreedyScorePolicy",
                    n_ap=2, n_sta_per_ap=2, n_links=3)


# ── action validation ───────────────────────────────────────────────────────

def test_validate_action_ok_and_normalized() -> None:
    links = [[1] * N_STA for _ap in range(N_AP)]  # plain lists are coerced
    act = validate_action(
        PolicyAction(selected_links=links, map_mode=0), N_AP, N_STA, N_LINKS)
    assert isinstance(act.selected_links, np.ndarray)
    assert act.selected_links.shape == (N_AP, N_STA)
    assert act.selected_links.dtype.kind == "i"
    assert act.map_mode == 0


def test_validate_action_bad_shape_raises() -> None:
    links = np.zeros((N_AP, N_STA + 1), dtype=np.int64)
    with pytest.raises(ValueError, match="shape"):
        validate_action(
            PolicyAction(selected_links=links, map_mode=0), N_AP, N_STA, N_LINKS)


def test_validate_action_link_out_of_range_raises() -> None:
    links = np.zeros((N_AP, N_STA), dtype=np.int64)
    links[0, 0] = N_LINKS  # one past the last valid link index
    with pytest.raises(ValueError, match="link"):
        validate_action(
            PolicyAction(selected_links=links, map_mode=0), N_AP, N_STA, N_LINKS)


def test_validate_action_bad_map_mode_raises() -> None:
    links = np.zeros((N_AP, N_STA), dtype=np.int64)
    with pytest.raises(ValueError, match="map_mode"):
        validate_action(
            PolicyAction(selected_links=links, map_mode=3), N_AP, N_STA, N_LINKS)


def test_validate_action_none_map_mode_raises_value_error() -> None:
    links = np.zeros((N_AP, N_STA), dtype=np.int64)
    with pytest.raises(ValueError, match="map_mode"):
        validate_action(
            PolicyAction(selected_links=links, map_mode=None),  # type: ignore[arg-type]
            N_AP, N_STA, N_LINKS)


def test_validate_action_rejects_non_policy_action() -> None:
    with pytest.raises(ValueError, match="PolicyAction"):
        validate_action(
            np.zeros((N_AP, N_STA)), N_AP, N_STA, N_LINKS)  # type: ignore[arg-type]


# ── example policy (examples/policies/greedy_score.py) ──────────────────────

def _greedy(cbr_weight: float = 1.0):
    return load_policy(
        "examples.policies.greedy_score.GreedyScorePolicy",
        n_ap=N_AP, n_sta_per_ap=N_STA, n_links=N_LINKS, cbr_weight=cbr_weight,
    )


def test_greedy_score_prefers_best_csi_when_cbr_flat() -> None:
    csi = np.zeros((N_STA, N_LINKS), dtype=np.float32)
    csi[:, 2] = 1.0  # link 2 has the best CSI everywhere
    act = validate_action(
        _greedy().act(_mk_obs(csi=csi), _masks()), N_AP, N_STA, N_LINKS)
    assert (act.selected_links == 2).all()


def test_greedy_score_avoids_congested_link() -> None:
    csi = np.zeros((N_STA, N_LINKS), dtype=np.float32)
    csi[:, 0] = 0.6   # best CSI on link 0 ...
    csi[:, 1] = 0.5
    cbr = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # ... but link 0 saturated
    act = validate_action(
        _greedy(cbr_weight=1.0).act(_mk_obs(csi=csi, cbr=cbr), _masks()),
        N_AP, N_STA, N_LINKS)
    assert (act.selected_links == 1).all()


def test_greedy_score_respects_link_mask() -> None:
    csi = np.zeros((N_STA, N_LINKS), dtype=np.float32)
    csi[:, 0] = 1.0  # best CSI on link 0, but link 0 is not in the allowed set
    mask = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    obs = _mk_obs(csi=csi, link_mask=mask)
    act = validate_action(
        _greedy().act(obs, _masks(mask)), N_AP, N_STA, N_LINKS)
    assert (act.selected_links != 0).all()


def test_greedy_score_map_mode_is_none() -> None:
    act = validate_action(
        _greedy().act(_mk_obs(), _masks()), N_AP, N_STA, N_LINKS)
    assert act.map_mode == 0
