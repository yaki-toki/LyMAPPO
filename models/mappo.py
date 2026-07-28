"""MAPPO loss + GAE advantage + Interference-weighted FedAvg aggregator.

Implements the OBSS-weighted federated update of formulation eq. (10) - (11)
and the PPO clipping model of Lemma 3.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:-1]
    return advantages, returns


def mappo_actor_loss(
    log_probs_old: torch.Tensor,
    log_probs_new: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float = 0.2,
) -> torch.Tensor:
    """clipped surrogate (Schulman 2017; Lemma 3 of formulation)."""
    ratio = (log_probs_new - log_probs_old).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip) * advantages
    return -torch.min(surr1, surr2).mean()


def critic_loss(
    values_pred: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    return ((values_pred - returns) ** 2).mean()


class InterferenceWeightedAggregator:
    """formulation eq. (10): w_i = OBSS_i / sum_i OBSS_i, [w_lo, w_hi] clip.

    Enforces the [bar w / N_AP, hi w / N_AP] bound of Assumption A4
    (eq. ass:weights).
    """

    def __init__(
        self,
        n_ap: int,
        w_min: float = 0.5,
        w_max: float = 2.0,
    ) -> None:
        self.n_ap = n_ap
        self.lo = w_min / n_ap
        self.hi = w_max / n_ap

    def compute_weights(self, obss_per_ap: np.ndarray) -> np.ndarray:
        raw = np.maximum(obss_per_ap, 1e-6)
        if raw.sum() <= 0:
            return np.full(self.n_ap, 1.0 / self.n_ap)
        weights = raw / raw.sum()
        # Bounded-simplex projection: a plain clip followed by renormalize
        # breaks the bound again, so the residual mass is redistributed in
        # proportion to the slack of each component, satisfying sum=1 and
        # [lo, hi] at the same time (actual A4 enforcement).
        for _ in range(self.n_ap + 1):
            weights = np.clip(weights, self.lo, self.hi)
            total = float(weights.sum())
            if abs(total - 1.0) < 1e-9:
                break
            if total > 1.0:
                slack = weights - self.lo
                weights = weights - slack * (total - 1.0) / slack.sum()
            else:
                headroom = self.hi - weights
                weights = weights + headroom * (1.0 - total) / headroom.sum()
        return weights / weights.sum()

    def aggregate_grads(
        self,
        grads_per_ap: List[Dict[str, torch.Tensor]],
        weights: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        agg: Dict[str, torch.Tensor] = {}
        device = next(iter(grads_per_ap[0].values())).device
        w_t = torch.from_numpy(weights.astype(np.float32)).to(device)
        for name in grads_per_ap[0].keys():
            stacked = torch.stack([g[name] for g in grads_per_ap], dim=0)
            shape = [self.n_ap] + [1] * (stacked.dim() - 1)
            agg[name] = (stacked * w_t.view(*shape)).sum(dim=0)
        return agg


class UniformAggregator(InterferenceWeightedAggregator):
    """Vanilla FedAvg: w_i = 1/N (the aggregation scheme of du2024fedwifi).

    For the C2 ablation and for reproducing the du2024fedwifi SOTA. Only
    compute_weights is overridden to a uniform weighting that ignores OBSS;
    aggregate_grads is reused from the parent class.
    """

    def compute_weights(self, obss_per_ap: np.ndarray) -> np.ndarray:
        return np.full(self.n_ap, 1.0 / self.n_ap, dtype=np.float32)


class QFFLAggregator(InterferenceWeightedAggregator):
    """q-FFL (Li et al., ICLR 2020) style loss-weighted FedAvg -- R2 baseline.

    The client loss F_k of the original algorithm is mapped to the per-AP
    rollout cost (negative mean shaped reward): w_k proportional to
    (min-max normalized cost + floor)^q. The floor preserves the q-FFL property
    that even the lowest-cost client does not go to 0. Since this is a fairness
    (fair-FL) oriented weighting, the A4 clip is not applied (fidelity to the
    original method takes priority).
    """

    Q_POWER = 1.0
    COST_FLOOR = 0.25

    def compute_weights(self, cost_per_ap: np.ndarray) -> np.ndarray:
        cost = np.asarray(cost_per_ap, dtype=np.float64)
        spread = float(cost.max() - cost.min())
        if spread <= 0:
            return np.full(self.n_ap, 1.0 / self.n_ap)
        normalized = (cost - cost.min()) / spread
        weights = (normalized + self.COST_FLOOR) ** self.Q_POWER
        return weights / weights.sum()


class AFLAggregator(InterferenceWeightedAggregator):
    """AFL (Mohri et al., ICML 2019) style worst-client weighting -- R2 baseline.

    The mixture lambda of agnostic FL is updated by stochastic mirror ascent:
    lambda_k <- lambda_k * exp(eta * normalized cost). As the rounds progress
    the mass concentrates on the worst client. The state (lambda) is kept across
    rounds and, for fidelity to the original method, the A4 clip is not applied.
    """

    ETA = 0.5

    def __init__(self, n_ap: int) -> None:
        super().__init__(n_ap)
        self.lmbda = np.full(n_ap, 1.0 / n_ap)

    def compute_weights(self, cost_per_ap: np.ndarray) -> np.ndarray:
        cost = np.asarray(cost_per_ap, dtype=np.float64)
        spread = float(cost.max() - cost.min())
        if spread > 0:
            normalized = (cost - cost.min()) / spread
            self.lmbda = self.lmbda * np.exp(self.ETA * normalized)
            self.lmbda = self.lmbda / self.lmbda.sum()
        return self.lmbda.copy()


def make_aggregator(mode: str, n_ap: int) -> InterferenceWeightedAggregator:
    """Aggregation mode -> aggregator instance.

    - "iw"      : measured link-overlap weighting (CL submission method)
    - "zq"      : virtual-queue (dual) pressure weighting (WCL revision
                  proposal, option B)
    - "hybrid"  : overlap x (1 + Z) product weighting (option B')
    - "uniform" : UniformAggregator (vanilla FedAvg / du2024fedwifi proxy)
    - "qffl"    : q-FFL style loss weighting (fair-FL baseline)
    - "afl"     : AFL style worst-client weighting (fair-FL baseline)

    zq/hybrid differ only in the measure while normalize + A4 clip are the same,
    so InterferenceWeightedAggregator is reused (the measure choice is handled
    by compute_agg_measure). FedProx is excluded from the baselines because in
    this training structure (immediate aggregation every epoch on a shared
    actor, no local multi-step) the proximal term is always 0, which makes it
    identical to FedAvg.
    """
    if mode in ("iw", "zq", "hybrid"):
        return InterferenceWeightedAggregator(n_ap)
    if mode == "uniform":
        return UniformAggregator(n_ap)
    if mode == "qffl":
        return QFFLAggregator(n_ap)
    if mode == "afl":
        return AFLAggregator(n_ap)
    raise ValueError(f"unknown aggregation mode: {mode!r}")


def compute_obss_measure(
    actions_per_ap: Dict[int, Dict],
    n_ap: int,
    n_links: int,
) -> np.ndarray:
    """Overlap count with other APs on the links each AP uses (eq. 9 simplified).

    P4 skeleton stage: 1.0 is used instead of alpha_{i'}^{tot}. From P5 on it is
    weighted by the actual airtime from env.cbr.
    """
    ap_uses_link = np.zeros((n_ap, n_links), dtype=bool)
    for ap_id in range(n_ap):
        la = np.asarray(actions_per_ap[ap_id]["link_active"])
        ap_uses_link[ap_id] = la.sum(axis=0) > 0

    obss = np.zeros(n_ap, dtype=np.float32)
    for i in range(n_ap):
        for ip in range(n_ap):
            if i == ip:
                continue
            if (ap_uses_link[i] & ap_uses_link[ip]).any():
                obss[i] += 1.0
    return obss


def compute_z_measure(info: Dict, n_ap: int) -> np.ndarray:
    """per-AP virtual-queue (dual variable) pressure measure: 1 + Z_i^(99) + Z_i^(99.9).

    Z is the dual variable of the CMDP constraint, so this weighting is a
    primal-dual coupling: the gradient of an AP that has accumulated constraint
    violations takes a larger share in the federated update. If every AP is
    feasible (Z=0) the measure is the all-ones vector -> a natural fallback to
    uniform. It also behaves as uniform on older env info that lacks the
    Z_per_ap_* keys.
    """
    z99 = np.asarray(info.get("Z_per_ap_99", np.zeros(n_ap)), dtype=np.float32)
    z999 = np.asarray(info.get("Z_per_ap_99_9", np.zeros(n_ap)), dtype=np.float32)
    return 1.0 + z99 + z999


def compute_agg_measure(
    mode: str,
    last_actions: Dict[int, Dict] | None,
    info: Dict,
    n_ap: int,
    n_links: int,
    cost_per_ap: np.ndarray | None = None,
) -> np.ndarray:
    """aggregation mode -> source measure of the non-uniform weighting.

    - "iw"      : measured link-overlap of the previous rollout
                  (compute_obss_measure)
    - "zq"      : virtual-queue pressure 1 + Z (compute_z_measure)
    - "hybrid"  : overlap x (1 + Z) product
    - "uniform" : all-ones vector (value irrelevant, UniformAggregator ignores it)
    - "qffl"/"afl": the raw per-AP rollout cost (negative mean shaped reward) is
      passed through -- the transform is handled by the compute_weights of that
      aggregator
    """
    if mode == "iw":
        return compute_obss_measure(last_actions, n_ap, n_links)
    if mode == "zq":
        return compute_z_measure(info, n_ap)
    if mode == "hybrid":
        obss = np.maximum(
            compute_obss_measure(last_actions, n_ap, n_links), 1e-6
        )
        return obss * compute_z_measure(info, n_ap)
    if mode == "uniform":
        return np.ones(n_ap, dtype=np.float32)
    if mode in ("qffl", "afl"):
        if cost_per_ap is None:
            raise ValueError(f"{mode!r} aggregation requires cost_per_ap")
        return np.asarray(cost_per_ap, dtype=np.float32)
    raise ValueError(f"unknown aggregation measure mode: {mode!r}")
