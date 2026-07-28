"""MAPPO 손실 + GAE advantage + Interference-weighted FedAvg aggregator.

formulation 식 (10) - (11) 의 OBSS-가중 federated update 와
Lemma 3 의 PPO clipping 모델을 구현.
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
    """formulation 식 (10): w_i = OBSS_i / sum_i OBSS_i, [w_lo, w_hi] clip.

    Assumption A4 (식 ass:weights) 의 [bar w / N_AP, hi w / N_AP] bound 강제.
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
        # Bounded-simplex projection: 단순 clip 후 renormalize 는 bound 를
        # 다시 깨뜨리므로, 잔여 질량을 여유 있는 성분에 비례 재분배하며
        # sum=1 과 [lo, hi] 를 동시에 만족시킨다 (A4 실제 강제).
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
    """Vanilla FedAvg: w_i = 1/N (du2024fedwifi 의 aggregation 방식).

    C2 ablation 및 du2024fedwifi SOTA 재현용. compute_weights 만 OBSS 를
    무시한 uniform 으로 override 하고, aggregate_grads 는 부모 클래스 재사용.
    """

    def compute_weights(self, obss_per_ap: np.ndarray) -> np.ndarray:
        return np.full(self.n_ap, 1.0 / self.n_ap, dtype=np.float32)


class QFFLAggregator(InterferenceWeightedAggregator):
    """q-FFL (Li et al., ICLR 2020) 스타일 loss-가중 FedAvg — R2 비교군.

    원 알고리즘의 클라이언트 손실 F_k 를 per-AP rollout 비용 (−평균 shaped
    reward) 으로 대응: w_k ∝ (min-max 정규화 비용 + floor)^q. floor 는
    최저-비용 클라이언트도 0 이 되지 않게 하는 q-FFL 의 성질을 보존한다.
    공정성 (fair-FL) 지향 가중이므로 A4 clip 은 적용하지 않는다 (원 기법
    충실성 우선).
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
    """AFL (Mohri et al., ICML 2019) 스타일 worst-client 가중 — R2 비교군.

    agnostic FL 의 mixture lambda 를 stochastic mirror ascent 로 갱신:
    lambda_k <- lambda_k * exp(eta * 정규화 비용). 라운드가 지날수록 최악
    클라이언트로 질량이 집중된다. 상태 (lambda) 를 라운드 간 유지하며,
    원 기법 충실성을 위해 A4 clip 은 적용하지 않는다.
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

    - "iw"      : 실측 링크-overlap 가중 (CL 제출판 기법)
    - "zq"      : 가상 큐(dual) 압력 가중 (WCL 개정 제안 기법, 옵션 B)
    - "hybrid"  : overlap x (1 + Z) 곱 가중 (옵션 B')
    - "uniform" : UniformAggregator (vanilla FedAvg / du2024fedwifi proxy)
    - "qffl"    : q-FFL 스타일 loss-가중 (fair-FL 비교군)
    - "afl"     : AFL 스타일 worst-client 가중 (fair-FL 비교군)

    zq/hybrid 는 measure 만 다르고 normalize + A4 clip 은 동일하므로
    InterferenceWeightedAggregator 를 재사용한다 (measure 선택은
    compute_agg_measure 가 담당). FedProx 는 본 학습 구조 (공유 actor 에서
    epoch 마다 즉시 aggregation, local multi-step 없음) 에서 proximal 항이
    항상 0 이라 FedAvg 와 동일해지므로 비교군에서 제외.
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
    """각 AP 가 사용하는 link 의 다른 AP 와의 overlap 카운트 (식 9 단순화).

    P4 골격 단계: alpha_{i'}^{tot} 대신 1.0 사용. P5 부터 env.cbr 의 실제 airtime 으로 가중.
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
    """per-AP 가상 큐(dual 변수) 압력 measure: 1 + Z_i^(99) + Z_i^(99.9).

    Z 는 CMDP 제약의 dual 변수이므로 이 가중은 primal-dual 결합:
    제약 위반이 누적된 AP 의 gradient 가 연합 update 에서 더 큰 몫을 갖는다.
    전 AP 가 feasible (Z=0) 이면 1 벡터 -> uniform 으로 자연 회귀.
    Z_per_ap_* 키가 없는 구버전 env info 에도 uniform 으로 동작.
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
    """aggregation mode -> 비균등 가중의 원천 measure.

    - "iw"      : 직전 rollout 의 실측 링크-overlap (compute_obss_measure)
    - "zq"      : 가상 큐 압력 1 + Z (compute_z_measure)
    - "hybrid"  : overlap x (1 + Z) 곱
    - "uniform" : 1 벡터 (UniformAggregator 가 무시하므로 값 무관)
    - "qffl"/"afl": per-AP rollout 비용 (−평균 shaped reward) 원본 전달 —
      변환은 해당 aggregator 의 compute_weights 가 담당
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
            raise ValueError(f"{mode!r} aggregation 은 cost_per_ap 가 필요")
        return np.asarray(cost_per_ap, dtype=np.float32)
    raise ValueError(f"unknown aggregation measure mode: {mode!r}")
