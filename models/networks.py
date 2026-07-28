"""Actor / Critic MLP + state encoding / action decoding helpers.

formulation 의 Section IV (CTDE) 에 대응:
- Actor (decentralized): per-AP local state -> (link logits, map_mode logits)
- Critic (centralized): joint state -> scalar value (training only)

State 인코딩 (per AP):
    csi (n_sta, n_links) + queue (n_sta, n_links, 4) + hol_age (n_sta, n_links)
    + cbr (n_links,) + Z_99 (n_sta,) + Z_99_9 (n_sta,) -> 1-D float vector

Action 디코딩 (per AP):
    logits = (n_sta * n_links + n_map_modes,) ->
        per-STA link 선택 (categorical) + map_mode (categorical)
    환경 action dict 의 EDCA / AMPDU / link_active 는 default 값 + sampled link 의
    one-hot 으로 결정. P4 후속 단계에서 EDCA 도 학습 대상으로 확장 예정.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim.python.mac.edca import (
    DEFAULT_AIFSN,
    DEFAULT_CWMAX,
    DEFAULT_CWMIN,
    DEFAULT_TXOP_US,
)


N_MAP_MODES = 3  # none / Co-TDMA / Co-OFDMA
AMPDU_FIXED = 16


# P6 (obs 정규화): persist 학습 + 정직한 회계에서는 buffer 길이가 수천,
# HOL age 가 수백 ms 까지 자라므로 raw 값은 MLP 를 포화시키고 train/eval
# 분포를 어긋나게 한다. 큐는 log1p (heavy-tail 압축), HOL 은 deadline 단위,
# Z 는 z_clip 단위로 정규화한다. csi/cbr 은 원래 소규모라 유지.
# ★드라이버(train_ns3.py/feddrl.py)는 시작 시 CLI 값으로 이 둘을 덮어써
# 정규화가 실제 deadline999/z_clip 을 추적하게 한다 (P1/P6 수정: 구 Z_NORM=100
# 은 z_clip=10 과 10× 불일치 → actor 가 Z 를 [0,0.1] 로만 봤음).
HOL_NORM_S = 1e-2   # 기본 L^(99.9) = 10 ms; 드라이버가 deadline999 로 덮어씀
Z_NORM = 10.0       # 기본 z_clip = 10;   드라이버가 --z-clip 으로 덮어씀


def encode_obs(obs_ap: Dict[str, np.ndarray]) -> np.ndarray:
    """per-AP obs dict -> flat float32 vector (P6: 정규화, P7: link_mask).

    link_mask 가 없는 구형 obs (예: 갱신 전 ns-3 bridge) 는 전 링크 사용
    가능 (대칭) 으로 간주한다.
    """
    cbr = obs_ap["cbr"].ravel().astype(np.float32)
    link_mask = obs_ap.get("link_mask")
    if link_mask is None:
        link_mask = np.ones_like(cbr)
    parts = [
        obs_ap["csi"].ravel().astype(np.float32),
        np.log1p(obs_ap["queue"].ravel().astype(np.float32)),
        obs_ap["hol_age"].ravel().astype(np.float32) / HOL_NORM_S,
        cbr,
        obs_ap["Z_99"].ravel().astype(np.float32) / Z_NORM,
        obs_ap["Z_99_9"].ravel().astype(np.float32) / Z_NORM,
        link_mask.ravel().astype(np.float32),  # P7: AP 의 링크 집합 K_i
    ]
    return np.concatenate(parts)


def encode_joint(obs: Dict[int, Dict[str, np.ndarray]]) -> np.ndarray:
    """모든 AP 의 obs 를 concat 한 joint state (centralized critic 입력)."""
    return np.concatenate([encode_obs(obs[ap]) for ap in sorted(obs.keys())])


def mask_link_logits(logits: torch.Tensor, link_mask, n_sta: int,
                     n_links: int) -> torch.Tensor:
    """per-AP 링크집합 K_i 강제: 비허용 링크 logit 을 큰 음수로 눌러 정책이
    그 밴드를 절대 선택하지 않게 한다. 학습(train_ns3)과 평가(feddrl.py)가
    반드시 같은 마스킹을 써야 train/eval 행동분포가 일치한다 (P3 수정).
    logits: (..., n_sta*n_links + N_MAP_MODES)."""
    out = logits.clone()
    lead = out.shape[:-1]
    grid = out[..., : n_sta * n_links].view(*lead, n_sta, n_links)
    disallow = torch.tensor([float(m) == 0.0 for m in link_mask],
                            dtype=torch.bool, device=out.device)
    grid[..., disallow] = -1e9
    return out


def mask_map_logits(logits: torch.Tensor, n_sta: int, n_links: int,
                    n_map_modes: int) -> torch.Tensor:
    """map_mode 를 0(no coord)으로 동결: mode 1..K-1 logit 을 큰 음수로.
    (Co-TDMA 는 16× 직렬화 붕괴, Co-OFDMA 는 현 시나리오서 no-op — 학습·평가
    동일 지점에 적용해 행동분포 일치를 보장한다.)"""
    out = logits.clone()
    base = n_sta * n_links
    if n_map_modes > 1:
        out[..., base + 1: base + n_map_modes] = -1e9
    return out


def state_dim_of(env) -> int:
    n_sta = env.cfg.n_sta_per_ap
    n_links = env.cfg.n_links
    return (
        n_sta * n_links
        + n_sta * n_links * 4
        + n_sta * n_links
        + n_links
        + n_sta
        + n_sta
        + n_links  # P7: link_mask
    )


def action_dim_of(env) -> int:
    return env.cfg.n_sta_per_ap * env.cfg.n_links + N_MAP_MODES


class ActorMLP(nn.Module):
    """per-AP decentralized actor."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorLSTM(nn.Module):
    """Recurrent per-AP actor (korolev2023rlmlo closer-proxy).

    State → encoder (1 FC layer) → LSTM cell → output FC. The LSTM hidden
    state is maintained externally by the caller and reset per episode.
    Used in conjunction with --aggregation none + per-AP independent actors
    to approximate the MH-RSAC family (single-MLD, recurrent, no parameter
    sharing) lifted to our multi-AP action space.

    NOTE: this proxy keeps PPO on-policy training and does NOT implement
    SAC's entropy regularisation or twin Q-target. The closer reimpl
    therefore isolates the *recurrent + independent* contribution of
    korolev2023rlmlo from the off-policy SAC details.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
        )
        self.lstm = nn.LSTMCell(hidden, hidden)
        self.head = nn.Linear(hidden, action_dim)

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Returns (logits, new_hidden). x shape: (B, state_dim)."""
        z = self.encoder(x)
        if hidden is None:
            hidden = self.init_state(x.size(0), x.device)
        h, c = self.lstm(z, hidden)
        return self.head(h), (h, c)


class LSTMActorWrapper(nn.Module):
    """Backward-compat shim: makes ActorLSTM behave like ActorMLP for callers
    that don't manage hidden state explicitly.

    The wrapper holds (h, c) as buffers and detaches them on each forward
    so PPO's k-epoch updates do not backprop through previous slots.
    External code calls ``reset_hidden()`` at episode boundary; otherwise
    the hidden state evolves automatically inside the rollout.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.lstm_actor = ActorLSTM(state_dim, action_dim, hidden=hidden)
        self._h: torch.Tensor | None = None
        self._c: torch.Tensor | None = None

    def reset_hidden(self, batch_size: int, device: torch.device) -> None:
        self._h, self._c = self.lstm_actor.init_state(batch_size, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._h is None or self._h.size(0) != x.size(0):
            self.reset_hidden(x.size(0), x.device)
        logits, (h, c) = self.lstm_actor(x, (self._h, self._c))
        # Detach to keep PPO's multi-epoch updates well-defined.
        self._h = h.detach()
        self._c = c.detach()
        return logits


class CriticMLP(nn.Module):
    """centralized critic — joint state -> V(s) (n_heads=1, 하위호환) 또는
    per-AP 가치 벡터 V_i(s) (n_heads=N_AP).

    F10 (P4 수정): 단일 스칼라 V 가 cross-AP 평균 return 을 예측하면 per-AP
    GAE 의 bootstrap 항이 자기 AP 보상과 어긋나 — 이질 부하(load-spread>0)
    에서 advantage 가 정확히 편향된다. per-AP head 는 CTDE(joint 입력)를
    유지하면서 각 AP 의 baseline 을 분리한다."""

    def __init__(self, joint_state_dim: int, hidden: int = 64,
                 n_heads: int = 1, n_quantiles: int = 1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_quantiles = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(joint_state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_heads * n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.n_quantiles > 1:
            # QR critic (B): (..., n_heads, K) — head 별 return 분위수.
            return out.view(*out.shape[:-1], self.n_heads, self.n_quantiles)
        return out.squeeze(-1) if self.n_heads == 1 else out


def sample_action(
    logits: torch.Tensor,
    n_sta: int,
    n_links: int,
    rng: np.random.Generator,
    deterministic: bool = False,
) -> Tuple[Dict, Dict[str, np.ndarray], float]:
    """Categorical sampling: 각 STA 링크 + map_mode."""
    logits_np = logits.detach().cpu().numpy().astype(np.float64)
    link_logits = logits_np[: n_sta * n_links].reshape(n_sta, n_links)
    map_logits = logits_np[n_sta * n_links : n_sta * n_links + N_MAP_MODES]

    selected_links = np.zeros(n_sta, dtype=np.int64)
    log_prob = 0.0
    mlo_dist = np.zeros((n_sta, n_links), dtype=np.float64)
    for s in range(n_sta):
        lp = link_logits[s] - link_logits[s].max()
        ex = np.exp(lp)
        probs = ex / ex.sum()
        if deterministic:
            link = int(np.argmax(link_logits[s]))
        else:
            link = int(rng.choice(n_links, p=probs))
        selected_links[s] = link
        log_prob += float(np.log(max(probs[link], 1e-12)))
        mlo_dist[s] = probs

    mp = map_logits - map_logits.max()
    ex_m = np.exp(mp)
    probs_m = ex_m / ex_m.sum()
    if deterministic:
        map_mode = int(np.argmax(map_logits))
    else:
        map_mode = int(rng.choice(N_MAP_MODES, p=probs_m))
    log_prob += float(np.log(max(probs_m[map_mode], 1e-12)))

    link_active = np.zeros((n_sta, n_links), dtype=np.int32)
    for s in range(n_sta):
        link_active[s, selected_links[s]] = 1

    action = {
        "mlo_dist": mlo_dist,
        "edca_cwmin": np.array(DEFAULT_CWMIN, dtype=np.int32),
        "edca_cwmax": np.array(DEFAULT_CWMAX, dtype=np.int32),
        "edca_aifsn": np.array(DEFAULT_AIFSN, dtype=np.int32),
        "edca_txop_us": np.array(DEFAULT_TXOP_US, dtype=np.float64),
        "map_mode": map_mode,
        "ampdu_len": np.full((n_sta, n_links), AMPDU_FIXED, dtype=np.int32),
        "link_active": link_active,
    }
    indices = {"selected_links": selected_links, "map_mode": map_mode}
    return action, indices, log_prob


def log_prob_of(
    logits: torch.Tensor,
    selected_links: torch.Tensor,
    map_modes: torch.Tensor,
    n_sta: int,
    n_links: int,
) -> torch.Tensor:
    """배치 log-prob 재계산 (PPO ratio 계산용). logits: (B, action_dim)."""
    link_logits = logits[..., : n_sta * n_links].view(-1, n_sta, n_links)
    map_logits = logits[..., n_sta * n_links : n_sta * n_links + N_MAP_MODES]

    link_log_probs = F.log_softmax(link_logits, dim=-1)
    sel = selected_links.unsqueeze(-1)
    sta_logp = link_log_probs.gather(-1, sel).squeeze(-1).sum(-1)

    map_log_probs = F.log_softmax(map_logits, dim=-1)
    map_logp = map_log_probs.gather(-1, map_modes.unsqueeze(-1)).squeeze(-1)

    return sta_logp + map_logp
