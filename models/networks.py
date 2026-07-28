"""Actor / Critic MLP + state encoding / action decoding helpers.

Corresponds to Section IV (CTDE) of the formulation:
- Actor (decentralized): per-AP local state -> (link logits, map_mode logits)
- Critic (centralized): joint state -> scalar value (training only)

State encoding (per AP):
    csi (n_sta, n_links) + queue (n_sta, n_links, 4) + hol_age (n_sta, n_links)
    + cbr (n_links,) + Z_99 (n_sta,) + Z_99_9 (n_sta,) -> 1-D float vector

Action decoding (per AP):
    logits = (n_sta * n_links + n_map_modes,) ->
        per-STA link selection (categorical) + map_mode (categorical)
    EDCA / AMPDU / link_active in the environment action dict are set from the
    default values plus the one-hot of the sampled link. A later P4 stage will
    extend EDCA to be learned as well.
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


# P6 (obs normalization): with persistent training + honest accounting the
# buffer length reaches thousands and the HOL age grows to hundreds of ms, so
# raw values saturate the MLP and skew the train/eval distributions. The queue
# is normalized with log1p (heavy-tail compression), HOL in deadline units and
# Z in z_clip units. csi/cbr are already small-scale and are kept as is.
# NOTE: the drivers (train_ns3.py/feddrl.py) overwrite these two with the CLI
# values at startup so the normalization tracks the actual deadline999/z_clip
# (P1/P6 fix: the old Z_NORM=100 was a 10x mismatch with z_clip=10, so the
# actor only ever saw Z in [0,0.1]).
HOL_NORM_S = 1e-2   # default L^(99.9) = 10 ms; driver overwrites with deadline999
Z_NORM = 10.0       # default z_clip = 10;   driver overwrites with --z-clip


def encode_obs(obs_ap: Dict[str, np.ndarray]) -> np.ndarray:
    """per-AP obs dict -> flat float32 vector (P6: normalization, P7: link_mask).

    Legacy obs without link_mask (e.g. a pre-update ns-3 bridge) is treated as
    having every link available (symmetric).
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
        link_mask.ravel().astype(np.float32),  # P7: link set K_i of the AP
    ]
    return np.concatenate(parts)


def encode_joint(obs: Dict[int, Dict[str, np.ndarray]]) -> np.ndarray:
    """Joint state concatenating the obs of every AP (centralized critic input)."""
    return np.concatenate([encode_obs(obs[ap]) for ap in sorted(obs.keys())])


def mask_link_logits(logits: torch.Tensor, link_mask, n_sta: int,
                     n_links: int) -> torch.Tensor:
    """Enforce the per-AP link set K_i: push disallowed link logits to a large
    negative value so the policy never selects those bands. Training (train_ns3)
    and evaluation (feddrl.py) must use the same masking for the train/eval
    action distributions to match (P3 fix).
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
    """Freeze map_mode to 0 (no coord): push the mode 1..K-1 logits to a large
    negative value. (Co-TDMA collapses under 16x serialization, Co-OFDMA is a
    no-op in the current scenario -- applied at the same point in training and
    evaluation to guarantee matching action distributions.)"""
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
    """centralized critic -- joint state -> V(s) (n_heads=1, backward compatible)
    or the per-AP value vector V_i(s) (n_heads=N_AP).

    F10 (P4 fix): if a single scalar V predicts the cross-AP mean return, the
    bootstrap term of the per-AP GAE is misaligned with that AP's own reward --
    the advantage is biased exactly under heterogeneous load (load-spread>0).
    The per-AP head keeps CTDE (joint input) while separating the baseline of
    each AP."""

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
            # QR critic (B): (..., n_heads, K) -- per-head return quantiles.
            return out.view(*out.shape[:-1], self.n_heads, self.n_quantiles)
        return out.squeeze(-1) if self.n_heads == 1 else out


def sample_action(
    logits: torch.Tensor,
    n_sta: int,
    n_links: int,
    rng: np.random.Generator,
    deterministic: bool = False,
) -> Tuple[Dict, Dict[str, np.ndarray], float]:
    """Categorical sampling: per-STA link + map_mode."""
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
    """Batched log-prob recomputation (for the PPO ratio). logits: (B, action_dim)."""
    link_logits = logits[..., : n_sta * n_links].view(-1, n_sta, n_links)
    map_logits = logits[..., n_sta * n_links : n_sta * n_links + N_MAP_MODES]

    link_log_probs = F.log_softmax(link_logits, dim=-1)
    sel = selected_links.unsqueeze(-1)
    sta_logp = link_log_probs.gather(-1, sel).squeeze(-1).sum(-1)

    map_log_probs = F.log_softmax(map_logits, dim=-1)
    map_logp = map_log_probs.gather(-1, map_modes.unsqueeze(-1)).squeeze(-1)

    return sta_logp + map_logp
