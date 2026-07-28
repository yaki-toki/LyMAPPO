"""Finite-state Markov CSI channel per Assumption A1 of the formulation.

Per-link CSI h_{j,k}(t) lives on a finite state space H = {0, ..., n_states-1}
representing quantized SNR levels. Transition kernel P_h has a "stay-or-jump"
structure (stay with `persistence`, uniform over remaining states otherwise);
this gives a doubly stochastic matrix with uniform stationary distribution mu_h
and finite mixing time tau_mix = O(1 / (1 - persistence)).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class MarkovChannel:
    n_states: int = 8
    persistence: float = 0.7
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        if not (0.0 < self.persistence < 1.0):
            raise ValueError("persistence must be in (0, 1)")
        off = (1.0 - self.persistence) / (self.n_states - 1)
        P = np.full((self.n_states, self.n_states), off)
        np.fill_diagonal(P, self.persistence)
        self.P = P
        self.mu = np.full(self.n_states, 1.0 / self.n_states)

    @property
    def tau_mix(self) -> float:
        # 1 / (1 - lambda_2). For our doubly stochastic stay-or-jump kernel
        # the second eigenvalue is `persistence - off`.
        off = (1.0 - self.persistence) / (self.n_states - 1)
        gap = 1.0 - (self.persistence - off)
        return 1.0 / max(gap, 1e-12)

    def sample_initial(self, shape: tuple) -> np.ndarray:
        return self.rng.choice(self.n_states, size=shape, p=self.mu).astype(np.int32)

    def step(self, h: np.ndarray) -> np.ndarray:
        flat = h.ravel()
        out = np.empty_like(flat)
        for s in range(self.n_states):
            mask = flat == s
            n_mask = int(mask.sum())
            if n_mask:
                out[mask] = self.rng.choice(self.n_states, size=n_mask, p=self.P[s])
        return out.reshape(h.shape).astype(np.int32)

    def snr_db(self, h: np.ndarray) -> np.ndarray:
        return 3.0 * h.astype(np.float64)

    def rate_bps(self, h: np.ndarray, bandwidth_hz: float = 20e6) -> np.ndarray:
        snr_lin = 10.0 ** (self.snr_db(h) / 10.0)
        return bandwidth_hz * np.log2(1.0 + snr_lin)
