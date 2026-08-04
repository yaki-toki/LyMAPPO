"""Example custom policy: per-STA greedy link score = CSI − w·CBR.

Demonstrates the plug-in API surface (``models/policy_api.py``): reading the
per-AP obs dicts, honoring the allowed-link mask, and returning a
``PolicyAction``. Run it against the built-in baselines with:

    python3 feddrl.py --seed 0 \
        --policy examples.policies.greedy_score.GreedyScorePolicy \
        ... (same protocol flags as any evaluation run)

This is a teaching example, not a strong baseline: it balances link quality
(CSI) against congestion (CBR) with a fixed weight and no hysteresis, so it
can flap between links under a fast-varying channel.
"""
from __future__ import annotations

from typing import List

import numpy as np

from models.policy_api import Policy, PolicyAction


class GreedyScorePolicy(Policy):
    """Pick, per STA, the allowed link maximizing ``csi − cbr_weight·cbr``."""

    def __init__(self, cbr_weight: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cbr_weight = float(cbr_weight)

    def act(self, obs: List[dict], link_masks: np.ndarray) -> PolicyAction:
        links = np.zeros((self.n_ap, self.n_sta_per_ap), dtype=np.int64)
        for ap in range(self.n_ap):
            csi = np.asarray(obs[ap]["csi"], dtype=np.float64)        # (n_sta, n_links)
            cbr = np.asarray(obs[ap]["cbr"], dtype=np.float64)        # (n_links,)
            score = csi - self.cbr_weight * cbr[None, :]
            # Disallowed links must never win the argmax.
            score[:, link_masks[ap] <= 0] = -np.inf
            links[ap] = np.argmax(score, axis=1)
        return PolicyAction(selected_links=links, map_mode=0)
