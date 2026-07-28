"""IEEE 802.11 EDCA - 4 access categories (VO, VI, BE, BK).

Every (STA, link) pair holds 4 AC queues. The EDCA parameters (CWmin, CWmax,
AIFSN, TXOP) are exposed as the eq. (2) action variables of the formulation.
The defaults are the IEEE 802.11-2020 standard EDCA parameters.

This module provides an approximate Bianchi-style contention model: in each slot
a transmission is attempted per AC with probability tau = 2 / (cw + 1); on a
collision CW is doubled (saturating at cwmax) and on success it is reset to cwmin.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple

import numpy as np


AC_NAMES = ("VO", "VI", "BE", "BK")
N_AC = 4

DEFAULT_CWMIN = (3, 7, 15, 15)
DEFAULT_CWMAX = (7, 15, 1023, 1023)
DEFAULT_AIFSN = (2, 2, 3, 7)
DEFAULT_TXOP_US = (1504.0, 3008.0, 0.0, 0.0)


@dataclass
class Packet:
    ac: int
    arrive_t: float
    size_bits: int
    deadline_t: float


@dataclass
class EDCAQueue:
    """Per (STA, link) EDCA buffer set with 4 ACs."""

    cwmin: np.ndarray = field(
        default_factory=lambda: np.array(DEFAULT_CWMIN, dtype=np.int32)
    )
    cwmax: np.ndarray = field(
        default_factory=lambda: np.array(DEFAULT_CWMAX, dtype=np.int32)
    )
    aifsn: np.ndarray = field(
        default_factory=lambda: np.array(DEFAULT_AIFSN, dtype=np.int32)
    )
    txop_us: np.ndarray = field(
        default_factory=lambda: np.array(DEFAULT_TXOP_US, dtype=np.float64)
    )
    queues: List[Deque[Packet]] = field(
        default_factory=lambda: [deque() for _ in range(N_AC)]
    )
    cw: np.ndarray = field(
        default_factory=lambda: np.array(DEFAULT_CWMIN, dtype=np.int32)
    )
    backoff: np.ndarray = field(default_factory=lambda: np.zeros(N_AC, dtype=np.int32))

    def enqueue(self, pkt: Packet) -> None:
        self.queues[pkt.ac].append(pkt)

    def queue_lengths(self) -> np.ndarray:
        return np.array([len(q) for q in self.queues], dtype=np.int32)

    def hol_age(self, t: float) -> np.ndarray:
        out = np.zeros(N_AC, dtype=np.float64)
        for a, q in enumerate(self.queues):
            if q:
                out[a] = t - q[0].arrive_t
        return out

    def apply_action(
        self,
        cwmin: np.ndarray,
        cwmax: np.ndarray,
        aifsn: np.ndarray,
        txop_us: np.ndarray,
    ) -> None:
        self.cwmin = np.clip(cwmin, 1, 1023).astype(np.int32)
        self.cwmax = np.clip(cwmax, self.cwmin, 1023).astype(np.int32)
        self.aifsn = np.clip(aifsn, 1, 15).astype(np.int32)
        self.txop_us = np.clip(txop_us, 0.0, 5440.0).astype(np.float64)

    def serve(self, ac: int, n_pkts: int, t: float) -> Tuple[np.ndarray, int]:
        latencies = []
        bits = 0
        for _ in range(n_pkts):
            if not self.queues[ac]:
                break
            p = self.queues[ac].popleft()
            latencies.append(t - p.arrive_t)
            bits += p.size_bits
        return np.array(latencies, dtype=np.float64), bits

    def reset_cw(self, ac: int) -> None:
        self.cw[ac] = self.cwmin[ac]

    def clear(self) -> None:
        self.queues = [deque() for _ in range(N_AC)]
        self.cw = self.cwmin.copy()
        self.backoff.fill(0)

    # --------- Bianchi-style contention model ---------

    def bianchi_tx_prob(self, ac: int) -> float:
        """Per-slot transmission attempt probability tau = 2 / (cw + 1) (Bianchi 1998).

        A separate backoff state is held per AC. Returns 0 if the queue is empty.
        """
        if not self.queues[ac]:
            return 0.0
        return 2.0 / float(self.cw[ac] + 1)

    def tx_prob(self, ac: int) -> float:
        """tau for the shared-buffer (MLD upper-MAC) model -- no own-queue-empty gate.

        Since P1, the data lives in the per-STA shared buffer and this object holds
        only per-(STA,link) contention state, so checking for an empty buffer is the
        caller's responsibility.
        """
        return 2.0 / float(self.cw[ac] + 1)

    def on_success(self, ac: int) -> None:
        """Successful transmission -> reset CW to cwmin."""
        self.cw[ac] = self.cwmin[ac]

    def on_collision(self, ac: int) -> None:
        """Collision -> double CW (saturating at cwmax)."""
        new_cw = min(2 * int(self.cw[ac]), int(self.cwmax[ac]))
        self.cw[ac] = np.int32(new_cw)
