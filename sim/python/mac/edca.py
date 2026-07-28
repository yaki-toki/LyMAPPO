"""IEEE 802.11 EDCA - 4 액세스 카테고리 (VO, VI, BE, BK).

각 (STA, link) 쌍이 4 개의 AC 큐를 보유한다. EDCA 파라미터 (CWmin, CWmax,
AIFSN, TXOP) 는 formulation 의 식 (2) action 변수로 노출된다. 기본값은
IEEE 802.11-2020 표준 EDCA 파라미터.

이 모듈은 Bianchi-style 근사 contention 모델을 제공한다: 각 슬롯에서 AC
별로 transmission 시도 확률 tau = 2 / (cw + 1) 로 송신 시도, 충돌 발생 시
CW 를 두 배로 증가 (cwmax 에서 saturate), 성공 시 cwmin 으로 reset.
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

    # --------- Bianchi-style contention 모델 ---------

    def bianchi_tx_prob(self, ac: int) -> float:
        """슬롯 당 송신 시도 확률 tau = 2 / (cw + 1) (Bianchi 1998 모델).

        AC 별로 별도 backoff 상태 보유. 큐가 비어 있으면 0 반환.
        """
        if not self.queues[ac]:
            return 0.0
        return 2.0 / float(self.cw[ac] + 1)

    def tx_prob(self, ac: int) -> float:
        """공유-버퍼 (MLD 상위-MAC) 모델용 tau — 자체 큐 비어있음 게이트 없음.

        P1 이후 데이터는 per-STA 공유 버퍼에 있고 이 객체는 per-(STA,link)
        contention 상태만 보유하므로, 버퍼 비어있음 확인은 호출자 몫이다.
        """
        return 2.0 / float(self.cw[ac] + 1)

    def on_success(self, ac: int) -> None:
        """송신 성공 → CW 를 cwmin 으로 reset."""
        self.cw[ac] = self.cwmin[ac]

    def on_collision(self, ac: int) -> None:
        """충돌 발생 → CW 를 두 배 (cwmax 에서 saturate)."""
        new_cw = min(2 * int(self.cw[ac]), int(self.cwmax[ac]))
        self.cw[ac] = np.int32(new_cw)
