"""WLAN MARL 환경: N_AP × N_STA × K MLO 링크 + dual UHR 제약.

formulation Section II 의 Constrained MDP 를 구현한다:
- State (식 1): AP·STA·링크 별 CSI, queue, HOL age, CBR, virtual queue
- Action (식 2): MLO 링크 분포, MAP 모드, EDCA 파라미터, A-MPDU, 링크 활성화
- Dual UHR 제약 (식 4): 두 virtual queue Z_j^(99), Z_j^(99.9)
- Shaped reward (식 8): Lyapunov-style penalty 적용

Multi-AP coordination 모델:
- map_mode 0 (none): 각 AP 가 독립 contention
- map_mode 1 (Co-TDMA): 슬롯마다 round-robin 으로 1 개 AP 만 송신
- map_mode 2 (Co-OFDMA): 모든 AP 가 동시 송신, rate 를 1/N_AP 으로 분할
- 모드는 AP 다수결로 결정 (협조형 시나리오 가정)

OBSS 간섭 모델:
- 동일 링크 k 에서 여러 AP 가 active 하면 effective rate 를 분할
- formulation 식 (9) 의 OBSS_i 를 추정값으로 obs.cbr 에 저장 (사후 step 에서 갱신)

Gym-style 외부 API 이지만 MARL 을 위해 dict-of-dicts (AP 별 obs/action).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Tuple

import numpy as np

from sim.python.channel.markov import MarkovChannel
from sim.python.mac.edca import (
    DEFAULT_AIFSN,
    DEFAULT_CWMAX,
    DEFAULT_CWMIN,
    DEFAULT_TXOP_US,
    N_AC,
    EDCAQueue,
    Packet,
)


L_UHR_S = {99: 5e-3, 99.9: 10e-3}
EPS_UHR = {99: 1e-2, 99.9: 1e-3}
PERCENTILES = (99, 99.9)


@dataclass
class WLANConfig:
    n_ap: int = 4
    n_sta_per_ap: int = 5
    n_links: int = 3
    slot_dt_s: float = 9e-6
    horizon: int = 2000
    arrival_pps: float = 200.0
    pkt_size_bits: int = 12000
    bandwidth_hz: float = 20e6
    seed: int = 0
    V: float = 1.0
    # AP 별 사용 가능한 link 인덱스. None 이면 전체 link 사용 (대칭 토폴로지).
    # 예: [[0,1],[0,1],[1,2],[2]] -> AP0/1 은 link 0/1, AP2 는 1/2, AP3 는 2 만.
    # 비대칭 토폴로지에서 OBSS overlap 이 AP 별로 달라져 식 (10) 의
    # interference-weighted FedAvg 가중치가 명확한 비균등 패턴을 형성한다.
    ap_link_sets: list | None = None
    # True 시 env.reset() 에서 Lyapunov virtual queue Z_j^(l) 을 초기화하지 않음.
    # rollout 사이에도 Z 가 누적되어 정책이 long-horizon Lyapunov 신호를 학습.
    # baseline 평가는 default False (각 seed 마다 clean Z 시작) 를 유지.
    persistent_z: bool = False
    # 가상 큐 Z 의 상한 Z_max. 제안 기법 정식 정의 (eq:vqueue) 의 일부로
    # 항상 활성화되며, saturation regime 에서 Z 발산으로 인한 policy collapse
    # (페널티항이 throughput항을 압도하여 정책이 "전송 최소화" attractor 로
    # 수렴) 를 차단한다. 기본값 100 은 drift constant B 와 같은 차수.
    # feasible regime 에서는 Z 가 Z_max 에 도달하지 않으므로 결과 불변;
    # None 으로 설정하면 cap 없는 ablation 모드 (saturation collapse 재현용).
    z_clip: float | None = 100.0
    # Airtime-consistent contention (recalibration v2). None => v1 behaviour:
    # contention is per-AP (perfect spatial reuse), so network-wide link
    # concentration is not penalised — this is why surrogate-trained policies
    # concentrate links and fail to transfer to ns-3's shared per-link channels.
    # Set to G: partition the n_ap APs into G contention groups PER LINK; within
    # a group all selecting STAs share one channel (one winner per group/link/
    # slot). G = n_ap reproduces v1; G = 1 is a fully shared channel (ns-3
    # topology, no reuse); intermediate G models partial path-loss reuse. Calibrate
    # G so the surrogate's fixed-RSSI feasibility curve matches ns-3's.
    reuse_groups: int | None = None


class WLANEnv:
    """gym-style MARL env. step(actions) where actions: dict[ap_id -> action]."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: WLANConfig | None = None) -> None:
        self.cfg = cfg or WLANConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.channel = MarkovChannel(rng=self.rng)
        self.t_step = 0
        self.t = 0.0

        self.n_sta_total = self.cfg.n_ap * self.cfg.n_sta_per_ap
        self.sta_to_ap = np.repeat(np.arange(self.cfg.n_ap), self.cfg.n_sta_per_ap)

        # AP 별 사용 가능 링크 mask (식 1 의 channel set K_i 정의).
        self.ap_link_mask = np.ones(
            (self.cfg.n_ap, self.cfg.n_links), dtype=np.int32
        )
        if self.cfg.ap_link_sets is not None:
            assert len(self.cfg.ap_link_sets) == self.cfg.n_ap, (
                f"ap_link_sets must have {self.cfg.n_ap} entries"
            )
            self.ap_link_mask.fill(0)
            for ap_id, link_list in enumerate(self.cfg.ap_link_sets):
                for k in link_list:
                    if 0 <= k < self.cfg.n_links:
                        self.ap_link_mask[ap_id, k] = 1

        self.queues: List[List[EDCAQueue]] = [
            [EDCAQueue() for _ in range(self.cfg.n_links)]
            for _ in range(self.n_sta_total)
        ]
        # P1: MLD 상위-MAC 공유 버퍼 — 도착은 per-STA 버퍼에 쌓이고, 전송
        # 링크는 서비스 시점의 링크 선택이 결정한다 (링크별 큐 좌초 제거).
        # self.queues 는 per-(STA,link) EDCA contention 상태 전용으로 유지.
        self.buffers: List[Deque[Packet]] = [
            deque() for _ in range(self.n_sta_total)
        ]
        # P2: 도착 순서 (FIFO) 로 append-only 도착 시각 기록. crossed 포인터가
        # "deadline 을 넘긴 순간 1회 집계" 를 O(1) amortized 로 전진시킨다.
        self._arrive_ts: List[List[float]] = [
            [] for _ in range(self.n_sta_total)
        ]
        self._arrived = np.zeros(self.n_sta_total, dtype=np.int64)
        self._served = np.zeros(self.n_sta_total, dtype=np.int64)
        # v2 airtime model: step index until which each (link, contention-group)
        # channel is busy transmitting a won A-MPDU (TXOP occupancy). Only used
        # when cfg.reuse_groups is set; makes A-MPDU service airtime-consistent.
        self._link_busy_until = np.zeros(
            (self.cfg.n_links, self.cfg.n_ap), dtype=np.int64
        )
        self._crossed = {
            ell: np.zeros(self.n_sta_total, dtype=np.int64)
            for ell in PERCENTILES
        }
        self._decided = {
            ell: np.zeros(self.n_sta_total, dtype=np.int64)
            for ell in PERCENTILES
        }
        self.csi = self.channel.sample_initial((self.n_sta_total, self.cfg.n_links))
        self.cbr = np.zeros((self.cfg.n_ap, self.cfg.n_links))

        self.Z = {ell: np.zeros(self.n_sta_total) for ell in PERCENTILES}
        self._violation_count = {ell: 0 for ell in PERCENTILES}
        self._packet_count = 0
        self._packets_per_ap = np.zeros(self.cfg.n_ap, dtype=np.int64)

    def reset(self) -> Dict[int, Dict[str, np.ndarray]]:
        self.t_step = 0
        self.t = 0.0
        self.csi = self.channel.sample_initial((self.n_sta_total, self.cfg.n_links))
        self.cbr.fill(0.0)
        for q_per_sta in self.queues:
            for q in q_per_sta:
                q.clear()
        self.buffers = [deque() for _ in range(self.n_sta_total)]
        self._arrive_ts = [[] for _ in range(self.n_sta_total)]
        self._arrived.fill(0)
        self._served.fill(0)
        self._link_busy_until.fill(0)
        # Z 는 cfg.persistent_z 에 따라 보존 / 초기화.
        for ell in PERCENTILES:
            if not self.cfg.persistent_z:
                self.Z[ell].fill(0.0)
            self._violation_count[ell] = 0
            self._crossed[ell].fill(0)
            self._decided[ell].fill(0)
        self._packet_count = 0
        self._packets_per_ap.fill(0)
        return self._observations()

    def step(
        self, actions: Dict[int, Dict[str, Any]]
    ) -> Tuple[
        Dict[int, Dict[str, np.ndarray]],
        Dict[int, float],
        Dict[int, bool],
        Dict[str, float],
    ]:
        # 1. Poisson 도착: 모든 STA 에 BE (AC=2) 패킷을 per-STA 공유 버퍼로
        #    적재 (P1: 링크는 전송 시점에 결정 — MLD 상위-MAC 큐 모델).
        n_arr = self.rng.poisson(
            self.cfg.arrival_pps * self.cfg.slot_dt_s, size=self.n_sta_total
        )
        deadline = self.t + L_UHR_S[99.9]
        for j, n in enumerate(n_arr):
            self._arrived[j] += int(n)
            for _ in range(int(n)):
                self._arrive_ts[j].append(self.t)
                self.buffers[j].append(
                    Packet(
                        ac=2,
                        arrive_t=self.t,
                        size_bits=self.cfg.pkt_size_bits,
                        deadline_t=deadline,
                    )
                )

        # 1b. AP 별 link mask 강제 적용 — formulation 식 (1) 의 K_i 정의 반영.
        #     AP 가 사용 불가능한 link 의 link_active 를 0 으로 강제 (in-place).
        for ap_id in range(self.cfg.n_ap):
            la = np.asarray(actions[ap_id]["link_active"])
            actions[ap_id]["link_active"] = (
                la * self.ap_link_mask[ap_id][np.newaxis, :]
            ).astype(np.int32)

        # 2. EDCA 파라미터 설치 (action -> queue).
        for ap_id, ap_action in actions.items():
            sta_ids = np.where(self.sta_to_ap == ap_id)[0]
            for j in sta_ids:
                for k in range(self.cfg.n_links):
                    self.queues[j][k].apply_action(
                        ap_action["edca_cwmin"],
                        ap_action["edca_cwmax"],
                        ap_action["edca_aifsn"],
                        ap_action["edca_txop_us"],
                    )

        # 3. MAP 모드: 다수결로 글로벌 모드 결정 (협조형).
        map_modes = [int(actions[a]["map_mode"]) for a in range(self.cfg.n_ap)]
        global_map_mode = max(set(map_modes), key=map_modes.count)

        # 4. 링크별 OBSS overlap 계산: 각 link k 에서 active 한 AP 수.
        ap_link_active = np.zeros((self.cfg.n_ap, self.cfg.n_links))
        for ap_id in range(self.cfg.n_ap):
            la = np.asarray(actions[ap_id]["link_active"])  # (n_sta_per_ap, n_links)
            md = np.asarray(actions[ap_id]["mlo_dist"])
            # AP 가 link k 를 사용하는지: STA 중 1 명이라도 active+positive prob 면 사용.
            ap_link_active[ap_id] = (la.sum(axis=0) * (md.sum(axis=0) > 0)).astype(float)
        per_link_active_aps = (ap_link_active > 0).astype(int).sum(axis=0)
        per_link_active_aps = np.maximum(per_link_active_aps, 1)
        self.cbr = ap_link_active / max(self.n_sta_total / self.cfg.n_ap, 1)

        # 5. Co-TDMA 의 경우: 슬롯마다 한 AP 만 송신 가능.
        if global_map_mode == 1:
            tdma_winner = self.t_step % self.cfg.n_ap
        else:
            tdma_winner = None

        # 6. AP × link 별 Bianchi contention.
        rewards = {ap_id: 0.0 for ap_id in range(self.cfg.n_ap)}
        latencies_collected: List[float] = []
        # P2/P3: 이번 슬롯의 per-STA 위반 이벤트 (늦은 서비스 + 나이 초과).
        self._v_slot = {
            ell: np.zeros(self.n_sta_total, dtype=np.int64)
            for ell in PERCENTILES
        }

        if self.cfg.reuse_groups is not None:
            # v2 recalibration: airtime-consistent shared-channel contention.
            self._contend_shared(
                actions, tdma_winner, global_map_mode, rewards,
                latencies_collected)
        for ap_id in range(self.cfg.n_ap):
            if self.cfg.reuse_groups is not None:
                break  # v2 path handled the whole slot above
            if tdma_winner is not None and ap_id != tdma_winner:
                continue
            sta_ids = np.where(self.sta_to_ap == ap_id)[0]
            ap_act = actions[ap_id]
            for k in range(self.cfg.n_links):
                # 6a. AP-link 의 contention 후보 STA 와 시도 확률 수집.
                contenders: List[Tuple[int, float]] = []
                for j in sta_ids:
                    if not self.buffers[j]:
                        # P1: 보낼 패킷이 없으면 contention 미참여.
                        continue
                    local_idx = j - ap_id * self.cfg.n_sta_per_ap
                    mlo_dist = np.asarray(ap_act["mlo_dist"][local_idx])
                    la = np.asarray(ap_act["link_active"][local_idx])
                    if la[k] == 0 or mlo_dist[k] <= 0:
                        continue
                    if int(np.argmax(mlo_dist * la)) != k:
                        continue
                    tau = self.queues[j][k].tx_prob(ac=2)
                    if tau > 0:
                        contenders.append((j, tau))
                if not contenders:
                    continue

                # 6b. 각 STA 가 송신 시도하는지 sample.
                attempts = []
                for j, tau in contenders:
                    if self.rng.random() < tau:
                        attempts.append(j)
                if not attempts:
                    continue
                if len(attempts) >= 2:
                    # 충돌: 모든 시도자 CW 두 배.
                    for j in attempts:
                        self.queues[j][k].on_collision(ac=2)
                    continue

                # 6c. 단일 송신자: A-MPDU 만큼 serve, OBSS + Co-OFDMA 비례 적용.
                j = attempts[0]
                local_idx = j - ap_id * self.cfg.n_sta_per_ap
                ampdu_len = int(np.asarray(ap_act["ampdu_len"][local_idx])[k])
                obss_factor = 1.0 / per_link_active_aps[k]
                co_ofdma_factor = (
                    1.0 / self.cfg.n_ap if global_map_mode == 2 else 1.0
                )
                # rate-aware: csi 가 좋을수록 더 많이 serve.
                rate_ratio = float(
                    self.channel.rate_bps(self.csi[j, k:k + 1])[0]
                    / self.channel.rate_bps(np.array([self.channel.n_states - 1]))[0]
                )
                effective_n = max(
                    1,
                    int(round(ampdu_len * rate_ratio * obss_factor * co_ofdma_factor)),
                )
                # P1: 공유 버퍼에서 FIFO serve — 링크 k 는 전송 수단일 뿐,
                # 패킷이 링크에 결박되지 않는다. 늦은 서비스 위반은
                # _serve_buffer 가 _v_slot 에 기록 (중복 집계 방지 포함).
                lats, bits = self._serve_buffer(j, effective_n)
                if bits > 0:
                    self.queues[j][k].on_success(ac=2)
                    latencies_collected.extend(lats.tolist())
                    self._packets_per_ap[ap_id] += int(lats.size)
                    rewards[ap_id] += self.cfg.V * (
                        float(bits) / self.cfg.pkt_size_bits
                    )

        # 6b. P2: 나이-초과 집계 — 서비스 여부와 무관하게 큐 내 나이가
        #     L^(l) 를 넘는 순간 위반 1회 (FIFO 단조성 -> 포인터 전진).
        for j in range(self.n_sta_total):
            ats = self._arrive_ts[j]
            n_arr_j = int(self._arrived[j])
            for ell in PERCENTILES:
                c = int(self._crossed[ell][j])
                lim = L_UHR_S[ell]
                while c < n_arr_j and self.t - ats[c] > lim:
                    self._v_slot[ell][j] += 1
                    c += 1
                self._crossed[ell][j] = c

        # 6c. P3: Lyapunov 페널티 + Z 갱신 — v 에 좌초 (나이 초과) 포함.
        #     penalty 는 갱신 전 Z(t) 를 사용 (drift-plus-penalty 표준),
        #     allowance 는 이번 슬롯의 '결정 완료' 패킷 수에 비례.
        for j in range(self.n_sta_total):
            ap_id_j = int(self.sta_to_ap[j])
            penalty = 0.0
            for ell in PERCENTILES:
                v_j = int(self._v_slot[ell][j])
                decided_new = max(
                    int(self._served[j]), int(self._crossed[ell][j])
                )
                delta_decided = decided_new - int(self._decided[ell][j])
                self._decided[ell][j] = decided_new
                if v_j:
                    penalty += float(self.Z[ell][j]) * v_j
                    self._violation_count[ell] += v_j
                if v_j or delta_decided:
                    self.Z[ell][j] += v_j - EPS_UHR[ell] * delta_decided
                    z_val = max(0.0, float(self.Z[ell][j]))
                    if self.cfg.z_clip is not None:
                        z_val = min(self.cfg.z_clip, z_val)
                    self.Z[ell][j] = z_val
            if penalty:
                rewards[ap_id_j] -= penalty

        self._packet_count += int(len(latencies_collected))

        # 7. 채널 진화 + 시간 증가.
        self.csi = self.channel.step(self.csi)
        self.t_step += 1
        self.t += self.cfg.slot_dt_s

        done = self.t_step >= self.cfg.horizon
        dones = {ap_id: done for ap_id in range(self.cfg.n_ap)}
        # P2: 분모 = '결정 완료' 패킷 (서비스됨 or 나이-초과 확정). 좌초
        # 패킷을 생존자 통계에서 빼던 기존 회계를 대체한다.
        decided_totals = {
            ell: int(np.maximum(self._served, self._crossed[ell]).sum())
            for ell in PERCENTILES
        }
        info = {
            "p99_violation_rate": (
                self._violation_count[99] / max(decided_totals[99], 1)
            ),
            "p99_9_violation_rate": (
                self._violation_count[99.9] / max(decided_totals[99.9], 1)
            ),
            "decided_99": decided_totals[99],
            "decided_99_9": decided_totals[99.9],
            "packets_arrived": int(self._arrived.sum()),
            "mean_Z_99": float(self.Z[99].mean()),
            "mean_Z_99_9": float(self.Z[99.9].mean()),
            # per-AP 가상 큐 압력 (zq/hybrid aggregation 의 dual-변수 measure).
            "Z_per_ap_99": [
                float(self.Z[99][self.sta_to_ap == ap].mean())
                for ap in range(self.cfg.n_ap)
            ],
            "Z_per_ap_99_9": [
                float(self.Z[99.9][self.sta_to_ap == ap].mean())
                for ap in range(self.cfg.n_ap)
            ],
            "packets_served": self._packet_count,
            "packets_per_ap": self._packets_per_ap.copy(),
            "global_map_mode": int(global_map_mode),
        }
        return self._observations(), rewards, dones, info

    def _contend_shared(
        self,
        actions: Dict[int, Dict[str, Any]],
        tdma_winner: int | None,
        global_map_mode: int,
        rewards: Dict[int, float],
        latencies_collected: List[float],
    ) -> None:
        """Airtime-consistent per-link contention (cfg.reuse_groups=G).

        Each link is one channel shared by a group of APs (no per-AP spatial
        reuse), matching ns-3 where all APs on a link share one YansWifiChannel.
        Network-wide link concentration raises the contender count per group/
        link -> more collisions -> less service, so a policy that piles STAs onto
        one link is penalised (as in ns-3). G=1 is a fully shared channel; larger
        G models partial path-loss reuse. Mutates rewards/latencies_collected.
        """
        n_ap = self.cfg.n_ap
        g = max(1, min(int(self.cfg.reuse_groups), n_ap))
        for k in range(self.cfg.n_links):
            for grp in range(g):
                if self.t_step < self._link_busy_until[k, grp]:
                    continue  # channel still busy transmitting a prior A-MPDU
                contenders: List[Tuple[int, int, float]] = []
                for ap_id in range(n_ap):
                    if (ap_id * g) // n_ap != grp:
                        continue
                    if tdma_winner is not None and ap_id != tdma_winner:
                        continue
                    ap_act = actions[ap_id]
                    for j in np.where(self.sta_to_ap == ap_id)[0]:
                        if not self.buffers[j]:
                            continue
                        local_idx = j - ap_id * self.cfg.n_sta_per_ap
                        mlo_dist = np.asarray(ap_act["mlo_dist"][local_idx])
                        la = np.asarray(ap_act["link_active"][local_idx])
                        if la[k] == 0 or mlo_dist[k] <= 0:
                            continue
                        if int(np.argmax(mlo_dist * la)) != k:
                            continue
                        tau = self.queues[j][k].tx_prob(ac=2)
                        if tau > 0:
                            contenders.append((int(j), ap_id, tau))
                if not contenders:
                    continue
                attempts = [
                    (j, ap_id) for (j, ap_id, tau) in contenders
                    if self.rng.random() < tau
                ]
                if not attempts:
                    continue
                if len(attempts) >= 2:
                    for j, _ap in attempts:
                        self.queues[j][k].on_collision(ac=2)
                    continue
                j, ap_id = attempts[0]
                local_idx = j - ap_id * self.cfg.n_sta_per_ap
                ampdu_len = int(
                    np.asarray(actions[ap_id]["ampdu_len"][local_idx])[k]
                )
                # Co-OFDMA splits the channel across APs => smaller per-AP burst.
                max_frames = max(1, int(round(
                    ampdu_len * (1.0 / n_ap if global_map_mode == 2 else 1.0)
                )))
                lats, bits = self._serve_buffer(j, max_frames)
                if bits > 0:
                    self.queues[j][k].on_success(ac=2)
                    latencies_collected.extend(lats.tolist())
                    self._packets_per_ap[ap_id] += int(lats.size)
                    rewards[ap_id] += self.cfg.V * (
                        float(bits) / self.cfg.pkt_size_bits
                    )
                    # Airtime occupancy: the won A-MPDU holds the channel for its
                    # transmission time. CSI enters HERE as rate (not as burst
                    # size), so lower CSI / bigger bursts => longer busy => less
                    # throughput — the physical coupling ns-3 has and v1 lacked.
                    rate_bps = float(
                        self.channel.rate_bps(self.csi[j, k:k + 1])[0]
                    )
                    tx_slots = int(np.ceil(
                        bits / max(rate_bps, 1.0) / self.cfg.slot_dt_s
                    ))
                    self._link_busy_until[k, grp] = self.t_step + max(1, tx_slots)

    def _serve_buffer(self, j: int, n_pkts: int) -> Tuple[np.ndarray, int]:
        """P1: per-STA 공유 버퍼에서 FIFO 로 최대 n_pkts 개 서비스.

        P2: 늦은 서비스 (lat > L^(l)) 는 crossed 포인터 이전 인덱스면 이미
        나이-초과로 집계된 패킷이므로 중복 집계하지 않는다. 서비스된 구간은
        두 percentile 모두 '결정 완료' 로 포인터를 전진시킨다.
        """
        buf = self.buffers[j]
        lats: List[float] = []
        bits = 0
        for _ in range(n_pkts):
            if not buf:
                break
            pkt = buf.popleft()
            idx = int(self._served[j])
            lat = self.t - pkt.arrive_t
            lats.append(lat)
            bits += pkt.size_bits
            self._served[j] += 1
            for ell in PERCENTILES:
                if idx >= self._crossed[ell][j] and lat > L_UHR_S[ell]:
                    self._v_slot[ell][j] += 1
        for ell in PERCENTILES:
            if self._crossed[ell][j] < self._served[j]:
                self._crossed[ell][j] = self._served[j]
        return np.array(lats, dtype=np.float64), bits

    def _observations(self) -> Dict[int, Dict[str, np.ndarray]]:
        obs: Dict[int, Dict[str, np.ndarray]] = {}
        for ap_id in range(self.cfg.n_ap):
            sta_ids = np.where(self.sta_to_ap == ap_id)[0]
            ap_csi = self.csi[sta_ids]
            # P1: 공유 버퍼 길이/HOL age 를 전 링크 열에 동일하게 노출
            # (obs shape 유지: (n_sta, n_links, N_AC) / (n_sta, n_links)).
            buf_len = np.array(
                [len(self.buffers[j]) for j in sta_ids], dtype=np.float32
            )
            ap_q = np.zeros(
                (len(sta_ids), self.cfg.n_links, N_AC), dtype=np.float32
            )
            ap_q[:, :, 2] = buf_len[:, None]  # BE (AC=2) 열에 배치
            hol = np.array(
                [
                    self.t - self.buffers[j][0].arrive_t
                    if self.buffers[j]
                    else 0.0
                    for j in sta_ids
                ],
                dtype=np.float32,
            )
            ap_hol = np.tile(hol[:, None], (1, self.cfg.n_links))
            obs[ap_id] = {
                "csi": ap_csi.astype(np.float32),
                "queue": ap_q.astype(np.float32),
                "hol_age": ap_hol.astype(np.float32),
                "cbr": self.cbr[ap_id].astype(np.float32),
                "Z_99": self.Z[99][sta_ids].astype(np.float32),
                "Z_99_9": self.Z[99.9][sta_ids].astype(np.float32),
                # P7: 공유 (연합) actor 가 자기 AP 의 링크 집합 K_i 를 관측
                # 해야 비대칭 토폴로지에서 조건부 특화가 가능하다. 이것이
                # 없으면 shared actor 는 링크-집합 blind (none arm 만 유리).
                "link_mask": self.ap_link_mask[ap_id].astype(np.float32),
            }
        return obs
