"""WLAN MARL environment: N_AP x N_STA x K MLO links + dual UHR constraints.

Implements the Constrained MDP of formulation Section II:
- State (eq. 1): CSI, queue, HOL age, CBR, virtual queue per AP/STA/link
- Action (eq. 2): MLO link distribution, MAP mode, EDCA parameters, A-MPDU,
  link activation
- Dual UHR constraints (eq. 4): two virtual queues Z_j^(99), Z_j^(99.9)
- Shaped reward (eq. 8): Lyapunov-style penalty applied

Multi-AP coordination model:
- map_mode 0 (none): each AP contends independently
- map_mode 1 (Co-TDMA): only one AP transmits per slot, round-robin
- map_mode 2 (Co-OFDMA): all APs transmit at once, rate split by 1/N_AP
- The mode is decided by AP majority vote (cooperative scenario assumption)

OBSS interference model:
- If several APs are active on the same link k, the effective rate is split
- OBSS_i of formulation eq. (9) is stored in obs.cbr as an estimate (refreshed
  in the following step)

Gym-style external API, but dict-of-dicts (per-AP obs/action) for MARL.
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
    # Usable link indices per AP. None means all links are used (symmetric topology).
    # e.g. [[0,1],[0,1],[1,2],[2]] -> AP0/1 use links 0/1, AP2 uses 1/2, AP3 only 2.
    # In an asymmetric topology the OBSS overlap differs per AP, so the
    # interference-weighted FedAvg weights of eq. (10) form a clearly non-uniform
    # pattern.
    ap_link_sets: list | None = None
    # If True, env.reset() does not reinitialize the Lyapunov virtual queue Z_j^(l).
    # Z then accumulates across rollouts and the policy learns a long-horizon
    # Lyapunov signal. Baseline evaluation keeps the default False (clean Z start
    # for every seed).
    persistent_z: bool = False
    # Upper bound Z_max of the virtual queue Z. Part of the formal definition of
    # the proposed method (eq:vqueue), hence always active; it blocks the policy
    # collapse caused by Z divergence in the saturation regime (the penalty term
    # overwhelms the throughput term and the policy converges to a "minimize
    # transmission" attractor). The default 100 is the same order as the drift
    # constant B. In the feasible regime Z never reaches Z_max, so results are
    # unchanged; setting it to None gives an ablation mode without the cap (to
    # reproduce the saturation collapse).
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

        # Usable-link mask per AP (channel set K_i defined in eq. 1).
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
        # P1: MLD upper-MAC shared buffer -- arrivals accumulate in the per-STA
        # buffer and the transmit link is decided by the link selection at service
        # time (removes per-link queue stranding).
        # self.queues is kept exclusively for per-(STA,link) EDCA contention state.
        self.buffers: List[Deque[Packet]] = [
            deque() for _ in range(self.n_sta_total)
        ]
        # P2: append-only record of arrival times in arrival (FIFO) order. The
        # crossed pointer advances "count once at the moment the deadline is
        # passed" in O(1) amortized time.
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
        # Z is preserved or reset according to cfg.persistent_z.
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
        # 1. Poisson arrivals: load BE (AC=2) packets for every STA into the
        #    per-STA shared buffer (P1: the link is decided at transmit time --
        #    MLD upper-MAC queue model).
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

        # 1b. Enforce the per-AP link mask -- reflects the K_i definition of
        #     formulation eq. (1). Forces link_active to 0 for links an AP cannot
        #     use (in-place).
        for ap_id in range(self.cfg.n_ap):
            la = np.asarray(actions[ap_id]["link_active"])
            actions[ap_id]["link_active"] = (
                la * self.ap_link_mask[ap_id][np.newaxis, :]
            ).astype(np.int32)

        # 2. Install EDCA parameters (action -> queue).
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

        # 3. MAP mode: the global mode is decided by majority vote (cooperative).
        map_modes = [int(actions[a]["map_mode"]) for a in range(self.cfg.n_ap)]
        global_map_mode = max(set(map_modes), key=map_modes.count)

        # 4. Per-link OBSS overlap: number of APs active on each link k.
        ap_link_active = np.zeros((self.cfg.n_ap, self.cfg.n_links))
        for ap_id in range(self.cfg.n_ap):
            la = np.asarray(actions[ap_id]["link_active"])  # (n_sta_per_ap, n_links)
            md = np.asarray(actions[ap_id]["mlo_dist"])
            # Whether the AP uses link k: used if any STA is active with positive prob.
            ap_link_active[ap_id] = (la.sum(axis=0) * (md.sum(axis=0) > 0)).astype(float)
        per_link_active_aps = (ap_link_active > 0).astype(int).sum(axis=0)
        per_link_active_aps = np.maximum(per_link_active_aps, 1)
        self.cbr = ap_link_active / max(self.n_sta_total / self.cfg.n_ap, 1)

        # 5. Co-TDMA case: only one AP may transmit per slot.
        if global_map_mode == 1:
            tdma_winner = self.t_step % self.cfg.n_ap
        else:
            tdma_winner = None

        # 6. Bianchi contention per AP x link.
        rewards = {ap_id: 0.0 for ap_id in range(self.cfg.n_ap)}
        latencies_collected: List[float] = []
        # P2/P3: per-STA violation events in this slot (late service + age exceeded).
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
                # 6a. Collect the contending STAs of this AP-link and their attempt probs.
                contenders: List[Tuple[int, float]] = []
                for j in sta_ids:
                    if not self.buffers[j]:
                        # P1: no packet to send -> does not join the contention.
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

                # 6b. Sample whether each STA attempts a transmission.
                attempts = []
                for j, tau in contenders:
                    if self.rng.random() < tau:
                        attempts.append(j)
                if not attempts:
                    continue
                if len(attempts) >= 2:
                    # Collision: double the CW of every attempting STA.
                    for j in attempts:
                        self.queues[j][k].on_collision(ac=2)
                    continue

                # 6c. Single transmitter: serve up to A-MPDU, scaled by OBSS + Co-OFDMA.
                j = attempts[0]
                local_idx = j - ap_id * self.cfg.n_sta_per_ap
                ampdu_len = int(np.asarray(ap_act["ampdu_len"][local_idx])[k])
                obss_factor = 1.0 / per_link_active_aps[k]
                co_ofdma_factor = (
                    1.0 / self.cfg.n_ap if global_map_mode == 2 else 1.0
                )
                # rate-aware: the better the csi, the more is served.
                rate_ratio = float(
                    self.channel.rate_bps(self.csi[j, k:k + 1])[0]
                    / self.channel.rate_bps(np.array([self.channel.n_states - 1]))[0]
                )
                effective_n = max(
                    1,
                    int(round(ampdu_len * rate_ratio * obss_factor * co_ofdma_factor)),
                )
                # P1: FIFO service from the shared buffer -- link k is only the
                # transmission means, packets are not bound to a link. Late-service
                # violations are recorded in _v_slot by _serve_buffer (including
                # double-count prevention).
                lats, bits = self._serve_buffer(j, effective_n)
                if bits > 0:
                    self.queues[j][k].on_success(ac=2)
                    latencies_collected.extend(lats.tolist())
                    self._packets_per_ap[ap_id] += int(lats.size)
                    rewards[ap_id] += self.cfg.V * (
                        float(bits) / self.cfg.pkt_size_bits
                    )

        # 6b. P2: age-exceeded accounting -- regardless of service, one violation
        #     at the moment the in-queue age exceeds L^(l) (FIFO monotonicity ->
        #     pointer advance).
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

        # 6c. P3: Lyapunov penalty + Z update -- v includes stranding (age
        #     exceeded). The penalty uses Z(t) before the update (standard
        #     drift-plus-penalty), and the allowance is proportional to the number
        #     of packets 'decided' in this slot.
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

        # 7. Channel evolution + time advance.
        self.csi = self.channel.step(self.csi)
        self.t_step += 1
        self.t += self.cfg.slot_dt_s

        done = self.t_step >= self.cfg.horizon
        dones = {ap_id: done for ap_id in range(self.cfg.n_ap)}
        # P2: denominator = 'decided' packets (served, or confirmed age-exceeded).
        # Replaces the previous accounting that dropped stranded packets from the
        # survivor statistics.
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
            # per-AP virtual queue pressure (dual-variable measure of zq/hybrid aggregation).
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
        """P1: serve at most n_pkts packets FIFO from the per-STA shared buffer.

        P2: a late service (lat > L^(l)) at an index before the crossed pointer has
        already been counted as age-exceeded, so it is not counted twice. The served
        range advances the pointer to 'decided' for both percentiles.
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
            # P1: expose the shared buffer length / HOL age identically across all
            # link columns (obs shape kept: (n_sta, n_links, N_AC) / (n_sta, n_links)).
            buf_len = np.array(
                [len(self.buffers[j]) for j in sta_ids], dtype=np.float32
            )
            ap_q = np.zeros(
                (len(sta_ids), self.cfg.n_links, N_AC), dtype=np.float32
            )
            ap_q[:, :, 2] = buf_len[:, None]  # placed in the BE (AC=2) column
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
                # P7: the shared (federated) actor must observe its own AP's link
                # set K_i for conditional specialization to be possible in an
                # asymmetric topology. Without it the shared actor is link-set
                # blind (only the none arm benefits).
                "link_mask": self.ap_link_mask[ap_id].astype(np.float32),
            }
        return obs
