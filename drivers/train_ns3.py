"""train_ns3.py — ns-3-in-the-loop federated MAPPO trainer (redesign).

설계 전환: Python surrogate 를 제거하고 ns-3 를 직접 RL 환경으로 사용한다.
실제 동작 (802.11 PHY/MAC, packet-level DES) = ns-3, 학습 (PPO + federated
aggregation) = Python. 두 환경은 ns3-ai 셰어드 메모리로 연동한다.

기존 ``feddrl.py`` (eval-only driver) 와 동일한 EnvMsg/ActMsg 인터페이스,
동일한 obs 스키마 (csi/queue/hol_age/cbr/Z_99/Z_99_9/link_mask), 동일한
Lyapunov dual 갱신을 재사용한다. 새로 추가되는 것은 rollout 수집 + per-AP
reward 계산뿐이며, GAE/PPO/aggregation 은 ``models.mappo`` 를 100% 재사용한다.

MDP 정렬 (drift-plus-penalty):
    read k 의 EnvMsg 는 slot (k-1) — 즉 직전 action a_{k-1} — 의 served/violation
    을 담고 있다. 따라서
        g_{k-1} = (v_{k-1} - eps * served_{k-1}) / N_STA
        r_{k-1} = V * u_{k-1} - Z_{k-1} · g_{k-1}      (PRE-update dual)
        Z_k     = clip(Z_{k-1} + g_{k-1}, 0, z_clip)   (dual ascent)
    obs_k 에는 갱신된 Z_k 가 실린다. transition 은
        (obs_{k-1}[Z_{k-1}], a_{k-1}, r_{k-1}, obs_k[Z_k]) 로 정합한다.

액션 공간 v1 = link + mode:
    per-STA link 선택 (학습) + 단일 global map_mode (ActMsg 가 하나만 지원 →
    per-AP 샘플의 다수결로 실행). PPO ratio 는 per-AP 샘플 map_mode 로 계산하여
    일관성을 유지한다 (실행은 다수결). per-AP 개별 mode 는 struct 확장이 필요한
    v2 과제로 남긴다.

실행 (WSL Ubuntu, ns-3 example 디렉터리 안에서):
    PYTHONDONTWRITEBYTECODE=1 python3 -B train_ns3.py \
        --aggregation zq --seed 0 --arrival-pps 5500 \
        --iterations 200 --rollout 64 \
        --ckpt-out $REPO_ROOT/models/checkpoints/ns3_zq_s0.pt
"""
from __future__ import annotations

import argparse
import csv as _csv
import datetime as _dt
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

N_AP = 16
N_STA_PER_AP = 5
N_LINKS = 3
EPS_99 = 1e-2
EPS_999 = 1e-3
MACRO_SLOT_MS = 5.0  # feddrl_scenario.cc kMacroSlot 와 일치해야 함.


def _mask_link_logits(logits: Any, link_mask: Any, n_sta: int,
                      n_links: int) -> Any:
    """models.networks.mask_link_logits 위임 — 학습·평가가 단일 구현을 공유해야
    train/eval 행동분포가 일치한다 (P3 수정; 상세 docstring 은 networks.py)."""
    from models.networks import mask_link_logits  # type: ignore

    return mask_link_logits(logits, link_mask, n_sta, n_links)


def _mask_map_logits(logits: Any, n_sta: int, n_links: int,
                     n_map_modes: int) -> Any:
    """models.networks.mask_map_logits 위임 (P3 수정; 상세는 networks.py)."""
    from models.networks import mask_map_logits  # type: ignore

    return mask_map_logits(logits, n_sta, n_links, n_map_modes)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ns-3-in-the-loop federated MAPPO")
    p.add_argument("--aggregation", type=str, default="zq",
                   choices=["none", "uniform", "zq", "iw", "hybrid",
                            "qffl", "afl", "cluster"],
                   help="none=독립 local actor (FedAvg 없음); cluster=이질성-"
                        "인지 연합(같은 링크집합 K_i 그룹끼리만 uniform FedAvg "
                        "— cross-K 특화 파괴 진단에서 도출된 처방); 나머지는 "
                        "전역 가중 FedAvg 방식.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arrival-pps", type=float, default=5500.0)
    p.add_argument("--iterations", type=int, default=200,
                   help="PPO update 횟수 (각 update 는 --rollout slot 소비).")
    p.add_argument("--rollout", type=int, default=64,
                   help="update 당 매크로 슬롯 수.")
    p.add_argument("--allow-coord", action="store_true",
                   help="map_mode(Co-TDMA/OFDMA) 탐색 허용. 기본 off = mode0 "
                        "동결. 16-AP dense 에서 Co-TDMA 는 16× 직렬화로 백로그를 "
                        "폭발시켜 sim 을 붕괴시키고 지연에도 무용하므로 기본 동결.")
    p.add_argument("--macro-slot-ms", type=float, default=20.0,
                   help="RL 결정+측정 창 길이(ms). 길수록 슬롯당 패킷 수가 많아 "
                        "reward/Z 분산이 낮아진다. ns-3 scenario 로 전달된다.")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lyapunov-v", type=float, default=1.0,
                   help="drift-plus-penalty 의 utility 가중 V.")
    p.add_argument("--eps99", type=float, default=1e-2,
                   help="99% 지연 위반율 목표 (P(delay>deadline99) ≤ eps99). 목표가 "
                        "달성가능 영역이어야 Z 가 이질적으로 살아있어 zq 가중이 "
                        "의미를 갖는다 (Slater 가정 A5).")
    p.add_argument("--eps999", type=float, default=1e-3,
                   help="99.9% 지연 위반율 목표 (P(delay>deadline999) ≤ eps999).")
    p.add_argument("--deadline99-ms", type=float, default=5.0,
                   help="p99 데드라인(ms). 밀도-스케일: 4-AP=5, 16-AP dense=10 "
                        "(구조적 tail 바닥이 밀도와 함께 커짐). ns-3 로 전달.")
    p.add_argument("--deadline999-ms", type=float, default=10.0,
                   help="p99.9 데드라인(ms). 4-AP=10, 16-AP dense=20. ns-3 로 전달.")
    p.add_argument("--load-spread", type=float, default=0.0,
                   help="AP별 부하 이질성 [0,1]. 0=동종(전 AP arrivalPps). >0 이면 "
                        "AP0=arrivalPps ~ AP15=(1-load_spread)*arrivalPps 선형 gradient "
                        "→ per-AP 제약압력(Z) 이질화(연합/zq 검증 조건). ns-3 로 전달. "
                        "평가도 동일 값이어야 함.")
    p.add_argument("--seg-suffix", type=str, default="",
                   help="ns3-ai 공유메모리 객체 이름의 런별 고유 suffix. 동시 실행 "
                        "학습 잡마다 다른 값을 주어 세그먼트 충돌을 막는다(병렬화). "
                        "빈 값=라이브러리 기본(단일 실행). ns-3 와 이름이 일치해야 함.")
    p.add_argument("--chan-dwell-ms", type=float, default=200.0,
                   help="F6 Markov 채널 평균 상태 체류(ms); 0=정적(레거시). "
                        "ns-3 로 전달. 평가(feddrl.py)와 동일 값이어야 함.")
    p.add_argument("--het-bands", type=int, default=1,
                   help="F7 이질 밴드(2.4/5/6GHz, 20/40/80MHz); 0=레거시. "
                        "ns-3 로 전달. 평가와 동일 값이어야 함.")
    p.add_argument("--link2-width", type=int, default=80,
                   help="link2(6GHz) 채널폭 MHz: 80(기본)|40. ns-3 전달, 평가 일치 필수.")
    p.add_argument("--drain-target", type=int, default=8,
                   help="F8 공유버퍼 drain-on-demand per-link MAC 큐 목표 깊이; "
                        "0=direct-send 레거시. ns-3 로 전달, 평가와 동일 필수.")
    p.add_argument("--price-beta", type=float, default=0.0,
                   help="가격-결합(PX): 이웃 AP 의 dual(Z99+Z999)을 밴드별 "
                        "혼잡가격으로 보상에 결합 — r_i -= beta * sum_k "
                        "usage_{i,k} * price_k^{-i}. 네트워크 라그랑지안의 "
                        "교차항(공유밴드 airtime 외부효과) 1차 복원. 0=off.")
    p.add_argument("--quantiles", type=int, default=0,
                   help="QR critic(B): per-AP head 당 분위수 개수(0=스칼라 "
                        "레거시). 분위 회귀(pinball)로 return 분포를 학습.")
    p.add_argument("--cvar-alpha", type=float, default=0.25,
                   help="QR critic 사용 시 actor advantage 의 baseline 을 "
                        "하위 alpha 분위 평균(CVaR)으로 — 최악-경로 return "
                        "(tail 위반 스파이크 지배)을 우선 개선.")
    p.add_argument("--no-lyapunov", action="store_true",
                   help="ablation: Lyapunov 제약 결합 제거 — 보상=V·u 만(Z 벌점 "
                        "없음), obs 의 Z 특징 0 고정. 'Z 메커니즘이 UHR 달성의 "
                        "필수 요소인가'를 분리 검정 (리뷰어 필수 질문).")
    p.add_argument("--z-clip", type=float, default=10.0,
                   help="dual 변수 Z 상한. rate-form drift(≤1/슬롯) 기준 10 이면 "
                        "충분하며 reward 스케일을 학습가능 범위로 유지한다.")
    p.add_argument("--shared-actor", action="store_true",
                   help="#7 MAPPO baseline: ONE actor shared across all 16 APs "
                        "(parameter sharing), trained on POOLED experience with the "
                        "same drift-plus-penalty reward. Distinct from FedAvg "
                        "(averages 16 per-AP actors). Mutually exclusive with --cpo.")
    p.add_argument("--cpo", action="store_true",
                   help="#7 constrained-PPO (PID-Lagrangian) baseline: 16 "
                        "independent per-AP actors + a conventional per-AP "
                        "Lagrange multiplier λ updated per-iteration on the "
                        "episodic mean UHR violation (NOT our per-slot clipped "
                        "drift; λ not observed by the policy). Distinct constraint "
                        "mechanism. Mutually exclusive with --shared-actor.")
    p.add_argument("--cpo-kp", type=float, default=0.25,
                   help="CPO PID proportional gain (Stooke et al. 2020).")
    p.add_argument("--cpo-ki", type=float, default=1.0,
                   help="CPO PID integral gain (= classic dual-ascent step).")
    p.add_argument("--cpo-kd", type=float, default=0.0,
                   help="CPO PID derivative gain (0 = PI-Lagrangian).")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--eps-clip", type=float, default=0.2)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=1e-3)
    p.add_argument("--agg-period", type=int, default=1,
                   help="FedAvg communication period: aggregate + broadcast the "
                        "per-AP local actors every N iterations (each iteration = "
                        "1 rollout + args.epochs local PPO epochs). 1 = every "
                        "round. Ignored for aggregation=none (never aggregates).")
    p.add_argument("--episode-ms", type=float, default=0.0,
                   help="0 이면 iterations·rollout 에서 자동 산출.")
    p.add_argument("--ns3-path", type=str, default="../../../../")
    p.add_argument("--repo-root", type=str, default=os.environ.get("REPO_ROOT", "."))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--ckpt-out", type=str, required=True)
    p.add_argument("--train-log-csv", type=str, default=None)
    return p


def _ns3_obs_and_reward(
    env_msg: Any,
    z99: np.ndarray,
    z999: np.ndarray,
    v_weight: float,
    z_clip: float,
    eps99: float,
    eps999: float,
    masks: np.ndarray,
    no_lyap: bool = False,
    cpo: bool = False,
    lam99: np.ndarray | None = None,
    lam999: np.ndarray | None = None,
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """EnvMsg → (per-AP obs dict 리스트, reward, 갱신 Z99, 갱신 Z999, served,
    v99, v999).

    reward 는 PRE-update dual (z99/z999 인자) 로, obs 의 Z 는 POST-update 로
    채운다 (drift-plus-penalty 정렬). 모든 카운트는 per-STA 크기로 정규화한다
    (feddrl.py / wlan_env.step 와 동일).
    """
    queue_len = np.frombuffer(env_msg.queueLen(), dtype=np.uint32).reshape(
        N_AP, N_STA_PER_AP).astype(np.float32)
    hol_us = np.frombuffer(env_msg.holUs(), dtype=np.uint32).reshape(
        N_AP, N_STA_PER_AP).astype(np.float32)
    cbr = np.frombuffer(env_msg.cbr(), dtype=np.uint8).reshape(
        N_AP, N_LINKS).astype(np.float32) / 255.0
    served = np.frombuffer(env_msg.served(), dtype=np.uint32).astype(np.float32)
    v99 = np.frombuffer(env_msg.violation99(), dtype=np.uint32).astype(np.float32)
    v999 = np.frombuffer(env_msg.violation999(), dtype=np.uint32).astype(np.float32)
    dropped = np.frombuffer(env_msg.dropped(), dtype=np.uint32).astype(np.float32)
    # CSI: ns-3 물리 채널이 소유·측정한 per-STA-per-link 품질을 obs 로 받는다
    # (해석식 재유도 없음). shape (N_AP, N_STA_PER_AP, N_LINKS).
    csi_all = np.frombuffer(env_msg.csi(), dtype=np.float32).reshape(
        N_AP, N_STA_PER_AP, N_LINKS)

    # 제약은 rate 형 (eq:uhr): P(delay>L) ≤ eps, per-slot 분모 decided =
    # served + dropped (dropped = MaxDelay=deadline999 aged-out + queue-full;
    # ns-3 가 EnvMsg.dropped 로 보고). 드롭 패킷은 두 deadline 을 모두 놓친
    # 위반이므로 v 에 포함 — eq:uhr 의 "decided = served or aged out" 를
    # per-slot 에서 성립시켜 dual 신호의 survivor-bias 를 제거한다 (E1 수정).
    # rate 형 drift 는 O(1) 유계로 카운트형(포화 시 Z 폭발)보다 안정적이며
    # per-패킷 rate 제약에 충실하다 (논문 eq:queue 를 rate 형으로 재서술 예정).
    decided = np.maximum(served + dropped, 1.0)
    g99 = (v99 + dropped) / decided - eps99
    g999 = (v999 + dropped) / decided - eps999
    utility = served / N_STA_PER_AP            # delivered throughput (drop 무이득)
    if no_lyap:
        # ablation: 제약 결합 제거 — 순수 throughput 보상. Z 는 진단용으로만
        # 갱신하고 보상·obs 에서 배제 (obs 는 아래에서 0 고정).
        reward = v_weight * utility
    elif cpo:
        # CPO / constrained-PPO baseline (#7): conventional Lagrangian on the
        # SAME UHR constraint but with a per-AP multiplier λ that is (a) FIXED
        # within the rollout and updated per-ITERATION by a PID controller on
        # the episodic mean violation (NOT the per-slot clipped drift of ours),
        # and (b) NOT part of the observation (obs Z 특징 0 고정, no_lyap 와
        # 동일). Per-slot penalty uses the UN-normalized drift g (same scale as
        # our reward, so λ·g ~ O(utility) at a violation and the critic target
        # stays well-conditioned); λ itself adapts on the budget-NORMALIZED
        # aggregate violation (train loop) so the p99/p999 duals both reach a
        # useful magnitude despite the per-iteration (slow) update cadence. This
        # is textbook Lagrangian-PPO, distinct from our rate-form drift-plus-
        # penalty (Z per-slot, clipped, observed by the policy).
        reward = v_weight * utility - (lam99 * g99 + lam999 * g999)
    else:
        reward = v_weight * utility - (z99 * g99 + z999 * g999)
    new_z99 = np.clip(z99 + g99, 0.0, z_clip)
    new_z999 = np.clip(z999 + g999, 0.0, z_clip)
    zero_z = no_lyap or cpo

    obs_per_ap: List[Dict[str, np.ndarray]] = []
    for ap in range(N_AP):
        queue = np.zeros((N_STA_PER_AP, N_LINKS, 4), dtype=np.float32)
        queue[:, 0, 0] = queue_len[ap]
        hol_age = np.zeros((N_STA_PER_AP, N_LINKS), dtype=np.float32)
        hol_age[:, 0] = hol_us[ap] * 1e-6
        csi = csi_all[ap]
        obs_per_ap.append({
            "csi": csi,
            "queue": queue,
            "hol_age": hol_age,
            "cbr": cbr[ap],
            "Z_99": (np.zeros(N_STA_PER_AP, dtype=np.float32) if zero_z
                     else np.full(N_STA_PER_AP, new_z99[ap],
                                  dtype=np.float32)),
            "Z_99_9": (np.zeros(N_STA_PER_AP, dtype=np.float32) if zero_z
                       else np.full(N_STA_PER_AP, new_z999[ap],
                                    dtype=np.float32)),
            "link_mask": masks[ap],
        })
    return obs_per_ap, reward, new_z99, new_z999, served, v99, v999, dropped


def collect_rollout_ns3(
    msg: Any,
    actor: Any,
    actors_local: List[Any] | None,
    rollout: int,
    rng: np.random.Generator,
    device: str,
    z99: np.ndarray,
    z999: np.ndarray,
    v_weight: float,
    z_clip: float,
    eps99: float,
    eps999: float,
    masks: np.ndarray,
    encode_obs: Any,
    sample_action: Any,
    n_map_modes: int,
    allow_coord: bool = False,
    no_lyap: bool = False,
    price_beta: float = 0.0,
    cpo: bool = False,
    lam99: np.ndarray | None = None,
    lam999: np.ndarray | None = None,
) -> Dict[str, Any]:
    """ns-3 스트림에서 rollout 개의 (state, action, reward) transition 을 모은다.

    rollout+1 회 read 하여 마지막 read 는 GAE bootstrap obs 로만 쓴다. reward
    r_{k-1} 은 read k 에서 계산되어 직전 read 의 action 에 귀속된다.
    """
    import torch  # type: ignore

    is_local = actors_local is not None
    states: List[List[np.ndarray]] = [[] for _ in range(N_AP)]
    sel_links: List[List[np.ndarray]] = [[] for _ in range(N_AP)]
    map_modes: List[List[int]] = [[] for _ in range(N_AP)]
    log_probs: List[List[float]] = [[] for _ in range(N_AP)]
    rewards: List[List[float]] = [[] for _ in range(N_AP)]
    joints: List[np.ndarray] = []
    last_link_active: Dict[int, Dict[str, np.ndarray]] = {}
    served_acc = np.zeros(N_AP, dtype=np.float64)
    v99_acc = np.zeros(N_AP, dtype=np.float64)
    v999_acc = np.zeros(N_AP, dtype=np.float64)
    dropped_acc = np.zeros(N_AP, dtype=np.float64)
    finished = False
    # 가격-결합(PX): 직전 슬롯 액션의 밴드별 사용비중 (reward r_{k-1} 에 귀속).
    prev_usage = np.zeros((N_AP, N_LINKS), dtype=np.float32)

    for k in range(rollout + 1):
        # PRE-update dual 가격 (Z·g 항과 동일 시점 규약): 밴드 k 의 이웃 가격
        # = sum_{j != i, k in K_j} (Z99_j + Z999_j).
        if price_beta > 0.0:
            price_pre = (z99 + z999).astype(np.float32)          # (N_AP,)
            band_price = masks.T @ price_pre                     # (N_LINKS,)
        msg.PyRecvBegin()
        if msg.PyGetFinished():
            msg.PyRecvEnd()
            finished = True
            break
        env_msg = msg.GetCpp2PyStruct()
        (obs_per_ap, reward_prev, z99, z999,
         served, v99, v999, dropped) = _ns3_obs_and_reward(
            env_msg, z99, z999, v_weight, z_clip, eps99, eps999,
            masks, no_lyap=no_lyap, cpo=cpo, lam99=lam99, lam999=lam999)
        feats = [encode_obs(obs_per_ap[ap]) for ap in range(N_AP)]
        joints.append(np.concatenate(feats))
        msg.PyRecvEnd()

        if k >= 1:
            if price_beta > 0.0:
                # 교차항 복원: r_i -= beta * sum_k usage_{i,k} * price_k^{-i}
                # (price_k^{-i} = band_price_k - masks[i,k]*price_pre[i]).
                own = masks * price_pre[:, None]                # (N_AP, L)
                cross = (prev_usage * (band_price[None, :] - own)).sum(axis=1)
                reward_prev = reward_prev - price_beta * cross
            for ap in range(N_AP):
                rewards[ap].append(float(reward_prev[ap]))
            served_acc += served
            v99_acc += v99
            v999_acc += v999
            dropped_acc += dropped

        # 액션 샘플링 + 전송 (마지막 read 는 bootstrap: ns-3 를 계속 진행시키기
        # 위해 전송은 하되 학습 데이터로 기록하지 않는다).
        msg.PySendBegin()
        act_msg = msg.GetPy2CppStruct()
        mode_votes: List[int] = []
        step: List[Tuple[Dict[str, np.ndarray], float]] = []
        for ap in range(N_AP):
            ft = torch.from_numpy(feats[ap]).float().to(device).unsqueeze(0)
            net = actors_local[ap] if is_local else actor
            with torch.no_grad():
                logits = net(ft).squeeze(0)
            logits = _mask_link_logits(logits, masks[ap], N_STA_PER_AP, N_LINKS)
            if not allow_coord:
                logits = _mask_map_logits(logits, N_STA_PER_AP, N_LINKS,
                                          n_map_modes)
            _, indices, logp = sample_action(
                logits, N_STA_PER_AP, N_LINKS, rng, deterministic=False)
            step.append((indices, logp))
            for sta in range(N_STA_PER_AP):
                act_msg.set_selected_link(
                    ap, sta, int(indices["selected_links"][sta]))
            mode_votes.append(int(indices["map_mode"]))
        act_msg.mapMode = int(
            np.bincount(mode_votes, minlength=n_map_modes).argmax())
        if price_beta > 0.0:
            for ap in range(N_AP):
                links = step[ap][0]["selected_links"].astype(np.int64)
                cnt = np.bincount(links, minlength=N_LINKS)[:N_LINKS]
                prev_usage[ap] = cnt / float(N_STA_PER_AP)
        msg.PySendEnd()

        if k < rollout:
            for ap in range(N_AP):
                indices, logp = step[ap]
                links = indices["selected_links"].astype(np.int64)
                states[ap].append(feats[ap])
                sel_links[ap].append(links)
                map_modes[ap].append(int(indices["map_mode"]))
                log_probs[ap].append(float(logp))
                la = np.zeros((N_STA_PER_AP, N_LINKS), dtype=np.int32)
                la[np.arange(N_STA_PER_AP), links] = 1
                last_link_active[ap] = {"link_active": la}

    # GAE 정합을 위해 rewards 길이에 맞춰 truncate (early-finish 안전장치).
    t = min(len(rewards[0]), len(states[0])) if rewards[0] else 0
    out_states = {ap: np.asarray(states[ap][:t], dtype=np.float32)
                  for ap in range(N_AP)}
    out_sel = {ap: np.asarray(sel_links[ap][:t], dtype=np.int64)
               for ap in range(N_AP)}
    out_mm = {ap: np.asarray(map_modes[ap][:t], dtype=np.int64)
              for ap in range(N_AP)}
    out_lp = {ap: np.asarray(log_probs[ap][:t], dtype=np.float32)
              for ap in range(N_AP)}
    out_rw = {ap: np.asarray(rewards[ap][:t], dtype=np.float32)
              for ap in range(N_AP)}
    joint_states = np.asarray(joints[:t + 1], dtype=np.float32)

    served_sum = float(served_acc.sum())
    # eq:uhr 정합: 분모 decided = served + dropped, 분자 v = late + dropped.
    decided_acc = np.maximum(served_acc + dropped_acc, 1.0)
    p99_per_ap = (v99_acc + dropped_acc) / decided_acc
    p999_per_ap = (v999_acc + dropped_acc) / decided_acc
    last_info = {
        "Z_per_ap_99": z99.copy(),
        "Z_per_ap_99_9": z999.copy(),
        "p99_violation_rate": float(
            (v99_acc.sum() + dropped_acc.sum()) / max(decided_acc.sum(), 1.0)),
        "p99_9_violation_rate": float(
            (v999_acc.sum() + dropped_acc.sum()) / max(decided_acc.sum(), 1.0)),
        # per-AP 위반율 (F5 ckpt 기준 = per-AP CMDP infeasibility 용).
        "p99_per_ap": p99_per_ap.copy(),
        "p99_9_per_ap": p999_per_ap.copy(),
        "mean_Z_99": float(z99.mean()),
        "mean_Z_99_9": float(z999.mean()),
        # zq 진단: per-AP Z 의 표준편차. 0 이면 전 AP 동일(=zq 가 uniform 으로
        # 퇴화), 클수록 제약 압력이 이질적(=zq 가중이 의미를 가짐). 99.9 쪽도
        # 같은 진단 (eps999 가 더 엄격해 더 일찍 전-AP 포화될 수 있음).
        "std_Z_99": float(z99.std()),
        "std_Z_99_9": float(z999.std()),
        # 포화/측정창 진단: 한 측정창(슬롯)당 AP 평균 delivered 패킷 수.
        "served_per_slot": served_sum / max(N_AP * t, 1),
        # coordination 진단: 전 AP·슬롯 샘플 중 map_mode!=0(Co-TDMA/OFDMA gating)
        # 비율. allow_coord off 면 항상 0. on 이면 정책이 조정을 실제로 얼마나
        # 부르는지 → coord-ON 결과 해석의 핵심(0 이면 coord-ON≡coord-OFF).
        "frac_coord": float(np.mean([
            (out_mm[ap] != 0).mean() if len(out_mm[ap]) else 0.0
            for ap in range(N_AP)])) if t else 0.0,
    }
    return {
        "states": out_states, "sel_links": out_sel, "map_modes": out_mm,
        "log_probs": out_lp, "rewards": out_rw, "joint_states": joint_states,
        "last_actions": last_link_active, "last_info": last_info,
        "z99": z99, "z999": z999, "T": t, "finished": finished,
    }


def _open_log(path: str | None):
    if not path:
        return None, None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["iter", "timestamp_iso", "actor_loss", "critic_loss",
              "avg_return", "p99_train", "p99_9_train", "mean_Z_99_train",
              "mean_Z_99_9_train", "T"]
    f = open(path, "w", newline="")
    w = _csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    return f, w


def train(args: argparse.Namespace) -> int:
    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    import torch  # type: ignore
    import torch.optim as optim  # type: ignore

    from models.mappo import (  # type: ignore
        compute_agg_measure, compute_gae, critic_loss, make_aggregator,
        mappo_actor_loss,
    )
    from models.networks import (  # type: ignore
        ActorMLP, CriticMLP, N_MAP_MODES, action_dim_of, encode_obs,
        log_prob_of, sample_action, state_dim_of,
    )
    from sim.python.env.wlan_env import WLANConfig, WLANEnv  # type: ignore
    from feddrl import _link_masks  # type: ignore

    # P1/P6 수정: obs 정규화 상수를 실제 설정에 연동 — Z 는 z_clip 단위,
    # HoL 은 deadline999 단위 (구: Z/100 고정 → actor 가 Z∈[0,0.1] 만 봄).
    import models.networks as _networks  # type: ignore
    _networks.Z_NORM = args.z_clip
    _networks.HOL_NORM_S = args.deadline999_ms * 1e-3

    device = torch.device(args.device)
    cfg = WLANConfig(seed=args.seed, horizon=1, n_ap=N_AP,
                     n_sta_per_ap=N_STA_PER_AP, n_links=N_LINKS)
    env = WLANEnv(cfg)
    state_dim = state_dim_of(env)
    action_dim = action_dim_of(env)
    joint_dim = state_dim * N_AP
    masks = _link_masks()

    # ALL methods keep per-AP LOCAL actors (capacity-matched to 'none'). Federation
    # is now proper FedAvg: each AP runs args.epochs local PPO epochs on its own
    # actor, then every agg_period iterations the per-AP actor PARAMETERS are
    # weighted-averaged (measure -> weights) and broadcast back to all APs. 'none'
    # never aggregates (fully independent). Critic stays centralized on the joint
    # state (CTDE). Previously the federated path used ONE shared actor with per-step
    # gradient averaging (1/16 the actor capacity of 'none') -- an unfair confound.
    # #7 new learned baselines:
    #   --shared-actor (MAPPO): ONE actor shared across all 16 APs, trained on
    #     POOLED experience (canonical MAPPO parameter sharing, Yu et al. 2022).
    #     Distinct from FedAvg which averages 16 per-AP actors each round.
    #   --cpo (constrained-PPO): 16 independent per-AP actors + conventional
    #     per-AP PID-Lagrangian on the UHR constraint (see _ns3_obs_and_reward).
    # Neither federates. Both keep the centralized per-AP-head critic (CTDE).
    shared_mode = bool(getattr(args, "shared_actor", False))
    cpo_mode = bool(getattr(args, "cpo", False))
    if shared_mode and cpo_mode:
        raise ValueError("--shared-actor 와 --cpo 는 동시 지정 불가")

    is_local = not shared_mode
    do_agg = (args.aggregation != "none") and not shared_mode and not cpo_mode
    if shared_mode:
        # MAPPO: single shared actor + single optimizer, pooled PPO update.
        actor = ActorMLP(state_dim, action_dim).to(device)
        opt_shared = optim.Adam(actor.parameters(), lr=args.lr_actor)
        actors_local = None
        opts_local = None
    else:
        actor = None
        actors_local = [ActorMLP(state_dim, action_dim).to(device)
                        for _ in range(N_AP)]
        opts_local = [optim.Adam(a.parameters(), lr=args.lr_actor)
                      for a in actors_local]
        opt_shared = None
    aggregator = (make_aggregator(args.aggregation, N_AP)
                  if do_agg and args.aggregation != "cluster" else None)

    # CPO per-AP PID-Lagrangian state (multiplier λ + integral + prev cost).
    lam99 = np.zeros(N_AP, dtype=np.float32)
    lam999 = np.zeros(N_AP, dtype=np.float32)
    pid_I99 = np.zeros(N_AP, dtype=np.float64)
    pid_I999 = np.zeros(N_AP, dtype=np.float64)
    pid_prev99 = np.zeros(N_AP, dtype=np.float64)
    pid_prev999 = np.zeros(N_AP, dtype=np.float64)

    # F10 (P4 수정): per-AP value heads — joint 입력(CTDE)은 유지하되 각 AP 의
    # GAE baseline 을 자기 AP 의 return 에 맞춘다 (구: 단일 V=cross-AP 평균이
    # 이질 부하에서 per-AP advantage 를 편향).
    # B (QR critic): quantiles>0 이면 head 당 K 분위수 — actor advantage 는
    # 하위 alpha 분위 평균(CVaR) baseline (최악-경로 return 우선 개선).
    n_q = max(1, args.quantiles)
    critic = CriticMLP(joint_dim, n_heads=N_AP, n_quantiles=n_q).to(device)
    opt_critic = optim.Adam(critic.parameters(), lr=args.lr_critic)

    print(f"[init] state_dim={state_dim} action_dim={action_dim} "
          f"joint_dim={joint_dim} n_ap={N_AP} agg={args.aggregation}")

    # ns-3 를 하나의 긴 에피소드로 띄운다 (persistent env). 필요 슬롯 = 학습에
    # 쓰는 슬롯 + bootstrap + 여유.
    slot_ms = args.macro_slot_ms
    if args.episode_ms > 0:
        episode_ms = args.episode_ms
    else:
        need_slots = args.iterations * (args.rollout + 1) + 50
        episode_ms = need_slots * slot_ms
    print(f"[init] macro_slot_ms={slot_ms:.0f} episode_ms={episode_ms:.0f} "
          f"(~{episode_ms / slot_ms:.0f} slots)")

    import ns3ai_feddrl_py as py_binding  # type: ignore
    from ns3ai_utils import Experiment  # type: ignore

    setting = {
        "seed": args.seed,
        "arrivalPps": args.arrival_pps,
        "baselineTag": f"ns3train_{args.aggregation}",
        "episodeDurationMs": episode_ms,
        "macroSlotMs": slot_ms,
        "deadline99Ms": args.deadline99_ms,
        "deadline999Ms": args.deadline999_ms,
        "loadSpread": args.load_spread,
        "chanDwellMs": args.chan_dwell_ms,
        "hetBands": args.het_bands,
        "drainTarget": args.drain_target,
        "link2Width": args.link2_width,
    }
    seg = args.seg_suffix
    if seg:
        # Unique shm names, identical to the scenario's segSuffix derivation.
        setting["segSuffix"] = seg
        exp = Experiment("ns3ai_feddrl", args.ns3_path, py_binding,
                         handleFinish=True,
                         segName=f"ns3ai_seg_{seg}",
                         cpp2pyMsgName=f"ns3ai_c2p_{seg}",
                         py2cppMsgName=f"ns3ai_p2c_{seg}",
                         lockableName=f"ns3ai_lock_{seg}")
    else:
        exp = Experiment("ns3ai_feddrl", args.ns3_path, py_binding,
                         handleFinish=True)
    msg = exp.run(setting=setting, show_output=True)

    log_f, log_w = _open_log(args.train_log_csv)
    z99 = np.zeros(N_AP, dtype=np.float32)
    z999 = np.zeros(N_AP, dtype=np.float32)
    rng = np.random.default_rng(args.seed)

    def _save_ckpt(path):
        # All methods save 16 per-AP actor state_dicts under "actors" so the eval
        # driver (feddrl.py._load_actors, mode="local") applies actors[ap] uniformly.
        # MAPPO (--shared-actor) has ONE actor -> save 16 identical copies of it so
        # every AP evaluates the shared policy (byte-identical to loading via
        # actor[0]); CPO / none keep the 16 distinct per-AP actors.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if shared_mode:
            sd = actor.state_dict()
            actor_sds = [sd for _ in range(N_AP)]
            arm_tag = "mappo_shared"
        else:
            actor_sds = [a.state_dict() for a in actors_local]
            arm_tag = "cpo" if cpo_mode else args.aggregation
        torch.save({
            "actors": actor_sds,
            "critic": critic.state_dict(),
            "mode": "local", "n_ap": N_AP,
            "aggregation": args.aggregation,
            "arm": arm_tag,
        }, path)

    # Best-checkpoint tracking: persistent-env training drifts/degrades past the
    # ~iter 20-40 sweet spot, so the FINAL policy is often not the best.
    # 선택 기준 = per-AP CMDP infeasibility (F5 수정): 네트워크 평균 p99 는
    # 위반 AP 를 준수 AP 들이 가려 CMDP(∀i 제약)와 불일치했다. 점수는
    #   mean_i [ max(0, p99_i-eps99)/eps99 + max(0, p999_i-eps999)/eps999 ]
    # — 전 AP feasible 이면 0, 제약 위반 초과분에 비례. 논문 eq:cmdp 정합.
    best_score = float("inf")
    score_hist: List[float] = []
    saved_any = False
    _BEST_WARMUP = 8
    _BEST_SMOOTH = 5

    def _cmdp_infeasibility(info: Dict) -> float:
        p99a = np.asarray(info["p99_per_ap"], dtype=np.float64)
        p999a = np.asarray(info["p99_9_per_ap"], dtype=np.float64)
        ex99 = np.maximum(0.0, p99a - args.eps99) / args.eps99
        ex999 = np.maximum(0.0, p999a - args.eps999) / args.eps999
        return float(np.mean(ex99 + ex999))

    try:
        for it in range(args.iterations):
            roll = collect_rollout_ns3(
                msg, actor, actors_local, args.rollout, rng, device,
                z99, z999, args.lyapunov_v, args.z_clip,
                args.eps99, args.eps999, masks,
                encode_obs, sample_action, N_MAP_MODES,
                allow_coord=args.allow_coord, no_lyap=args.no_lyapunov,
                price_beta=args.price_beta,
                cpo=cpo_mode, lam99=lam99, lam999=lam999)
            z99, z999 = roll["z99"], roll["z999"]
            t = roll["T"]
            if t < 2:
                print(f"[iter {it}] ns-3 stream ended (T={t}); stopping.")
                break

            states = roll["states"]
            sel_links = roll["sel_links"]
            map_modes = roll["map_modes"]
            log_probs_old = roll["log_probs"]
            rewards = roll["rewards"]
            joint_states = roll["joint_states"]
            last_actions = roll["last_actions"]
            last_info = roll["last_info"]

            joint_t = torch.from_numpy(joint_states).float().to(device)
            with torch.no_grad():
                q_full = critic(joint_t).cpu().numpy()
            if n_q > 1:
                # QR: v_mean(불편 baseline, critic 타깃용) + v_cvar(actor 용).
                v_mean = q_full.mean(axis=-1)                    # (T+1, N_AP)
                k_low = max(1, int(np.ceil(args.cvar_alpha * n_q)))
                v_cvar = np.sort(q_full, axis=-1)[..., :k_low].mean(axis=-1)
            else:
                v_mean = q_full
                v_cvar = q_full

            advantages_per_ap = {}
            returns_per_ap = {}
            for ap in range(N_AP):
                # actor advantage: CVaR baseline (n_q==1 이면 동일 경로).
                adv, _ = compute_gae(rewards[ap], v_cvar[:, ap],
                                     args.gamma, args.gae_lambda)
                # critic 타깃 return: 불편 mean baseline 으로 산출.
                _, ret = compute_gae(rewards[ap], v_mean[:, ap],
                                     args.gamma, args.gae_lambda)
                advantages_per_ap[ap] = (adv - adv.mean()) / (adv.std() + 1e-8)
                returns_per_ap[ap] = ret

            cost_per_ap = np.array(
                [-float(np.mean(rewards[ap])) for ap in range(N_AP)],
                dtype=np.float32)

            actor_loss_avg = 0.0
            if shared_mode:
                # MAPPO: ONE shared actor, POOLED PPO update. Per-AP forward is
                # needed (link masks K_i differ per AP), but gradients accumulate
                # into the single shared parameter set and one optimizer step is
                # taken per epoch -> equal-weight pooling of all 16 APs' experience.
                for _ in range(args.epochs):
                    opt_shared.zero_grad()
                    a_loss_sum = 0.0
                    for ap in range(N_AP):
                        s_t = torch.from_numpy(states[ap]).float().to(device)
                        sel_t = torch.from_numpy(sel_links[ap]).long().to(device)
                        mm_t = torch.from_numpy(map_modes[ap]).long().to(device)
                        lp_old = torch.from_numpy(
                            log_probs_old[ap]).float().to(device)
                        adv_t = torch.from_numpy(
                            advantages_per_ap[ap]).float().to(device)
                        logits_new = actor(s_t)
                        logits_new = _mask_link_logits(
                            logits_new, masks[ap], N_STA_PER_AP, N_LINKS)
                        if not args.allow_coord:
                            logits_new = _mask_map_logits(
                                logits_new, N_STA_PER_AP, N_LINKS, N_MAP_MODES)
                        lp_new = log_prob_of(
                            logits_new, sel_t, mm_t, N_STA_PER_AP, N_LINKS)
                        a_loss = mappo_actor_loss(
                            lp_old, lp_new, adv_t, args.eps_clip)
                        (a_loss / N_AP).backward()   # pooled: mean over APs
                        a_loss_sum += a_loss.item()
                    opt_shared.step()
                    actor_loss_avg = a_loss_sum / N_AP
            else:
                for _ in range(args.epochs):
                    actor_loss_sum = 0.0
                    for ap in range(N_AP):
                        s_t = torch.from_numpy(states[ap]).float().to(device)
                        sel_t = torch.from_numpy(sel_links[ap]).long().to(device)
                        mm_t = torch.from_numpy(map_modes[ap]).long().to(device)
                        lp_old = torch.from_numpy(
                            log_probs_old[ap]).float().to(device)
                        adv_t = torch.from_numpy(
                            advantages_per_ap[ap]).float().to(device)

                        # local PPO step on THIS AP's own actor (advantage from the
                        # shared centralized critic -> CTDE).
                        opts_local[ap].zero_grad()
                        logits_new = actors_local[ap](s_t)
                        logits_new = _mask_link_logits(
                            logits_new, masks[ap], N_STA_PER_AP, N_LINKS)
                        if not args.allow_coord:
                            logits_new = _mask_map_logits(
                                logits_new, N_STA_PER_AP, N_LINKS, N_MAP_MODES)
                        lp_new = log_prob_of(
                            logits_new, sel_t, mm_t, N_STA_PER_AP, N_LINKS)
                        a_loss = mappo_actor_loss(
                            lp_old, lp_new, adv_t, args.eps_clip)
                        a_loss.backward()
                        opts_local[ap].step()
                        actor_loss_sum += a_loss.item()
                    actor_loss_avg = actor_loss_sum / N_AP

            # FedAvg communication round: after the local epochs, weighted-average
            # the per-AP actor PARAMETERS and broadcast to all APs. compute_weights
            # normalizes (sum=1), so aggregate_grads over the param tensors yields
            # the weighted MEAN of the models (FedAvg). 'none' skips this entirely,
            # leaving 16 fully-independent actors (capacity-matched upper baseline).
            if do_agg and (it % args.agg_period == 0):
                if args.aggregation == "cluster":
                    # 이질성-인지 연합: 같은 링크집합 K_i 그룹끼리만 uniform
                    # FedAvg. cross-K 파라미터 평균이 per-AP 특화를 파괴한다는
                    # 진단(none>uniform>zq>iw 단조 악화)에서 도출 — 특화(K_i)는
                    # 보존하고 같은 구조 AP 간 경험만 공유(분산 감소).
                    groups: Dict[tuple, List[int]] = {}
                    for a in range(N_AP):
                        groups.setdefault(
                            tuple(float(x) for x in masks[a]), []).append(a)
                    with torch.no_grad():
                        for members in groups.values():
                            plists = [dict(actors_local[a].named_parameters())
                                      for a in members]
                            for n in plists[0].keys():
                                mean_p = torch.stack(
                                    [pl[n] for pl in plists], dim=0).mean(dim=0)
                                for pl in plists:
                                    pl[n].copy_(mean_p)
                else:
                    measure = compute_agg_measure(
                        args.aggregation, last_actions, last_info,
                        N_AP, N_LINKS, cost_per_ap=cost_per_ap)
                    weights = aggregator.compute_weights(measure)
                    params_per_ap = [
                        {n: p.detach().clone()
                         for n, p in actors_local[ap].named_parameters()}
                        for ap in range(N_AP)]
                    agg_params = aggregator.aggregate_grads(
                        params_per_ap, weights)
                    with torch.no_grad():
                        for ap in range(N_AP):
                            for n, p in actors_local[ap].named_parameters():
                                p.copy_(agg_params[n])
                # F11 (P5 수정): broadcast 후 per-AP Adam 1/2차 모멘트 리셋.
                # 평균 전 로컬 궤적의 적응 스텝을 평균된 가중치에 적용하면
                # 이론이 분석하는 θ+ηg 로컬 업데이트가 아니며(라운드 간 옵티마
                # 이저 메모리 누수), 연합 arm 의 bimodal 붕괴 후보 원인.
                for ap in range(N_AP):
                    opts_local[ap] = optim.Adam(
                        actors_local[ap].parameters(), lr=args.lr_actor)

            # CPO PID-Lagrangian update (per-AP, per-ITERATION). Uses the
            # rollout-aggregate violation rate (last_info) as the constraint
            # signal, normalized by the budget so p99/p999 share scale:
            #   c = violation_rate/ε − 1   (>0 = infeasible)
            #   I ← clip(I + Ki·c, 0, z_clip)        (integral term = classic dual)
            #   λ ← clip(Kp·[c]_+ + I + Kd·[c−c_prev]_+, 0, z_clip)
            # λ takes effect NEXT iteration's rollout reward. Slow (per-iteration)
            # dual on the EXPECTED constraint — the textbook Lagrangian-PPO recipe,
            # distinct from our per-slot clipped drift Z that the actor observes.
            if cpo_mode:
                p99a = np.asarray(last_info["p99_per_ap"], dtype=np.float64)
                p999a = np.asarray(last_info["p99_9_per_ap"], dtype=np.float64)
                c99 = p99a / args.eps99 - 1.0
                c999 = p999a / args.eps999 - 1.0
                kp, ki, kd = args.cpo_kp, args.cpo_ki, args.cpo_kd
                zc = args.z_clip
                pid_I99 = np.clip(pid_I99 + ki * c99, 0.0, zc)
                pid_I999 = np.clip(pid_I999 + ki * c999, 0.0, zc)
                d99 = np.maximum(0.0, c99 - pid_prev99)
                d999 = np.maximum(0.0, c999 - pid_prev999)
                lam99 = np.clip(
                    kp * np.maximum(0.0, c99) + pid_I99 + kd * d99,
                    0.0, zc).astype(np.float32)
                lam999 = np.clip(
                    kp * np.maximum(0.0, c999) + pid_I999 + kd * d999,
                    0.0, zc).astype(np.float32)
                pid_prev99, pid_prev999 = c99, c999

            # F10: critic 타깃 = per-AP returns 행렬 (T, N_AP) — head i 가
            # AP i 의 return 을 학습 (구: cross-AP 평균 스칼라).
            ap_returns = np.stack(
                [returns_per_ap[ap] for ap in range(N_AP)], axis=1)
            ap_returns_t = torch.from_numpy(ap_returns).float().to(device)
            if n_q > 1:
                taus = torch.arange(
                    0.5, n_q, 1.0, device=device) / n_q       # (K,)
            c_loss_val = 0.0
            for _ in range(args.epochs):
                opt_critic.zero_grad()
                values_pred = critic(joint_t[:-1])
                if n_q > 1:
                    # 분위 회귀 (pinball loss): head 별 return 분포 적합.
                    u = ap_returns_t.unsqueeze(-1) - values_pred  # (T,N,K)
                    c_loss = torch.max(taus * u, (taus - 1.0) * u).mean()
                else:
                    c_loss = critic_loss(values_pred, ap_returns_t)
                c_loss.backward()
                opt_critic.step()
                c_loss_val = c_loss.item()

            avg_return = float(
                np.mean([rewards[ap].sum() for ap in range(N_AP)]))
            if log_w is not None:
                log_w.writerow({
                    "iter": it,
                    "timestamp_iso": _dt.datetime.now().isoformat(
                        timespec="seconds"),
                    "actor_loss": float(actor_loss_avg),
                    "critic_loss": float(c_loss_val),
                    "avg_return": avg_return,
                    "p99_train": last_info["p99_violation_rate"],
                    "p99_9_train": last_info["p99_9_violation_rate"],
                    "mean_Z_99_train": last_info["mean_Z_99"],
                    "mean_Z_99_9_train": last_info["mean_Z_99_9"],
                    "T": t,
                })
                log_f.flush()
            dual_str = (
                f"lam99={lam99.mean():.2f}±{lam99.std():.2f} "
                f"lam999={lam999.mean():.2f}±{lam999.std():.2f}"
                if cpo_mode else
                f"Z99={last_info['mean_Z_99']:.2f}±{last_info['std_Z_99']:.2f} "
                f"Z999={last_info['mean_Z_99_9']:.2f}"
                f"±{last_info['std_Z_99_9']:.2f}")
            print(f"[iter {it:3d}] T={t} aloss={actor_loss_avg:+.4f} "
                  f"closs={c_loss_val:.3f} ret={avg_return:+.2f} "
                  f"srv/slot={last_info['served_per_slot']:.1f} "
                  f"p99={last_info['p99_violation_rate']:.4f} "
                  f"p99.9={last_info['p99_9_violation_rate']:.4f} "
                  f"{dual_str} "
                  f"coord={last_info['frac_coord']*100:.0f}%")

            score_hist.append(_cmdp_infeasibility(last_info))
            if it >= _BEST_WARMUP:
                score = float(np.mean(score_hist[-_BEST_SMOOTH:]))
                if score < best_score:
                    best_score = score
                    _save_ckpt(args.ckpt_out)
                    saved_any = True
    finally:
        if log_f is not None:
            log_f.close()
        del exp

    if not saved_any:
        # never improved past warmup (very short run) -> save the final policy.
        _save_ckpt(args.ckpt_out)
        print(f"[done] checkpoint (final; no best tracked) -> {args.ckpt_out}")
    else:
        print(f"[done] best checkpoint (smoothed CMDP infeasibility="
              f"{best_score:.4f}) -> {args.ckpt_out}")
    return 0


def main() -> int:
    return train(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
