"""train_ns3.py — ns-3-in-the-loop federated MAPPO trainer (redesign).

Design shift: the Python surrogate is removed and ns-3 is used directly as the
RL environment. Actual behavior (802.11 PHY/MAC, packet-level DES) = ns-3,
learning (PPO + federated aggregation) = Python. The two sides are coupled
through ns3-ai shared memory.

The same EnvMsg/ActMsg interface as the existing ``feddrl.py`` (eval-only
driver), the same obs schema (csi/queue/hol_age/cbr/Z_99/Z_99_9/link_mask) and
the same Lyapunov dual update are reused. Only rollout collection + per-AP
reward computation are new; GAE/PPO/aggregation reuse ``models.mappo`` 100%.

MDP alignment (drift-plus-penalty):
    The EnvMsg of read k carries the served/violation counts of slot (k-1),
    i.e. of the previous action a_{k-1}. Therefore
        g_{k-1} = (v_{k-1} - eps * served_{k-1}) / N_STA
        r_{k-1} = V * u_{k-1} - Z_{k-1} · g_{k-1}      (PRE-update dual)
        Z_k     = clip(Z_{k-1} + g_{k-1}, 0, z_clip)   (dual ascent)
    obs_k carries the updated Z_k. A transition is assembled as
        (obs_{k-1}[Z_{k-1}], a_{k-1}, r_{k-1}, obs_k[Z_k]).

Action space v1 = link + mode:
    per-STA link selection (learned) + a single global map_mode (ActMsg
    supports only one, so the per-AP samples are executed by majority vote).
    The PPO ratio is computed with each AP's own sampled map_mode to stay
    consistent (execution uses the majority vote). Per-AP individual modes
    need a struct extension and are left as a v2 task.

Execution (WSL Ubuntu, inside the ns-3 example directory):
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
MACRO_SLOT_MS = 5.0  # Must match kMacroSlot in feddrl_scenario.cc.


def _mask_link_logits(logits: Any, link_mask: Any, n_sta: int,
                      n_links: int) -> Any:
    """Delegates to models.networks.mask_link_logits -- train and eval share a
    single implementation so their action distributions match (P3 fix; full
    docstring in networks.py)."""
    from models.networks import mask_link_logits  # type: ignore

    return mask_link_logits(logits, link_mask, n_sta, n_links)


def _mask_map_logits(logits: Any, n_sta: int, n_links: int,
                     n_map_modes: int) -> Any:
    """Delegates to models.networks.mask_map_logits (P3 fix; details in
    networks.py)."""
    from models.networks import mask_map_logits  # type: ignore

    return mask_map_logits(logits, n_sta, n_links, n_map_modes)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ns-3-in-the-loop federated MAPPO")
    p.add_argument("--aggregation", type=str, default="zq",
                   choices=["none", "uniform", "zq", "iw", "hybrid",
                            "qffl", "afl", "cluster"],
                   help="none=independent local actors (no FedAvg); "
                        "cluster=heterogeneity-aware federation (uniform "
                        "FedAvg only within groups sharing the same link set "
                        "K_i -- prescribed by the diagnosis that cross-K "
                        "averaging destroys specialization); the others are "
                        "global weighted FedAvg variants.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arrival-pps", type=float, default=5500.0)
    p.add_argument("--iterations", type=int, default=200,
                   help="PPO update count (each update consumes --rollout "
                        "slots).")
    p.add_argument("--rollout", type=int, default=64,
                   help="Macro slots consumed per update.")
    p.add_argument("--allow-coord", action="store_true",
                   help="Allow map_mode (Co-TDMA/OFDMA) exploration. Default "
                        "off = frozen at mode0. In the 16-AP dense setting "
                        "Co-TDMA serializes 16x, exploding the backlog and "
                        "collapsing the sim while giving no delay benefit, so "
                        "it stays frozen by default.")
    p.add_argument("--macro-slot-ms", type=float, default=20.0,
                   help="RL decision+measurement window length (ms). Longer "
                        "windows carry more packets per slot, lowering "
                        "reward/Z variance. Forwarded to the ns-3 scenario.")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lyapunov-v", type=float, default=1.0,
                   help="Utility weight V in drift-plus-penalty.")
    p.add_argument("--eps99", type=float, default=1e-2,
                   help="99% delay violation-rate target "
                        "(P(delay>deadline99) <= eps99). The target must lie "
                        "in the achievable region so that Z stays alive and "
                        "heterogeneous, which is what makes the zq weighting "
                        "meaningful (Slater assumption A5).")
    p.add_argument("--eps999", type=float, default=1e-3,
                   help="99.9% delay violation-rate target "
                        "(P(delay>deadline999) <= eps999).")
    p.add_argument("--deadline99-ms", type=float, default=5.0,
                   help="p99 deadline (ms). Density-scaled: 4-AP=5, 16-AP "
                        "dense=10 (the structural tail floor grows with "
                        "density). Forwarded to ns-3.")
    p.add_argument("--deadline999-ms", type=float, default=10.0,
                   help="p99.9 deadline (ms). 4-AP=10, 16-AP dense=20. "
                        "Forwarded to ns-3.")
    p.add_argument("--load-spread", type=float, default=0.0,
                   help="Per-AP load heterogeneity in [0,1]. 0=homogeneous "
                        "(arrivalPps for every AP). >0 gives a linear gradient "
                        "AP0=arrivalPps ~ AP15=(1-load_spread)*arrivalPps -> "
                        "heterogeneous per-AP constraint pressure (Z), the "
                        "precondition for validating federation/zq. Forwarded "
                        "to ns-3. Evaluation must use the same value.")
    p.add_argument("--seg-suffix", type=str, default="",
                   help="Per-run unique suffix for the ns3-ai shared-memory "
                        "object names. Give each concurrent training job a "
                        "different value to avoid segment collisions "
                        "(parallelism). Empty = library default (single run). "
                        "Must match the name used on the ns-3 side.")
    p.add_argument("--chan-dwell-ms", type=float, default=200.0,
                   help="F6 Markov channel mean state dwell time (ms); "
                        "0=static (legacy). Forwarded to ns-3. Evaluation "
                        "(feddrl.py) must use the same value.")
    p.add_argument("--het-bands", type=int, default=1,
                   help="F7 heterogeneous bands (2.4/5/6GHz, 20/40/80MHz); "
                        "0=legacy. Forwarded to ns-3. Evaluation must use the "
                        "same value.")
    p.add_argument("--link2-width", type=int, default=80,
                   help="link2 (6GHz) channel width in MHz: 80 (default)|40. "
                        "Forwarded to ns-3; evaluation must match.")
    p.add_argument("--drain-target", type=int, default=8,
                   help="F8 shared-buffer drain-on-demand target depth of the "
                        "per-link MAC queue; 0=direct-send legacy. Forwarded "
                        "to ns-3; evaluation must match.")
    p.add_argument("--price-beta", type=float, default=0.0,
                   help="Price coupling (PX): folds the neighboring APs' dual "
                        "(Z99+Z999) into the reward as a per-band congestion "
                        "price -- r_i -= beta * sum_k usage_{i,k} * "
                        "price_k^{-i}. First-order restoration of the cross "
                        "term of the network Lagrangian (shared-band airtime "
                        "externality). 0=off.")
    p.add_argument("--quantiles", type=int, default=0,
                   help="QR critic (B): number of quantiles per per-AP head "
                        "(0=scalar legacy). Learns the return distribution by "
                        "quantile (pinball) regression.")
    p.add_argument("--cvar-alpha", type=float, default=0.25,
                   help="With the QR critic, use the mean of the lowest alpha "
                        "quantiles (CVaR) as the actor advantage baseline -- "
                        "prioritizes improving the worst-case returns "
                        "(dominated by tail violation spikes).")
    p.add_argument("--no-lyapunov", action="store_true",
                   help="ablation: drop the Lyapunov constraint coupling -- "
                        "reward = V*u only (no Z penalty), Z features in obs "
                        "pinned to 0. Isolates the question 'is the Z "
                        "mechanism necessary to achieve UHR?' (a mandatory "
                        "reviewer question).")
    p.add_argument("--z-clip", type=float, default=10.0,
                   help="Upper bound on the dual variable Z. With the "
                        "rate-form drift (<=1 per slot) 10 is enough and keeps "
                        "the reward scale within a learnable range.")
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
                   help="0 = derived automatically from iterations * rollout.")
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
    """EnvMsg -> (list of per-AP obs dicts, reward, updated Z99, updated Z999,
    served, v99, v999).

    reward uses the PRE-update duals (the z99/z999 arguments) while the Z in
    obs is filled with the POST-update values (drift-plus-penalty alignment).
    All counts are normalized by the per-STA size (same as feddrl.py /
    wlan_env.step).
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
    # CSI: the per-STA-per-link quality owned and measured by the ns-3 physical
    # channel is taken as obs (no analytical re-derivation).
    # shape (N_AP, N_STA_PER_AP, N_LINKS).
    csi_all = np.frombuffer(env_msg.csi(), dtype=np.float32).reshape(
        N_AP, N_STA_PER_AP, N_LINKS)

    # The constraint is in rate form (eq:uhr): P(delay>L) <= eps, with the
    # per-slot denominator decided = served + dropped (dropped = MaxDelay=
    # deadline999 aged-out + queue-full; reported by ns-3 as EnvMsg.dropped).
    # Dropped packets missed both deadlines, so they count as violations in v
    # -- this makes the "decided = served or aged out" of eq:uhr hold per slot
    # and removes the survivor bias from the dual signal (E1 fix). The
    # rate-form drift is O(1)-bounded, hence more stable than the count form
    # (Z explodes under saturation), and is faithful to the per-packet rate
    # constraint (eq:queue in the paper is to be restated in rate form).
    decided = np.maximum(served + dropped, 1.0)
    g99 = (v99 + dropped) / decided - eps99
    g999 = (v999 + dropped) / decided - eps999
    utility = served / N_STA_PER_AP            # delivered throughput (no gain on drop)
    if no_lyap:
        # ablation: constraint coupling removed -- pure throughput reward. Z is
        # still updated for diagnostics only and excluded from the reward and
        # from obs (obs is pinned to 0 below).
        reward = v_weight * utility
    elif cpo:
        # CPO / constrained-PPO baseline (#7): conventional Lagrangian on the
        # SAME UHR constraint but with a per-AP multiplier λ that is (a) FIXED
        # within the rollout and updated per-ITERATION by a PID controller on
        # the episodic mean violation (NOT the per-slot clipped drift of ours),
        # and (b) NOT part of the observation (obs Z pinned to 0, as in
        # no_lyap). Per-slot penalty uses the UN-normalized drift g (same scale as
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
    """Collect rollout (state, action, reward) transitions from the ns-3 stream.

    Reads rollout+1 times; the last read is used only as the GAE bootstrap obs.
    The reward r_{k-1} is computed at read k and attributed to the action of
    the previous read.
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
    # Price coupling (PX): per-band usage share of the previous slot's action
    # (attributed to reward r_{k-1}).
    prev_usage = np.zeros((N_AP, N_LINKS), dtype=np.float32)

    for k in range(rollout + 1):
        # PRE-update dual price (same timing convention as the Z*g term):
        # neighbor price of band k
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
                # Cross-term restoration:
                # r_i -= beta * sum_k usage_{i,k} * price_k^{-i}
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

        # Sample and send the action (the last read is the bootstrap: it is
        # still sent to keep ns-3 advancing, but not recorded as training
        # data).
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

    # Truncate to the rewards length for GAE alignment (early-finish guard).
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
    # eq:uhr alignment: denominator decided = served + dropped, numerator
    # v = late + dropped.
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
        # Per-AP violation rates (basis of the F5 ckpt criterion = per-AP
        # CMDP infeasibility).
        "p99_per_ap": p99_per_ap.copy(),
        "p99_9_per_ap": p999_per_ap.copy(),
        "mean_Z_99": float(z99.mean()),
        "mean_Z_99_9": float(z999.mean()),
        # zq diagnostic: std of the per-AP Z. 0 means all APs are identical
        # (=zq degenerates to uniform); larger means the constraint pressure is
        # heterogeneous (=zq weighting is meaningful). Same diagnostic on the
        # 99.9 side (eps999 is stricter, so all APs may saturate earlier).
        "std_Z_99": float(z99.std()),
        "std_Z_99_9": float(z999.std()),
        # Saturation / measurement-window diagnostic: mean delivered packets
        # per AP per measurement window (slot).
        "served_per_slot": served_sum / max(N_AP * t, 1),
        # Coordination diagnostic: fraction of all AP-slot samples with
        # map_mode!=0 (Co-TDMA/OFDMA gating). Always 0 when allow_coord is off;
        # when on, it shows how often the policy actually calls for
        # coordination -- key to reading coord-ON results (0 means
        # coord-ON == coord-OFF).
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

    # P1/P6 fix: tie the obs normalization constants to the actual settings --
    # Z in units of z_clip, HoL in units of deadline999 (was: fixed Z/100, so
    # the actor only ever saw Z in [0,0.1]).
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
        raise ValueError("--shared-actor and --cpo are mutually exclusive")

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

    # F10 (P4 fix): per-AP value heads -- the joint input (CTDE) is kept, but
    # each AP's GAE baseline is fit to its own AP's return (was: a single
    # V=cross-AP mean, which biased the per-AP advantage under heterogeneous
    # load).
    # B (QR critic): quantiles>0 gives K quantiles per head -- the actor
    # advantage uses the mean of the lowest alpha quantiles (CVaR) as baseline
    # (prioritizes improving worst-case returns).
    n_q = max(1, args.quantiles)
    critic = CriticMLP(joint_dim, n_heads=N_AP, n_quantiles=n_q).to(device)
    opt_critic = optim.Adam(critic.parameters(), lr=args.lr_critic)

    print(f"[init] state_dim={state_dim} action_dim={action_dim} "
          f"joint_dim={joint_dim} n_ap={N_AP} agg={args.aggregation}")

    # Launch ns-3 as one long episode (persistent env). Required slots = slots
    # used for training + bootstrap + margin.
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
    # Selection criterion = per-AP CMDP infeasibility (F5 fix): the network-mean
    # p99 let compliant APs mask violating ones and thus disagreed with the
    # CMDP (constraint for all i). The score is
    #   mean_i [ max(0, p99_i-eps99)/eps99 + max(0, p999_i-eps999)/eps999 ]
    # -- 0 when every AP is feasible, proportional to the constraint excess
    # otherwise. Consistent with eq:cmdp in the paper.
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
                # QR: v_mean (unbiased baseline, for the critic target)
                # + v_cvar (for the actor).
                v_mean = q_full.mean(axis=-1)                    # (T+1, N_AP)
                k_low = max(1, int(np.ceil(args.cvar_alpha * n_q)))
                v_cvar = np.sort(q_full, axis=-1)[..., :k_low].mean(axis=-1)
            else:
                v_mean = q_full
                v_cvar = q_full

            advantages_per_ap = {}
            returns_per_ap = {}
            for ap in range(N_AP):
                # actor advantage: CVaR baseline (identical path when n_q==1).
                adv, _ = compute_gae(rewards[ap], v_cvar[:, ap],
                                     args.gamma, args.gae_lambda)
                # critic target return: uses the unbiased mean baseline.
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
                    # Heterogeneity-aware federation: uniform FedAvg only
                    # within groups sharing the same link set K_i. Derived
                    # from the diagnosis that cross-K parameter averaging
                    # destroys per-AP specialization (monotone degradation
                    # none>uniform>zq>iw) -- keep the specialization (K_i) and
                    # share experience only among structurally identical APs
                    # (variance reduction).
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
                # F11 (P5 fix): reset the per-AP Adam 1st/2nd moments after the
                # broadcast. Applying adaptive steps built from the local
                # pre-average trajectory to the averaged weights is no longer
                # the theta+eta*g local update the theory analyzes (optimizer
                # state leaks across rounds), and is a candidate cause of the
                # bimodal collapse of the federated arms.
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

            # F10: critic target = per-AP returns matrix (T, N_AP) -- head i
            # learns AP i's return (was: a cross-AP mean scalar).
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
                    # Quantile regression (pinball loss): fit the return
                    # distribution of each head.
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
