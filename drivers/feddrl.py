"""feddrl.py -- ns3-ai shared-memory driver for the FedDRL ns-3 scenario.

The Python side is the shared-memory creator, and ``ns3ai_utils.Experiment``
launches the ns-3 scenario (``feddrl_scenario``) as a child process (same
structure as apb.py). Every macro slot it receives an EnvMsg and sends an
ActMsg.

Policy modes:
- ``--ckpt`` omitted -> stub policy (all STAs on link 0, mapMode=0).
- ``--ckpt <path>`` -> load a trained actor checkpoint. The fields available in
  the ns-3 EnvMsg (queueLen, holUs, cbr, served) are best-effort mapped to the
  obs schema of the Python sim (csi/queue/hol_age/cbr/Z_99/Z_99_9), then the
  actor MLP samples link/mapMode. CSI and the Z queues are absent on the ns-3
  side, so placeholder values are used -- part of the sim2sim gap originates
  from this obs schema difference (for the cross-validation report).

Running (inside WSL Ubuntu):
    cd ~/ns-3-dev/contrib/ai/examples/feddrl
    python3 feddrl.py --seed 0 --arrival-pps 5500              # stub
    python3 feddrl.py --seed 0 --ckpt $REPO_ROOT/models/checkpoints/feddrl_arr5500_v10.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any, List, Optional

import numpy as np

N_AP = 16
N_STA_PER_AP = 5
N_LINKS = 3

_LINK_MASKS_CACHE: Optional[np.ndarray] = None

# per-AP running dual variables, reconstructed from ns-3 per-slot counts with
# THE SAME rule as training (train_ns3.py._ns3_obs_and_reward): rate-form
# g = (v+drop)/decided - eps, Z <- clip(Z+g, 0, Z_CLIP). (Old: the count-form,
# unclipped, /N_STA reconstruction had different dynamics/scale from training
# = an OOD Z input -- P3 fix.)
# Fresh process per episode -> 0 init. Constants overwritten via CLI (main).
_Z99_RUNNING = np.zeros(N_AP, dtype=np.float32)
_Z999_RUNNING = np.zeros(N_AP, dtype=np.float32)
EPS_99 = 1e-2
EPS_999 = 1e-3
Z_CLIP = 10.0
NO_LYAP = False  # True when evaluating a no-Z ablation ckpt (set in main)

# ── r3 #1 per-slot metric diagnostic ────────────────────────────────────────
# Remark 1 reviewer check: compare the eq.(1) per-slot-RATE metric
# (1/T)·Σ_t v_t/d_t against the reported PACKET metric Σ_t v_t / Σ_t d_t, per AP,
# for BOTH p99 and p99.9. Per-slot (v_t, d_t) come straight from EnvMsg each slot
# with the SAME rate-form as training (train_ns3._ns3_obs_and_reward):
#   d_t = served_t + dropped_t   (decided = delivered or aged-out; survivor-bias free)
#   v99_t  = violation99_t  + dropped_t   (a drop misses BOTH deadlines)
#   v999_t = violation999_t + dropped_t
# Pure reads of EnvMsg — no side effects, no ns-3 rebuild.
_METRIC_ON = False
_METRIC_ACC: Optional[dict] = None

# ── r3 #5 federation action-divergence: per-slot per-AP obs (106-d) dump ──────
# Reuses the exact encode_obs feature the actor policy already builds each slot;
# saved to a .npy of shape (T, N_AP, state_dim) for OFFLINE (Windows) divergence
# analysis (FedAvg common actor vs independent per-AP actors on the SAME obs).
_OBS_DUMP_ON = False
_OBS_LIST: List[np.ndarray] = []


def _metric_init() -> dict:
    return {
        "sum_v99": np.zeros(N_AP, dtype=np.float64),
        "sum_v999": np.zeros(N_AP, dtype=np.float64),
        "sum_d": np.zeros(N_AP, dtype=np.float64),
        "sum_r99": np.zeros(N_AP, dtype=np.float64),
        "sum_r999": np.zeros(N_AP, dtype=np.float64),
        "n_pos": np.zeros(N_AP, dtype=np.int64),
        "n_zero": np.zeros(N_AP, dtype=np.int64),
        "dmin": np.full(N_AP, np.inf, dtype=np.float64),
        "dmax": np.zeros(N_AP, dtype=np.float64),
    }


def _metric_accumulate(env_msg: Any, acc: dict) -> None:
    """Accumulate per-slot rate-form (v_t, d_t) per AP from EnvMsg (pure read)."""
    viol99 = np.frombuffer(env_msg.violation99(), dtype=np.uint32).astype(np.float64)
    viol999 = np.frombuffer(env_msg.violation999(), dtype=np.uint32).astype(np.float64)
    served = np.frombuffer(env_msg.served(), dtype=np.uint32).astype(np.float64)
    dropped = np.frombuffer(env_msg.dropped(), dtype=np.uint32).astype(np.float64)
    d = served + dropped
    v99 = viol99 + dropped
    v999 = viol999 + dropped
    pos = d > 0
    for a in range(N_AP):
        if pos[a]:
            acc["sum_v99"][a] += v99[a]
            acc["sum_v999"][a] += v999[a]
            acc["sum_d"][a] += d[a]
            acc["sum_r99"][a] += v99[a] / d[a]
            acc["sum_r999"][a] += v999[a] / d[a]
            acc["n_pos"][a] += 1
            if d[a] < acc["dmin"][a]:
                acc["dmin"][a] = d[a]
            if d[a] > acc["dmax"][a]:
                acc["dmax"][a] = d[a]
        else:
            acc["n_zero"][a] += 1


def _metric_report(acc: dict) -> None:
    """Emit one [METRIC_AP] line per AP: both metric forms + true per-slot d range."""
    for a in range(N_AP):
        n = max(int(acc["n_pos"][a]), 1)
        perslot_p99 = acc["sum_r99"][a] / n
        perslot_p999 = acc["sum_r999"][a] / n
        denom = max(acc["sum_d"][a], 1.0)
        packet_p99 = acc["sum_v99"][a] / denom
        packet_p999 = acc["sum_v999"][a] / denom
        dmin = acc["dmin"][a] if np.isfinite(acc["dmin"][a]) else 0.0
        dmax = acc["dmax"][a]
        sys.stdout.write(
            f"[METRIC_AP] {a},perslot_p99={perslot_p99:.6f},"
            f"packet_p99={packet_p99:.6f},perslot_p999={perslot_p999:.6f},"
            f"packet_p999={packet_p999:.6f},dmin_slot={dmin:.0f},"
            f"dmax_slot={dmax:.0f},npos={int(acc['n_pos'][a])},"
            f"nzero={int(acc['n_zero'][a])}\n"
        )
    sys.stdout.flush()

def _link_masks() -> np.ndarray:
    """Per-AP multi-hot link masks ``(N_AP, N_LINKS)`` — P7 obs alignment.

    Reuses the same ``make_asymmetric_link_sets`` source of truth as training so
    the ns-3 obs layout matches what the actor was trained on. Without this the
    ns-3 bridge omits ``link_mask`` and ``encode_obs`` falls back to all-ones
    (every AP sees every link), which is wrong for the asymmetric OBSS topology
    and invalidates sim2sim eval. Lazily imported: ``repo_root`` is already on
    ``sys.path`` by the time actor mode runs (see ``_load_actors``).
    """
    global _LINK_MASKS_CACHE
    if _LINK_MASKS_CACHE is None:
        from models.topology import make_asymmetric_link_sets  # type: ignore

        masks = np.zeros((N_AP, N_LINKS), dtype=np.float32)
        for ap, links in enumerate(make_asymmetric_link_sets(N_AP)):
            masks[ap, list(links)] = 1.0
        _LINK_MASKS_CACHE = masks
    return _LINK_MASKS_CACHE


def _load_actors(
    ckpt_path: str, repo_root: str, device: str
) -> tuple[List[Any], Any]:
    """Load the repo's train_feddrl checkpoint and return (actors, sample_action).

    Auto-detects whether the checkpoint has an ``actors`` (local mode) or an
    ``actor`` (shared mode) key. In shared mode all N_AP actors share the same
    weights.
    """
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import torch  # type: ignore

    from models.networks import (  # type: ignore
        ActorMLP,
        action_dim_of,
        sample_action as _sample_action,
        state_dim_of,
    )
    from sim.python.env.wlan_env import WLANConfig, WLANEnv  # type: ignore

    # The input/output dim of the actor MLP is determined by the env config.
    cfg = WLANConfig(seed=0, horizon=1, n_ap=N_AP,
                     n_sta_per_ap=N_STA_PER_AP, n_links=N_LINKS)
    env = WLANEnv(cfg)
    state_dim = state_dim_of(env)
    action_dim = action_dim_of(env)

    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    is_local = payload.get("mode") == "local" or "actors" in payload

    actors: List[Any] = []
    if is_local:
        for sd in payload["actors"]:
            net = ActorMLP(state_dim, action_dim).to(device)
            net.load_state_dict(sd)
            net.eval()
            actors.append(net)
    else:
        net = ActorMLP(state_dim, action_dim).to(device)
        net.load_state_dict(payload["actor"])
        net.eval()
        actors = [net for _ in range(N_AP)]

    return actors, _sample_action


def _ns3_obs_to_python(env_msg: Any) -> List[dict]:
    """Convert the C++ EnvMsg to the per-AP obs dict list expected by the actor.

    The ns-3 EnvMsg only carries queueLen, holUs, cbr, served; csi and the Z
    queues are absent, so they are filled with placeholders (csi=0.5, Z=0).
    """
    queue_len = np.frombuffer(env_msg.queueLen(), dtype=np.uint32).reshape(
        N_AP, N_STA_PER_AP)
    hol_us = np.frombuffer(env_msg.holUs(), dtype=np.uint32).reshape(
        N_AP, N_STA_PER_AP)
    cbr_raw = np.frombuffer(env_msg.cbr(), dtype=np.uint8).reshape(
        N_AP, N_LINKS).astype(np.float32) / 255.0
    # CSI: taken as obs from the value the ns-3 physical channel owns and
    # measures (no analytic re-derivation).
    csi_all = np.frombuffer(env_msg.csi(), dtype=np.float32).reshape(
        N_AP, N_STA_PER_AP, N_LINKS)

    masks = _link_masks()
    # Dual variable Z: same rule as training (train_ns3.py._ns3_obs_and_reward)
    # -- rate form g = (v+drop)/decided - eps, Z <- clip(Z+g, 0, Z_CLIP).
    # decided = served + dropped (includes the eq:uhr aged-out, removes
    # survivor bias).
    viol99 = np.frombuffer(env_msg.violation99(), dtype=np.uint32).astype(np.float32)
    viol999 = np.frombuffer(env_msg.violation999(), dtype=np.uint32).astype(np.float32)
    served = np.frombuffer(env_msg.served(), dtype=np.uint32).astype(np.float32)
    dropped = np.frombuffer(env_msg.dropped(), dtype=np.uint32).astype(np.float32)
    decided = np.maximum(served + dropped, 1.0)
    g99 = (viol99 + dropped) / decided - EPS_99
    g999 = (viol999 + dropped) / decided - EPS_999
    if not NO_LYAP:  # a no-Z ablation ckpt keeps Z features at 0 (as in training)
        for a in range(N_AP):
            _Z99_RUNNING[a] = min(
                Z_CLIP, max(0.0, float(_Z99_RUNNING[a] + g99[a])))
            _Z999_RUNNING[a] = min(
                Z_CLIP, max(0.0, float(_Z999_RUNNING[a] + g999[a])))
    out: List[dict] = []
    for ap in range(N_AP):
        # queue (n_sta, n_links, 4): backlog by AC. ns-3 side assumes single AC.
        queue = np.zeros((N_STA_PER_AP, N_LINKS, 4), dtype=np.float32)
        queue[:, 0, 0] = queue_len[ap].astype(np.float32)
        # hol_age (n_sta, n_links): μs -> seconds.
        hol_age = np.zeros((N_STA_PER_AP, N_LINKS), dtype=np.float32)
        hol_age[:, 0] = hol_us[ap].astype(np.float32) * 1e-6
        # per-STA-per-link CSI (value measured and delivered by the ns-3 channel).
        csi = csi_all[ap]
        # Z: per-AP running dual variable, broadcast to the STAs of the AP.
        z99 = np.full((N_STA_PER_AP,), _Z99_RUNNING[ap], dtype=np.float32)
        z999 = np.full((N_STA_PER_AP,), _Z999_RUNNING[ap], dtype=np.float32)
        out.append({
            "csi": csi,
            "queue": queue,
            "hol_age": hol_age,
            "cbr": cbr_raw[ap],
            "Z_99": z99,
            "Z_99_9": z999,
            "link_mask": masks[ap],  # P7: link set K_i of the AP (encode_obs match)
        })
    return out


def _stub_policy(act_msg: Any) -> None:
    for ap in range(N_AP):
        for sta in range(N_STA_PER_AP):
            act_msg.set_selected_link(ap, sta, 0)
    act_msg.mapMode = 0


def _rssi_policy(env_msg: Any, act_msg: Any) -> None:
    """Fixed RSSI-MLO policy: per-STA best-CSI available link, mode none.

    Corresponds to _override_action_rssi_link (link=argmax csi) of the surrogate
    -- the common baseline used in calibration Stage 2 to exclude learning and
    compare the pure dynamics of the two simulators. CSI uses the obs value
    measured and delivered by ns-3 as is.
    """
    masks = _link_masks()
    csi_all = np.frombuffer(env_msg.csi(), dtype=np.float32).reshape(
        N_AP, N_STA_PER_AP, N_LINKS)
    for ap in range(N_AP):
        for sta in range(N_STA_PER_AP):
            avail = [l for l in range(N_LINKS) if masks[ap][l] > 0]
            best = max(avail, key=lambda l: csi_all[ap, sta, l]) if avail else 0
            act_msg.set_selected_link(ap, sta, best)
    act_msg.mapMode = 0


def _slci_policy(env_msg: Any, act_msg: Any) -> None:
    """SLCI (Lopez-Raventos & Bellalta, IEEE WCL 11(7), 2022): put the STAs of
    each BSS on the least-congested (CBR) link among the allowed set K_i -- the
    published least-congested-interface heuristic. ICC'23 MH-RSAC published
    numbers against this baseline, so it anchors the chain comparison. CBR is
    the obs value measured and delivered by ns-3."""
    masks = _link_masks()
    cbr = np.frombuffer(env_msg.cbr(), dtype=np.uint8).reshape(
        N_AP, N_LINKS).astype(np.float32)
    for ap in range(N_AP):
        avail = [l for l in range(N_LINKS) if masks[ap][l] > 0]
        best = min(avail, key=lambda l: cbr[ap, l]) if avail else 0
        for sta in range(N_STA_PER_AP):
            act_msg.set_selected_link(ap, sta, best)
    act_msg.mapMode = 0


def _actor_policy(
    env_msg: Any,
    act_msg: Any,
    actors: List[Any],
    sample_action_fn: Any,
    rng: np.random.Generator,
    device: str,
    no_coord: bool = True,
) -> None:
    import torch  # type: ignore

    from models.networks import (  # type: ignore
        N_MAP_MODES,
        encode_obs,
        mask_link_logits,
        mask_map_logits,
    )

    obs_per_ap = _ns3_obs_to_python(env_msg)
    masks = _link_masks()
    map_modes = []
    feats_all: List[np.ndarray] = [] if _OBS_DUMP_ON else []
    for ap in range(N_AP):
        feat = encode_obs(obs_per_ap[ap])
        if _OBS_DUMP_ON:
            feats_all.append(feat.astype(np.float32))
        with torch.no_grad():
            logits = actors[ap](
                torch.from_numpy(feat).to(device).unsqueeze(0)
            ).squeeze(0)
        # P3 fix: logit masking at the same point and with the same
        # implementation as training -- without it the evaluation policy can
        # argmax into a disallowed link / frozen mode (train/eval action
        # distribution mismatch).
        logits = mask_link_logits(logits, masks[ap], N_STA_PER_AP, N_LINKS)
        if no_coord:
            logits = mask_map_logits(logits, N_STA_PER_AP, N_LINKS, N_MAP_MODES)
        _, indices, _ = sample_action_fn(
            logits, N_STA_PER_AP, N_LINKS, rng, deterministic=True,
        )
        for sta in range(N_STA_PER_AP):
            link = int(indices["selected_links"][sta])
            act_msg.set_selected_link(ap, sta, link)
        map_modes.append(int(indices["map_mode"]))
    if _OBS_DUMP_ON:
        _OBS_LIST.append(np.stack(feats_all, axis=0))  # (N_AP, state_dim)
    # map_mode can differ per AP but ActMsg carries a single mapMode ->
    # majority vote.
    act_msg.mapMode = int(np.bincount(map_modes, minlength=3).argmax())
    if os.environ.get("FEDDRL_DEBUG"):
        qsum = int(np.frombuffer(env_msg.queueLen(), dtype=np.uint32).sum())
        csum = int(np.frombuffer(env_msg.cbr(), dtype=np.uint8).sum())
        back = np.frombuffer(act_msg.selectedLink(), dtype=np.int8).reshape(
            N_AP, N_STA_PER_AP)
        v99 = np.frombuffer(env_msg.violation99(), dtype=np.uint32).tolist()
        srv = np.frombuffer(env_msg.served(), dtype=np.uint32).tolist()
        sys.stderr.write(
            f"[dbg] served={srv} viol99={v99} "
            f"Z99={_Z99_RUNNING.round(2).tolist()} "
            f"mode={act_msg.mapMode} links={back[0].tolist()}\n"
        )


def _run_loop(
    msg_interface: Any,
    actors: Optional[List[Any]],
    sample_action_fn: Any,
    rng: np.random.Generator,
    device: str,
    rssi: bool = False,
    slci: bool = False,
    no_coord: bool = False,
    balance_links: bool = False,
) -> int:
    n_slots = 0
    mode_hist = [0, 0, 0]   # how often each map_mode is sent (0 = no coord)
    link_hist = [0, 0, 0]   # per-link STA-selection counts (concentration)
    try:
        while True:
            msg_interface.PyRecvBegin()
            if msg_interface.PyGetFinished():
                msg_interface.PyRecvEnd()
                break
            env_msg = msg_interface.GetCpp2PyStruct()
            if _METRIC_ON and _METRIC_ACC is not None:
                _metric_accumulate(env_msg, _METRIC_ACC)

            msg_interface.PyRecvEnd()

            msg_interface.PySendBegin()
            act_msg = msg_interface.GetPy2CppStruct()
            if rssi:
                _rssi_policy(env_msg, act_msg)
            elif slci:
                _slci_policy(env_msg, act_msg)
            elif actors is None:
                _stub_policy(act_msg)
            else:
                _actor_policy(env_msg, act_msg, actors, sample_action_fn,
                              rng, device, no_coord=no_coord)
            if no_coord:
                act_msg.mapMode = 0
            if balance_links:
                for ap in range(N_AP):
                    for sta in range(N_STA_PER_AP):
                        act_msg.set_selected_link(
                            ap, sta, (ap * N_STA_PER_AP + sta) % N_LINKS)
            # Record what was actually sent this slot for mechanism diagnosis:
            # coordination-gating frequency and link concentration.
            m = int(act_msg.mapMode)
            if 0 <= m < 3:
                mode_hist[m] += 1
            for lk in np.frombuffer(act_msg.selectedLink(), dtype=np.int8).tolist():
                if 0 <= lk < N_LINKS:
                    link_hist[lk] += 1
            msg_interface.PySendEnd()
            n_slots += 1
    except Exception as e:  # pragma: no cover
        exc_type, exc_value, exc_traceback = sys.exc_info()
        sys.stderr.write(f"[feddrl.py] Exception in loop: {e}\n")
        traceback.print_tb(exc_traceback)
        raise
    sys.stderr.write(f"[ACTDIAG] modes={mode_hist} links={link_hist}\n")
    return n_slots


def main() -> int:
    parser = argparse.ArgumentParser(description="ns3-ai FedDRL driver")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrival-pps", type=float, default=5500.0)
    parser.add_argument(
        "--episode-ms",
        type=float,
        default=180.0,
        help="Measurement episode length (ms) after warmup; larger gives more "
             "packets for a stable KPI at low load (feasibility probing).",
    )
    parser.add_argument(
        "--macro-slot-ms",
        type=float,
        default=20.0,
        help="RL decision/measurement window (ms) forwarded to ns-3. MUST match "
             "the value used in training (train_ns3.py --macro-slot-ms=20); a "
             "mismatch makes the policy see a different obs/action cadence than it "
             "was trained on. Omitting it let ns-3 default to 5ms (train/eval skew).",
    )
    parser.add_argument(
        "--settle-ms",
        type=float,
        default=0.0,
        help="Queue-settle warmup (ms) after the association warmup. Traffic "
             "starts with empty queues; KPI counting is reset after this settle "
             "so the queue-buildup transient is excluded -> steady-state UHR. "
             "Default 0 = legacy (transient-contaminated). Use e.g. 1000 for the "
             "steady-state eval; policies need no retrain (measurement-only).",
    )
    parser.add_argument(
        "--load-spread",
        type=float,
        default=0.0,
        help="Per-AP load heterogeneity in [0,1] forwarded to the scenario. "
             "0 = homogeneous; >0 gives AP0 the full arrivalPps down to AP15 at "
             "(1-load_spread)*arrivalPps -> heterogeneous per-AP Z. MUST match the "
             "value used in training.",
    )
    parser.add_argument(
        "--uhr-ac",
        action="store_true",
        help="Enable the dedicated learning-controlled UHR access category: "
             "route UHR flows to AC_VI whose per-BSS EDCA is set at runtime. "
             "Off = legacy single-class AC_BE (byte-identical).",
    )
    parser.add_argument(
        "--bg-pps",
        type=float,
        default=0.0,
        help="Per-band best-effort background load (pkts/s) on AC_BE to make the "
             "medium contention-limited. 0 = no background (legacy).",
    )
    parser.add_argument(
        "--bg-link-skew",
        type=float,
        default=0.0,
        help="Per-link background asymmetry in [0,1] (sum preserved): skew=1 "
             "clears link0 and loads the top link, creating a clear link for UHR "
             "steering to exploit. 0 = symmetric background (legacy).",
    )
    parser.add_argument(
        "--uhr-edca-spread",
        type=float,
        default=0.0,
        help="Static UHR-AC heuristic strength in [0,1] (validation only): sets "
             "each BSS's UHR-AC aggressiveness by load rank (AP0 hottest -> most "
             "aggressive). 0 = all neutral. Overridden by the RL action.",
    )
    parser.add_argument(
        "--uhr-level",
        type=int,
        default=-1,
        help="Uniform UHR-AC level override (0..3) applied to ALL BSSs; -1 = use "
             "the spread heuristic. Diagnostic: does within-BSS EDCA move the tail?",
    )
    parser.add_argument(
        "--rts-cts",
        type=int,
        default=4692000,
        help="RTS/CTS threshold (bytes); data frames >= this use RTS/CTS. 0 = RTS "
             "on every data frame (hidden-terminal protection). Default off.",
    )
    parser.add_argument(
        "--bg-edca",
        action="store_true",
        help="Diagnostic: apply per-BSS UHR level to AC_BE too, to test whether "
             "EDCA reallocates airtime for the clean backlogged OnOff background.",
    )
    parser.add_argument(
        "--uhr-onoff",
        action="store_true",
        help="Diagnostic: drive UHR via a backlogged OnOff generator (isolate the "
             "g_staBuf traffic architecture).",
    )
    parser.add_argument(
        "--baseline-tag",
        type=str,
        default="ns3_feddrl_ai",
        help="CSV baseline column tag passed to the ns-3 scenario.",
    )
    parser.add_argument(
        "--deadline99-ms",
        type=float,
        default=5.0,
        help="p99 UHR deadline (ms) forwarded to the scenario. Density-scaled: "
             "5 at 4-AP, 10 at 16-AP dense (structural tail floor precludes 5ms).",
    )
    parser.add_argument(
        "--deadline999-ms",
        type=float,
        default=10.0,
        help="p99.9 UHR deadline (ms) forwarded to the scenario. 10 at 4-AP, "
             "20 at 16-AP dense.",
    )
    parser.add_argument(
        "--seg-suffix",
        type=str,
        default="",
        help="Unique suffix for ns3-ai shared-memory object names so concurrent "
             "eval jobs don't collide. Empty = library defaults. Must match ns-3.",
    )
    parser.add_argument(
        "--ns3-path",
        type=str,
        default="../../../../",
        help=(
            "ns-3 root relative to the example directory. feddrl/ is at "
            "contrib/ai/examples/feddrl/ (4 levels deep)."
        ),
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "Trained actor checkpoint (.pt). If omitted, the stub policy is used."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=os.environ.get("REPO_ROOT", "."),
        help="LyMAPPO repo root (for importing models/).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for actor inference (cpu / cuda).",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="auto",
        choices=["auto", "rssi", "stub", "slci"],
        help="auto=ckpt actor (if present)/stub, rssi=fixed RSSI-MLO policy "
             "(for calibration), slci=published least-congested-interface "
             "heuristic (Lopez-Raventos & Bellalta, IEEE WCL 2022): pick the "
             "minimum-CBR link among the allowed links.",
    )
    parser.add_argument(
        "--no-coord",
        action="store_true",
        help="Force mapMode=0 (disable Co-TDMA gating) — ablation to isolate the "
             "coordination-delay contribution to a policy's ns-3 violation.",
    )
    parser.add_argument(
        "--balance-links",
        action="store_true",
        help="Override the policy's link choice with round-robin (perfectly "
             "balanced) — confirms link concentration is the cause of a policy's "
             "ns-3 violation excess.",
    )
    parser.add_argument(
        "--chan-dwell-ms", type=float, default=200.0,
        help="F6 Markov channel mean state dwell (ms). 0 = static channel "
             "(legacy).",
    )
    parser.add_argument(
        "--het-bands", type=int, default=1,
        help="F7 heterogeneous bands (2.4/5/6GHz, widths 20/40/80MHz). "
             "0 = legacy identical channel.",
    )
    parser.add_argument(
        "--link2-width", type=int, default=80,
        help="link2 (6GHz) channel width in MHz: 80 (default) or 40 (mitigates "
             "the churn retry storm).",
    )
    parser.add_argument(
        "--drain-target", type=int, default=8,
        help="F8 target per-link MAC queue depth for the shared-buffer "
             "drain-on-demand. 0 = direct-send legacy (A/B).",
    )
    parser.add_argument(
        "--z-clip", type=float, default=10.0,
        help="Upper bound of the dual variable Z -- MUST match training "
             "(train_ns3.py --z-clip). The eval Z reconstruction needs the same "
             "rule/scale as training to avoid OOD inputs.",
    )
    parser.add_argument(
        "--eps99", type=float, default=1e-2,
        help="99% violation-rate target -- must match training (used in the Z "
             "reconstruction).",
    )
    parser.add_argument(
        "--eps999", type=float, default=1e-3,
        help="99.9% violation-rate target -- must match training (used in the Z "
             "reconstruction).",
    )
    parser.add_argument(
        "--no-lyapunov", action="store_true",
        help="For evaluating a no-Z ablation ckpt: keep the Z features of obs "
             "fixed at 0 as in training (skips the Z reconstruction).",
    )
    parser.add_argument(
        "--metric-log", action="store_true",
        help="r3 #1: accumulate per-slot (v99,v999,d) per AP and emit a "
             "[METRIC_AP] line at end with both eq.(1) per-slot-rate and packet "
             "metric forms + true per-slot d_min/d_max (Remark 1 verification).",
    )
    parser.add_argument(
        "--obs-dump", type=str, default=None,
        help="r3 #5: dump per-slot per-AP encoded obs (T,N_AP,state_dim) to this "
             ".npy path for offline federation action-divergence analysis.",
    )
    parser.add_argument(
        "--hol-norm-ms", type=float, default=None,
        help="HoL normalization reference for obs (ms). Default None = use "
             "deadline999. When the eval deadline is set differently from "
             "training (e.g. the 20/40ms deadline recalibration eval), pass the "
             "training-time value (20) to prevent an obs distribution shift.",
    )
    args = parser.parse_args()

    # Align the eval Z-reconstruction constants with the training setup
    # (module globals -- _ns3_obs_to_python).
    global EPS_99, EPS_999, Z_CLIP, NO_LYAP, _METRIC_ON, _METRIC_ACC, _OBS_DUMP_ON
    EPS_99 = args.eps99
    EPS_999 = args.eps999
    Z_CLIP = args.z_clip
    NO_LYAP = args.no_lyapunov
    _METRIC_ON = bool(args.metric_log)
    if _METRIC_ON:
        _METRIC_ACC = _metric_init()
    _OBS_DUMP_ON = args.obs_dump is not None

    # Ensure the WLAN repo root is importable for BOTH actor and RSSI paths.
    # _link_masks() pulls make_asymmetric_link_sets from models.topology;
    # in RSSI mode _load_actors (which used to add this) never runs.
    if args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    # P1/P6 fix: tie the obs normalization constants to the actual setup -- they
    # must equal training for the encode_obs input distribution to match (old:
    # a fixed Z/100 meant the actor could not see Z).
    # If --hol-norm-ms is given, the KPI deadline and the obs normalization are
    # decoupled (recalibration eval).
    import models.networks as _networks  # type: ignore
    _networks.Z_NORM = args.z_clip
    _networks.HOL_NORM_S = (
        args.hol_norm_ms if args.hol_norm_ms is not None
        else args.deadline999_ms) * 1e-3

    actors: Optional[List[Any]] = None
    sample_action_fn: Any = None
    if args.ckpt:
        if not os.path.isfile(args.ckpt):
            sys.stderr.write(f"[feddrl.py] ckpt not found: {args.ckpt}\n")
            return 2
        actors, sample_action_fn = _load_actors(
            args.ckpt, args.repo_root, args.device
        )
        sys.stderr.write(
            f"[feddrl.py] loaded actor ckpt: {args.ckpt}\n"
        )

    rng = np.random.default_rng(args.seed)

    import ns3ai_feddrl_py as py_binding  # type: ignore
    from ns3ai_utils import Experiment  # type: ignore

    target = "ns3ai_feddrl"
    setting = {
        "seed": args.seed,
        "arrivalPps": args.arrival_pps,
        "baselineTag": args.baseline_tag,
        "episodeDurationMs": args.episode_ms,
        "macroSlotMs": args.macro_slot_ms,
        "deadline99Ms": args.deadline99_ms,
        "deadline999Ms": args.deadline999_ms,
        "settleMs": args.settle_ms,
        "loadSpread": args.load_spread,
        "uhrAc": 1 if args.uhr_ac else 0,
        "bgPps": args.bg_pps,
        "bgLinkSkew": args.bg_link_skew,
        "chanDwellMs": args.chan_dwell_ms,
        "hetBands": args.het_bands,
        "drainTarget": args.drain_target,
        "link2Width": args.link2_width,
        "uhrEdcaSpread": args.uhr_edca_spread,
        "uhrLevel": args.uhr_level,
        "rtsCts": args.rts_cts,
        "bgEdca": 1 if args.bg_edca else 0,
        "uhrOnoff": 1 if args.uhr_onoff else 0,
    }
    seg = args.seg_suffix
    if seg:
        setting["segSuffix"] = seg
        exp = Experiment(target, args.ns3_path, py_binding, handleFinish=True,
                         segName=f"ns3ai_seg_{seg}",
                         cpp2pyMsgName=f"ns3ai_c2p_{seg}",
                         py2cppMsgName=f"ns3ai_p2c_{seg}",
                         lockableName=f"ns3ai_lock_{seg}")
    else:
        exp = Experiment(target, args.ns3_path, py_binding, handleFinish=True)
    msg_interface = exp.run(setting=setting, show_output=True)

    n_slots = 0
    try:
        n_slots = _run_loop(
            msg_interface, actors, sample_action_fn, rng, args.device,
            rssi=(args.policy == "rssi"),
            slci=(args.policy == "slci"),
            no_coord=args.no_coord,
            balance_links=args.balance_links,
        )
    finally:
        sys.stderr.write(
            f"[feddrl.py] handled {n_slots} macro slots; cleaning up.\n"
        )
        del exp
    if _METRIC_ON and _METRIC_ACC is not None:
        _metric_report(_METRIC_ACC)
    if _OBS_DUMP_ON and _OBS_LIST:
        arr = np.stack(_OBS_LIST, axis=0).astype(np.float32)  # (T, N_AP, dim)
        os.makedirs(os.path.dirname(args.obs_dump), exist_ok=True)
        np.save(args.obs_dump, arr)
        sys.stderr.write(
            f"[feddrl.py] obs dump saved: {args.obs_dump} shape={arr.shape}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
