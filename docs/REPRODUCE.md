# Reproduction and verification procedure

This document describes the experiment batteries, the settled protocol,
and the statistics conventions, so that results can be independently
re-run and checked.

## Settled protocol (all batteries share it)

| Parameter | Value |
|---|---|
| Topology | 16 AP (4×4 grid, 8 m pitch), 5 STA/AP on 3 m ring |
| Links | 2.4 GHz/20 MHz, 5 GHz/40 MHz, 6 GHz/40 MHz; per-AP link sets tiled `{0,1},{0,1},{1,2},{2}` |
| Load | λ = 60 pkt/s/STA (CBR 1500 B), zero-sum linear spread 0.6 (×1.6 → ×0.4) |
| Channel | 3-state Markov h∈{0.9,0.7,0.5}, mean dwell 200 ms |
| Deadlines / targets | (L99, L999) = (20, 40) ms; ε99 = 1e-2, ε999 = 1e-3 |
| Macro-slot | 20 ms |
| Training | 100 iterations × 32-slot rollouts (persistent episode) |
| Evaluation | best checkpoint (per-AP CMDP infeasibility score), held-out channel seeds; main protocol 110 s episodes, development protocol 11 s |

## Wall-clock expectations (single machine)

- One **11 s** evaluation episode ≈ **~3 minutes** wall-clock.
- One **110 s** evaluation episode ≈ **~45 minutes** wall-clock.
- KPI lines are emitted **at the end** of an episode — a silent long run
  is normal. Do not kill runs with SIGINT expecting partial KPIs.
- Keep the number of parallel runs ≤ physical cores.

## Batteries (`scripts/`)

| Script | What it runs |
|---|---|
| `_final_matrix.sh` | main arms: independent base (s0-7), federated zq/uniform/iw variants, RR/RSSI baselines |
| `_seed_ext.sh` | seed extension s8-15 for the full method and its base (n=16) |
| `_ab_arms.sh` | PX / QR component ablation arms |
| `_fairfl_arms.sh` | q-FFL and AFL federated arms |
| `_cluster_arm.sh` | clustered FL (equal link-set groups) |
| `_noz_arm.sh` | no-Z ablation (reward = Vu only) |
| `_slci_evals.sh`, `_slci_fill.sh` | SLCI static baseline over evaluation seeds |
| `_load_sweep.sh` | λ ∈ {40, 80} retraining sweep |
| `_beta_sweep.sh` | PX weight β sensitivity |
| `_alphak_sweep.sh` | quantile-critic α / K sensitivity |
| `_longeval.sh`, `_longeval2.sh` | 110 s main-protocol re-evaluation of stored checkpoints |
| `_r3_baselines_full.sh` | learned baselines: shared-actor MAPPO and PPO-Lagrangian training |
| `_r3_eval11s.sh` | like-for-like 11 s evaluation of the three learned arms |

All scripts auto-detect `REPO_ROOT` and default `NS3_ROOT=$HOME/ns-3-dev`;
results land under `$REPO_ROOT/results/`.

## Aggregation and statistics

```bash
python3 -B scripts/_core_cmp_analyze.py results/<dir>
python3 -B scripts/_r3_baselines_analyze.py    # learned-baseline table
```

Conventions (enforced by the analyzers, please keep them when comparing):

- The statistical unit is the **training seed** for learned arms
  (each value averaged over held-out channel seeds) and the
  **evaluation channel seed** for static baselines. **No pooled tests.**
- Report **mean ± std across seeds** — never "k of n compliant" counts
  alone, and never selected seeds.
- Pairwise comparisons: two-sided Mann-Whitney at seed level; apply
  Holm correction within a declared test family.
- Pre-specified metrics: per-AP feasibility count (of 16),
  worst-AP p99.9, network p99, network p99.9. Here p99/p99.9 denote
  deadline-violation *rates* against the 20/40 ms deadlines, not latency
  percentiles.

## Documented exclusions

- Evaluation channel seed **11** triggers a pre-existing ns-3.40 PHY
  assertion (`phy-entity.cc:509`) and is excluded a priori (substituted).
- A small number of cells crash with the same assertion at any episode
  length; they are excluded and must be listed when reporting.
- Figures: `python3 -B scripts/figures/fig_*.py` reads `results/` and
  writes `figs/`.
