# LyMAPPO — ns-3-in-the-loop RL platform for Wi-Fi 8 (802.11bn) UHR experimentation

LyMAPPO is a reproducible simulation platform for studying **Ultra-High
Reliability (UHR)** — the per-BSS deadline-violation-rate targets of IEEE
802.11bn / Wi-Fi 8 — in **dense multi-band WLANs**, with reinforcement
learning agents trained *directly against ns-3* (full-stack 802.11
contention), not against an abstract surrogate.

It provides three things:

1. **An environment**: a 16-AP / 80-STA dense enterprise scenario on
   ns-3.40, with three bands (2.4/5/6 GHz), *asymmetric* per-AP link sets,
   shared-buffer multi-link operation (service-time routing),
   3-state Markov channels, and **survivor-bias-free accounting**
   (every packet is *decided* — delivered or aged out — and dropped
   packets count as violations). The learner interacts through a
   20 ms macro-slot shared-memory interface (ns3-ai `EnvMsg`/`ActMsg`).
2. **Agents and baselines** trained in the same loop, on equal terms:
   LyMAPPO (independent per-AP PPO + rate-form Lyapunov duals + price
   coupling + quantile critic), shared-actor MAPPO, independent
   PPO-Lagrangian, federated aggregation (FedAvg, q-FFL, AFL,
   clustered FL), and static policies (round-robin, RSSI-greedy, SLCI).
3. **A verification procedure**: scripted experiment batteries with
   pre-specified metrics, seed-level statistics (mean±std,
   Mann-Whitney, Holm correction), and documented crash/seed rules —
   so third parties can re-run and check the numbers, or plug in their
   own policy and compare.

<p align="center"><img src="assets/fig_topology.png" alt="16-AP dense multi-band topology: 4x4 grid, asymmetric link sets tiled by column, zero-sum load spread" width="640"></p>

## Fidelity scope (please read)

The PHY/MAC base is **802.11ax**; MLO is emulated with per-link
single-link devices (not native 802.11be MLO), and 802.11bn features
under standardization (MAPC: Co-SR / Co-TDMA) are **not** implemented —
the coordination field exists in the action interface but is frozen to
`none`. What is 11bn-oriented here is the *problem*: per-BSS
deadline-violation-rate constraints at density. Native 11be MLO and
draft-inspired MAPC prototypes are on the [roadmap](docs/ROADMAP.md).

## Quick start

The simulation always builds and runs on **Ubuntu Linux** — either
natively or inside **WSL2 on Windows**; native Windows builds of ns-3
are not supported. Pick your track below. See
[docs/INSTALL.md](docs/INSTALL.md) for details and troubleshooting.

### Track A — Ubuntu 22.04 / 24.04 (native Linux)

```bash
# 1. dependencies (once, interactive sudo)
sudo apt-get install -y build-essential cmake ninja-build git python3-dev \
    pybind11-dev libgtk-3-dev libxml2-dev libssl-dev libsqlite3-dev sqlite3 \
    libboost-dev libboost-program-options-dev libboost-system-dev \
    libboost-filesystem-dev libprotobuf-dev protobuf-compiler gcc-12 g++-12

# 2. ns-3.40 + this repo
git clone --depth 1 --branch ns-3.40 https://gitlab.com/nsnam/ns-3-dev.git ~/ns-3-dev
git clone https://github.com/yaki-toki/LyMAPPO.git && cd LyMAPPO
pip install -r requirements.txt

# 3. build (installs ns3-ai, syncs scenario+drivers, builds ns-3)
bash install.sh

# 4. smoke: one short training run (a few minutes)
bash scripts/smoke.sh
```

### Track B — Windows 10/11 (via WSL2)

One-time setup — install WSL2 with Ubuntu (administrator PowerShell,
then reboot):

```powershell
wsl --install -d Ubuntu-24.04
```

Open the Ubuntu shell (`wsl`) and follow **Track A inside the WSL
filesystem** (clone into `~`, not under `/mnt/c` or `/mnt/d`): the WSL
filesystem is much faster for the ns-3 build and a Linux checkout
guarantees LF line endings (`.gitattributes` enforces this either way).
`install.sh` automatically isolates the build from the inherited
Windows PATH, so a standard Windows development setup does not
interfere.

Two things work directly with **Windows Python** (no ns-3, no WSL
needed) — handy for developing policies and plotting on the host:

```powershell
git clone https://github.com/yaki-toki/LyMAPPO.git ; cd LyMAPPO
pip install -r requirements.txt
python -B -m pytest tests -q          # host-side unit tests (32 tests)
python -B scripts/figures/fig_topology.py   # figures from results/ CSVs
```

You can also drive WSL runs from PowerShell without opening a shell:

```powershell
wsl bash -c "cd ~/LyMAPPO && bash scripts/smoke.sh"
```

## Repository layout

| Path | Contents |
|---|---|
| `scenario/` | ns-3 scenario (C++): topology, channels, MLO queueing, KPI accounting, ns3-ai message interface |
| `drivers/` | `train_ns3.py` (training: LyMAPPO / MAPPO / PPO-Lagrangian / FL arms), `feddrl.py` (evaluation + static policies) |
| `models/` | `networks.py` (actor/critic, observation encoding), `mappo.py` (PPO/GAE, aggregators) |
| `scripts/` | experiment batteries (`_*.sh`) + `_core_cmp_analyze.py` (seed-level statistics) |
| `scripts/figures/` | figure generation from result CSVs |
| `sim/` | Python-side environment utilities shared with the drivers (link-set definitions, EDCA constants) and a fast standalone surrogate environment — **not** the evidence path of the paper (that is ns-3), kept for unit testing and quick iteration |
| `tests/` | unit tests (aggregation, FL baselines, observation encoding, surrogate environment) |
| `docs/` | INSTALL, REPRODUCE, ROADMAP |

## Run an experiment

Every battery script follows the same pattern: it syncs `drivers/` and
`models/` into the ns-3 example directory, then launches training or
evaluation runs whose KPI lines are appended to CSV/text files under
`results/`. Paths are controlled by two environment variables
(defaults shown):

```bash
REPO_ROOT=<this checkout>   # auto-detected by the scripts
NS3_ROOT=$HOME/ns-3-dev
```

Example — train one LyMAPPO seed and evaluate it:

```bash
cd $NS3_ROOT/contrib/ai/examples/feddrl
python3 train_ns3.py --aggregation none --price-beta 0.05 --seed 0 \
    --arrival-pps 60 --iterations 100 --rollout 32 --macro-slot-ms 20 \
    --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 \
    --ckpt-out $REPO_ROOT/results/my_run_s0.pt
```

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for the full battery list,
the settled protocol constants, expected wall-clock times, and the
statistics conventions used to compare arms.

## Representative results

Approximate numbers from the settled operating point (λ = 60 pkt/s/STA,
deadlines 20/40 ms, 110 s evaluation episodes, held-out channel seeds;
mean ± std across seeds — reproduce with the batteries in
[docs/REPRODUCE.md](docs/REPRODUCE.md)):

| Policy | Feasible BSSs (of 16) ↑ | Network p99 ↓ | Network p99.9 ↓ | Delivered (of 4800 pkt/s) |
|---|---|---|---|---|
| **LyMAPPO (full)** | **15.1 ± 0.6** | **0.0017** | **0.0013** | 4794 |
| Round-robin | 9.0 ± 1.2 | 0.0103 | 0.0040 | 4783 |
| RSSI-greedy | 7.8 ± 0.7 | 0.0158 | 0.0068 | 4773 |
| SLCI (least-congested) | 3.9 ± 0.4 | 0.0423 | 0.0097 | 4754 |

Every arm delivers ≥ 99 % of offered traffic, so the differences above
are tail-shape differences, not capacity differences. The feasibility
and network-p99 margins over the static baselines are significant under
multiple-comparison (Holm) correction; the deeper p99.9 tails are
weaker (raw seed-level only vs. round-robin). Learned baselines trained
in the same loop land closer — LyMAPPO 14.6 ± 0.9 vs. PPO-Lagrangian
13.8 ± 1.5 vs. shared-actor MAPPO 12.4 ± 3.2 feasible BSSs (11 s
protocol) — a point-estimate lead that is within noise at n = 8 seeds.
Federated parameter aggregation (FedAvg, q-FFL, AFL, clustered FL)
stays *below* independent learning on held-out conditions (9.8–11.2
feasible BSSs), which is the platform's core cautionary finding.

<p align="center"><img src="assets/fig_seeds.png" alt="Feasibility vs network p99 across arms (seed-level mean and dispersion)" width="560"></p>

<p align="center"><img src="assets/fig_sweep.png" alt="Load sweep 40-80 pkt/s/STA: feasible APs and network p99 per policy" width="560"></p>

## Develop your own policy

Two entry points, in increasing depth:

- **Evaluation-only policy**: add a branch to `feddrl.py --policy`
  (see `rr` / `rssi` / `slci`) that maps the per-slot observation to a
  per-STA link choice.
- **Trained policy**: `train_ns3.py` exposes the training loop —
  observation encoding (106-D per AP: CSI, log backlog, HoL age, CBR,
  duals, link mask), masked per-STA link logits, and reward shaping —
  behind CLI switches. A cleaner plug-in `Policy` API is planned
  (see [roadmap](docs/ROADMAP.md)).

## Known issues

- ns-3 channel seed 11 triggers a pre-existing PHY assertion in
  ns-3.40 (`phy-entity.cc:509`); the batteries exclude and document it.
- ns3-ai is cloned from upstream master; if a future upstream change
  breaks the build, pin the version noted in `docs/INSTALL.md`.

## License

- `scenario/` (C++, links against ns-3): **GPL-2.0** — see `scenario/LICENSE`.
- Everything else (Python, scripts, docs): **MIT** — see `LICENSE`.

## Citation

If you use this platform, please cite the repository (see
`CITATION.cff`). The accompanying paper is under review; the citation
will be updated on publication.
