# Installation

Target environment: **Ubuntu 22.04 / 24.04** — native Linux, or WSL2 on
Windows. Native Windows builds of ns-3 are not supported.

## 0. Windows hosts: WSL2 setup (skip on native Linux)

```powershell
# administrator PowerShell, then reboot
wsl --install -d Ubuntu-24.04
```

Recommendations for WSL2:

- Clone both this repo and ns-3 **inside the WSL filesystem** (`~`),
  not under `/mnt/c` / `/mnt/d`: builds are several times faster and
  the checkout is guaranteed LF.
- `install.sh` pins the build PATH to Linux system directories, so
  Windows toolchains (mingw, Anaconda, CUDA) inherited into WSL's PATH
  cannot leak headers into the ns-3 build.
- Host-side unit tests (`python -B -m pytest tests -q`) and the figure
  scripts also run with Windows Python — only the ns-3 build and the
  actual simulations require the Ubuntu side.
- To script runs from PowerShell: `wsl bash -c "cd ~/LyMAPPO && bash scripts/smoke.sh"`.

| Component | Version |
|---|---|
| ns-3 | **3.40** (pinned — the scenario and the build workarounds below are validated against it) |
| ns3-ai | upstream master (tested 2026-07); ctrl-plane = shared memory |
| Compiler | g++ 11/12 (**not** GCC 13 — see workarounds) |
| Python | 3.10+ with `requirements.txt` (numpy, scipy, torch, matplotlib, pytest) |
| CMake | 3.24+ |

## 1. System packages (once)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git \
    python3-dev python3-pip pybind11-dev libgtk-3-dev libxml2-dev \
    libssl-dev libsqlite3-dev sqlite3 \
    libboost-dev libboost-program-options-dev libboost-system-dev \
    libboost-filesystem-dev libprotobuf-dev protobuf-compiler \
    gcc-12 g++-12
```

## 2. ns-3.40

```bash
git clone --depth 1 --branch ns-3.40 https://gitlab.com/nsnam/ns-3-dev.git ~/ns-3-dev
```

(Any location works; export `NS3_ROOT=/path/to/ns-3-dev` if not `~/ns-3-dev`.)

## 3. Python packages

```bash
pip install -r requirements.txt
```

## 4. Build

```bash
bash install.sh
```

`install.sh` performs, idempotently:

1. clones ns3-ai into `$NS3_ROOT/contrib/ai` and runs its installer;
2. prunes ns3-ai examples that are incompatible with the ns-3.40 API
   (`rate-control/thompson-sampling`, `rl-tcp`, `multi-bss`);
3. syncs `scenario/` + `drivers/` + `models/{mappo,networks}.py` into
   `$NS3_ROOT/contrib/ai/examples/feddrl/` and registers the
   subdirectory in the examples CMakeLists;
4. configures ns-3 (`default` profile, examples on, tests off) and builds
   with `-DNS3_AI_AVAILABLE`.

## Build workarounds (why install.sh does what it does)

These were all hit in practice; `install.sh` applies them automatically.

| Issue | Workaround |
|---|---|
| GCC 13.3 (Ubuntu 24.04 default) internal compiler error (`try_forward_edges`, cfgcleanup) on ns-3.40 callback headers | force `CC=gcc-12 CXX=g++-12` when available |
| `optimized` profile (`-O3 -march=native`) triggers the same ICE class | build profile `default` (-O2) |
| WSL inherits the Windows PATH; mingw/Anaconda headers leak into CMake include search and break the stats module / cairo linking | PATH fixed to Linux-only during build; `CPATH` etc. unset |
| Some ns3-ai bundled examples do not compile against ns-3.40 | pruned once, marker-protected |
| ns3-ai needs boost + protobuf beyond the ns-3 docs | included in the apt line above |

## 5. Verify

```bash
bash scripts/smoke.sh   # short training run; ends with a saved checkpoint
python3 -B -m pytest tests/ -q   # host-side unit tests (no ns-3 needed)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pybind11` not found | `pip install pybind11 && export pybind11_DIR=$(python -m pybind11 --cmakedir)` |
| ns3-ai build failure | check ns-3 version is exactly 3.40; re-run `install.sh` |
| Shell scripts fail with `\r` errors | your checkout mangled line endings; `.gitattributes` enforces LF — re-checkout or `git config core.autocrlf input` |
| Runs are extremely slow | evaluation episodes are long by design; see wall-clock expectations in REPRODUCE.md. Keep concurrent runs ≤ CPU cores. |
