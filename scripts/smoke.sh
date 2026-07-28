#!/usr/bin/env bash
# smoke.sh — minimal end-to-end check: a short LyMAPPO training run
# (few iterations, small rollout) against the built ns-3 scenario.
# Expected: [it ...] progress lines, then a saved checkpoint. ~minutes.
set -u
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export NS3_ROOT="${NS3_ROOT:-$HOME/ns-3-dev}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2

EX="$NS3_ROOT/contrib/ai/examples/feddrl"
if [ ! -d "$EX" ]; then
    echo "[smoke] ERROR: $EX not found — run install.sh first"
    exit 1
fi
cp -f "$REPO_ROOT"/drivers/*.py "$EX/"
cp -f "$REPO_ROOT/models/mappo.py" "$REPO_ROOT/models/networks.py" "$EX/"
mkdir -p "$REPO_ROOT/results"
cd "$EX" || exit 1

python3 -B train_ns3.py \
    --aggregation none --seed 0 \
    --arrival-pps 60 --iterations 8 --rollout 32 --macro-slot-ms 20 \
    --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 \
    --link2-width 40 --eps99 1e-2 --eps999 1e-3 \
    --ckpt-out "$REPO_ROOT/results/smoke_s0.pt"
rc=$?
if [ $rc -eq 0 ] && [ -f "$REPO_ROOT/results/smoke_s0.pt" ]; then
    echo "[smoke] OK — checkpoint written to results/smoke_s0.pt"
else
    echo "[smoke] FAILED (rc=$rc)"
    exit 1
fi
