#!/usr/bin/env bash
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export NS3_ROOT="${NS3_ROOT:-$HOME/ns-3-dev}"
# Review response (weakness #1): rare-event re-evaluation at 110 s episodes
# (10x) -> ~33k decided/AP/run, ~1e5 per train-seed across es{10,12,18}.
# Scope: T1 arms only (pxqr n=16, none n=16, baselines n=8 eval seeds).
# Checkpoints reused from final_l2w40 (no retraining).
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EX=$NS3_ROOT/contrib/ai/examples/feddrl
SRC=$REPO_ROOT/results/final_l2w40
RES=$REPO_ROOT/results/longeval
mkdir -p "$RES"
cp -f $REPO_ROOT/drivers/feddrl.py "$EX/feddrl.py"
cd "$EX" || exit 1
EC="--arrival-pps 60 --episode-ms 110000 --macro-slot-ms 20 --settle-ms 1000 --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 --link2-width 40 --no-coord"
one() { OF="$1"; TAG="$2"; shift 2
  timeout --signal=INT --kill-after=30 21600 \
    python3 -u -B "$EX/feddrl.py" $EC --seg-suffix "$TAG" "$@" \
    2>&1 | grep -aE '^\[(KPI|KPI_AP|DIAG)|Traceback|Error' > "$RES/$OF"
  pkill -9 -f "segSuffix=$TAG" 2>/dev/null
}
NPAR=0
q() { one "$@" & NPAR=$((NPAR+1)); if [ "$NPAR" -ge 10 ]; then wait; NPAR=0; fi; }
for ARM in pxqr none; do
  for TS in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    for ES in 10 12 18; do
      q "eval_${ARM}_ts${TS}_es${ES}.txt" "le_${ARM}_${TS}_${ES}" \
        --seed "$ES" --ckpt "$SRC/ckpt_${ARM}_s${TS}.pt"
    done
  done
done
for ES in 10 12 13 14 15 16 17 18; do
  q "base_rr_es${ES}.txt" "le_rr_${ES}" --seed "$ES" --policy stub --balance-links
  q "base_rssi_es${ES}.txt" "le_rs_${ES}" --seed "$ES" --policy rssi
  q "base_slci_es${ES}.txt" "le_sl_${ES}" --seed "$ES" --policy slci
done
wait
echo "=== long-eval complete ==="
ls "$RES"/*.txt | wc -l
