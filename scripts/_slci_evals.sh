#!/usr/bin/env bash
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export NS3_ROOT="${NS3_ROOT:-$HOME/ns-3-dev}"
# SLCI published baseline (Lopez-Raventos & Bellalta, IEEE WCL 2022) at the
# settled protocol: 8 eval seeds {10,12..18} (es11 excluded), 11s episodes.
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EX=$NS3_ROOT/contrib/ai/examples/feddrl
RES=$REPO_ROOT/results/final_l2w40
cp -f $REPO_ROOT/drivers/feddrl.py "$EX/feddrl.py"
cd "$EX" || exit 1
EVAL_COMMON="--arrival-pps 60 --episode-ms 11000 --macro-slot-ms 20 --settle-ms 1000 --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 --link2-width 40 --no-coord"
one() { OF="$1"; TAG="$2"; shift 2
  timeout --signal=INT --kill-after=20 2400 \
    python3 -u -B "$EX/feddrl.py" $EVAL_COMMON --seg-suffix "$TAG" "$@" \
    2>&1 | grep -aE '^\[(KPI|KPI_AP|DIAG)|Traceback|Error' > "$RES/$OF"
  pkill -9 -f "segSuffix=$TAG" 2>/dev/null
}
NPAR=0
q() { one "$@" & NPAR=$((NPAR+1)); if [ "$NPAR" -ge 8 ]; then wait; NPAR=0; fi; }
for ES in 10 12 13 14 15 16 17 18; do
  q "base_slci_es${ES}.txt" "sl_${ES}" --seed "$ES" --policy slci
done
wait
echo "=== slci evals done ==="
ls "$RES"/base_slci_*.txt | wc -l
