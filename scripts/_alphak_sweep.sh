#!/usr/bin/env bash
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export NS3_ROOT="${NS3_ROOT:-$HOME/ns-3-dev}"
# Review sensitivity grid, remaining axes: CVaR alpha in {0.1, 0.5} (K=8)
# and quantile count K in {16, 32} (alpha=0.25); beta=0.05 fixed everywhere.
# alpha=0.25/K=8 == existing pxqr arm (reused). s0-7, evals es{10,12,18}.
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EX=$NS3_ROOT/contrib/ai/examples/feddrl
RES=$REPO_ROOT/results/alphak_sweep
mkdir -p "$RES"
: > "$RES/done11.txt"
cp -f $REPO_ROOT/drivers/train_ns3.py "$EX/train_ns3.py"
cp -f $REPO_ROOT/models/mappo.py "$EX/mappo.py"
cd "$EX" || exit 1
TC="--arrival-pps 60 --iterations 100 --rollout 32 --macro-slot-ms 20 --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 --link2-width 40 --eps99 1e-2 --eps999 1e-3"
EC="--arrival-pps 60 --episode-ms 11000 --macro-slot-ms 20 --settle-ms 1000 --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 --link2-width 40 --no-coord"
tr_run() { ARM="$1"; A="$2"; K="$3"; S="$4"
  TAG="ak_${ARM}_${S}"
  ( timeout --signal=INT --kill-after=30 14400 \
      python3 -u -B "$EX/train_ns3.py" --aggregation none --seed "$S" \
      $TC --price-beta 0.05 --quantiles "$K" --cvar-alpha "$A" \
      --seg-suffix "$TAG" \
      --ckpt-out "$RES/ckpt_${ARM}_s${S}.pt" \
      --train-log-csv "$RES/train_${ARM}_s${S}.csv" \
      > "$RES/log_${ARM}_s${S}.txt" 2>&1
    echo "rc=$? train ${ARM} s${S}" >> "$RES/done11.txt" ) &
  sleep 3
}
for S in 0 1 2 3 4 5 6 7; do
  tr_run pxa01 0.1 8 "$S"
  tr_run pxa05 0.5 8 "$S"
  tr_run pxk16 0.25 16 "$S"
  tr_run pxk32 0.25 32 "$S"
done
wait
echo "=== alpha/K trainings done ==="
one() { OF="$1"; TAG="$2"; shift 2
  timeout --signal=INT --kill-after=20 2400 \
    python3 -u -B "$EX/feddrl.py" $EC --seg-suffix "$TAG" "$@" \
    2>&1 | grep -aE '^\[(KPI|KPI_AP|DIAG)|Traceback|Error' > "$RES/$OF"
  pkill -9 -f "segSuffix=$TAG" 2>/dev/null
}
NPAR=0
q() { one "$@" & NPAR=$((NPAR+1)); if [ "$NPAR" -ge 8 ]; then wait; NPAR=0; fi; }
for ARM in pxa01 pxa05 pxk16 pxk32; do
  for TS in 0 1 2 3 4 5 6 7; do
    for ES in 10 12 18; do
      q "eval_${ARM}_ts${TS}_es${ES}.txt" "ake_${ARM}_${TS}_${ES}" \
        --seed "$ES" --ckpt "$RES/ckpt_${ARM}_s${TS}.pt"
    done
  done
done
wait
echo "=== alpha/K sweep complete ==="
cat "$RES/done11.txt" | sort | uniq -c | tail -3
