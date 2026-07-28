#!/usr/bin/env bash
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export NS3_ROOT="${NS3_ROOT:-$HOME/ns-3-dev}"
# Heterogeneity-aware federation arm (cluster: FedAvg only within same-K_i groups),
# derived from the diagnosed failure mechanism (cross-K specialization destruction).
# Settled protocol, n=8, evals at es{10,12,18}. Into final_l2w40/.
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EX=$NS3_ROOT/contrib/ai/examples/feddrl
RES=$REPO_ROOT/results/final_l2w40
cp -f $REPO_ROOT/drivers/train_ns3.py "$EX/train_ns3.py"
cd "$EX" || exit 1
: > "$RES/done4.txt"
TRAIN_COMMON="--arrival-pps 60 --iterations 100 --rollout 32 --macro-slot-ms 20 --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 --link2-width 40 --eps99 1e-2 --eps999 1e-3"
EVAL_COMMON="--arrival-pps 60 --episode-ms 11000 --macro-slot-ms 20 --settle-ms 1000 --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 --link2-width 40 --no-coord"
for S in 0 1 2 3 4 5 6 7; do
  TAG="cl_${S}"
  ( timeout --signal=INT --kill-after=30 14400 \
      python3 -u -B "$EX/train_ns3.py" --aggregation cluster --seed "$S" \
      $TRAIN_COMMON --seg-suffix "$TAG" \
      --ckpt-out "$RES/ckpt_cluster_s${S}.pt" \
      --train-log-csv "$RES/train_cluster_s${S}.csv" \
      > "$RES/log_cluster_s${S}.txt" 2>&1
    echo "rc=$? train cluster s${S}" >> "$RES/done4.txt" ) &
  sleep 4
done
wait
echo "=== cluster trainings done ==="
one() { OF="$1"; TAG="$2"; shift 2
  timeout --signal=INT --kill-after=20 2400 \
    python3 -u -B "$EX/feddrl.py" $EVAL_COMMON --seg-suffix "$TAG" "$@" \
    2>&1 | grep -aE '^\[(KPI|KPI_AP|DIAG)|Traceback|Error' > "$RES/$OF"
  pkill -9 -f "segSuffix=$TAG" 2>/dev/null
}
NPAR=0
q() { one "$@" & NPAR=$((NPAR+1)); if [ "$NPAR" -ge 10 ]; then wait; NPAR=0; fi; }
for TS in 0 1 2 3 4 5 6 7; do
  for ES in 10 12 18; do
    q "eval_cluster_ts${TS}_es${ES}.txt" "ce_${TS}_${ES}" \
      --seed "$ES" --ckpt "$RES/ckpt_cluster_s${TS}.pt"
  done
done
wait
echo "=== cluster arm complete ==="
cat "$RES/done4.txt"
