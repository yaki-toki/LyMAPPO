#!/usr/bin/env bash
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export NS3_ROOT="${NS3_ROOT:-$HOME/ns-3-dev}"
# r3 #7 FULL: train MAPPO(shared) + CPO(PID-Lagrangian), seeds 0-7, 100 it x 32
# rollout, SAME env as the full method. Then eval on the 110s protocol {10,12,18}.
# Launch ONLY after the smoke passes. Mirrors experiments/_ab_arms.sh structure.
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EX=$NS3_ROOT/contrib/ai/examples/feddrl
RES=$REPO_ROOT/results/r3_baselines
mkdir -p "$RES"
cp -f $REPO_ROOT/drivers/train_ns3.py "$EX/train_ns3.py"
cp -f $REPO_ROOT/drivers/feddrl.py "$EX/feddrl.py"
cp -f $REPO_ROOT/models/mappo.py "$EX/mappo.py"
cp -f $REPO_ROOT/models/networks.py "$EX/networks.py"
cd "$EX" || exit 1
: > "$RES/done_train.txt"
TC="--arrival-pps 60 --iterations 100 --rollout 32 --macro-slot-ms 20 --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 --link2-width 40 --eps99 1e-2 --eps999 1e-3"
EC="--arrival-pps 60 --episode-ms 110000 --macro-slot-ms 20 --settle-ms 1000 --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 --link2-width 40 --no-coord"

# --- training phase (concurrency cap 8) ---
NTR=0
tr_run() { ARM="$1"; S="$2"; shift 2
  TAG="r3f_${ARM}_${S}"
  ( timeout --signal=INT --kill-after=30 14400 \
      python3 -u -B "$EX/train_ns3.py" --aggregation none --seed "$S" \
      $TC "$@" --seg-suffix "$TAG" \
      --ckpt-out "$RES/ckpt_${ARM}_s${S}.pt" \
      --train-log-csv "$RES/train_${ARM}_s${S}.csv" \
      > "$RES/log_${ARM}_s${S}.txt" 2>&1
    pkill -9 -f "segSuffix=$TAG" 2>/dev/null
    echo "rc=$? train ${ARM} s${S}" >> "$RES/done_train.txt" ) &
  NTR=$((NTR+1)); sleep 3
  if [ "$NTR" -ge 16 ]; then wait; NTR=0; fi
}
for S in 0 1 2 3 4 5 6 7; do
  tr_run mappo "$S" --shared-actor
  tr_run cpo   "$S" --cpo
done
wait
echo "=== trainings done ==="
cat "$RES/done_train.txt" | sort | uniq -c | tail

# --- eval phase (110s protocol, concurrency cap 10) ---
: > "$RES/done_eval.txt"
one() { OF="$1"; TAG="$2"; shift 2
  timeout --signal=INT --kill-after=20 4800 \
    python3 -u -B "$EX/feddrl.py" $EC --seg-suffix "$TAG" "$@" \
    2>&1 | grep -aE '^\[(KPI|KPI_AP|DIAG)|Traceback|Error' > "$RES/$OF"
  pkill -9 -f "segSuffix=$TAG" 2>/dev/null
  echo "$OF" >> "$RES/done_eval.txt"
}
NPAR=0
q() { one "$@" & NPAR=$((NPAR+1)); if [ "$NPAR" -ge 10 ]; then wait; NPAR=0; fi; }
for ARM in mappo cpo; do
  for TS in 0 1 2 3 4 5 6 7; do
    for ES in 10 12 18; do
      q "eval_${ARM}_ts${TS}_es${ES}.txt" "r3e_${ARM}_${TS}_${ES}" \
        --seed "$ES" --ckpt "$RES/ckpt_${ARM}_s${TS}.pt"
    done
  done
done
wait
echo "=== r3 #7 full battery complete ==="
ls "$RES"/eval_*.txt | wc -l
