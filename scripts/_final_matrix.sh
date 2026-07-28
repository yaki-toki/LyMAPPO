#!/usr/bin/env bash
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export NS3_ROOT="${NS3_ROOT:-$HOME/ns-3-dev}"
# FINAL evidence matrix under the settled protocol:
#   L60, spread 0.6, Markov dwell 200ms {0.9,0.7,0.5}, het bands with link2=40MHz
#   (retry-storm fix, verified), drain 8, deadlines 20/40ms (train == eval),
#   eps 1e-2/1e-3, 100 it, rollout 32.
# Phase 1: 17 trainings (none s0-7, zq/uniform/iw s0-2) + rr/rssi x es10-17.
# Phase 2: 51 ckpt evals (17 x es10-12). All into results/final_l2w40/.
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EX=$NS3_ROOT/contrib/ai/examples/feddrl
RES=$REPO_ROOT/results/final_l2w40
mkdir -p "$RES"
cp -f $REPO_ROOT/drivers/train_ns3.py "$EX/train_ns3.py"
cp -f $REPO_ROOT/drivers/feddrl.py "$EX/feddrl.py"
cp -f $REPO_ROOT/models/mappo.py "$EX/mappo.py"
cd "$EX" || exit 1
: > "$RES/done.txt"
TRAIN_COMMON="--arrival-pps 60 --iterations 100 --rollout 32 --macro-slot-ms 20 --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 --link2-width 40 --eps99 1e-2 --eps999 1e-3"
EVAL_COMMON="--arrival-pps 60 --episode-ms 11000 --macro-slot-ms 20 --settle-ms 1000 --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 --link2-width 40 --no-coord"
tr_run() { AGG="$1"; S="$2"
  TAG="fm_${AGG}_${S}"
  ( timeout --signal=INT --kill-after=30 14400 \
      python3 -u -B "$EX/train_ns3.py" --aggregation "$AGG" --seed "$S" \
      $TRAIN_COMMON --seg-suffix "$TAG" \
      --ckpt-out "$RES/ckpt_${AGG}_s${S}.pt" \
      --train-log-csv "$RES/train_${AGG}_s${S}.csv" \
      > "$RES/log_${AGG}_s${S}.txt" 2>&1
    echo "rc=$? train ${AGG} s${S}" >> "$RES/done.txt" ) &
  sleep 4
}
bl() { OF="$1"; TAG="$2"; shift 2
  ( timeout --signal=INT --kill-after=20 2400 \
      python3 -u -B "$EX/feddrl.py" $EVAL_COMMON --seg-suffix "$TAG" "$@" \
      2>&1 | grep -aE '^\[(KPI|KPI_AP|DIAG)|Traceback' > "$RES/$OF"
    pkill -9 -f "segSuffix=$TAG" 2>/dev/null
    echo "rc=0 base $OF" >> "$RES/done.txt" ) &
}
for S in 0 1 2 3 4 5 6 7; do tr_run none "$S"; done
for AGG in zq uniform iw; do for S in 0 1 2; do tr_run "$AGG" "$S"; done; done
for ES in 10 11 12 13 14 15 16 17; do
  bl "base_rr_es${ES}.txt" "fm_rr_${ES}" --seed "$ES" --policy stub --balance-links
  bl "base_rssi_es${ES}.txt" "fm_rssi_${ES}" --seed "$ES" --policy rssi
done
wait
echo "=== phase 1 (trainings + baselines) done ==="
cat "$RES/done.txt" | sort | uniq -c | tail -3
one() { OF="$1"; TAG="$2"; shift 2
  timeout --signal=INT --kill-after=20 2400 \
    python3 -u -B "$EX/feddrl.py" $EVAL_COMMON --seg-suffix "$TAG" "$@" \
    2>&1 | grep -aE '^\[(KPI|KPI_AP|DIAG)|Traceback|Error' > "$RES/$OF"
  pkill -9 -f "segSuffix=$TAG" 2>/dev/null
}
NPAR=0
q() { one "$@" & NPAR=$((NPAR+1)); if [ "$NPAR" -ge 11 ]; then wait; NPAR=0; fi; }
for TS in 0 1 2 3 4 5 6 7; do
  for ES in 10 11 12; do
    q "eval_none_ts${TS}_es${ES}.txt" "fe_no_${TS}_${ES}" \
      --seed "$ES" --ckpt "$RES/ckpt_none_s${TS}.pt"
  done
done
for AGG in zq uniform iw; do
  for TS in 0 1 2; do
    for ES in 10 11 12; do
      q "eval_${AGG}_ts${TS}_es${ES}.txt" "fe_${AGG}_${TS}_${ES}" \
        --seed "$ES" --ckpt "$RES/ckpt_${AGG}_s${TS}.pt"
    done
  done
done
wait
echo "=== final matrix complete ==="
ls "$RES"/eval_*.txt "$RES"/base_*.txt | wc -l
