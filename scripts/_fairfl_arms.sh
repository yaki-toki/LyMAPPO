#!/usr/bin/env bash
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export NS3_ROOT="${NS3_ROOT:-$HOME/ns-3-dev}"
# Published fair-FL replacement arms (user: remove iw/zq traces, use published
# methods): q-FFL (Li ICLR'20) + AFL (Mohri ICML'19), already implemented in
# mappo.py (--aggregation qffl/afl). Settled protocol, s0-7, evals es{10,12,18}.
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EX=$NS3_ROOT/contrib/ai/examples/feddrl
RES=$REPO_ROOT/results/final_l2w40
cp -f $REPO_ROOT/drivers/train_ns3.py "$EX/train_ns3.py"
cp -f $REPO_ROOT/models/mappo.py "$EX/mappo.py"
cd "$EX" || exit 1
: > "$RES/done7.txt"
TC="--arrival-pps 60 --iterations 100 --rollout 32 --macro-slot-ms 20 --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 --link2-width 40 --eps99 1e-2 --eps999 1e-3"
EC="--arrival-pps 60 --episode-ms 11000 --macro-slot-ms 20 --settle-ms 1000 --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 --link2-width 40 --no-coord"
for AGG in qffl afl; do
  for S in 0 1 2 3 4 5 6 7; do
    TAG="ff_${AGG}_${S}"
    ( timeout --signal=INT --kill-after=30 14400 \
        python3 -u -B "$EX/train_ns3.py" --aggregation "$AGG" --seed "$S" \
        $TC --seg-suffix "$TAG" \
        --ckpt-out "$RES/ckpt_${AGG}_s${S}.pt" \
        --train-log-csv "$RES/train_${AGG}_s${S}.csv" \
        > "$RES/log_${AGG}_s${S}.txt" 2>&1
      echo "rc=$? train ${AGG} s${S}" >> "$RES/done7.txt" ) &
    sleep 3
  done
done
wait
echo "=== fair-FL trainings done ==="
one() { OF="$1"; TAG="$2"; shift 2
  timeout --signal=INT --kill-after=20 2400 \
    python3 -u -B "$EX/feddrl.py" $EC --seg-suffix "$TAG" "$@" \
    2>&1 | grep -aE '^\[(KPI|KPI_AP|DIAG)|Traceback|Error' > "$RES/$OF"
  pkill -9 -f "segSuffix=$TAG" 2>/dev/null
}
NPAR=0
q() { one "$@" & NPAR=$((NPAR+1)); if [ "$NPAR" -ge 10 ]; then wait; NPAR=0; fi; }
for AGG in qffl afl; do
  for TS in 0 1 2 3 4 5 6 7; do
    for ES in 10 12 18; do
      q "eval_${AGG}_ts${TS}_es${ES}.txt" "ffe_${AGG}_${TS}_${ES}" \
        --seed "$ES" --ckpt "$RES/ckpt_${AGG}_s${TS}.pt"
    done
  done
done
wait
echo "=== fair-FL arms complete ==="
cat "$RES/done7.txt" | sort | uniq -c | tail -3
