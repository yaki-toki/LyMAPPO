#!/usr/bin/env bash
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export NS3_ROOT="${NS3_ROOT:-$HOME/ns-3-dev}"
# Option-2 strengthening (user: "strengthening options 1,2,3 in sequence"): load
# sweep lambda in {40,80} pps/STA (60 = existing final_l2w40 operating point).
# pxqr trainings s0-7 per load + evals es{10,12,18}; baselines rr/rssi/slci
# eval-only.
# Protocol otherwise identical to _seed_ext.sh (deadlines 20/40ms fixed).
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EX=$NS3_ROOT/contrib/ai/examples/feddrl
cp -f $REPO_ROOT/drivers/train_ns3.py "$EX/train_ns3.py"
cp -f $REPO_ROOT/drivers/feddrl.py "$EX/feddrl.py"
cp -f $REPO_ROOT/models/mappo.py "$EX/mappo.py"
cd "$EX" || exit 1
for L in 40 80; do
  RES="$REPO_ROOT/results/sweep_l${L}"
  mkdir -p "$RES"
  : > "$RES/done9.txt"
  TC="--arrival-pps $L --iterations 100 --rollout 32 --macro-slot-ms 20 --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 --link2-width 40 --eps99 1e-2 --eps999 1e-3"
  for S in 0 1 2 3 4 5 6 7; do
    TAG="ls${L}_pxqr_${S}"
    ( timeout --signal=INT --kill-after=30 14400 \
        python3 -u -B "$EX/train_ns3.py" --aggregation none --seed "$S" \
        $TC --price-beta 0.05 --quantiles 8 --cvar-alpha 0.25 \
        --seg-suffix "$TAG" \
        --ckpt-out "$RES/ckpt_pxqr_s${S}.pt" \
        --train-log-csv "$RES/train_pxqr_s${S}.csv" \
        > "$RES/log_pxqr_s${S}.txt" 2>&1
      echo "rc=$? train l${L} s${S}" >> "$RES/done9.txt" ) &
    sleep 3
  done
done
wait
echo "=== sweep trainings done ==="
one() { RES="$1"; OF="$2"; TAG="$3"; L="$4"; shift 4
  EC="--arrival-pps $L --episode-ms 11000 --macro-slot-ms 20 --settle-ms 1000 --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 --link2-width 40 --no-coord"
  timeout --signal=INT --kill-after=20 2400 \
    python3 -u -B "$EX/feddrl.py" $EC --seg-suffix "$TAG" "$@" \
    2>&1 | grep -aE '^\[(KPI|KPI_AP|DIAG)|Traceback|Error' > "$RES/$OF"
  pkill -9 -f "segSuffix=$TAG" 2>/dev/null
}
NPAR=0
q() { one "$@" & NPAR=$((NPAR+1)); if [ "$NPAR" -ge 10 ]; then wait; NPAR=0; fi; }
for L in 40 80; do
  RES="$REPO_ROOT/results/sweep_l${L}"
  for TS in 0 1 2 3 4 5 6 7; do
    for ES in 10 12 18; do
      q "$RES" "eval_pxqr_ts${TS}_es${ES}.txt" "lse${L}_${TS}_${ES}" "$L" \
        --seed "$ES" --ckpt "$RES/ckpt_pxqr_s${TS}.pt"
    done
  done
  for ES in 10 12 18; do
    q "$RES" "base_rr_es${ES}.txt" "lsr${L}_rr_${ES}" "$L" \
      --seed "$ES" --policy stub --balance-links
    q "$RES" "base_rssi_es${ES}.txt" "lsr${L}_rs_${ES}" "$L" \
      --seed "$ES" --policy rssi
    q "$RES" "base_slci_es${ES}.txt" "lsr${L}_sl_${ES}" "$L" \
      --seed "$ES" --policy slci
  done
done
wait
echo "=== load sweep complete ==="
for L in 40 80; do
  cat "$REPO_ROOT/results/sweep_l${L}/done9.txt" | sort | uniq -c | tail -2
done
