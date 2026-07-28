#!/usr/bin/env bash
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export NS3_ROOT="${NS3_ROOT:-$HOME/ns-3-dev}"
# #7 learned-baseline comparison at the 11s DEVELOPMENT PROTOCOL (paper-validated: 11s vs 110s
# point estimates agree within 0.4 APs). Three arms on identical protocol for apples-to-apples:
#   pxqr = LyMAPPO(full)  [final_l2w40]   mappo, cpo [r3_baselines]
# 8 train seeds x channel seeds {10,12,18} = 72 evals. NPAR=12 (32 cores), 900s SIGKILL (>=3x the
# ~4-5min single-eval wall). Outputs eval11_{arm}_ts{S}_es{ES}.txt with [KPI]/[KPI_AP]/[DIAG].
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EX=$NS3_ROOT/contrib/ai/examples/feddrl
R=$REPO_ROOT/results/r3_baselines
SRC=$REPO_ROOT/results/final_l2w40
EC="--arrival-pps 60 --episode-ms 11000 --macro-slot-ms 20 --settle-ms 1000 --load-spread 0.6 --deadline99-ms 20 --deadline999-ms 40 --link2-width 40 --no-coord"
cd "$EX" || exit 1
ckpt(){ if [ "$1" = pxqr ]; then echo "$SRC/ckpt_pxqr_s$2.pt"; else echo "$R/ckpt_$1_s$2.pt"; fi; }
one(){ OF="$1"; TAG="$2"; CK="$3"; ES="$4"
  timeout --signal=KILL 900 \
    python3 -u -B "$EX/feddrl.py" $EC --seg-suffix "$TAG" --seed "$ES" --ckpt "$CK" \
    2>&1 | grep -aE '^\[(KPI|KPI_AP|DIAG)' > "$R/$OF"
  pkill -9 -f "segSuffix=$TAG" 2>/dev/null; }
NPAR=0
q(){ one "$@" & NPAR=$((NPAR+1)); if [ "$NPAR" -ge 12 ]; then wait; NPAR=0; fi; }
for S in 0 1 2 3 4 5 6 7; do
  for ARM in pxqr mappo cpo; do
    for ES in 10 12 18; do
      q "eval11_${ARM}_ts${S}_es${ES}.txt" "x_${ARM}_${S}_${ES}" "$(ckpt $ARM $S)" "$ES"
    done
  done
done
wait
echo "=== R3 EVAL11 DONE ==="
NZ=$(for f in "$R"/eval11_*_ts*_es*.txt; do [ -s "$f" ] && echo x; done | wc -l)
echo "nonzero eval11 files: $NZ / 72"
for A in pxqr mappo cpo; do echo "-- $A ts0 es10 --"; grep -aE '^\[KPI\]' "$R/eval11_${A}_ts0_es10.txt" 2>/dev/null; done
