"""r3 #7 eval aggregation for the new learned baselines (mappo, cpo).

Reuses the pre-registered metrics + multi-seed convention of
experiments/_core_cmp_analyze.py: per train-seed = mean over the 3 eval seeds,
then mean±std over train seeds. Reports feasible/16, net p99, net p99.9,
worst-AP p99.9 for mappo and cpo (and pxqr if its evals are present) so the new
Table I/II rows sit next to LyMAPPO full.
"""
from __future__ import annotations

import glob
import os
import re
import statistics as st
import sys

RES = sys.argv[1] if len(sys.argv) > 1 else \
    "results/r3_baselines"
EPS99, EPS999 = 1e-2, 1e-3

AP_RE = re.compile(
    r"^\[KPI_AP\] (\d+),decided=(\d+),rx=\d+,lost=(\d+),"
    r"p99=([0-9.]+),p999=([0-9.]+)")


def parse(path: str) -> dict | None:
    net_p99 = net_p999 = None
    ap_p99: dict[int, float] = {}
    ap_p999: dict[int, float] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("[KPI] "):
                parts = line[6:].strip().split(",")
                net_p99, net_p999 = float(parts[5]), float(parts[6])
            m = AP_RE.match(line)
            if m:
                a = int(m.group(1))
                ap_p99[a] = float(m.group(4))
                ap_p999[a] = float(m.group(5))
    if net_p99 is None or len(ap_p999) != 16:
        return None
    feas = sum(1 for a in range(16)
               if ap_p99[a] <= EPS99 and ap_p999[a] <= EPS999)
    return {
        "feasible": feas,
        "net_p99": net_p99,
        "net_p999": net_p999,
        "worst_p999": max(ap_p999.values()),
        "worst_p99": max(ap_p99.values()),
    }


def fmt(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{st.mean(vals):.4f}±{st.stdev(vals):.4f}"


def main() -> None:
    metrics = ["feasible", "net_p99", "net_p999", "worst_p999"]
    bad: list[str] = []
    print(f"{'arm':10s} " + " ".join(f"{m:>16s}" for m in metrics))
    for arm in ("mappo", "cpo", "pxqr"):
        per_train: dict[str, list[dict]] = {}
        for path in sorted(glob.glob(f"{RES}/eval_{arm}_ts*_es*.txt")):
            ts = re.search(r"_ts(\d+)_", path).group(1)
            r = parse(path)
            if r is None:
                bad.append(os.path.basename(path))
                continue
            per_train.setdefault(ts, []).append(r)
        if not per_train:
            continue
        agg = {m: [] for m in metrics}
        for ts, runs in sorted(per_train.items()):
            for m in metrics:
                agg[m].append(st.mean(r[m] for r in runs))
        print(f"{arm:10s} " + " ".join(f"{fmt(agg[m]):>16s}" for m in metrics))
        for ts, runs in sorted(per_train.items()):
            row = " ".join(f"{st.mean(r[m] for r in runs):>16.4f}" for m in metrics)
            print(f"  ts{ts} ({len(runs)}ev) {row}")
    if bad:
        print(f"\nINCOMPLETE/CRASHED ({len(bad)}): " + ", ".join(bad))


if __name__ == "__main__":
    main()
