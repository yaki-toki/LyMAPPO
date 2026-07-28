"""core_cmp Phase-B eval aggregation: computes the pre-registered metrics.

Metrics (registered before any verdict):
  - worst-AP p999 (max over 16 APs) -- the target of the zq mechanism
  - number of feasible APs (p99<=1e-2 AND p999<=1e-3)
  - hot-AP(0/4/8) p999 -- the local tail stub-RR could not meet
  - network p99/p999 (for reference)

Aggregation rules (multi-seed convention):
  - learned arms: mean over 3 eval seeds -> 1 value per train seed -> mean+/-std
    over 3 train seeds (respects train-seed clustering; the pooled 9-value std is
    for reference only)
  - baselines (RR/RSSI): mean+/-std over 3 eval seeds
"""
from __future__ import annotations

import glob
import os
import re
import statistics as st
import sys

# Takes the results directory as an argument, so it serves both core_cmp (old env)
# and newenv_cmp (new env).
RES = sys.argv[1] if len(sys.argv) > 1 else (
    "results/core_cmp")
EPS99, EPS999 = 1e-2, 1e-3
HOT = (0, 4, 8)

AP_RE = re.compile(
    r"^\[KPI_AP\] (\d+),decided=(\d+),rx=\d+,lost=(\d+),p99=([0-9.]+),p999=([0-9.]+)")


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
        return None  # crashed / incomplete run — report, don't silently drop
    feas = sum(1 for a in range(16)
               if ap_p99[a] <= EPS99 and ap_p999[a] <= EPS999)
    return {
        "worst_p999": max(ap_p999.values()),
        "worst_p99": max(ap_p99.values()),
        "feasible": feas,
        "hot_p999": max(ap_p999[a] for a in HOT),
        "net_p99": net_p99,
        "net_p999": net_p999,
    }


def fmt(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{st.mean(vals):.4f}±{st.stdev(vals):.4f}"


def main() -> None:
    metrics = ["worst_p999", "feasible", "hot_p999", "net_p99", "net_p999"]
    bad: list[str] = []
    per_arm_seed: dict[str, dict[str, list[float]]] = {}
    print(f"{'arm':10s} " + " ".join(f"{m:>16s}" for m in metrics))
    for arm in ("zq", "uniform", "none", "iw", "cluster", "noz",
                "qffl", "afl", "px", "qr", "pxqr",
                "pxb001", "pxb0025", "pxb01",
                "pxa01", "pxa05", "pxk16", "pxk32"):
        per_train: dict[str, list[dict]] = {}
        for path in sorted(glob.glob(f"{RES}/eval_{arm}_ts*_es*.txt")):
            ts = re.search(r"_ts(\d+)_", path).group(1)
            r = parse(path)
            if r is None:
                bad.append(os.path.basename(path))
                continue
            per_train.setdefault(ts, []).append(r)
        # mean over eval seeds per train seed -> mean+/-std across train seeds
        agg = {m: [] for m in metrics}
        for ts, runs in sorted(per_train.items()):
            for m in metrics:
                agg[m].append(st.mean(r[m] for r in runs))
        print(f"{arm:10s} " + " ".join(f"{fmt(agg[m]):>16s}" for m in metrics))
        for ts, runs in sorted(per_train.items()):
            row = " ".join(
                f"{st.mean(r[m] for r in runs):>16.4f}" for m in metrics)
            print(f"  ts{ts} ({len(runs)}ev) {row}")
        per_arm_seed[arm] = agg
    for arm in ("rr", "rssi", "slci"):
        runs = []
        # Handles both the old-env (eval_*) and new-env (base_*) file names.
        paths = (sorted(glob.glob(f"{RES}/eval_{arm}_es*.txt"))
                 or sorted(glob.glob(f"{RES}/base_{arm}_es*.txt")))
        for path in paths:
            r = parse(path)
            if r is None:
                bad.append(os.path.basename(path))
                continue
            runs.append(r)
        vals = {m: [r[m] for r in runs] for m in metrics}
        print(f"{arm:10s} " + " ".join(f"{fmt(vals[m]):>16s}" for m in metrics))
        per_arm_seed[arm] = vals
    if bad:
        print(f"\nINCOMPLETE/CRASHED ({len(bad)}): " + ", ".join(bad))

    # Seed-level Mann-Whitney U (two-sided): learned arms use train-seed values
    # (mean over eval seeds), baselines use eval-seed values -- follows the
    # convention that forbids anticonservative pooling.
    def _mwu(a: list[float], b: list[float]) -> float:
        try:
            from scipy.stats import mannwhitneyu  # type: ignore
            return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except Exception:
            # Normal approximation (no tie correction) -- sufficient for n>=8.
            n1, n2 = len(a), len(b)
            u = sum(1 for x in a for y in b if x < y) \
                + 0.5 * sum(1 for x in a for y in b if x == y)
            mu = n1 * n2 / 2.0
            sd = (n1 * n2 * (n1 + n2 + 1) / 12.0) ** 0.5
            if sd == 0:
                return 1.0
            z = abs(u - mu) / sd
            import math
            return float(math.erfc(z / math.sqrt(2.0)))

    print("\n[stats] seed-level Mann-Whitney (two-sided):")
    for m in ("net_p99", "worst_p999", "hot_p999", "feasible"):
        for a, b in (("none", "rr"), ("none", "rssi"), ("none", "zq"),
                     ("none", "uniform"), ("zq", "uniform"),
                     ("cluster", "none"), ("cluster", "uniform"),
                     ("none", "noz"),
                     ("none", "qffl"), ("none", "afl"),
                     ("qffl", "uniform"), ("afl", "uniform"),
                     ("none", "px"), ("none", "qr"), ("none", "pxqr"),
                     ("pxqr", "rr"), ("pxqr", "rssi"), ("pxqr", "slci")):
            if a in per_arm_seed and b in per_arm_seed:
                va, vb = per_arm_seed[a][m], per_arm_seed[b][m]
                if va and vb:
                    print(f"  {m:12s} {a:>7s}(n={len(va)}) vs {b:<7s}"
                          f"(n={len(vb)}): p={_mwu(va, vb):.4f}")


if __name__ == "__main__":
    main()
