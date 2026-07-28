"""core_cmp Phase-B eval 집계: 사전등록 지표 산출.

지표 (판정 전 등록):
  - worst-AP p999 (max over 16 APs)  — zq 메커니즘의 표적
  - feasible AP 수 (p99<=1e-2 AND p999<=1e-3)
  - hot-AP(0/4/8) p999 — stub-RR 이 못 맞추던 국소 tail
  - 네트워크 p99/p999 (참고)

집계 규칙 (다중시드 관례):
  - 학습 arm: eval seed 3개 평균 → train seed 당 1값 → train seed 3개의 mean±std
    (train-seed clustering 존중; pooled 9값 std 는 참고로만)
  - baseline(RR/RSSI): eval seed 3개 mean±std
"""
from __future__ import annotations

import glob
import os
import re
import statistics as st
import sys

# 결과 디렉터리를 인자로 받아 core_cmp(구 환경)와 newenv_cmp(새 환경) 겸용.
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
        # train-seed 당 eval-seed 평균 → train-seed 간 mean±std
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
        # 구 환경(eval_*) / 새 환경(base_*) 파일명 겸용.
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

    # 시드 단위 Mann-Whitney U (양측): 학습 arm 은 train-seed 값(eval-seed 평균),
    # baseline 은 eval-seed 값 — anticonservative pooling 금지 관례 준수.
    def _mwu(a: list[float], b: list[float]) -> float:
        try:
            from scipy.stats import mannwhitneyu  # type: ignore
            return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except Exception:
            # 정규근사 (동률 무보정) — n>=8 에서 충분.
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
