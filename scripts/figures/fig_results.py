"""v3 G2+G3: per-AP feasibility 히트맵 + arm별 시드 분포 스트립플롯.

입력: results/final_l2w40 의 eval_{arm}_ts{N}_es{M}.txt / base_{arm}_es{M}.txt
  ([KPI_AP] a,decided=..,rx=..,lost=..,p99=..,p999=..)
출력: docs/v3/figs/fig_feasmap.{png,pdf}, fig_seeds.{png,pdf}
사용: python -B fig_results.py [arm1 arm2 ...]  (기본: 현재 확정 arm 목록)
"""
from __future__ import annotations

import glob
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = (r"D:\WLAN\experiments\results\final_l2w40" if os.name == "nt"
       else "results/final_l2w40")
OUT = (r"D:\WLAN\docs\v3\figs" if os.name == "nt"
       else "figs")
EPS99, EPS999 = 1e-2, 1e-3
AP_RE = re.compile(
    r"^\[KPI_AP\] (\d+),decided=\d+,rx=\d+,lost=\d+,p99=([0-9.]+),p999=([0-9.]+)")
LABEL = {"pxqr": "LyMAPPO (full)", "none": "w/o PX+QR",
         "px": "w/o QR", "qr": "w/o PX",
         "uniform": "FedAvg", "cluster": "clust. FL",
         "qffl": "q-FFL", "afl": "AFL", "noz": "no-Z",
         "rr": "RR", "rssi": "RSSI", "slci": "SLCI"}


def _parse_run(path):
    ap99, ap999 = {}, {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = AP_RE.match(line)
        if m:
            ap99[int(m.group(1))] = float(m.group(2))
            ap999[int(m.group(1))] = float(m.group(3))
    return (ap99, ap999) if len(ap999) == 16 else None


def runs_of(arm):
    pats = ([f"{RES}/eval_{arm}_ts*_es*.txt"]
            if arm not in ("rr", "rssi", "slci")
            else [f"{RES}/base_{arm}_es*.txt"])
    out = []
    for pat in pats:
        for path in sorted(glob.glob(pat)):
            r = _parse_run(path)
            if r:
                out.append(r)
    return out


def seed_runs(arm):
    """시드 단위 그룹 (표와 동일 통계단위): 학습 arm=train-seed, baseline=eval-seed."""
    out = {}
    if arm in ("rr", "rssi", "slci"):
        for path in sorted(glob.glob(f"{RES}/base_{arm}_es*.txt")):
            r = _parse_run(path)
            if r:
                out[re.search(r"_es(\d+)\.", path).group(1)] = [r]
    else:
        for path in sorted(glob.glob(f"{RES}/eval_{arm}_ts*_es*.txt")):
            r = _parse_run(path)
            if r:
                key = re.search(r"_ts(\d+)_", path).group(1)
                out.setdefault(key, []).append(r)
    return out


def main():
    arms = sys.argv[1:] or ["pxqr", "none", "px", "qr", "uniform", "qffl",
                            "afl", "cluster", "rr", "rssi", "slci"]
    arms = [a for a in arms if runs_of(a)]
    # --- G2: per-AP feasibility rate 히트맵 (arm x AP) ---
    mat = np.zeros((len(arms), 16))
    for i, arm in enumerate(arms):
        rs = runs_of(arm)
        for a in range(16):
            ok = [1.0 if (r99[a] <= EPS99 and r999[a] <= EPS999) else 0.0
                  for r99, r999 in rs]
            mat[i, a] = float(np.mean(ok))
    fig, ax = plt.subplots(figsize=(3.6, 0.175 * len(arms) + 0.9))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(16))
    ax.set_xticklabels([f"{a}" for a in range(16)], fontsize=7)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([LABEL.get(a, a) for a in arms], fontsize=8.5)
    for a in range(16):  # 링크집합 그룹 경계 표시
        if a % 4 == 0 and a:
            ax.axvline(a - 0.5, color="white", lw=1.4)
    ax.set_xlabel("AP index (K tiling: {2.4,5} x2 | {5,6} | {6})",
                  fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=7.5)
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_feasmap.png", dpi=170, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig_feasmap.pdf", bbox_inches="tight")
    # --- G3: mean±std Pareto 뷰 (x=feasible, y=net p99 symlog) ---
    STYLE = {  # arm: (색, 마커) — plot_pareto 관례(검정 테두리, Ours=진녹 별)
        "pxqr": ("#005a32", "*"), "none": ("#41ab5d", "o"),
        "px": ("#74c476", "P"), "qr": ("#a1d99b", "h"),
        "uniform": ("#888888", "^"), "qffl": ("#e6ab02", "X"),
        "afl": ("#bb3754", "v"), "cluster": ("#8a5fbf", "p"),
        "rr": ("#7570b3", "D"), "rssi": ("#d95f02", "s"),
        "slci": ("#8c510a", "8")}
    fig2, ax2 = plt.subplots(figsize=(3.7, 3.1))
    for arm in arms:
        feas, p99s = [], []
        for runs in seed_runs(arm).values():  # 시드수준 (표와 동일 단위)
            feas.append(float(np.mean(
                [sum(1 for a in range(16)
                     if r99[a] <= EPS99 and r999[a] <= EPS999)
                 for r99, r999 in runs])))
            p99s.append(float(np.mean(
                [float(np.mean([r99[a] for a in range(16)]))
                 for r99, r999 in runs])))
        fx, fy = float(np.mean(feas)), float(np.mean(p99s))
        sx, sy = float(np.std(feas)), float(np.std(p99s))
        c, mk = STYLE.get(arm, ("#333333", "o"))
        ax2.errorbar(fx, fy, xerr=sx, yerr=[[min(sy, fy)], [sy]],
                     fmt=mk, ms=11 if mk == "*" else 7.5, mfc=c,
                     mec="black", mew=0.7, ecolor=c, elinewidth=1.1,
                     capsize=2.2, label=LABEL.get(arm, arm), zorder=3)
    ax2.set_yscale("symlog", linthresh=1e-4)
    ax2.set_ylim(0, 0.12)
    ax2.axhline(EPS99, ls="--", c="red", lw=1.1)
    ax2.text(0.4, EPS99 * 1.35, r"UHR $\varepsilon_{99}=10^{-2}$",
             color="red", fontsize=8)
    ax2.set_xlim(-0.7, 16.9)
    ax2.set_xticks([0, 4, 8, 12, 16])
    ax2.set_xlabel("feasible APs (of 16)", fontsize=9)
    ax2.set_ylabel("network $p_{99}$", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.grid(alpha=0.2)
    # 범례: 행우선 2행(6+5) 배치 — matplotlib 은 열우선이라 인덱스 재배열
    handles, labels = ax2.get_legend_handles_labels()
    half = (len(handles) + 1) // 2
    order = []
    for k in range(half):
        order.append(k)
        if half + k < len(handles):
            order.append(half + k)
    ax2.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=half,
               fontsize=6.0, frameon=True, edgecolor="0.5",
               columnspacing=0.75, handletextpad=0.25, handlelength=1.1,
               borderpad=0.4)
    fig2.tight_layout()
    fig2.savefig(f"{OUT}/fig_seeds.png", dpi=170, bbox_inches="tight")
    fig2.savefig(f"{OUT}/fig_seeds.pdf", bbox_inches="tight")
    print("saved:", OUT, "arms:", arms)


if __name__ == "__main__":
    main()
