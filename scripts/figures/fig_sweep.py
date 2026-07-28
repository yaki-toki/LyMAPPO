"""v3 G5: load sweep curves (feasible + net p99 vs lambda) -- graceful degradation.

Input: results/final_l2w40 (lambda=60), results/sweep_l40, results/sweep_l80
Output: docs/v3/figs/fig_sweep.{png,pdf}
Statistical unit: learned arm=train-seed (mean over eval seeds), baseline=eval-seed.
"""
from __future__ import annotations

import glob
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = (r"D:\WLAN\experiments\results" if os.name == "nt"
        else "results")
OUT = (r"D:\WLAN\docs\v3\figs" if os.name == "nt"
       else "figs")
DIRS = {40: f"{BASE}/sweep_l40", 60: f"{BASE}/final_l2w40",
        80: f"{BASE}/sweep_l80"}
EPS99, EPS999 = 1e-2, 1e-3
AP_RE = re.compile(
    r"^\[KPI_AP\] (\d+),decided=\d+,rx=\d+,lost=\d+,p99=([0-9.]+),p999=([0-9.]+)")
STYLE = {"pxqr": ("#005a32", "*", "LyMAPPO (full)"),
         "rr": ("#7570b3", "D", "RR"),
         "rssi": ("#d95f02", "s", "RSSI"),
         "slci": ("#8c510a", "8", "SLCI")}


def _parse(path):
    ap99, ap999 = {}, {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = AP_RE.match(line)
        if m:
            ap99[int(m.group(1))] = float(m.group(2))
            ap999[int(m.group(1))] = float(m.group(3))
    return (ap99, ap999) if len(ap999) == 16 else None


def seed_stats(res, arm):
    """(feasible mean, std, netp99 mean, std, n) -- seed level."""
    groups = {}
    if arm == "pxqr":
        for p in sorted(glob.glob(f"{res}/eval_pxqr_ts*_es*.txt")):
            r = _parse(p)
            if r:
                key = re.search(r"_ts(\d+)_", p).group(1)
                groups.setdefault(key, []).append(r)
    else:
        for p in sorted(glob.glob(f"{res}/base_{arm}_es*.txt")):
            r = _parse(p)
            if r:
                groups[re.search(r"_es(\d+)\.", p).group(1)] = [r]
    feas, p99s = [], []
    for runs in groups.values():
        feas.append(float(np.mean(
            [sum(1 for a in range(16)
                 if r99[a] <= EPS99 and r999[a] <= EPS999)
             for r99, r999 in runs])))
        p99s.append(float(np.mean(
            [float(np.mean([r99[a] for a in range(16)]))
             for r99, r999 in runs])))
    return (np.mean(feas), np.std(feas), np.mean(p99s), np.std(p99s),
            len(feas))


def main():
    plt.rcParams.update({"font.size": 8.5})
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.7, 4.0), sharex=True)
    loads = sorted(DIRS)
    for arm, (c, mk, lab) in STYLE.items():
        F, Fs, P, Ps = [], [], [], []
        for lam in loads:
            f, fs, p, ps, n = seed_stats(DIRS[lam], arm)
            F.append(f); Fs.append(fs); P.append(p); Ps.append(ps)
        kw = dict(c=c, marker=mk, ms=8 if mk == "*" else 5.5, mec="black",
                  mew=0.6, lw=1.2, capsize=2.2, elinewidth=1.0)
        axa.errorbar(loads, F, yerr=Fs, label=lab, **kw)
        axb.errorbar(loads, P, yerr=[np.minimum(Ps, P), Ps], **kw)
    axa.set_ylabel("feasible APs (of 16)", fontsize=8.5)
    axa.set_ylim(0, 16.9)
    axa.axhline(16, ls=":", c="#888888", lw=0.8)
    axa.grid(alpha=0.2)
    axa.tick_params(labelsize=7.5)
    axa.legend(fontsize=7, loc="lower left", frameon=True, edgecolor="0.5",
               handletextpad=0.4)
    axb.set_yscale("log")
    axb.axhline(EPS99, ls="--", c="red", lw=1.0)
    axb.text(41, EPS99 * 1.35, r"$\varepsilon_{99}$", color="red",
             fontsize=8)
    axb.set_ylabel("network $p_{99}$", fontsize=8.5)
    axb.set_xlabel(r"offered load $\lambda$ (pkt/s per STA)", fontsize=8.5)
    axb.set_xticks(loads)
    axb.grid(alpha=0.2)
    axb.tick_params(labelsize=7.5)
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_sweep.png", dpi=170, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig_sweep.pdf", bbox_inches="tight")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
