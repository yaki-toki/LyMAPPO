"""v3 G4: training dynamics (train p99 + dual backlog) -- visualizes the
training-time benefit of the aggregation.

Input: results/final_l2w40/train_{arm}_s{S}.csv
  (iter,timestamp_iso,actor_loss,critic_loss,avg_return,p99_train,
   p99_9_train,mean_Z_99_train,mean_Z_99_9_train,T)
Output: docs/v3/figs/fig_training.{png,pdf}
Usage: python -B fig_training.py [arm1 arm2 ...]
"""
from __future__ import annotations

import csv
import glob
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = (r"D:\WLAN\experiments\results\final_l2w40" if os.name == "nt"
       else "results/final_l2w40")
OUT = (r"D:\WLAN\docs\v3\figs" if os.name == "nt"
       else "figs")
EPS99 = 1e-2
P99_FLOOR = 1e-4  # zero clip for the log axis (stated in the caption)
COLOR = {"pxqr": "#005a32", "none": "#1f6fb4", "cluster": "#8a5fbf",
         "uniform": "#888888", "qffl": "#e6ab02", "afl": "#d95f02"}
LABEL = {"pxqr": "LyMAPPO (full)", "none": "w/o PX+QR (base)",
         "cluster": "clustered FL", "uniform": "FedAvg",
         "qffl": "q-FFL", "afl": "AFL"}


def series(arm):
    """List of per-seed (p99[iter], z[iter]) arrays for the arm."""
    runs = []
    for path in sorted(glob.glob(f"{RES}/train_{arm}_s*.csv")):
        p99, z = [], []
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p99.append(float(row["p99_train"]))
                z.append(float(row["mean_Z_99_train"])
                         + float(row["mean_Z_99_9_train"]))
        if len(p99) >= 100:  # completed runs only (skip partial CSVs of a running battery)
            runs.append((np.array(p99), np.array(z)))
    return runs


STYLE = {  # arm: (color, line style, marker) -- to distinguish the mean lines
    "pxqr": ("#005a32", "-", "*"), "none": ("#1f6fb4", "--", "o"),
    "cluster": ("#8a5fbf", "-.", "s"), "uniform": ("#555555", ":", "^")}


def main():
    arms = sys.argv[1:] or ["pxqr", "none", "cluster", "uniform"]
    plt.rcParams.update({"font.size": 8.5})
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.7, 3.9), sharex=True)
    per_arm = {a: series(a) for a in arms}
    per_arm = {a: r for a, r in per_arm.items() if r}
    n_it = min(min(len(p) for p, _ in r) for r in per_arm.values())
    it = np.arange(n_it)
    # Shared min-max envelope (all runs of all arms) -- replaces the seed spaghetti
    all_p = np.stack([np.maximum(p[:n_it], P99_FLOOR)
                      for r in per_arm.values() for p, _ in r])
    all_z = np.stack([z[:n_it] for r in per_arm.values() for _, z in r])
    axa.fill_between(it, all_p.min(0), all_p.max(0), color="0.78",
                     alpha=0.4, lw=0, label="all runs (min-max)")
    axb.fill_between(it, all_z.min(0), all_z.max(0), color="0.78",
                     alpha=0.4, lw=0)
    for arm, runs in per_arm.items():
        c, ls, _ = STYLE.get(arm, ("#333333", "-", ""))
        p_mat = np.stack([np.maximum(p[:n_it], P99_FLOOR) for p, _ in runs])
        z_mat = np.stack([z[:n_it] for _, z in runs])
        kw = dict(c=c, ls=ls, lw=1.0)
        axa.plot(it, np.exp(np.log(p_mat).mean(axis=0)),
                 label=LABEL.get(arm, arm), **kw)
        axb.plot(it, z_mat.mean(axis=0), **kw)
    axa.axhline(EPS99, ls="--", c="red", lw=1.0)
    axa.text(2, EPS99 * 1.4, r"$\varepsilon_{99}$", color="red", fontsize=8)
    axa.set_yscale("log")
    axa.set_ylabel("training-window $p_{99}$", fontsize=8.5)
    axa.grid(alpha=0.2)
    axa.tick_params(labelsize=7.5)
    axb.set_ylabel(r"dual backlog $\bar Z^{99}+\bar Z^{99.9}$", fontsize=8.5)
    axb.set_xlabel("training iteration", fontsize=8.5)
    axb.grid(alpha=0.2)
    axb.tick_params(labelsize=7.5)
    # Legend: same style as Fig.3 -- bottom box, row-major 3+2 layout
    handles, labels = axa.get_legend_handles_labels()
    half = (len(handles) + 1) // 2
    order = []
    for k in range(half):
        order.append(k)
        if half + k < len(handles):
            order.append(half + k)
    axb.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=half,
               fontsize=6.8, frameon=True, edgecolor="0.5",
               columnspacing=0.8, handletextpad=0.4, handlelength=1.6)
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_training.png", dpi=170, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig_training.pdf", bbox_inches="tight")
    print("saved:", OUT, "arms:", [a for a in arms if series(a)])


if __name__ == "__main__":
    main()
