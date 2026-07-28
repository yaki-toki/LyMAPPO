"""v3 G1: 시뮬레이션 토폴로지 다이어그램 (구현 상수와 1:1).

feddrl_scenario.cc 확정값: 16 AP 4x4 격자 pitch 8 m, AP 당 5 STA 반경 3 m 링,
K_i = (i%4) 타일 {0,1},{0,1},{1,2},{2}; 밴드 link0=2.4G/20M, link1=5G/40M,
link2=6G/40M; zero-sum load-spread 0.6 (AP0 x1.6 → AP15 x0.4).
출력: docs/v3/figs/fig_topology.{png,pdf}
"""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

PITCH = 8.0
RING = 3.0
N_AP, PER, COLS = 16, 5, 4
# K_i 그룹 (열 = i%4): 색은 plot_pareto 팔레트 관례 근처의 구분색.
GROUPS = {
    (1, 1, 0): ("#1f6fb4", r"$\mathcal{K}=\{2.4,5\}$ GHz"),
    (0, 1, 1): ("#2c8b57", r"$\mathcal{K}=\{5,6\}$ GHz"),
    (0, 0, 1): ("#bb3754", r"$\mathcal{K}=\{6\}$ GHz (single-link)"),
}
SETS = [(1, 1, 0), (1, 1, 0), (0, 1, 1), (0, 0, 1)]


def main() -> None:
    plt.rcParams.update({"font.size": 8.5})
    fig, ax = plt.subplots(figsize=(3.9, 4.3))
    # OBSS 도메인 오버레이: 같은 밴드를 쓰는 BSS 는 전부 하나의 CSMA 충돌
    # 도메인 (전 AP 상호 CS 범위). 밴드-공유 그래프 = OBSS 구조.
    #   2.4GHz: 열 0-1 (8 BSS) / 5GHz: 열 0-2 (12 BSS) / 6GHz: 열 2-3 (8 BSS)
    from matplotlib.patches import FancyBboxPatch
    bands = [  # (x0열, x1열, y-offset, 색, 라벨)
        (0, 1, 5.0, "#1f6fb4", "2.4 GHz (8 BSSs)"),
        (0, 2, 7.2, "#8a5fbf", "5 GHz OBSS (12 BSSs)"),
        (2, 3, 5.0, "#bb3754", "6 GHz (8 BSSs)"),
    ]
    # AP coverage(캐리어센스) 기반 OBSS: 모델 CS 반경 ~51 m (TxPower 16 dBm,
    # log-distance n=3, preamble 검출 -82 dBm) >> 배치 대각 ~34 m → 어느 AP 의
    # CS 원이든 전 배치를 덮는다 = 16 BSS 완전 중첩 OBSS. 대표로 AP5 의 원과
    # 최원거리 AP15 까지의 화살표를 표시 (16개 전부 그리면 완전히 겹침).
    CS_R = 51.0
    cx, cy = 1 * PITCH, 1 * PITCH  # AP5
    # 실척 인셋: CS 원(반경 ~51 m) 안에 4x4 격자 전체가 들어감을 증명.
    axin = ax.inset_axes([0.02, 0.02, 0.20, 0.20])
    axin.add_patch(Circle((cx, cy), CS_R, fill=True, fc="#f2c94c",
                          alpha=0.28, ec="#c9962a", ls="-.", lw=1.2))
    for a2 in range(N_AP):
        axin.plot((a2 % COLS) * PITCH, (a2 // COLS) * PITCH, marker="^",
                  ms=3.4, c=GROUPS[SETS[a2 % 4]][0], mec="black", mew=0.3)
    axin.set_xlim(cx - CS_R - 6, cx + CS_R + 6)
    axin.set_ylim(cy - CS_R - 6, cy + CS_R + 6)
    axin.set_aspect("equal")
    axin.set_xticks([]); axin.set_yticks([])
    axin.set_xlabel("CS to scale: AP5 range\n~51 m ⊃ all 16 BSSs",
                    fontsize=6.8, color="#8a6d1a", labelpad=2)
    for sp in axin.spines.values():
        sp.set_edgecolor("#c9962a")
    for c0, c1, yo, bc, lab in bands:
        x0, x1 = c0 * PITCH - 4.6, c1 * PITCH + 4.6
        ax.add_patch(FancyBboxPatch(
            (x0, -4.6), x1 - x0, 3 * PITCH + 4.6 + yo,
            boxstyle="round,pad=0.25", fc=bc, ec=bc, alpha=0.07, lw=1.1,
            ls="--", zorder=0))
        ax.text((x0 + x1) / 2, 3 * PITCH + yo - 0.5, lab, ha="center",
                fontsize=8.5, color=bc, alpha=0.95)
    for a in range(N_AP):
        col, row = a % COLS, a // COLS
        x, y = col * PITCH, row * PITCH
        key = SETS[a % 4]
        color, _ = GROUPS[key]
        mult = 1.0 + 0.6 * (1.0 - 2.0 * a / (N_AP - 1))  # zero-sum spread
        # STA 링
        ax.add_patch(Circle((x, y), RING, fill=False, ls=":", lw=0.6,
                            ec=color, alpha=0.55))
        for k in range(PER):
            th = 2 * math.pi * k / PER
            ax.plot(x + RING * math.cos(th), y + RING * math.sin(th),
                    marker="o", ms=3.2, mec="black", mew=0.3, c=color,
                    alpha=0.85)
        # AP
        ax.plot(x, y, marker="^", ms=10, c=color, mec="black", mew=0.7)
        ax.annotate(f"AP{a}", (x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8)
        if a in (0, N_AP - 1):  # 부하 배수는 끝점만 (스프레드는 하단 표기)
            mxy = (26, -4)  # 두 끝점 모두 우측 표기 (인셋·화살표 회피)
            ax.annotate(f"{mult:.1f}x", (x, y), xytext=mxy,
                        textcoords="offset points", ha="center",
                        fontsize=7.5, color="#444444")
    # 부하 gradient 화살표
    ax.annotate("", xy=(3 * PITCH + 3.5, 3 * PITCH), xytext=(-3.5, 0),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9,
                                ls="--"))
    ax.text(2.05 * PITCH, -6.2,
            "zero-sum load spread 0.6:\nAP0 1.6x  →  AP15 0.4x",
            ha="center", fontsize=7.8, color="#555555")
    handles = [Line2D([], [], marker="^", ls="", ms=8, c=c, mec="black",
                      label=l) for c, l in GROUPS.values()]
    handles += [
        Line2D([], [], marker="o", ls="", ms=4.5, c="#777777", mec="black",
               label="STA (dotted ring = placement, not coverage)"),
        Line2D([], [], ls="--", c="#aaaaaa",
               label="8 m pitch (all APs in mutual CS range)"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6.8,
              frameon=True, edgecolor="0.5", columnspacing=0.8,
              handletextpad=0.4)
    ax.set_xlabel("x (m)", fontsize=8.5)
    ax.set_ylabel("y (m)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_aspect("equal")
    ax.set_xlim(-6.5, 3 * PITCH + 6.5)
    ax.set_ylim(-9.5, 3 * PITCH + 9.0)
    ax.grid(alpha=0.15)
    out = (r"D:\WLAN\docs\v3\figs" if os.name == "nt"
           else "figs")
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{out}/fig_topology.png", dpi=170, bbox_inches="tight")
    fig.savefig(f"{out}/fig_topology.pdf", bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
