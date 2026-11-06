#!/usr/bin/env python3
"""
make_paper_figures.py -- redraw every figure of Section V from the logged results.

Reads  : <results>/ (eval_episodes.csv, train_curves.csv, per_emotion.csv, summary.csv,
         confusion_matrix.csv)  -- all written by run_experiments.py
Writes : <figdir>/*.png   (300 dpi, IEEE single-column width)

Nothing is drawn from a random generator; every mark comes from a logged value.
Colours: validated categorical palette (blue, orange, aqua, yellow, magenta) in fixed
order A,B,C,D,O. Because three of those colours are light on white paper, every series
also has its own line style / marker / hatch and is labelled, so the figures stay
readable in grayscale print.

Usage:  python make_paper_figures.py --results rerun_out --figdir figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CFG = ["A", "B", "C", "D", "O"]
NAME = {"A": "A Safety-only", "B": "B Reward-only", "C": "C Proposed",
        "D": "D State-only", "O": "O Oracle"}
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "O": "#e87ba4"}
LS = {"A": "-", "B": "--", "C": "-", "D": "-.", "O": ":"}
MK = {"A": "o", "B": "s", "C": "^", "D": "D", "O": "v"}
HATCH = {"A": "", "B": "//", "C": "", "D": "..", "O": "xx"}
SHORT = {"A": "A\nSafety", "B": "B\nReward", "C": "C\nProposed", "D": "D\nState", "O": "O\nOracle"}
EMO = ["Calm", "Stressed", "Fatigued", "Overloaded"]
INK, MUTED = "#0b0b0b", "#52514e"
W = 3.5  # IEEE column width (inches)

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})


def save(fig, figdir, name):
    fig.savefig(os.path.join(figdir, name))
    plt.close(fig)
    print("wrote", name)


def curve_fig(tc, setting, col, ylabel, name, figdir, hline=None, window=20):
    fig, ax = plt.subplots(figsize=(W, 2.45))
    for c in CFG:
        g = tc[(tc.setting == setting) & (tc.config == c)]
        piv = g.pivot(index="episode", columns="seed", values=col)
        ma = piv.rolling(window, min_periods=1).mean()          # moving average per seed
        m, s = ma.mean(axis=1), ma.std(axis=1)                  # mean and SD across seeds
        ax.plot(m.index, m.values, color=COL[c], ls=LS[c], lw=1.3, label=NAME[c])
        ax.fill_between(m.index, m - s, m + s, color=COL[c], alpha=0.10, lw=0)
    if hline is not None:
        ax.axvline(hline, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.annotate(r"$\epsilon$ reaches 0.01", (hline, ax.get_ylim()[0]), xytext=(-3, 4),
                    textcoords="offset points", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1000)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, handlelength=2.2)
    save(fig, figdir, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="rerun_out")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--setting", default="paper")
    a = ap.parse_args()
    os.makedirs(a.figdir, exist_ok=True)
    R = a.results
    tc = pd.read_csv(os.path.join(R, "train_curves.csv"))
    ev = pd.read_csv(os.path.join(R, "eval_episodes.csv"))
    pe = pd.read_csv(os.path.join(R, "per_emotion.csv"))
    st = a.setting
    eps_floor = np.log(0.01) / np.log(0.995)

    # ---- training curves -------------------------------------------------------------
    curve_fig(tc, st, "env_return", "Safety (environment) return / episode", "fig_train_envreturn.png", a.figdir)
    curve_fig(tc, st, "esc", "Alert escalations / episode", "fig_train_alerts.png", a.figdir)
    curve_fig(tc, st, "train_return", "Training return / episode (own reward)", "fig_train_objective.png",
              a.figdir, hline=eps_floor)

    # ---- evaluation: mean escalations per seed ----------------------------------------
    g = ev[ev.setting == st]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    for i, c in enumerate(CFG):
        sm = g[g.config == c].groupby("seed").esc.mean()
        ax.bar(i, sm.mean(), width=0.62, color=COL[c], alpha=0.85, hatch=HATCH[c], edgecolor="white", lw=0.6)
        ax.errorbar(i, sm.mean(), yerr=sm.std(ddof=1), color=INK, capsize=2.5, lw=0.8)
        ax.scatter([i] * len(sm), sm.values, color=INK, s=7, zorder=3)
        ax.text(i, sm.mean() + sm.std(ddof=1) + 3, f"{sm.mean():.1f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(5))
    ax.set_xticklabels([SHORT[c] for c in CFG])
    ax.set_ylabel("Mean escalations / episode")
    ax.set_ylim(0, 125)
    save(fig, a.figdir, "fig_eval_alerts.png")

    # ---- relative change C vs A --------------------------------------------------------
    A = g[g.config == "A"].esc.values
    C = g[g.config == "C"].esc.values

    def pct(new, old):
        return 100 * (new - old) / old
    vals = [pct(C.mean(), A.mean()), pct(C.std(ddof=1), A.std(ddof=1)),
            pct(C.max() - C.min(), A.max() - A.min())]
    labels = ["Mean", "Std. dev.", "Range span"]
    fig, ax = plt.subplots(figsize=(W, 2.0))
    bars = ax.bar(labels, vals, width=0.5, color=[COL["C"] if v < 0 else "#e34948" for v in vals])
    ax.axhline(0, color=INK, lw=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (1.5 if v >= 0 else -1.5), f"{v:+.1f}%",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK)
    ax.set_ylabel("Change of C vs A (%)")
    ax.set_ylim(min(vals) - 8, max(vals) + 8)
    ax.tick_params(axis="x", pad=2)
    save(fig, a.figdir, "fig_relative_change.png")

    # ---- safety-return distribution ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(W, 2.2))
    data = [g[g.config == c].env_return.values for c in CFG]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                    flierprops=dict(marker="o", markersize=2.5, markeredgecolor=MUTED, alpha=0.6),
                    medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED))
    for patch, c in zip(bp["boxes"], CFG):
        patch.set(facecolor=COL[c], alpha=0.75, hatch=HATCH[c], edgecolor="white")
    for i, d in enumerate(data, start=1):
        ax.text(i, 105.5, f"{np.mean(d):.1f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([SHORT[c] for c in CFG])
    ax.set_ylabel("Safety (environment) return / episode")
    ax.set_ylim(top=109)
    save(fig, a.figdir, "fig_return_distribution.png")

    # ---- escalation rate by true emotion ------------------------------------------------
    p = pe[pe.setting == st]
    fig, ax = plt.subplots(figsize=(W, 2.4))
    w = 0.16
    for i, c in enumerate(CFG):
        rate = [float(p[(p.config == c) & (p.true_emotion == e)].escalation_rate.iloc[0]) for e in EMO]
        ax.bar(np.arange(4) + (i - 2) * w, rate, w, color=COL[c], hatch=HATCH[c], edgecolor="white",
               lw=0.5, label=NAME[c])
    ax.set_xticks(range(4))
    ax.set_xticklabels(EMO)
    ax.set_ylabel("Fraction of steps escalated")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    save(fig, a.figdir, "fig_alerts_by_emotion.png")

    # ---- confusion matrix -----------------------------------------------------------------
    cm = pd.read_csv(os.path.join(R, "confusion_matrix.csv"), index_col=0).values
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    ax.grid(False)
    ax.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7.5,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(EMO, rotation=35, ha="right"); ax.set_yticklabels(EMO)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    for sp in ax.spines.values():
        sp.set_visible(True)
    save(fig, a.figdir, "fig_confusion_matrix.png")


if __name__ == "__main__":
    main()
