#!/usr/bin/env python3
"""
Figures from the logs written by dqn_atari.py and replay_ablation.py.

    python plot_results.py --runs runs/breakout_cpu runs/breakout_cpu_seed1 --ablation runs/ablation --out figures

Writes:
    figures/atari_training.png    training-game score, evaluation score and average max-Q (cf. Figure 2 of the paper)
    figures/replay_ablation.png   CartPole: uniform replay vs no replay, mean and min-max band over seeds
    figures/value_trace.png       max-Q along one game of the trained agent (cf. Figure 3), if watch_agent.py was run
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORANGE, BLUE, AQUA, GREY, INK, GRID = "#eb6834", "#2a78d6", "#1baf7a", "#6b6a66", "#0b0b0b", "#e1e0d9"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titlesize": 11, "axes.titleweight": "bold",
                     "legend.frameon": False, "figure.dpi": 150})


def read(path):
    return list(csv.DictReader(open(path)))


def atari(runs, out, bin_steps=25_000):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    colours, notes = [ORANGE, BLUE, AQUA], []
    for i, run in enumerate(runs):
        cfg = json.load(open(os.path.join(run, "config.json")))
        log, games = read(os.path.join(run, "log.csv")), read(os.path.join(run, "train_games.csv"))
        extras = [n for n, on in (("target network", cfg.get("target_update")), ("Huber loss", cfg.get("huber")),
                                  ("life-loss terminal", cfg.get("life_loss_terminal"))) if on]
        label = ("2013 algorithm" if not extras else "+ 2015 additions") + f", seed {cfg['seed']}"
        notes = notes + ["2015 additions = " + ", ".join(extras)] if extras else notes
        k, c = cfg["frame_skip"], colours[i % 3]
        steps = np.array([int(g["step"]) for g in games])
        score = np.array([float(g["score"]) for g in games])
        edges = np.arange(0, steps.max() + bin_steps, bin_steps)
        mid = [(a + bin_steps / 2) * k / 1e6 for a in edges[:-1] if ((steps >= a) & (steps < a + bin_steps)).any()]
        mean = [score[(steps >= a) & (steps < a + bin_steps)].mean() for a in edges[:-1] if ((steps >= a) & (steps < a + bin_steps)).any()]
        ax[0].plot(mid, mean, color=c, lw=2, label=label)
        frames = [int(r["frames"]) / 1e6 for r in log]
        ax[1].plot(frames, [float(r["eval_score"]) for r in log], color=c, lw=2, label=label)
        ax[2].plot(frames, [float(r["avg_max_q"]) for r in log], color=c, lw=2, label=label)
    game = cfg["game"]
    ax[0].set_title(f"{game}: score per training game (eps-greedy)\nmean per {bin_steps * k // 1000}k frames")
    ax[1].set_title(f"{game}: evaluation score\neps = 0.05 for {cfg['eval_steps']:,} steps")
    ax[2].set_title(f"Average max-Q on {cfg['holdout_states']} held-out states\n(the paper's smoother metric, Sec 5.1)")
    for a in ax:
        a.set_xlabel("training frames (millions)")
        a.set_ylim(bottom=0)
    ax[0].axhline(1.2, color=GREY, lw=1, label="random play in the paper (1.2)")
    check = os.path.join(runs[0], "trick_check_model_last.json")
    if os.path.exists(check):
        left = json.load(open(check))["constant_policies"].get("LEFT")
        if left is not None:
            ax[1].axhline(left, color=GREY, lw=1, label=f"always pressing LEFT ({left})")
    for a in ax[:2]:
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
    if notes:
        fig.text(0.99, 0.01, notes[0], ha="right", fontsize=8, color=GREY)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "atari_training.png"))

    trace = os.path.join(runs[0], "watch", "value_trace.csv")
    if os.path.exists(trace):
        rows = read(trace)
        fig, a = plt.subplots(figsize=(8, 3))
        a.plot([int(r["step"]) for r in rows], [float(r["max_q"]) for r in rows], color=ORANGE, lw=1.5)
        hits = [int(r["step"]) for r in rows if float(r["reward"]) > 0]
        a.plot(hits, [float(rows[h]["max_q"]) for h in hits], "o", color=BLUE, ms=5, label="reward received")
        a.set_title("Predicted value max-Q along one game of the trained agent")
        a.set_xlabel("agent step")
        a.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "value_trace.png"))


def ablation(path, out):
    """One panel per learning rate, so no agent is shown only at a learning rate that suits another one."""
    rows, summary = read(os.path.join(path, "curves.csv")), json.load(open(os.path.join(path, "summary.json")))
    names = {"replay": "Uniform replay (the paper)", "recent32": "Last 32 transitions, no replay", "online": "Latest transition only (online)"}
    colours = {"replay": ORANGE, "recent32": BLUE, "online": AQUA}
    lrs = summary["config"]["lrs"]
    fig, ax = plt.subplots(1, len(lrs), figsize=(13, 3.9), sharey=True)
    for a, lr in zip(ax, lrs):
        for arm in ("replay", "recent32", "online"):
            sel = [r for r in rows if r["arm"] == arm and float(r["lr"]) == lr]
            steps = sorted({int(r["step"]) for r in sel})
            mean = [np.mean([float(r["eval_return"]) for r in sel if int(r["step"]) == s]) for s in steps]
            final = summary["summary"][f"{arm}|{lr:g}"]
            a.plot(np.array(steps) / 1000, mean, color=colours[arm], lw=2, label=names[arm])
            a.text(steps[-1] / 1000 + 0.6, final["mean"], f"{int(final['mean'] + 0.5)}", color=INK, fontsize=8, va="center")
        a.set_title(f"learning rate {lr:g}")
        a.set_xlabel("environment steps (thousands)")
        a.set_xlim(0, steps[-1] / 1000 + 3)
        a.set_ylim(0, 510)
    ax[0].set_ylabel("evaluation return (eps = 0.05)")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)
    fig.suptitle(f"CartPole-v1, mean of {summary['config']['seeds']} seeds. Number at the right: mean return over the last 10 evaluations",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "replay_ablation.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=["runs/breakout_cpu"])
    p.add_argument("--ablation", default="runs/ablation")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = [r for r in args.runs if os.path.exists(os.path.join(r, "log.csv"))]
    if runs:
        atari(runs, args.out)
    if os.path.exists(os.path.join(args.ablation, "summary.json")):
        ablation(args.ablation, args.out)
    print("wrote", sorted(os.listdir(args.out)))
