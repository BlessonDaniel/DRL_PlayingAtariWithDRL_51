#!/usr/bin/env python3
"""
Why experience replay?  A small ablation of the paper's Section 4 argument.

Mnih et al. (2013) give three reasons for sampling minibatches from a replay memory instead
of learning from each transition as it arrives: (1) every transition is reused in many
updates, (2) random sampling breaks the correlation between consecutive samples, and
(3) the training distribution is averaged over many past policies, which avoids feedback
loops and oscillation. A full Atari ablation needs GPU-days, so we test the same claim on
CartPole-v1, where one run takes about half a minute on a laptop CPU.

Three agents that are identical except for WHICH transitions each gradient step uses:

    online     the single most recent transition          (plain online Q-learning, Sec 2)
    recent32   the 32 most recent consecutive transitions (same batch size as replay,
                                                           but correlated and on-policy)
    replay     32 transitions drawn uniformly from all stored transitions (Algorithm 1)

Same Q-network, same RMSProp optimiser, same eps-greedy schedule (1.0 -> 0.1 over the first
10% of steps, as in the paper), one gradient step per environment step, no target network.
Each arm is run for several seeds at several learning rates, so that no arm is handicapped
by a learning rate that only suits another arm.

Usage:
    python replay_ablation.py --out runs/ablation            # 3 arms x 3 learning rates x 5 seeds
"""
import argparse
import csv
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARMS = ("online", "recent32", "replay")


def make_q():
    return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))


@torch.no_grad()
def evaluate(q, env, rng, episodes, eps=0.05):
    """Same protocol as the Atari runs: eps-greedy with eps = 0.05, average undiscounted return."""
    total = 0.0
    for _ in range(episodes):
        s, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        while not done:
            a = int(rng.integers(2)) if rng.random() < eps else int(q(torch.from_numpy(s)).argmax())
            s, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated
    return total / episodes


def run(arm, lr, seed, steps, eval_every, eval_episodes, gamma=0.99, batch=32):
    torch.manual_seed(seed)
    rng, eval_rng = np.random.default_rng(seed), np.random.default_rng(10_000 + seed)
    env, eval_env = gym.make("CartPole-v1"), gym.make("CartPole-v1")
    q = make_q()
    opt = torch.optim.RMSprop(q.parameters(), lr=lr, alpha=0.95, eps=0.01)

    # every transition of the run is kept, so "replay" samples from the whole history
    S = np.zeros((steps, 4), np.float32)
    S2 = np.zeros((steps, 4), np.float32)
    A = np.zeros(steps, np.int64)
    R = np.zeros(steps, np.float32)
    T = np.zeros(steps, np.float32)

    curve = []
    s, _ = env.reset(seed=seed)
    for t in range(steps):
        eps = max(0.1, 1.0 - 0.9 * t / (0.1 * steps))
        if rng.random() < eps:
            a = int(rng.integers(2))
        else:
            with torch.no_grad():
                a = int(q(torch.from_numpy(s)).argmax())
        s2, r, terminated, truncated, _ = env.step(a)
        S[t], A[t], R[t], S2[t], T[t] = s, a, r, s2, float(terminated)   # a time-limit cut is not terminal
        s = env.reset()[0] if (terminated or truncated) else s2

        n = t + 1
        if n >= batch:
            if arm == "online":
                idx = np.array([t])
            elif arm == "recent32":
                idx = np.arange(n - batch, n)
            else:
                idx = rng.integers(n, size=batch)
            sb, ab, rb = torch.from_numpy(S[idx]), torch.from_numpy(A[idx]), torch.from_numpy(R[idx])
            s2b, tb = torch.from_numpy(S2[idx]), torch.from_numpy(T[idx])
            with torch.no_grad():
                y = rb + gamma * (1.0 - tb) * q(s2b).max(1).values
            loss = F.mse_loss(q(sb).gather(1, ab.unsqueeze(1)).squeeze(1), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if n % eval_every == 0:
            curve.append((n, evaluate(q, eval_env, eval_rng, eval_episodes)))
    return curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/ablation")
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-episodes", type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(1)
    os.makedirs(args.out, exist_ok=True)

    rows, t0 = [], time.time()
    for lr in args.lrs:
        for arm in ARMS:
            for seed in range(args.seeds):
                curve = run(arm, lr, seed, args.steps, args.eval_every, args.eval_episodes)
                rows += [(arm, lr, seed, step, round(ret, 2)) for step, ret in curve]
                last = np.mean([ret for _, ret in curve[-10:]])
                print(f"arm={arm:9s} lr={lr:g} seed={seed} mean return over last 10 evals={last:6.1f} "
                      f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
                with open(os.path.join(args.out, "curves.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["arm", "lr", "seed", "step", "eval_return"])
                    w.writerows(rows)

    # summary: mean over the last 10 evaluations of each run, then mean / std across seeds
    summary = {}
    for arm in ARMS:
        for lr in args.lrs:
            finals = []
            for seed in range(args.seeds):
                rets = [r[4] for r in rows if r[0] == arm and r[1] == lr and r[2] == seed]
                finals.append(float(np.mean(rets[-10:])))
            summary[f"{arm}|{lr:g}"] = dict(arm=arm, lr=lr, mean=round(float(np.mean(finals)), 1),
                                            std=round(float(np.std(finals)), 1), per_seed=[round(x, 1) for x in finals])
    best = {arm: max((v for v in summary.values() if v["arm"] == arm), key=lambda v: v["mean"]) for arm in ARMS}
    out = dict(config=vars(args), summary=summary, best_lr_per_arm=best, minutes=round((time.time() - t0) / 60, 1))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nmean return over the last 10 evaluations (mean +/- std over seeds):")
    for arm in ARMS:
        print(f"  {arm:9s} " + "   ".join(f"lr={lr:g}: {summary[f'{arm}|{lr:g}']['mean']:6.1f} +/- "
                                          f"{summary[f'{arm}|{lr:g}']['std']:5.1f}" for lr in args.lrs))


if __name__ == "__main__":
    main()
