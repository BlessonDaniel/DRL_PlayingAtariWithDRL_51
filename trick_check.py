#!/usr/bin/env python3
"""
Is the agent really reading the screen, or has it found a blind trick?

Two checks on a saved checkpoint:
  1. which action the greedy policy picks, step by step, over a few evaluation games (eps = 0.05)
  2. what the "always press the same button" policies score under the same protocol

If one action dominates and its constant-action policy scores the same as the agent, the score comes
from a fixed habit rather than from looking at the ball. In deterministic Breakout a paddle parked at
the left wall returns the first serves by itself, so this is worth checking before trusting a score.

Usage:
    python trick_check.py --run runs/breakout_cpu --checkpoint model_last.pt
"""
import argparse
import json
import os

import numpy as np
import torch

from dqn_atari import AtariEnv, QNetwork, as_batch


def play(env, rng, choose, eps, games, max_steps=5000):
    scores, picks, values = [], [], []
    for _ in range(games):
        state, score = env.reset(), 0.0
        for _ in range(max_steps):
            greedy, value = choose(state)
            picks.append(greedy)
            values.append(value)
            action = int(rng.integers(env.n_actions)) if rng.random() < eps else greedy
            state, reward, terminated, truncated = env.step(action)
            score += reward
            if terminated or truncated:
                break
        scores.append(score)
    return scores, picks, values


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/breakout_cpu")
    p.add_argument("--checkpoint", default="model_last.pt")
    p.add_argument("--games", type=int, default=10, help="games played by the agent")
    p.add_argument("--baseline-games", type=int, default=20, help="games played by each constant-action policy")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    cfg = json.load(open(os.path.join(args.run, "config.json")))
    env = AtariEnv(cfg["game"], cfg["frame_skip"], args.seed, cfg["crop_top"])
    names = env.env.unwrapped.get_action_meanings()
    q = QNetwork(env.n_actions)
    q.load_state_dict(torch.load(os.path.join(args.run, args.checkpoint), map_location="cpu"))
    q.eval()

    def agent(state):
        values = q(as_batch(state, "cpu"))[0]
        return int(values.argmax()), float(values.max())

    scores, picks, values = play(env, np.random.default_rng(args.seed), agent, args.eps, args.games)
    share = {names[a]: round(picks.count(a) / len(picks), 3) for a in range(env.n_actions)}
    out = {"checkpoint": args.checkpoint, "agent_scores": scores, "agent_mean_score": round(float(np.mean(scores)), 2),
           "greedy_action_share": share, "max_q_min": round(min(values), 3), "max_q_max": round(max(values), 3),
           "max_q_std": round(float(np.std(values)), 3), "constant_policies": {}}
    for a in range(env.n_actions):
        s, _, _ = play(env, np.random.default_rng(args.seed), lambda state, a=a: (a, 0.0), args.eps, args.baseline_games)
        out["constant_policies"][names[a]] = round(float(np.mean(s)), 2)
    name = "trick_check_" + os.path.splitext(args.checkpoint)[0] + ".json"
    json.dump(out, open(os.path.join(args.run, name), "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
