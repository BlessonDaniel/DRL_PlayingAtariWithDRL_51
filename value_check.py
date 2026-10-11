#!/usr/bin/env python3
"""
Does the network over-estimate?  Compare what the agent PREDICTS with what it actually GETS.

For every step of a few evaluation games (eps = 0.05) we record
    predicted   max_a Q(s, a)                      the network's estimate of the discounted return
    realised    sum_k gamma^k * clip(r_{t+k})      the discounted (clipped) reward it then collected
and average both over all visited states. A large gap means the Q-values are too optimistic, the
bias of the max operator that Double DQN (van Hasselt et al., 2016) was later designed to remove.
The prediction is for the greedy policy and the games are played with eps = 0.05, so a small gap
is expected; a gap of several times is not.

Usage:
    python value_check.py --run runs/breakout_cpu --checkpoint model_last.pt --games 10
"""
import argparse
import json
import os

import numpy as np
import torch

from dqn_atari import AtariEnv, QNetwork, as_batch


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/breakout_cpu")
    p.add_argument("--checkpoint", default="model_last.pt")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--max-steps", type=int, default=5000, help="cap per game, in case the ball gets stuck in a loop")
    args = p.parse_args()

    cfg = json.load(open(os.path.join(args.run, "config.json")))
    gamma = cfg["gamma"]
    env = AtariEnv(cfg["game"], cfg["frame_skip"], args.seed, cfg["crop_top"])
    q = QNetwork(env.n_actions)
    q.load_state_dict(torch.load(os.path.join(args.run, args.checkpoint), map_location="cpu"))
    q.eval()
    rng = np.random.default_rng(args.seed)

    predicted, realised, scores = [], [], []
    for _ in range(args.games):
        state, values, rewards, score = env.reset(), [], [], 0.0
        for _ in range(args.max_steps):
            q_values = q(as_batch(state, "cpu"))[0]
            action = int(rng.integers(env.n_actions)) if rng.random() < args.eps else int(q_values.argmax())
            values.append(float(q_values.max()))
            state, reward, terminated, truncated = env.step(action)
            rewards.append(float(np.sign(reward)))             # the clipped reward the network was trained on
            score += reward
            if terminated or truncated:
                break
        g, returns = 0.0, []
        for r in reversed(rewards):                            # discounted return-to-go
            g = r + gamma * g
            returns.append(g)
        predicted += values
        realised += returns[::-1]
        scores.append(score)

    out = {"checkpoint": args.checkpoint, "games": args.games, "states": len(predicted),
           "mean_game_score": round(float(np.mean(scores)), 2),
           "mean_predicted_max_q": round(float(np.mean(predicted)), 3),
           "mean_realised_return": round(float(np.mean(realised)), 3)}
    out["ratio"] = round(out["mean_predicted_max_q"] / max(out["mean_realised_return"], 1e-9), 1)
    name = "value_check_" + os.path.splitext(args.checkpoint)[0] + ".json"
    json.dump(out, open(os.path.join(args.run, name), "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
