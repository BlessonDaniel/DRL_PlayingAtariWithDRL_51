#!/usr/bin/env python3
"""
Watch a trained agent: play games with the eps = 0.05 evaluation policy and save

    game_strip.png        a few RGB frames from the best game played (for slides / screenshots)
    preprocessing.png     one frame at each stage of phi: raw RGB -> grey 110x84 -> 84x84 crop
    value_trace.csv       max_a Q(s, a) at every step of that game (the paper's Figure 3 idea)
    game.gif              optional replay of the game (--gif)

Usage:
    python watch_agent.py --run runs/breakout_cpu --games 5 --gif
    python watch_agent.py --run runs/breakout_cpu --checkpoint model_best.pt --name watch_best
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
import torch

from dqn_atari import AtariEnv, QNetwork, as_batch


def upscale(img, k):
    return cv2.resize(img, (img.shape[1] * k, img.shape[0] * k), interpolation=cv2.INTER_NEAREST)


@torch.no_grad()
def play(q, env, rng, eps, max_steps=20_000):
    frames, values, rewards = [], [], []
    state, score = env.reset(), 0.0
    for _ in range(max_steps):
        q_values = q(as_batch(state, "cpu"))[0]
        action = int(rng.integers(env.n_actions)) if rng.random() < eps else int(q_values.argmax())
        frames.append(env.last_rgb.copy())
        values.append(float(q_values.max()))
        state, reward, terminated, truncated = env.step(action)
        rewards.append(reward)
        score += reward
        if terminated or truncated:
            break
    return score, frames, values, rewards


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/breakout_cpu")
    p.add_argument("--checkpoint", default="model_last.pt", help="model_last.pt (end of training) or model_best.pt (best evaluation)")
    p.add_argument("--name", default="watch", help="sub-folder of the run for the output")
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--gif", action="store_true")
    args = p.parse_args()

    cfg = json.load(open(os.path.join(args.run, "config.json")))
    env = AtariEnv(cfg["game"], cfg["frame_skip"], args.seed, cfg["crop_top"])
    q = QNetwork(env.n_actions)
    q.load_state_dict(torch.load(os.path.join(args.run, args.checkpoint), map_location="cpu"))
    q.eval()
    rng = np.random.default_rng(args.seed)

    games = [play(q, env, rng, args.eps) for _ in range(args.games)]
    scores = [g[0] for g in games]
    score, frames, values, rewards = max(games, key=lambda g: g[0])
    print(f"scores of {args.games} games (eps={args.eps}): {scores} -> keeping the best one "
          f"({score:.0f} points, {len(frames)} agent steps)")

    out = os.path.join(args.run, args.name)
    os.makedirs(out, exist_ok=True)

    # four frames spread over the game, with the running score printed under each
    picks = np.linspace(len(frames) * 0.12, len(frames) - 1, 4).astype(int)
    running = np.cumsum(rewards)
    on_screen = running - np.array(rewards)              # frame i was captured before the reward of step i
    tiles = []
    for i in picks:
        tile = upscale(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR), 3)
        bar = np.full((44, tile.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"step {i}   score {int(on_screen[i])}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        tiles.append(np.vstack([tile, bar]))
        tiles.append(np.full((tile.shape[0] + 44, 16, 3), 255, np.uint8))
    cv2.imwrite(os.path.join(out, "game_strip.png"), np.hstack(tiles[:-1]))
    for n, i in enumerate(picks):
        cv2.imwrite(os.path.join(out, f"frame_{n}.png"), upscale(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR), 4))

    # the three stages of phi on one mid-game frame
    rgb = frames[picks[1]]
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(grey, (84, 110), interpolation=cv2.INTER_AREA)
    crop = small[cfg["crop_top"]:cfg["crop_top"] + 84]
    cv2.imwrite(os.path.join(out, "phi_1_raw_rgb_210x160.png"), upscale(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 4))
    cv2.imwrite(os.path.join(out, "phi_2_grey_110x84.png"), upscale(small, 6))
    cv2.imwrite(os.path.join(out, "phi_3_crop_84x84.png"), upscale(crop, 6))

    with open(os.path.join(out, "value_trace.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "max_q", "reward", "score"])
        w.writerows(zip(range(len(values)), np.round(values, 4), rewards, running))

    with open(os.path.join(out, "watch.json"), "w") as f:
        json.dump({"checkpoint": args.checkpoint, "eps": args.eps, "scores_all_games": scores, "score": score,
                   "agent_steps": len(frames), "frames_shown": [{"step": int(i), "score": int(on_screen[i])} for i in picks]}, f, indent=1)

    if args.gif:
        from PIL import Image
        imgs = [Image.fromarray(upscale(f, 2)) for f in frames[::2]]
        imgs[0].save(os.path.join(out, "game.gif"), save_all=True, append_images=imgs[1:], duration=66, loop=0)
    print("wrote", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
