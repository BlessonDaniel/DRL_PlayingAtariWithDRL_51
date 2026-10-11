#!/usr/bin/env python3
"""
Deep Q-learning with experience replay -- a from-scratch reproduction of

    Mnih et al., "Playing Atari with Deep Reinforcement Learning",
    NIPS Deep Learning Workshop 2013, arXiv:1312.5602.

What the paper specifies is implemented as written:
  Sec 4.1  phi(.)      RGB -> grey-scale -> 110x84 -> crop 84x84 -> stack of the last 4 frames
  Sec 4.1  Q-network   conv 16@8x8/4 -> conv 32@4x4/2 -> FC 256 -> one linear output per action
  Alg. 1   learning    eps-greedy acting, store (phi, a, r, phi') in D, sample a uniform
                       minibatch of 32, one gradient step on (y - Q(phi, a))^2
  Sec 5    training    rewards clipped to {-1, 0, +1}, frame-skip k = 4, RMSProp,
                       eps annealed linearly 1.0 -> 0.1, evaluation with eps = 0.05
  Sec 5.1  metrics     average score per game + average max-Q on a held-out set of states

There is NO separate target network: targets use the current weights, exactly as in
Algorithm 1. The three stabilisers that the 2015 Nature paper added are available as
switches, all off by default, to see what they change:
    --target-update N        separate target network, copied every N updates
    --huber                  error clipping (Huber loss instead of squared error)
    --life-loss-terminal     a lost life ends the episode for learning purposes

What the paper does not report (learning rate, discount factor, RMSProp constants) is taken
from the Nature follow-up (Mnih et al., 2015): lr = 2.5e-4, gamma = 0.99.

Usage:
    python dqn_atari.py --game Breakout --steps 500000 --train-every 4 --out runs/breakout_cpu   # our CPU run
    python dqn_atari.py --game Breakout --paper-scale --out runs/breakout_full                     # needs a GPU + ~8 GB RAM
"""
import argparse
import csv
import json
import os
import time
from collections import deque

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

gym.register_envs(ale_py)
cv2.ocl.setUseOpenCL(False)

HISTORY = 4      # frames stacked by phi
SIDE = 84        # final image side


# --------------------------------------------------------------------------- pre-processing
def preprocess(rgb, crop_top=18):
    """Sec 4.1: 210x160 RGB -> grey-scale -> down-sample to 110x84 -> crop the 84x84 playing area."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(grey, (SIDE, 110), interpolation=cv2.INTER_AREA)   # cv2 takes (width, height)
    return np.ascontiguousarray(small[crop_top:crop_top + SIDE, :])    # copy, so the 110-row image is freed


class AtariEnv:
    """One ALE game with the paper's frame-skip and the phi(.) pre-processing + 4-frame stack.

    frameskip=k makes the emulator repeat the chosen action for k frames and return the last
    one, which is the frame-skipping described in Sec 5. Sticky actions are turned off because
    the 2013 experiments used the deterministic ALE.
    """

    def __init__(self, game, frame_skip=4, seed=0, crop_top=18):
        self.env = gym.make(f"ALE/{game}-v5", frameskip=frame_skip,
                            repeat_action_probability=0.0, full_action_space=False)
        self.n_actions = int(self.env.action_space.n)     # minimal ("valid") action set: 4-18 actions
        self.frames = deque(maxlen=HISTORY)
        self.crop_top = crop_top
        self._seed = seed
        self.last_rgb = None
        self.lives, self.life_lost = 0, False

    def reset(self):
        rgb, info = self.env.reset(seed=self._seed)
        self._seed = None                                  # seed only the first reset
        self.last_rgb = rgb
        self.lives, self.life_lost = info.get("lives", 0), False
        frame = preprocess(rgb, self.crop_top)
        for _ in range(HISTORY):
            self.frames.append(frame)
        return tuple(self.frames)

    def step(self, action):
        rgb, reward, terminated, truncated, info = self.env.step(int(action))
        self.last_rgb = rgb
        lives = info.get("lives", self.lives)
        self.life_lost, self.lives = lives < self.lives, lives   # only used by --life-loss-terminal
        self.frames.append(preprocess(rgb, self.crop_top))
        return tuple(self.frames), float(reward), terminated, truncated


# --------------------------------------------------------------------------- replay memory
class ReplayMemory:
    """The replay memory D: the last N transitions, sampled uniformly at random (Sec 4).

    A state is a tuple of 4 frame references, and consecutive states share 3 of them, so each
    stored transition costs roughly one 84x84 uint8 frame (~7 KB) instead of eight.
    """

    def __init__(self, capacity, rng):
        self.capacity, self.rng = capacity, rng
        self.data, self.pos = [], 0

    def __len__(self):
        return len(self.data)

    def push(self, state, action, reward, next_state, terminal):
        item = (state, action, reward, next_state, terminal)
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.pos] = item                     # overwrite the oldest transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, device):
        idx = self.rng.integers(len(self.data), size=batch_size)
        s = np.empty((batch_size, HISTORY, SIDE, SIDE), np.uint8)
        s2 = np.empty_like(s)
        a = np.empty(batch_size, np.int64)
        r = np.empty(batch_size, np.float32)
        d = np.empty(batch_size, np.float32)
        for b, i in enumerate(idx):
            state, a[b], r[b], next_state, d[b] = self.data[i]
            for k in range(HISTORY):
                s[b, k] = state[k]
                s2[b, k] = next_state[k]
        to = lambda x: torch.from_numpy(x).to(device)
        return to(s), to(a), to(r), to(s2), to(d)


# --------------------------------------------------------------------------- Q-network
class QNetwork(nn.Module):
    """Sec 4.1: the state goes in, one Q-value per action comes out (a single forward pass
    gives Q(s, a) for every action, instead of one pass per action)."""

    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(HISTORY, 16, kernel_size=8, stride=4)    # 84x84x4 -> 20x20x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)         # 20x20x16 -> 9x9x32
        self.fc = nn.Linear(32 * 9 * 9, 256)
        self.out = nn.Linear(256, n_actions)
        self.nhwc = False                      # set on CPU: NHWC memory layout makes the convolutions ~1.4x faster

    def forward(self, x):                      # x: uint8 [B, 4, 84, 84]
        x = x.float().div_(255.0)
        if self.nhwc:
            x = x.contiguous(memory_format=torch.channels_last)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc(x.flatten(1)))
        return self.out(x)


def as_batch(state, device):
    return torch.from_numpy(np.stack(state)).unsqueeze(0).to(device)


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(q, env, n_steps, eps, rng, device):
    """Sec 5.1 / 5.3: run an eps-greedy policy (eps = 0.05) for a fixed number of steps and
    average the un-clipped game score over the games completed in that time."""
    scores, score = [], 0.0
    state = env.reset()
    for _ in range(n_steps):
        if rng.random() < eps:
            action = rng.integers(env.n_actions)
        else:
            action = q(as_batch(state, device)).argmax(1).item()
        state, reward, terminated, truncated = env.step(action)
        score += reward
        if terminated or truncated:
            scores.append(score)
            score = 0.0
            state = env.reset()
    if not scores:                     # no game finished inside the budget: report the one in progress
        scores.append(score)
    return float(np.mean(scores)), len(scores)


def collect_holdout_states(game, frame_skip, crop_top, n_states, seed):
    """Sec 5.1: 'a fixed set of states collected by running a random policy before training'."""
    env, rng = AtariEnv(game, frame_skip, seed, crop_top), np.random.default_rng(seed)
    states, state = [], env.reset()
    while len(states) < n_states:
        state, _, terminated, truncated = env.step(rng.integers(env.n_actions))
        if rng.random() < 0.1:                             # thin out: neighbouring frames are near-duplicates
            states.append(np.stack(state))
        if terminated or truncated:
            state = env.reset()
    return np.stack(states)


@torch.no_grad()
def average_max_q(q, holdout, device):
    out = [q(torch.from_numpy(holdout[i:i + 256]).to(device)).max(1).values for i in range(0, len(holdout), 256)]
    return torch.cat(out).mean().item()


# --------------------------------------------------------------------------- training
def train(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    env = AtariEnv(args.game, args.frame_skip, args.seed, args.crop_top)
    eval_env = AtariEnv(args.game, args.frame_skip, args.seed + 1000, args.crop_top)
    eval_rng = np.random.default_rng(args.seed + 1000)
    holdout = collect_holdout_states(args.game, args.frame_skip, args.crop_top, args.holdout_states, args.seed + 2000)

    def make_network():
        net = QNetwork(env.n_actions).to(device)
        if device.type == "cpu":
            net = net.to(memory_format=torch.channels_last)
            net.nhwc = True
        return net

    q = make_network()
    target_q = make_network() if args.target_update else q            # 2013 paper: one network for both
    target_q.load_state_dict(q.state_dict())
    optimiser = torch.optim.RMSprop(q.parameters(), lr=args.lr, alpha=0.95, eps=0.01)
    memory = ReplayMemory(args.replay_size, rng)

    config = dict(vars(args), device=str(device), n_actions=env.n_actions,
                  parameters=sum(p.numel() for p in q.parameters()),
                  torch=torch.__version__, gymnasium=gym.__version__, ale_py=ale_py.__version__)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config))

    log_path = os.path.join(args.out, "log.csv")
    games_path = os.path.join(args.out, "train_games.csv")           # score of every training game (eps-greedy)
    with open(games_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "score"])
    fields = ["step", "frames", "minutes", "epsilon", "games", "train_score_last10", "eval_score",
              "eval_games", "avg_max_q", "loss", "steps_per_sec"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    def log_row(step, t0, eps, games, recent, losses, last_t, last_step):
        eval_score, eval_games = evaluate(q, eval_env, args.eval_steps, args.eval_eps, eval_rng, device)
        row = [step, step * args.frame_skip, round((time.time() - t0) / 60, 2), round(eps, 4), games,
               round(float(np.mean(recent)), 2) if recent else "", round(eval_score, 2), eval_games,
               round(average_max_q(q, holdout, device), 4),
               round(float(np.mean(losses)), 6) if losses else "",
               round((step - last_step) / max(time.time() - last_t, 1e-9), 1)]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(" | ".join(f"{k}={v}" for k, v in zip(fields, row)), flush=True)
        return eval_score

    t0 = last_t = time.time()
    last_step, games, updates, best = 0, 0, 0, -float("inf")
    recent, losses, score = deque(maxlen=10), [], 0.0
    log_row(0, t0, 1.0, 0, recent, losses, last_t, 0)       # untrained network = the "random" baseline
    last_t = time.time()

    state = env.reset()
    for step in range(1, args.steps + 1):
        # eps-greedy behaviour policy, eps annealed linearly from 1.0 to eps_end (Sec 5)
        eps = max(args.eps_end, 1.0 - (1.0 - args.eps_end) * step / args.eps_anneal_steps)
        if rng.random() < eps:
            action = int(rng.integers(env.n_actions))
        else:
            with torch.no_grad():
                action = q(as_batch(state, device)).argmax(1).item()

        next_state, reward, terminated, truncated = env.step(action)
        terminal = terminated or (args.life_loss_terminal and env.life_lost)
        memory.push(state, action, float(np.sign(reward)), next_state, float(terminal))    # clipped reward
        state, score = next_state, score + reward
        if terminated or truncated:
            recent.append(score)
            with open(games_path, "a", newline="") as f:
                csv.writer(f).writerow([step, score])
            games, score, state = games + 1, 0.0, env.reset()

        # one minibatch update per agent step (Algorithm 1)
        if step >= args.learning_starts and step % args.train_every == 0:
            s, a, r, s2, done = memory.sample(args.batch_size, device)
            with torch.no_grad():                                   # y_j = r_j (+ gamma * max_a' Q(phi_j+1, a'))
                y = r + args.gamma * (1.0 - done) * target_q(s2).max(1).values
            q_sa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q_sa, y) if args.huber else F.mse_loss(q_sa, y)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            updates += 1
            losses.append(loss.item())
            if args.target_update and updates % args.target_update == 0:
                target_q.load_state_dict(q.state_dict())

        if step % args.eval_every == 0 or step == args.steps:
            eval_score = log_row(step, t0, eps, games, recent, losses, last_t, last_step)
            torch.save(q.state_dict(), os.path.join(args.out, "model_last.pt"))
            if eval_score >= best:
                best = eval_score
                torch.save(q.state_dict(), os.path.join(args.out, "model_best.pt"))
            losses, last_t, last_step = [], time.time(), step

    print(f"done: {args.steps} steps ({args.steps * args.frame_skip} frames), {updates} updates, "
          f"best eval score {best:.2f}, {(time.time() - t0) / 60:.1f} min")


def parse_args():
    p = argparse.ArgumentParser(description="DQN (Mnih et al., 2013) on an Atari 2600 game")
    p.add_argument("--game", default="Breakout", help="ALE game name, e.g. Breakout, Pong, SpaceInvaders")
    p.add_argument("--out", default="runs/breakout")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=500_000, help="agent steps (= frames / frame_skip)")
    p.add_argument("--frame-skip", type=int, default=4, help="k in the paper; they use 3 for Space Invaders")
    p.add_argument("--crop-top", type=int, default=18, help="first row of the 84-row crop of the 110x84 image")
    p.add_argument("--replay-size", type=int, default=100_000, help="paper: 1,000,000")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.99, help="not given in the 2013 paper; Nature 2015 value")
    p.add_argument("--lr", type=float, default=2.5e-4, help="not given in the 2013 paper; Nature 2015 value")
    p.add_argument("--eps-end", type=float, default=0.1)
    p.add_argument("--eps-anneal-steps", type=int, default=100_000, help="paper: 1M frames = 250k steps")
    p.add_argument("--learning-starts", type=int, default=10_000, help="random steps before the first update")
    p.add_argument("--train-every", type=int, default=1, help="agent steps per minibatch update (paper: 1)")
    p.add_argument("--target-update", type=int, default=0,
                   help="0 = no target network (2013 paper); N = copy weights every N updates (Nature 2015)")
    p.add_argument("--huber", action="store_true", help="Huber loss = error clipping of Nature 2015 (default: squared error)")
    p.add_argument("--life-loss-terminal", action="store_true",
                   help="treat a lost life as the end of an episode when learning (Nature 2015; default: off)")
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--eval-steps", type=int, default=10_000, help="paper: 10,000 steps")
    p.add_argument("--eval-eps", type=float, default=0.05)
    p.add_argument("--holdout-states", type=int, default=500)
    p.add_argument("--threads", type=int, default=0, help="CPU threads for PyTorch (0 = default)")
    p.add_argument("--device", default="", help="cuda / mps / cpu (default: best available)")
    p.add_argument("--paper-scale", action="store_true",
                   help="10M frames, 1M replay, eps annealed over the first 1M frames")
    args = p.parse_args()
    if args.paper_scale:
        args.steps = 10_000_000 // args.frame_skip
        args.replay_size = 1_000_000
        args.eps_anneal_steps = 1_000_000 // args.frame_skip
        args.learning_starts = 50_000
        args.eval_every = 50_000
    return args


if __name__ == "__main__":
    train(parse_args())
