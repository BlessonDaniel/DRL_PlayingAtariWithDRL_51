#!/usr/bin/env python3
"""Sanity checks for dqn_atari.py. Run with:  python test_dqn.py"""
import numpy as np
import torch
import torch.nn.functional as F

from dqn_atari import HISTORY, SIDE, AtariEnv, QNetwork, ReplayMemory, as_batch, preprocess


def test_preprocessing_and_frame_stack():
    env = AtariEnv("Breakout", frame_skip=4, seed=0)
    state = env.reset()
    assert len(state) == HISTORY and all(f.shape == (SIDE, SIDE) and f.dtype == np.uint8 for f in state)
    assert preprocess(env.last_rgb).shape == (SIDE, SIDE)
    assert env.last_rgb.shape == (210, 160, 3), "ALE frames are 210x160 RGB (Sec 1 of the paper)"
    nxt, reward, terminated, truncated = env.step(1)
    # the stack slides by exactly one frame: phi_{t+1} shares three frames with phi_t
    assert all(nxt[i] is state[i + 1] for i in range(HISTORY - 1)) and nxt[-1] is not state[-1]
    assert 4 <= env.n_actions <= 18, "the paper: between 4 and 18 valid actions"


def test_replay_memory():
    rng = np.random.default_rng(0)
    mem = ReplayMemory(capacity=50, rng=rng)
    frame = lambda v: np.full((SIDE, SIDE), v, np.uint8)
    for t in range(80):                                   # more pushes than capacity: the oldest 30 are overwritten
        s = tuple(frame(t + k) for k in range(HISTORY))
        s2 = tuple(frame(t + 1 + k) for k in range(HISTORY))
        mem.push(s, t % 4, float(t % 3 - 1), s2, float(t % 10 == 0))
    assert len(mem) == 50
    s, a, r, s2, d = mem.sample(32, "cpu")
    assert s.shape == s2.shape == (32, HISTORY, SIDE, SIDE) and s.dtype == torch.uint8
    assert a.dtype == torch.int64 and r.dtype == torch.float32 and d.dtype == torch.float32
    t = s[:, 0, 0, 0].long()                              # each transition is tagged by its time index
    assert int(t.min()) >= 30, "transitions older than the capacity must be gone"
    assert torch.equal(s2[:, 0, 0, 0].long(), t + 1), "next state must be the state one step later"
    assert torch.equal(a, t % 4) and torch.equal(r, (t % 3 - 1).float()) and torch.equal(d, (t % 10 == 0).float())


def test_network():
    torch.manual_seed(0)
    q = QNetwork(n_actions=4)
    x = torch.randint(0, 256, (8, HISTORY, SIDE, SIDE), dtype=torch.uint8)
    out = q(x)
    assert out.shape == (8, 4)
    # 16 filters 8x8 on 4 channels, 32 filters 4x4 on 16 channels, 9*9*32 -> 256, 256 -> 4 (weights + biases)
    expected = (8 * 8 * 4 * 16 + 16) + (4 * 4 * 16 * 32 + 32) + (9 * 9 * 32 * 256 + 256) + (256 * 4 + 4)
    assert sum(p.numel() for p in q.parameters()) == expected == 677_172
    q_nhwc = QNetwork(4).to(memory_format=torch.channels_last)
    q_nhwc.load_state_dict(q.state_dict())
    q_nhwc.nhwc = True
    assert torch.allclose(out, q_nhwc(x), atol=1e-5), "the NHWC fast path must give the same Q-values"
    assert as_batch(tuple(np.zeros((SIDE, SIDE), np.uint8) for _ in range(HISTORY)), "cpu").shape == (1, HISTORY, SIDE, SIDE)


def test_target_and_learning_step():
    torch.manual_seed(0)
    q = QNetwork(4)
    s = torch.randint(0, 256, (32, HISTORY, SIDE, SIDE), dtype=torch.uint8)
    s2 = torch.randint(0, 256, (32, HISTORY, SIDE, SIDE), dtype=torch.uint8)
    a = torch.randint(0, 4, (32,))
    r = torch.randint(-1, 2, (32,)).float()
    terminal = (torch.arange(32) % 4 == 0).float()
    with torch.no_grad():
        y = r + 0.99 * (1 - terminal) * q(s2).max(1).values
    assert torch.equal(y[terminal == 1], r[terminal == 1]), "terminal transitions: y = r (Algorithm 1)"
    opt = torch.optim.RMSprop(q.parameters(), lr=2.5e-4, alpha=0.95, eps=0.01)
    losses = []
    for _ in range(60):                                    # the same batch again and again: the loss has to fall
        loss = F.mse_loss(q(s).gather(1, a.unsqueeze(1)).squeeze(1), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.75 * losses[0], (losses[0], losses[-1])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all checks passed")
