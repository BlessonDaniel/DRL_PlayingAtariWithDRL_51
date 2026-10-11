# DQN from scratch: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)

Deep Reinforcement Learning (AIMLCZG512), BITS Pilani WILP, Assignment Problem II, Group 51.

A PyTorch re-implementation of Algorithm 1 of [arXiv:1312.5602](https://arxiv.org/abs/1312.5602), a CPU-sized
training run on Atari Breakout, and a small ablation of experience replay. Everything on the "our reproduction"
slides of the presentation comes from the logs in `runs/`.

## Files

| File | What it is |
|---|---|
| `dqn_atari.py` | The agent: pre-processing, Q-network, replay memory, training loop, evaluation (about 300 lines) |
| `replay_ablation.py` | CartPole experiment: uniform replay vs two ways of learning without replay |
| `watch_agent.py` | Plays games with a saved checkpoint; saves frames, a GIF and the max-Q trace |
| `value_check.py` | Compares predicted max-Q with the discounted return the agent really collects |
| `trick_check.py` | Compares the agent with the four "always press one button" policies |
| `plot_results.py` | Figures from the logs |
| `test_dqn.py` | Sanity checks (replay memory, network size, targets, one learning step) |
| `DQN_Atari_2013.ipynb` | The same code as one Kaggle / Colab notebook |
| `runs/` | Configs, logs and checkpoints of the runs we report |
| `figures/` | Plots made by `plot_results.py` |

## Setup

```bash
pip install -r requirements.txt      # the Atari ROMs ship with ale-py, nothing else to download
python test_dqn.py
```

## Reproduce our numbers

```bash
# the 2013 algorithm, CPU budget (about 98 minutes on one CPU thread)
python dqn_atari.py --game Breakout --steps 500000 --train-every 4 --threads 1 --seed 0 --out runs/breakout_cpu

# the same run with the three stabilisers of the 2015 Nature paper switched on
python dqn_atari.py --game Breakout --steps 500000 --train-every 4 --threads 1 --seed 0 \
    --target-update 1000 --huber --life-loss-terminal --out runs/breakout_cpu_2015fixes

python replay_ablation.py --out runs/ablation            # 45 CartPole runs, about 43 CPU-minutes
python watch_agent.py --run runs/breakout_cpu --gif
python value_check.py --run runs/breakout_cpu
python trick_check.py --run runs/breakout_cpu
python plot_results.py --runs runs/breakout_cpu runs/breakout_cpu_2015fixes --ablation runs/ablation
```

With a GPU, drop `--train-every 4` (one update per agent step, as in Algorithm 1) and raise `--steps`;
`--paper-scale` runs the full 10M frames with a 1M-frame replay memory.

## What follows the paper and what does not

| | Paper (Sec. 4 and 5) | `dqn_atari.py` default | Our CPU runs |
|---|---|---|---|
| Pre-processing | grey, 110x84, crop 84x84, last 4 frames | same | same |
| Network | conv 16@8x8/4, conv 32@4x4/2, FC 256, one output per action | same (677,172 weights for 4 actions) | same |
| Learning | Q-learning, uniform replay, minibatch 32, RMSProp, no target network | same | same |
| Reward clipping, frame-skip | -1/0/+1, k = 4 | same | same |
| Exploration | eps 1.0 to 0.1 over the first 1M frames | over the first 0.4M frames | same as default |
| Evaluation | eps = 0.05 for a fixed number of steps | 10,000 steps | same |
| Updates | one per agent step | one per agent step | **one per 4 agent steps** |
| Frames, replay size | 10M, 1M | 2M, 100k | 2M, 100k |
| Learning rate, discount | not reported | 2.5e-4, 0.99 (from Nature 2015) | same |

One epoch in the paper's Figure 2 is 50,000 minibatch updates and the curves run for 100 epochs.
Our runs do 122,501 updates each, so they cover roughly the first 2.5 of those 100 epochs.

## Results

### Breakout, 500k agent steps (2M frames), seed 0, one CPU thread

| Run | Training score, last 0.4M frames | Evaluation: last / best checkpoint | Final agent, 10 games | Predicted max-Q vs realised return | Minutes |
|---|---|---|---|---|---|
| 2013 algorithm | 6.28 | 10.36 / 10.75 | 9.4 | 0.96 vs 0.69 | 98 |
| + 2015 additions | 3.55 | 4.78 / 11.00 | 6.0 | not comparable (per-life values) | 87 |

Baselines under the same protocol (eps = 0.05, 20 games): always LEFT 10.65, always FIRE 1.05, always NOOP 0.75, always RIGHT 0.1.
Pure random play (the first 10,000 training steps, 55 games): 1.20, against 1.2 for "Random" in Table 1 of the paper.

* Training score: mean game score of the eps-greedy training games. Evaluation: eps = 0.05 for 10,000 steps, as in the paper.
* Final agent: `trick_check.py` on `model_last.pt`. Predicted vs realised: `value_check.py` on `model_last.pt`.

**What the runs show.** Both agents improve on random play (1.20 points per game): the training score over the last 0.4M frames is 6.28 for the 2013 algorithm and 3.55 with the 2015 additions. The evaluation score is very noisy, between 0 and 11 from one checkpoint to the next. Watching the agents explains why: the high-scoring checkpoints park the paddle at the left wall. `trick_check.py` confirms it. A policy that always presses LEFT scores 10.65 (20 games, eps = 0.05); the final 2013-style agent scores 9.4 and the final agent with the 2015 additions 6.0 (10 games each), and the best checkpoint of each run picks LEFT at every single step. So with about 2.5% of the paper's updates the agents have found a one-button trick and are not tracking the ball yet. The 2015 additions gave faster early progress (4.01 against 1.98 at 0.4M frames) and Q-values that rise smoothly (0.03 to 0.62) instead of jumping between 1.06 and 2.23, but not a better score at this budget. With one seed per setting, treat the comparison as an observation, not a result.

**Over-estimation.** `value_check.py` on the 2013-style run: the final checkpoint predicts an average max-Q of 0.96 and collects a discounted return of 0.69; the best checkpoint predicts 1.45 against 0.68. (For the run with life-loss terminals the prediction is the return up to the next lost life, so it is not comparable with the whole-game return.)

**Reproducibility.** Re-running the first 100k frames with the same seed on the same machine gave an identical log.

![training curves](figures/atari_training.png)

### Replay ablation on CartPole-v1 (mean ± std over 5 seeds, mean return of the last 10 evaluations, maximum 500)

| Agent | lr 1e-4 | lr 3e-4 | lr 1e-3 |
|---|---|---|---|
| Uniform replay (the paper) | 293.9 ± 82.9 | 272.3 ± 38.0 | 315.5 ± 53.3 |
| Last 32 transitions, no replay | 108.6 ± 42.6 | 126.7 ± 37.5 | 402.5 ± 76.1 |
| Latest transition only (online) | 18.6 ± 3.3 | 49.3 ± 33.1 | 197.7 ± 77.0 |

Uniform replay is the only agent above 270 at all three learning rates, and plain online Q-learning is the worst at each one. At the highest learning rate the agent trained on its last 32 transitions scores higher than replay (402.5 against 315.5), so on a problem as easy as CartPole replay makes learning robust to the learning rate rather than being strictly necessary. All agents share the network, RMSProp settings, eps schedule and one update per step; 45 runs in total.

![replay ablation](figures/replay_ablation.png)

## Limits of this reproduction

* One seed per Atari setting and about 2.5% of the paper's updates: the Atari curves show the start of learning, not the scores of Table 1.
* ALE 0.12 through Gymnasium 1.4, not the 2013 emulator build.
* The ablation is on CartPole, a far easier problem than Atari; DeepMind's own Atari ablation is in Extended Data Table 3 of the Nature paper.

## References

1. V. Mnih et al. Playing Atari with Deep Reinforcement Learning. NIPS Deep Learning Workshop, 2013. arXiv:1312.5602
2. V. Mnih et al. Human-level control through deep reinforcement learning. Nature 518, 529-533, 2015. doi:10.1038/nature14236
3. M. Bellemare et al. The Arcade Learning Environment. JAIR 47, 253-279, 2013. arXiv:1207.4708
4. H. van Hasselt, A. Guez, D. Silver. Deep Reinforcement Learning with Double Q-learning. AAAI 2016. arXiv:1509.06461
