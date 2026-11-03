<!-- Author: wanghaobo -->

# Experiment 5 — Write MinAtar networks from scratch

Write a game policy as text, starting from zeros. The blog gives the rules and the MinAtar source, and does not let the network play before it is scored. No tools, no loss, no gradients.

## What the blog reports

- MinAtar Breakout, Seaquest, and Asterix. https://arxiv.org/abs/1903.03176
- MLP 600 → 32 → 32 → 6, tanh, same shape as their PPO networks, weights start at 0.
- Input: 10×10, six channels. Output: six action logits.
- Scoring in the post: 1,000 episodes, sticky actions 0.1, at most 10,000 steps. Logs used effort `xhigh`.
- Logs: https://github.com/RikiyaT/lm_understand-logs/tree/main/exp5_write_from_scratch

## This stub

One mock reply that contains a toy weight block. The script parses that block into `written_weights.json`. It does not play the game.

## TODOs

- `TODO(weight-io)`: start from zeros of the 600 → 32 → 32 → 6 shape, and include the game source in the prompt.
- `TODO(model-call)`: one tool-free reply that contains the full network.
- `TODO(eval)`: score episodes only after the weights are written.

```bash
python3 exp05_write_from_scratch/run.py
```
