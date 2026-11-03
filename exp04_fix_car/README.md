<!-- Author: wanghaobo -->

# Experiment 4 — Fix a car policy that stalls on a ramp

A PPO policy in Kinetix drives at a ramp and never backs up. The blog asks for one reply that explains the failure and writes new weights. No tools. The reported edit changes 3.5% of the weights and succeeds on the episodes they scored.

## What the blog reports

- Kinetix car-on-ramp: back up, then clear a red strip. https://arxiv.org/abs/2410.23208
- Actor: 679 → 128 (ReLU) → five layers of 128 (tanh) → 16, 171,664 weights.
- The model gets the weights, the scene, and one recorded episode. Logs used effort `high`.
- Logs: https://github.com/RikiyaT/lm_understand-logs/tree/main/exp4_fix_car

## This stub

One tool-free mock reply. If that reply contains the toy text format, the script writes `edited_weights.json` next to `answer.md`. The numbers are not a Kinetix edit.

## TODOs

- `TODO(weight-io)`: load the actor, the scene, and one failed episode.
- `TODO(model-call)`: one real tool-free completion. Do not add a training loop.
- `TODO(eval)`: roll out the edited actor. The blog used 1,024 episodes.

```bash
python3 exp04_fix_car/run.py
```
