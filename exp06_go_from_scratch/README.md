<!-- Author: wanghaobo -->

# Experiment 6 — Write a Go network that does not fit in the prompt

Write a 9×9 Go policy of about 1.5 million weights. The blog's harness gives a Python sandbox that can read and save tensors. The network starts at zero. A forward pass is a separate, counted budget: the network plays practice games or is evaluated on positions. Training, search, and running the net from inside Python are not allowed.

## What the blog reports

- Architecture: 243 → 512 × 6 → 82, ReLU, 1,480,274 weights.
- Forward-pass budgets: 0, 10, 25, and 50. With zero passes the model still writes the net in one Python run.
- Logs used reasoning effort `xhigh`.
- Logs: https://github.com/RikiyaT/lm_understand-logs/tree/main/exp6_go_from_scratch

## This stub

Runs one Python block in the local sandbox and saves a 2×2 stand-in named `L1.W`. It spends no forward passes.

## TODOs

- `TODO(weight-io)`: create the zero tensors for the Go architecture under `work/net/`.
- `TODO(model-call)`: a multi-turn tool-free client. This stub allows one code block.
- `TODO(forward-pass)`: a harness call that spends the budget, rather than Python calling the net.
- `TODO(eval)`: held-out games against the opponents the model never practiced on.

```bash
python3 exp06_go_from_scratch/run.py
```
