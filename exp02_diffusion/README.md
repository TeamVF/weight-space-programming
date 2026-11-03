<!-- Author: wanghaobo -->

# Experiment 2 — What does this diffusion model generate?

Read a small DDPM's weights and name the shape it draws. The blog forbids tools and forward passes. The model finds the shape in the output layer and draws an outline, including when the trained model fills the shape in.

## What the blog reports

- Three models, 12×12 grayscale: filled circles, squares, and triangles.
- MLP 160 → 64 → 64 → 144, ReLU, 23,824 weights. Input is the noisy image plus the diffusion step.
- No shape is named in the prompt. Logs used reasoning effort `high`.
- Logs: https://github.com/RikiyaT/lm_understand-logs/tree/main/exp2_diffusion

## This stub

One tool-free mock turn over a toy tensor. It does not train a DDPM.

## TODOs

- `TODO(weight-io)`: load one real DDPM checkpoint as text.
- `TODO(model-call)`: replace the mock client. Keep the turn tool-free.
- `TODO(eval)`: score the `Shape:` line.

```bash
python3 exp02_diffusion/run.py
```
