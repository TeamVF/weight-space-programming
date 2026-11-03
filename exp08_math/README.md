<!-- Author: wanghaobo -->

# Experiment 8 — Make a language model better at math

Edit Qwen3.5-0.8B by writing weights, with no gradient step. The blog reports that the model traced reasoning loops to one direction in the MLPs of layers 15 and 19 and changed 131,072 weights (0.016%). A follow-up on Gemma 3 270M, which rarely loops, did not add a math ability the model lacked.

## What the blog reports

- Qwen3.5-0.8B, thinking on. Practice items come from the GSM8K and MATH training sets, not the test sets.
- Forward-pass budgets: 10, 25, and 50. A pass answers a batch of problems or reads texts. No GPU in the editing kernel.
- The post's Qwen sampling note is on the model card: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Edited checkpoint: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
- Logs used effort `max`. Logs: https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math

## This stub

One sandbox turn. It writes a one-element placeholder named `edit` and does not open either model. It spends no forward passes.

## TODOs

- `TODO(weight-io)`: mount Qwen3.5-0.8B or Gemma 3 270M, or a diff against the public base weights.
- `TODO(model-call)`: a long tool-free session.
- `TODO(forward-pass)`: counted generations, separate from the Python kernel.
- `TODO(eval)`: score MATH-500, GSM8K, and IFEval without showing those sets to the editor.

```bash
python3 exp08_math/run.py
```
