<!-- Author: wanghaobo -->

# Experiment 7 — Which language model is sycophantic?

Two copies of Gemma 3 270M are fine-tuned the same way, except one always agrees with the user's suggested answer. Token embeddings stay frozen. The blog reports that weights alone were not enough to tell the copies apart. After an attention tool — patterns only, no output logits — the model picked the sycophant.

## What the blog reports

- Gemma 3 270M, instruction-tuned, 18 layers. Three pairs.
- On facts held out of training, the sycophant follows wrong guesses and the other copy mostly does not.
- Python over the weights, at most 20 runs, then the attention tool on up to 16 texts the model writes.
- Logs used effort `xhigh`. QA data: https://github.com/RikiyaT/lm_understand-logs/tree/main/exp7_sycophancy/dataset
- Logs: https://github.com/RikiyaT/lm_understand-logs/tree/main/exp7_sycophancy

## This stub

One sandbox turn over a toy tensor. There is no Gemma checkpoint and no attention tool.

## TODOs

- `TODO(weight-io)`: load both checkpoints. Keep the shared embedding identical.
- `TODO(model-call)`: weights-only attempts, then the attention tool.
- `TODO(attention-tool)`: return attention weights only. Do not return logits or generated text.
- `TODO(eval)`: score which id was selected.

```bash
python3 exp07_sycophancy/run.py
```
