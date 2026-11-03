<!-- Author: wanghaobo -->

# Experiment 3 — Which language, from the middle blocks?

Name the training corpus from the middle of a character-level transformer, after the input and output layers are removed. The blog also asks which heads copy earlier text. Python is allowed; running the network is not.

## What the blog reports

- Three models, 4 layers × 6 heads, width 384, about 7.16 million weights, 28 characters.
- Corpora: Python code, Japanese in romaji (Aozora Bunko), English Wikipedia. See [docs/references.md](../docs/references.md).
- The model sees the four middle blocks only, up to 100 Python runs, and no forward pass.
- It names the language twice each and matches the measured copying heads. The post points at Elhage et al. for that style of head. Logs used effort `xhigh`.
- Logs: https://github.com/RikiyaT/lm_understand-logs/tree/main/exp3_language

## This stub

One Python block in the local sandbox. `load` / `save` / `saved` see a toy tensor, not a transformer.

## TODOs

- `TODO(weight-io)`: expose the four middle blocks to the sandbox.
- `TODO(model-call)`: replace the mock. Keep the forward-pass ban.
- `TODO(eval)`: compare named heads with heads measured by a repeated-string probe.

```bash
python3 exp03_language/run.py
```
