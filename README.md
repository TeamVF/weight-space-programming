<!-- Author: wanghaobo -->

# weight-space-programming

Research and reproduce Weight-Space Programming experiments (LLM native read/write of neural network weights).

Weight-Space Programming, introduced by Takehi, Teoh, Zhang, and Khattab, asks a general-purpose language model to read and write a network's weights directly. The model is not an optimizer: it does not run gradient descent, and in the reported runs it is not allowed to write a training loop. Given a natural-language spec, and sometimes with no environment feedback at all, it reasons about what each part of a network computes and edits the numbers themselves.

The blog sorts the idea into three claims, across eight experiments: models can **read** weights (what a policy, a diffusion model, or a small language model was trained to do), **fix** weights (a stuck control policy, then a math failure in Qwen3.5-0.8B), and **write** weights from scratch (MinAtar policies, then a 1.5M-parameter Go network). Later experiments add a Python sandbox because the weights no longer fit in one prompt. The model still has no tools of its own; the harness runs the code and returns stdout.

This repository is a small offline harness for practicing that setup. It is not a reproduction of the Astra runs, and it does not ship their prompts, weights, or scores. Those logs are the public record of the post.

## References

- Blog: [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, Khattab, 2 November 2026).
- Logs and per-experiment READMEs: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs).
- Short list of the environments, checkpoints, and papers the blog points at for these experiments: [docs/references.md](docs/references.md).

## Layout

| Path | What it is |
| --- | --- |
| [`harness/`](harness/README.md) | Tool-free turn runner, mock LLM client, optional Python sandbox |
| [`wsp/`](wsp) | Toy weight IO (JSON, optional NumPy/PyTorch), run logs, TOML/env config |
| `exp01_which_game/` … `exp08_math/` | One stub per experiment in the blog and the logs repo |
| `runs/` | Created on first run. Gitignored |
| [`tests/test_smoke.py`](tests/test_smoke.py) | Imports, one mock turn, weight round-trip, two stub scripts |
| [`config.example.toml`](config.example.toml) | Copy to `config.toml` or point `WSP_CONFIG` at it |

The folder names follow the logs repo (`exp1_which_game` through `exp8_math`), with a zero so they sort in experiment order.

| Folder | Blog experiment | Logs | Tools in the post |
| --- | --- | --- | --- |
| [`exp01_which_game`](exp01_which_game) | Which game was this policy trained on? | [`exp1_which_game`](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp1_which_game) | none |
| [`exp02_diffusion`](exp02_diffusion) | What shape does this diffusion model draw? | [`exp2_diffusion`](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp2_diffusion) | none, no forward pass |
| [`exp03_language`](exp03_language) | Which language, from the middle blocks only? | [`exp3_language`](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp3_language) | Python sandbox, no forward pass |
| [`exp04_fix_car`](exp04_fix_car) | Fix a car policy that stalls on a ramp | [`exp4_fix_car`](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp4_fix_car) | none |
| [`exp05_write_from_scratch`](exp05_write_from_scratch) | Write MinAtar networks from zeros | [`exp5_write_from_scratch`](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp5_write_from_scratch) | none |
| [`exp06_go_from_scratch`](exp06_go_from_scratch) | Write a 1.5M-weight Go network | [`exp6_go_from_scratch`](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp6_go_from_scratch) | Python sandbox, budgeted forward passes |
| [`exp07_sycophancy`](exp07_sycophancy) | Which of two models is sycophantic? | [`exp7_sycophancy`](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp7_sycophancy) | Python sandbox, then an attention tool |
| [`exp08_math`](exp08_math) | Edit Qwen3.5-0.8B (and the Gemma follow-up) | [`exp8_math`](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math) | Python sandbox, budgeted forward passes |

Each experiment folder has a short README and a `run.py` stub. Search for `TODO(model-call)`, `TODO(weight-io)`, `TODO(forward-pass)`, and `TODO(eval)` before treating a stub as a real run.

## Smoke check

No API key, no network, and no extra packages. Python 3.11 or newer is enough.

```bash
make check
```

That runs `python3 -m unittest discover -s tests -v` (`PYTHON` overrides the interpreter). It imports `harness` and `wsp`, runs one mock tool-free turn, round-trips a toy weight file, and executes the experiment 1, 3, and 5 stubs.

## Run one stub

```bash
python3 exp01_which_game/run.py
python3 exp03_language/run.py
```

Output lands in `runs/<experiment>/<run_id>/` (`prompt.txt`, `answer.md`, `run.json`, and `tool_calls.jsonl` when the sandbox runs). The default client is `MockLLMClient`. `WSP_CLIENT=unconfigured` raises instead of calling a model; that class is the seam for a local tool-free client.

```bash
cp config.example.toml config.toml
# optional overrides: WSP_MODEL, WSP_EFFORT, WSP_RUNS_DIR, WSP_CONFIG, WSP_SANDBOX_TIMEOUT
```

Optional installs, not required to import or to pass `make check`:

```bash
pip install -e ".[weights]"   # numpy .npz IO
pip install -e ".[torch]"     # torch .pt IO
pip install -e ".[dev]"       # pytest
```
