<!-- Author: wanghaobo -->

# harness

Shared runner for one tool-free model turn, and an optional Python sandbox for the experiments that need it.

In the [logs repo](https://github.com/RikiyaT/lm_understand-logs), every turn is one Codex CLI call with shell, file editing, web search, and the other built-in tools switched off (`harness/codex_config.toml`, `harness/instructions.md`). Where code was allowed (experiments 3, 6, 7, and 8), the model wrote one Python block and their `harness/sandbox.py` ran it under bubblewrap. This package is a small, importable stand-in for that shape. It does not shell out to Codex and it does not require an API key.

```python
from harness import MockLLMClient, TurnRunner, load_instructions

runner = TurnRunner(
    MockLLMClient("Game: mock\nConfidence: 0\n"),
    instructions=load_instructions(),
    runs_root="runs",
    experiment="exp01_which_game",
)
record = runner.run_turn("Which game?", model="mock", effort="high")
assert record.tool_free
```

`run_turn` is a single completion. A Python fence in the reply is text, not a tool call.

`run_with_python` is the sandbox loop. The first fenced `python` block is executed, stdout is appended as the next user message, and the client is called once more. `run.json` keeps `audit.tool_free: true` and `audit.model_tools: []`. `audit.harness_python_runs` counts sandbox runs. That matches the logs repo's distinction: the model was tool-free, and the harness owned the code.

Each run directory contains:

| File | Contents |
| --- | --- |
| `instructions.md` | System text for the turn |
| `prompt.txt` | First user message |
| `messages.jsonl` | System, user, assistant, and any sandbox replies |
| `answer.md` | Last assistant message |
| `reasoning.txt` | Client reasoning string, when it sent one |
| `run.json` | Model, effort, tool audit, token usage, turn count |
| `tool_calls.jsonl` | One JSON object per sandbox run (`code`, `out`, `seconds`) |
| `work/` | Sandbox working directory (`net/*.json` via `load` / `save` / `saved`) |

## Clients

`MockLLMClient` is deterministic and offline. `UnconfiguredLLMClient.complete` raises `RuntimeError`. Implement that method for a local tool-free client.

TODO(model-call): a Codex-backed client should spawn `codex exec` (and `codex exec resume` for later turns) with tools disabled, and should record the command in `run.json`. Do not import a paid SDK at import time.

## Sandbox

`PythonSandbox` runs `python -I` in a subprocess with a timeout, a new process group, and an environment that keeps `PATH` and drops the parent environment (so API keys are not forwarded). Inside the snippet, `load(name)`, `save(name, array)`, and `saved()` read and write JSON lists under `work/net/`.

This is not bubblewrap, and it is not a security boundary. The logs repo's sandbox is the one that was used for the published runs. The timeout here defaults to 5 seconds (`WSP_SANDBOX_TIMEOUT` or `sandbox_timeout_s` in the TOML config). The published Go sandbox used a much longer limit.

TODO(forward-pass): experiments 6 and 8 also spend a budget of scored forward passes. Those calls are part of the harness, not arbitrary Python, and they are not implemented here.
