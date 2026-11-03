"""One tool-free model turn, plus an optional harness-owned Python loop.

Author: wanghaobo
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from harness.client import Completion, LLMClient
from harness.sandbox import PythonSandbox
from wsp.runs import RunLog

_FENCE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_INSTRUCTIONS = Path(__file__).with_name("instructions.md")


def load_instructions(path: Path | None = None) -> str:
    """Neutral system text. The default file is one sentence, on purpose.

    A leading ``Author:`` line is kept in the file and dropped here so it is
    not sent to the model.
    """
    source = path or _INSTRUCTIONS
    lines = [
        line
        for line in source.read_text(encoding="utf-8").splitlines()
        if not line.startswith("Author:")
    ]
    return "\n".join(lines).strip() + "\n"


def extract_python_block(text: str) -> str | None:
    """Return the first fenced Python block, if the reply has one."""
    match = _FENCE.search(text)
    if match is None:
        return None
    return match.group(1)


@dataclass
class TurnRecord:
    text: str
    reasoning: str
    model: str
    effort: str
    tool_free: bool
    run_dir: Path
    python_runs: int


class TurnRunner:
    """Run a conversation in which the model itself has no tools.

    `run_turn` is a single completion. It never executes code, even if the
    reply contains a Python fence.

    `run_with_python` is the loop used by experiments 3, 6, 7, and 8 in the
    blog: the model may emit one Python block, this process runs it, and the
    printed output comes back as the next user message. The model still has
    no shell, web, or file-edit tool of its own. `tool_free` stays true;
    `audit.harness_python_runs` counts sandbox runs.

    TODO(model-call): pass a real client. The mock client is the default in
    the experiment stubs.
    TODO(forward-pass): experiments 6 and 8 also budget scored forward passes.
    Those are not Python `eval` of the network. This runner does not implement
    them.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        instructions: str,
        runs_root: str | Path = "runs",
        experiment: str = "harness",
        sandbox: PythonSandbox | None = None,
        sandbox_timeout_s: float = 5,
    ) -> None:
        self.client = client
        self.instructions = instructions
        self.runs_root = Path(runs_root)
        self.experiment = experiment
        self.sandbox = sandbox or PythonSandbox(timeout_s=sandbox_timeout_s)

    def run_turn(
        self,
        prompt: str,
        *,
        model: str = "mock",
        effort: str = "high",
        run_id: str | None = None,
    ) -> TurnRecord:
        log = RunLog(self.runs_root, self.experiment, run_id)
        messages = self._open(log, prompt)
        completion = self.client.complete(messages, model=model, effort=effort)
        messages.append({"role": "assistant", "content": completion.text})
        self._finish(log, messages, completion, model=model, effort=effort, python_runs=0)
        return self._record(log, completion, model=model, effort=effort, python_runs=0)

    def run_with_python(
        self,
        prompt: str,
        *,
        model: str = "mock",
        effort: str = "high",
        max_code_turns: int = 1,
        run_id: str | None = None,
    ) -> TurnRecord:
        if max_code_turns < 0:
            raise ValueError("max_code_turns must be >= 0")
        log = RunLog(self.runs_root, self.experiment, run_id)
        messages = self._open(log, prompt)
        completion: Completion | None = None
        python_runs = 0
        for _ in range(max_code_turns + 1):
            completion = self.client.complete(messages, model=model, effort=effort)
            messages.append({"role": "assistant", "content": completion.text})
            code = extract_python_block(completion.text)
            if code is None or python_runs >= max_code_turns:
                break
            result = self.sandbox.run(code, log.directory / "work")
            python_runs += 1
            log.append_jsonl(
                "tool_calls.jsonl",
                {
                    "kind": "python",
                    "seconds": result.seconds,
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "code": code,
                    "out": result.stdout,
                },
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Sandbox output:\n{result.stdout}",
                }
            )
        if completion is None:
            raise RuntimeError("run_with_python made no model call")
        self._finish(
            log,
            messages,
            completion,
            model=model,
            effort=effort,
            python_runs=python_runs,
        )
        return self._record(log, completion, model=model, effort=effort, python_runs=python_runs)

    def _open(self, log: RunLog, prompt: str) -> list[dict[str, str]]:
        log.write_text("instructions.md", self.instructions)
        log.write_text("prompt.txt", prompt)
        return [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]

    def _finish(
        self,
        log: RunLog,
        messages: list[dict[str, str]],
        completion: Completion,
        *,
        model: str,
        effort: str,
        python_runs: int,
    ) -> None:
        log.write_jsonl("messages.jsonl", messages)
        log.write_text("answer.md", completion.text)
        if completion.reasoning:
            log.write_text("reasoning.txt", completion.reasoning + "\n")
        log.write_json(
            "run.json",
            {
                "experiment": self.experiment,
                "model": model,
                "effort": effort,
                "tool_free": True,
                "audit": {
                    "tool_free": True,
                    "model_tools": [],
                    "harness_python_runs": python_runs,
                },
                "usage": {
                    "input_tokens": completion.usage.input_tokens,
                    "output_tokens": completion.usage.output_tokens,
                },
                "turns": sum(1 for message in messages if message["role"] == "assistant"),
            },
        )

    def _record(
        self,
        log: RunLog,
        completion: Completion,
        *,
        model: str,
        effort: str,
        python_runs: int,
    ) -> TurnRecord:
        return TurnRecord(
            text=completion.text,
            reasoning=completion.reasoning,
            model=model,
            effort=effort,
            tool_free=True,
            run_dir=log.directory,
            python_runs=python_runs,
        )
