"""Offline entry point shared by the experiment stubs.

Author: wanghaobo
"""

from __future__ import annotations

import os

from harness.client import MockLLMClient, UnconfiguredLLMClient
from harness.runner import TurnRecord, TurnRunner, load_instructions
from wsp.config import load_config


def run_offline_stub(
    *,
    experiment: str,
    prompt: str,
    mock_reply: str,
    mock_final_reply: str | None = None,
    allow_python: bool = False,
    model: str | None = None,
    effort: str | None = None,
) -> TurnRecord:
    """Run one stub conversation and print the run directory.

    TODO(model-call): `WSP_CLIENT=mock` (the default) never calls a model.
    `WSP_CLIENT=unconfigured` raises, which is the seam for a real client.
    TODO(weight-io): callers should build `prompt` from a real checkpoint.
    The toy tensors in the stubs are only there so the path runs offline.
    """
    config = load_config()
    chosen_model = model or str(config["model"])
    chosen_effort = effort or str(config["effort"])
    client_kind = os.environ.get("WSP_CLIENT", "mock")
    if client_kind == "mock":
        client = MockLLMClient(mock_reply, final_reply=mock_final_reply)
    elif client_kind == "unconfigured":
        client = UnconfiguredLLMClient()
    else:
        raise SystemExit(
            f"Unknown WSP_CLIENT={client_kind!r}. Use 'mock' or 'unconfigured'."
        )
    runner = TurnRunner(
        client,
        instructions=load_instructions(),
        runs_root=str(config["runs_dir"]),
        experiment=experiment,
        sandbox_timeout_s=float(config["sandbox_timeout_s"]),
    )
    if allow_python:
        record = runner.run_with_python(
            prompt,
            model=chosen_model,
            effort=chosen_effort,
            max_code_turns=1,
        )
    else:
        record = runner.run_turn(prompt, model=chosen_model, effort=chosen_effort)
    print(record.run_dir)
    return record
