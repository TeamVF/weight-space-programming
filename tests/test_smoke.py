"""Import the harness and run one offline mock turn.

Author: wanghaobo
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness import MockLLMClient, PythonSandbox, TurnRunner, UnconfiguredLLMClient, load_instructions
from wsp.config import load_config
from wsp.weights import (
    count_parameters,
    format_weights_text,
    load_weights,
    parse_weights_text,
    save_weights,
    toy_tensors,
)


class SmokeTest(unittest.TestCase):
    def test_imports(self) -> None:
        import harness
        import wsp

        self.assertTrue(hasattr(harness, "TurnRunner"))
        self.assertTrue(hasattr(wsp, "load_config"))
        self.assertIn("Answer the user's request", load_instructions())

    def test_mock_turn_is_tool_free_and_does_not_run_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = TurnRunner(
                MockLLMClient("```python\nprint('should-not-run')\n```\nGame: mock\n"),
                instructions=load_instructions(),
                runs_root=tmp,
                experiment="exp01_which_game",
            )
            record = runner.run_turn("Which game?", model="mock", effort="high")
            self.assertTrue(record.tool_free)
            self.assertEqual(record.python_runs, 0)
            self.assertFalse((record.run_dir / "tool_calls.jsonl").exists())
            payload = json.loads((record.run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["audit"]["tool_free"])
            self.assertEqual(payload["audit"]["model_tools"], [])
            self.assertEqual(payload["audit"]["harness_python_runs"], 0)
            self.assertIn("Which game?", (record.run_dir / "prompt.txt").read_text(encoding="utf-8"))
            self.assertIn("Game: mock", (record.run_dir / "answer.md").read_text(encoding="utf-8"))

    def test_python_loop_runs_sandbox_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reply = "```python\nsave('probe', [1, 0])\nprint(saved())\n```\n"
            runner = TurnRunner(
                MockLLMClient(reply, final_reply="Language: mock\n"),
                instructions=load_instructions(),
                runs_root=tmp,
                experiment="exp03_language",
                sandbox_timeout_s=5,
            )
            record = runner.run_with_python("middle blocks", model="mock", effort="xhigh")
            self.assertTrue(record.tool_free)
            self.assertEqual(record.python_runs, 1)
            self.assertEqual(record.text, "Language: mock\n")
            calls = (record.run_dir / "tool_calls.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(calls), 1)
            logged = json.loads(calls[0])
            self.assertEqual(logged["returncode"], 0)
            self.assertIn("probe", logged["out"])
            saved = json.loads((record.run_dir / "work" / "net" / "probe.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, [1, 0])

    def test_weight_json_and_text_roundtrip(self) -> None:
        tensors = toy_tensors()
        self.assertEqual(count_parameters(tensors), 6)
        text = format_weights_text(tensors)
        self.assertEqual(parse_weights_text("prose\n\n" + text + "\nmore prose\n"), tensors)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_weights(Path(tmp) / "toy.json", tensors)
            self.assertEqual(load_weights(path), tensors)

    def test_numpy_npz_roundtrip_when_installed(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy is an optional extra")
        with tempfile.TemporaryDirectory() as tmp:
            path = save_weights(Path(tmp) / "toy.npz", toy_tensors())
            loaded = load_weights(path)
            self.assertEqual(sorted(loaded), ["W1", "b1"])
            self.assertEqual(loaded["b1"].tolist(), [0.0, 0.0])

    def test_unconfigured_client_raises_offline(self) -> None:
        client = UnconfiguredLLMClient()
        with self.assertRaises(RuntimeError):
            client.complete([{"role": "user", "content": "hi"}], model="mock", effort="high")

    def test_config_env_overrides_toml(self) -> None:
        keys = ("WSP_CONFIG", "WSP_MODEL", "WSP_RUNS_DIR", "WSP_SANDBOX_TIMEOUT")
        previous = {key: os.environ.get(key) for key in keys}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.toml"
                path.write_text(
                    'model = "from-file"\neffort = "low"\nruns_dir = "from-file"\n',
                    encoding="utf-8",
                )
                os.environ["WSP_CONFIG"] = str(path)
                os.environ["WSP_MODEL"] = "from-env"
                os.environ.pop("WSP_RUNS_DIR", None)
                os.environ["WSP_SANDBOX_TIMEOUT"] = "2.5"
                config = load_config()
            self.assertEqual(config["model"], "from-env")
            self.assertEqual(config["effort"], "low")
            self.assertEqual(config["runs_dir"], "from-file")
            self.assertEqual(config["sandbox_timeout_s"], 2.5)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_sandbox_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = PythonSandbox(timeout_s=0.3).run("import time\ntime.sleep(5)\nprint('no')\n", Path(tmp))
        self.assertTrue(result.timed_out)
        self.assertNotIn("no", result.stdout)

    def test_experiment_stubs_run_offline(self) -> None:
        self._run_stub("exp01_which_game/run.py", expect_python=False)
        self._run_stub("exp03_language/run.py", expect_python=True)
        self._run_stub("exp05_write_from_scratch/run.py", expect_python=False, expect_weights=True)

    def _run_stub(
        self,
        script: str,
        *,
        expect_python: bool,
        expect_weights: bool = False,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["WSP_RUNS_DIR"] = tmp
            env["WSP_CLIENT"] = "mock"
            env.pop("WSP_CONFIG", None)
            completed = subprocess.run(
                [sys.executable, str(ROOT / script)],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_dir = Path(completed.stdout.strip().splitlines()[-1])
            self.assertTrue((run_dir / "run.json").is_file())
            payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["audit"]["tool_free"])
            if expect_python:
                self.assertGreaterEqual(payload["audit"]["harness_python_runs"], 1)
            else:
                self.assertEqual(payload["audit"]["harness_python_runs"], 0)
            if expect_weights:
                written = json.loads((run_dir / "written_weights.json").read_text(encoding="utf-8"))
                self.assertIn("W1", written["tensors"])


if __name__ == "__main__":
    unittest.main()
