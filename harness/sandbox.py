"""Optional Python sandbox for experiments that allow one code block per turn.

This is a practice helper: a subprocess with a timeout and a stripped
environment. It is not the bubblewrap sandbox in the logs repo
(`harness/sandbox.py` there). Do not treat it as a security boundary.

Author: wanghaobo
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PRELUDE = """\
import json
import os
from pathlib import Path

WORK = Path(os.environ["WSP_WORK"])

def _safe_name(name):
    if not isinstance(name, str) or not name:
        raise ValueError("tensor name must be a non-empty string")
    if name in {".", ".."} or any(ch in name for ch in "/\\\\"):
        raise ValueError("tensor name must not contain a path")
    return name

def _path(name):
    return WORK / "net" / (_safe_name(name) + ".json")

def load(name):
    path = _path(name)
    if not path.exists():
        return []
    return json.loads(path.read_text())

def save(name, array):
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(array))
    return str(path)

def saved():
    directory = WORK / "net"
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.json"))
"""


@dataclass
class SandboxResult:
    stdout: str
    returncode: int
    timed_out: bool
    seconds: float


class PythonSandbox:
    """Run one Python snippet. `load` / `save` / `saved` talk to `work_dir/net`."""

    def __init__(self, timeout_s: float = 5, max_output: int = 20_000) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s
        self.max_output = max_output

    def run(self, code: str, work_dir: Path) -> SandboxResult:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        snippet = work_dir / "_wsp_snippet.py"
        snippet.write_text(PRELUDE + "\n" + code, encoding="utf-8")
        env = _child_env(work_dir)
        started = time.monotonic()
        process = subprocess.Popen(
            [sys.executable, "-I", str(snippet.name)],
            cwd=work_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_s)
            returncode = process.returncode if process.returncode is not None else 1
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(process.pid)
            stdout, stderr = process.communicate()
            returncode = process.returncode if process.returncode is not None else -1
        elapsed = round(time.monotonic() - started, 3)
        output = _clip(_combine(stdout or "", stderr or "", timed_out, self.timeout_s), self.max_output)
        return SandboxResult(
            stdout=output or "(no output)",
            returncode=returncode,
            timed_out=timed_out,
            seconds=elapsed,
        )


def _child_env(work_dir: Path) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "WSP_WORK": str(work_dir),
        "HOME": str(work_dir),
    }
    lang = os.environ.get("LANG")
    if lang:
        env["LANG"] = lang
    return env


def _combine(stdout: str, stderr: str, timed_out: bool, timeout_s: float) -> str:
    parts = [stdout]
    if stderr.strip():
        parts.append("\n[stderr]\n" + stderr)
    if timed_out:
        parts.append(f"\n[timed out after {timeout_s} s]")
    return "".join(parts).strip()


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[output cut at {limit} characters]"


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
