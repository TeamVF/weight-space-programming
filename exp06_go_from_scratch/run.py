"""Experiment 6 stub: write a Go network in the Python sandbox.

Author: wanghaobo
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wsp.stub import run_offline_stub
from wsp.weights import format_weights_text, toy_tensors

# Blog setup: 9x9 Go, MLP 243 -> 512 x 6 -> 82, ReLU, 1,480,274 weights,
# starting at zero. Python may read and save tensors. Forward passes are a
# separate budget (0, 10, 25, 50), not arbitrary Python. Logs used effort xhigh.
PROMPT = """\
Write a 9x9 Go policy into the weight tensors. They start at zero. You may return one Python block that calls load, save, and saved. Do not train on a dataset. A scored forward pass is not available in this stub.

Architecture reported in the blog: 243 -> 512 x 6 -> 82, ReLU. Input planes are the two stone colors and whose turn it is. Output is one logit per point plus pass.

TODO(weight-io): the text below is a toy tensor. A real run keeps the full net under work/net/.
TODO(forward-pass): the blog budgets 0, 10, 25, or 50 forward passes. This stub spends none.

{weights}
"""

MOCK_REPLY = """\
```python
save("L1.W", [[0.0, 0.0], [0.0, 0.0]])
print(saved())
```
"""

MOCK_FINAL = """\
Status: mock weights written
Forward passes used: 0
"""


def main() -> None:
    # TODO(weight-io): create the zero tensors named in the Go architecture.
    weights = format_weights_text(toy_tensors())
    # TODO(model-call): multi-turn client. This stub allows one Python block.
    # TODO(forward-pass): implement the budgeted rollout as a harness call.
    # TODO(eval): 2,000 games against the held-out opponents.
    run_offline_stub(
        experiment="exp06_go_from_scratch",
        prompt=PROMPT.format(weights=weights),
        mock_reply=MOCK_REPLY,
        mock_final_reply=MOCK_FINAL,
        allow_python=True,
        effort="xhigh",
    )


if __name__ == "__main__":
    main()
