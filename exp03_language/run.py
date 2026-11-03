"""Experiment 3 stub: name the training language from middle blocks only.

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

# Blog setup: character-level transformer, 4 layers x 6 heads, width 384,
# about 7.16M weights, trained on Python, romaji, or English. The model sees
# the four middle blocks only. Python is allowed; a forward pass is not.
# The logs used reasoning effort xhigh.
PROMPT = """\
Which language was this network trained on, and which attention heads copy earlier text?

You are given the middle transformer blocks only: no input embedding, no output layer, and no labels on the residual dimensions. You may return one Python block. That block may call load, save, and saved. Do not run the network or any layer on an input.

TODO(weight-io): the block below is a toy tensor, not a 7.16M-weight checkpoint.

{weights}
"""

MOCK_REPLY = """\
```python
present = saved()
print("tensors", present)
save("note", [1])
```
"""

MOCK_FINAL = """\
Language: mock
Copying heads: none
Confidence: 0
"""


def main() -> None:
    # TODO(weight-io): load the four middle blocks and expose them to load().
    weights = format_weights_text(toy_tensors())
    # TODO(model-call): replace the mock. Keep the forward-pass ban in the prompt.
    # TODO(eval): compare named heads with the measured copying heads.
    run_offline_stub(
        experiment="exp03_language",
        prompt=PROMPT.format(weights=weights),
        mock_reply=MOCK_REPLY,
        mock_final_reply=MOCK_FINAL,
        allow_python=True,
        effort="xhigh",
    )


if __name__ == "__main__":
    main()
