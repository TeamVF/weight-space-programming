"""Experiment 8 stub: edit a small language model toward better math.

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

# Blog setup: Qwen3.5-0.8B, thinking on, edited with no gradients. Forward-pass
# budgets were 10, 25, and 50. A Gemma 3 270M follow-up did not gain a new
# math ability. The logs used reasoning effort max.
PROMPT = """\
Make this language model better at math by editing its weights. Do not train and do not use gradients. You may return one Python block using load, save, and saved.

The blog's target is Qwen3.5-0.8B. A follow-up asks the same of Gemma 3 270M, which rarely loops. Neither checkpoint is loaded here.

TODO(weight-io): mount the real tensors, or a diff against the public base weights.
TODO(forward-pass): a forward pass is a budgeted generation, not a Python eval of the stack. Budgets in the post: 10, 25, 50. This stub spends none.

{weights}
"""

MOCK_REPLY = """\
```python
print("checkpoint", saved())
save("edit", [0])
```
"""

MOCK_FINAL = """\
Edit: none
Weights changed: 0
"""


def main() -> None:
    # TODO(weight-io): open Qwen3.5-0.8B, or the Gemma follow-up, as named tensors.
    weights = format_weights_text(toy_tensors())
    # TODO(model-call): a long tool-free session. Effort in the logs is max.
    # TODO(eval): MATH-500, GSM8K, IFEval, and the token-limit rate. Do not
    # show those sets to the editing model.
    run_offline_stub(
        experiment="exp08_math",
        prompt=PROMPT.format(weights=weights),
        mock_reply=MOCK_REPLY,
        mock_final_reply=MOCK_FINAL,
        allow_python=True,
        effort="max",
    )


if __name__ == "__main__":
    main()
