"""Experiment 4 stub: one-shot edit of a car policy that stalls on a ramp.

Author: wanghaobo
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wsp.stub import run_offline_stub
from wsp.weights import format_weights_text, parse_weights_text, save_weights, toy_tensors

# Blog setup: Kinetix actor, 679 -> 128 (ReLU) -> five layers of 128 (tanh)
# -> 16, 171,664 weights. One reply, no tools. The logs used effort high.
PROMPT = """\
This policy drives a car at a ramp and stalls. Explain the failure, then fix only that cause by writing new weights. One reply. No tools.

The actor reads 679 scene features and emits 16 motor and thruster logits.

TODO(weight-io): the block below is a toy tensor, not the Kinetix actor. A real prompt also includes the scene and one failed episode.

{weights}
"""

MOCK_REPLY = """\
The stub has no episode trace, so it does not edit a real policy.
Replacement weights, in the toy text format:

tensor W1 shape 2 2
0.000 0.000
0.000 0.000

tensor b1 shape 2
0.000 0.000
"""


def main() -> None:
    # TODO(weight-io): load the trained actor, the scene, and one recorded episode.
    weights = format_weights_text(toy_tensors())
    # TODO(model-call): one tool-free completion. Do not add a training loop.
    # TODO(eval): roll out the edited actor. The blog scored 1,024 episodes.
    record = run_offline_stub(
        experiment="exp04_fix_car",
        prompt=PROMPT.format(weights=weights),
        mock_reply=MOCK_REPLY,
        allow_python=False,
        effort="high",
    )
    parsed = parse_weights_text(record.text)
    if parsed:
        save_weights(record.run_dir / "edited_weights.json", parsed)


if __name__ == "__main__":
    main()
