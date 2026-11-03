"""Experiment 2 stub: name the shape a diffusion MLP was trained to draw.

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

# Blog setup: MLP 160 -> 64 -> 64 -> 144, ReLU, 23,824 weights. One model each
# for filled circles, squares, and triangles. No tools and no forward pass.
# The logs used reasoning effort high.
PROMPT = """\
What shape was this diffusion model trained to generate?

The network is an MLP with ReLU activations. Input: a noisy 12x12 grayscale image (144 values) and the diffusion step (16 values). Output: the clean image (144 values). Do not run a forward pass and do not use tools. Answer with a Shape line.

TODO(weight-io): the block below is a toy tensor, not a DDPM checkpoint.

{weights}
"""

MOCK_REPLY = """\
Observations: offline stub; these weights are not a diffusion model.
Reasoning: no model was called, and no forward pass was run.
Shape: mock
Confidence: 0
"""


def main() -> None:
    # TODO(weight-io): load one DDPM checkpoint and put the full weight text here.
    weights = format_weights_text(toy_tensors())
    # TODO(model-call): replace the mock client. Keep the turn tool-free.
    # TODO(eval): score the Shape line against circle / square / triangle.
    run_offline_stub(
        experiment="exp02_diffusion",
        prompt=PROMPT.format(weights=weights),
        mock_reply=MOCK_REPLY,
        allow_python=False,
        effort="high",
    )


if __name__ == "__main__":
    main()
