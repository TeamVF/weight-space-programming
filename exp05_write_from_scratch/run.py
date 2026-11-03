"""Experiment 5 stub: write a MinAtar policy from zeros, as text.

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

# Blog setup: MinAtar Breakout, Seaquest, or Asterix. MLP 600 -> 32 -> 32 -> 6,
# tanh, every weight starting at 0. One reply, no tools, no rollout before scoring.
# The logs used reasoning effort xhigh.
PROMPT = """\
Write the weights of a network that plays this game. The weights start at zero. Do not train, and do not use tools. Reply with the tensors in the text format shown below.

Game: unspecified MinAtar game (Breakout, Seaquest, or Asterix in the blog).
Input: 10x10 grid, 6 channels (player, danger, reward, projectile, trail, gauge).
Output: 6 action logits.
Architecture: 600 -> 32 -> 32 -> 6, tanh.

TODO(weight-io): a real prompt includes the game rules and source, and an all-zero network of that shape. The block below is a toy tensor so this stub can run.

{weights}
"""

MOCK_REPLY = """\
Offline stub. These are not MinAtar weights.

tensor W1 shape 2 2
0.250 -0.500
0.000 0.125

tensor b1 shape 2
0.000 0.000
"""


def main() -> None:
    # TODO(weight-io): start from zeros of shape 600 -> 32 -> 32 -> 6.
    weights = format_weights_text(toy_tensors())
    # TODO(model-call): one tool-free reply that contains the full network.
    # TODO(eval): 1,000 episodes, sticky actions 0.1, at most 10,000 steps.
    record = run_offline_stub(
        experiment="exp05_write_from_scratch",
        prompt=PROMPT.format(weights=weights),
        mock_reply=MOCK_REPLY,
        allow_python=False,
        effort="xhigh",
    )
    parsed = parse_weights_text(record.text)
    if parsed:
        save_weights(record.run_dir / "written_weights.json", parsed)


if __name__ == "__main__":
    main()
