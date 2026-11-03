"""Experiment 1 stub: name the game from raw policy weights.

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

# Blog setup: MLP 500 -> 32 -> 32 -> 5, tanh, 17,253 weights. Same net for
# Breakout, Pong, Pac-Man, and Freeway. The logs used reasoning effort high.
PROMPT = """\
Which game was this neural network trained on?

The network is an MLP with tanh activations. The input is a 10x10 grid in five channels (agent, moving objects, reward targets, obstacles, previous frame). The output is five action logits: no-op, up, down, left, right. No game is named below. Answer with a Game line and a Confidence line. Do not use tools.

TODO(weight-io): the block below is a toy tensor, not a PPO actor.

{weights}
"""

MOCK_REPLY = """\
Observations: offline stub; the tensor above is not a trained policy.
Reasoning: no model was called.
Game: mock
Description: placeholder
Confidence: 0
"""


def main() -> None:
    # TODO(weight-io): load one real actor and render every weight into the prompt.
    weights = format_weights_text(toy_tensors())
    # TODO(model-call): MockLLMClient is offline. Swap in a tool-free local client.
    # TODO(eval): map the Game line onto a label and write eval/scores.json.
    run_offline_stub(
        experiment="exp01_which_game",
        prompt=PROMPT.format(weights=weights),
        mock_reply=MOCK_REPLY,
        allow_python=False,
        effort="high",
    )


if __name__ == "__main__":
    main()
