<!-- Author: wanghaobo -->

# Experiment 1 — Which game was this network trained on?

Read a policy's raw weights and name the game. The blog asks this as an open question: four games share one architecture, and the model is not given the list.

## What the blog reports

- Grid versions of Breakout, Pong, Pac-Man, and Freeway, all 10×10, five shared actions.
- MLP 500 → 32 → 32 → 5, tanh, 17,253 weights. PPO, 1 billion steps, same settings per game.
- The model receives the weights as text and a description that does not name the game.
- No tools and no rollout. Logs used reasoning effort `high`.
- Logs: https://github.com/RikiyaT/lm_understand-logs/tree/main/exp1_which_game

## This stub

Writes a toy tensor, formats it as text, and runs one offline mock turn. It does not train a policy and does not score a guess.

## TODOs

- `TODO(weight-io)`: load a real actor and put every weight in the prompt.
- `TODO(model-call)`: replace `MockLLMClient` with a tool-free local client.
- `TODO(eval)`: map the `Game:` line to a label.

```bash
python3 exp01_which_game/run.py
```
