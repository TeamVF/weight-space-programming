<!-- Author: wanghaobo -->

# References

Canonical sources for this repo, then the environments, datasets, and papers the blog itself points to that matter for experiments 1–8. Titles and URLs below are taken from the blog page (its prose, inline links, and `<d-bibliography>`), the logs repo, or the arXiv record for a link the blog already uses. This is a short list, not the blog's full related-work bibliography.

## Canonical

Takehi, Rikiya, Jayden Teoh, Alex Zhang, and Omar Khattab. "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights." 2 November 2026.

https://rikiyat.github.io/blog/weight-space-programming/

```bibtex
@misc{takehi2026weightspace,
  title        = {{Sparks of Weight-Space Programming: LLMs can natively read \& write neural network weights}},
  author       = {Takehi, Rikiya and Teoh, Jayden and Zhang, Alex and Khattab, Omar},
  year         = {2026},
  month        = nov,
  howpublished = {\url{https://rikiyat.github.io/blog/weight-space-programming/}},
  note         = {Blog post}
}
```

Complete logs of the runs the post reports (prompts, replies, sandbox code, evaluations, networks):

https://github.com/RikiyaT/lm_understand-logs

The logs README is the map from each folder to a section of the post. Its `harness/` directory (`codex_config.toml`, `instructions.md`, `sandbox.py`) is the reference for how those runs disabled model tools and, where allowed, executed Python outside the model.

## Checkpoints and data the post links

- Qwen3.5-0.8B model card, including the note on thinking loops that experiment 8 discusses: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Edited Qwen3.5-0.8B from experiment 8: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
- Live demo of that edit: https://huggingface.co/spaces/akhaliq/weight-space-programming-demo
- Experiment 3, Python corpus manifest and rebuild script: https://github.com/RikiyaT/lm_understand-logs/tree/main/exp3_language/python_dataset
- Experiment 3, romaji text (Aozora Bunko, cleaned): https://huggingface.co/datasets/globis-university/aozorabunko-clean
- Experiment 3, English Wikipedia dump cited by the post (1 November 2023): https://huggingface.co/datasets/wikimedia/wikipedia
- Experiment 7, QA data: https://github.com/RikiyaT/lm_understand-logs/tree/main/exp7_sycophancy/dataset

## Experiment environments and methods

- Experiment 3, copying heads read from weights. Elhage, Nelson, et al. "A Mathematical Framework for Transformer Circuits." Transformer Circuits Thread, 2021. https://transformer-circuits.pub/2021/framework/index.html
- Experiment 4, car-on-ramp task. Matthews, Michael, Michael Beukman, Chris Lu, and Jakob Foerster. "Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks." ICLR 2025. https://arxiv.org/abs/2410.23208
- Experiment 5, Breakout, Seaquest, and Asterix. Young, Kenny, and Tian Tian. "MinAtar: An Atari-Inspired Testbed for Thorough and Reproducible Reinforcement Learning Experiments." arXiv:1903.03176, 2019. https://arxiv.org/abs/1903.03176
- Experiment 8, reasoning loops. Pipis, Charilaos, et al. "Wait, Wait, Wait... Why Do Reasoning Models Loop?" arXiv:2512.12895, 2025. https://arxiv.org/abs/2512.12895
- Experiment 8 describes the Qwen edit as removing one direction. Arditi, Andy, et al. "Refusal in Language Models Is Mediated by a Single Direction." arXiv:2406.11717, 2024. https://arxiv.org/abs/2406.11717
- Reading weights with a language model, cited in the post's related work. Kabra, Krishna, Constantin Venhoff, and Christian Schroeder de Witt. "Weight Oracles: Reading Neural Network Weights with Language Models." arXiv:2610.07334, 2026. https://arxiv.org/abs/2610.07334
- Hand-setting the weights of a tiny transformer, cited as the small-scale prior. Papailiopoulos, Dimitris. "AdderBoard: A Community Challenge for Tiny Arithmetic Transformers." 2026. https://github.com/anadim/AdderBoard
- Locating and editing a fact in the weights, grouped by the post with other interpretability edits. Meng, Kevin, David Bau, Alex Andonian, and Yonatan Belinkov. "Locating and Editing Factual Associations in GPT." NeurIPS 2022. https://arxiv.org/abs/2202.05262

## Links in the post's opening paragraph

The post's first technical paragraph links these three as the older ways weights get written. The arXiv titles are the titles on those records, which are not all the same as the anchor text.

- Karpathy, Andrej. "Software 2.0." https://karpathy.medium.com/software-2-0-a64152b37c35 (anchor text: "optimization")
- Schlag, Imanol, Kazuki Irie, and Jürgen Schmidhuber. "Linear Transformers Are Secretly Fast Weight Programmers." arXiv:2102.11174, 2021. https://arxiv.org/abs/2102.11174 (anchor text: "custom networks trained to write them")
- Lindner, David, et al. "Tracr: Compiled Transformers as a Laboratory for Interpretability." arXiv:2301.05062, 2023. https://arxiv.org/abs/2301.05062 (anchor text: "compilers from formal programs")
