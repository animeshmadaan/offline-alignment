# AuxDPO / χPO / AuxχPO at the Llama-3.1-8B scale

Code accompanying our CS497 report on **likelihood displacement** in offline preference optimisation. We train Llama-3.1-8B with LoRA adapters and compare four pairwise contrastive objectives on AlpacaFarm and MMLU-Pro:

- **DPO** — standard direct preference optimisation
- **AuxDPO** — auxiliary null-space variables that absorb reward misspecification
- **χPO** — χ²+KL link function for added pessimism
- **AuxχPO** — novel combination of the two

Trainers extend HuggingFace `transformers` / `accelerate`, with W&B logging. Entry point: [scripts/run.sh](scripts/run.sh). DPO loss variants live in [trainers/dpo.py](trainers/dpo.py) and [trainers/dpo_trainer.py](trainers/dpo_trainer.py).

## Acknowledgement

This codebase is built on top of [Asap7772/understanding-rlhf](https://github.com/Asap7772/understanding-rlhf), which provided the original AuxDPO trainer scaffolding and the `src/` vendored copies of `trl`, `trlx`, and `alpaca-farm`. Our contributions are the χPO and AuxχPO objectives, the LoRA-based Llama-3.1-8B training pipeline, and the MMLU-Pro evaluation setup.
