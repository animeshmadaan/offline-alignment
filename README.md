# Offline Alignment of LLMs

This code is a part of my undergraduate project on **likelihood displacement** in offline preference optimisation. I did this project with Prof. Sayak Ray Chowdhury at IIT Kanpur. We train Llama-3.1-8B with LoRA adapters and compare four pairwise contrastive objectives on AlpacaFarm and MMLU-Pro:

- **DPO**: standard direct preference optimisation
- **AuxDPO**: auxiliary null-space variables that absorb reward misspecification
- **χPO**: χ²+KL link function for added pessimism
- **AuxχPO**: novel combination of the two

## Acknowledgement

This codebase is built on top of [Asap7772/understanding-rlhf](https://github.com/Asap7772/understanding-rlhf), which provided the original DPO trainer pipeline. Our contributions are the AuxDPO, χPO, and AuxχPO objectives, the LoRA-based Llama-3.1-8B training pipeline, and the MMLU-Pro evaluation setup.
