---
base_model: HuggingFaceTB/SmolVLM-Base
library_name: peft
tags:
  - base_model:adapter:HuggingFaceTB/SmolVLM-Base
  - lora
  - robotics
  - vla
  - mobile-robot
license: apache-2.0
---

# SmolVLA — Action chunking variant (C consecutive actions per query)

**Paper:** [`../papers/paper_smolvla.tex`](../papers/paper_smolvla.tex)  
**Part of:** [SmolVLA project](../README.md) · **Recommended checkpoint:** `smolvlm_turtlebot_action_balanced/`

SmolVLM-Base (256M params) fine-tuned with LoRA on 15,883 TurtleBot 4 image-action pairs.
This checkpoint is the **Action chunking variant (C consecutive actions per query)** in the SmolVLA ablation series.

See the [root README](../README.md) and [paper](../papers/paper_smolvla.tex) for full results,
deployment instructions, and citation.

---

## Citation

```bibtex
@article{williams2026smolvla,
  title   = {SmolVLA: Lightweight On-Device Vision-Language-Action
             for Autonomous Mobile Robots in GPS-Denied Environments},
  author  = {Williams, Justin and Gupta, Kishor Datta and
             George, Roy and Sarkar, Mrinmoy},
  year    = {2026},
  note    = {Clark Atlanta University -- manuscript}
}
```

**Author:** Justin Williams · Clark Atlanta University · justinwilliamstech693@gmail.com
