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

# SmolVLA — Balanced Dataset Variant

**Paper:** [`../papers/paper_smolvla.tex`](../papers/paper_smolvla.tex)  
**Part of:** [SmolVLA project](../README.md)

This is the **class-balanced** variant of SmolVLA: SmolVLM-Base fine-tuned with LoRA on a
resampled TurtleBot 4 demonstration dataset that equalizes action-class frequency.
**Recommended deployment checkpoint** — highest overall action accuracy (85.2%).

---

## Model Details

| Field | Value |
|---|---|
| **Base model** | HuggingFaceTB/SmolVLM-Base (256M params) |
| **Fine-tuning** | LoRA (r=32, alpha=64, dropout 0.05) |
| **Dataset** | 15,883 image-action pairs (class-balanced sampling) |
| **Action vocabulary** | `forward`, `backward`, `left`, `right`, `stop`, `rotate` |
| **Action format** | `<direction>_<speed>_<duration>` (e.g., `forward_0.2_3.0s`) |
| **Developed by** | Justin Williams, Clark Atlanta University |

---

## Results vs Other Variants

| Variant | Overall Acc. | Left/Right Acc. |
|---|---|---|
| Base fine-tune | 77.9% | ~70% |
| **Balanced (this)** | **85.2%** | **~83%** |
| Temporal (k=2) | ~84% | ~81% |
| Chunked (C=4) | ~82% | ~80% |

Dataset balancing improves overall accuracy by **+7.3 points**, primarily recovering
underrepresented turning maneuvers: left (+12 pt), right (+12 pt), rotate (+14 pt).

---

## Deployment — Raspberry Pi 5 (CPU-only)

| Mode | Latency | Memory | Power |
|---|---|---|---|
| bf16 | 1.2 s/query | 510 MB | 6.5 W |
| 8-bit | 0.95 s/query | 260 MB | 5.8 W |
| GGUF Q4 | 0.72 s/query | 140 MB | 5.2 W |

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

**Author:** Justin Williams · Clark Atlanta University · [justinwilliamstech693@gmail.com](mailto:justinwilliamstech693@gmail.com)
