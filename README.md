# Offline Autonomy — Research Portfolio

**Justin Williams** · Ph.D. Candidate, Cyber-Physical Systems · Clark Atlanta University  
Supported by the **Air Force Research Laboratory (AFRL)** and the **Griffiss Institute**  
Advisor: Dr. Kishor Datta Gupta

---

This repository collects three independent research threads, each with a companion paper.
All systems share a common theme: **on-device, offline-capable AI for autonomous robots**
operating in GPS-denied or communication-contested environments.

---

## Research Projects

### 1. SmolVLA — Lightweight On-Device VLA for Mobile Robots
**Paper:** [`papers/paper_smolvla.tex`](papers/paper_smolvla.tex)

Fine-tunes **SmolVLM-Base (256M params)** with LoRA on 15,883 teleoperated TurtleBot 4 demonstrations to produce a robot policy that runs at **1.2 s/query on a Raspberry Pi 5 at 6.5 W** — no GPU, no cloud.

| Variant | Description | Checkpoint |
|---|---|---|
| Base fine-tune | Initial LoRA fine-tune | `smolvlm_turtlebot_action_ft/` |
| VLA | VLA-framed prompt format | `smolvlm_turtlebot_vla/` |
| Temporal | $k$-frame history input | `smolvlm_turtlebot_vla_temporal/` |
| Chunked | Predict $C$ actions per query | `smolvlm_turtlebot_vla_chunked/` |
| Balanced | Class-balanced dataset | `smolvlm_turtlebot_action_balanced/` |

**Key result:** Dataset balancing improves overall action accuracy from 77.9% → 85.2%; action chunking at $C=4$ reduces query rate 4×.

```bibtex
@article{williams2026smolvla,
  title   = {SmolVLA: Lightweight On-Device Vision-Language-Action for Autonomous Mobile Robots in GPS-Denied Environments},
  author  = {Williams, Justin and Gupta, Kishor Datta and George, Roy and Sarkar, Mrinmoy},
  year    = {2026},
  note    = {Clark Atlanta University — manuscript}
}
```

---

### 2. Satellite Captioning — Offline VLM for ISR
**Paper:** [`papers/paper_satellite.tex`](papers/paper_satellite.tex)

Fine-tunes **SmolVLM-Base** with LoRA on 9,828 geo-referenced satellite images for natural-language scene description, entirely offline at < 512 MB RAM.

| Metric | Zero-shot | Fine-tuned |
|---|---|---|
| BLEU-4 | 4.8 | **18.6** |
| CIDEr | 22.3 | **89.4** |
| Memory | 510 MB | 140 MB (Q4) |
| Latency | — | 1.8 s/image (RPi5 CPU) |

```bibtex
@article{williams2026satellite,
  title   = {Offline Satellite Image Captioning for Contested ISR: Fine-Tuning SmolVLM on Multi-Reference Aerial Imagery},
  author  = {Williams, Justin and Gupta, Kishor Datta and George, Roy and Sarkar, Mrinmoy},
  year    = {2026},
  note    = {Clark Atlanta University — manuscript}
}
```

---

### 3. OpenVLA-OFT Quantization — Edge VLA for Manipulation
**Paper:** [`openvla_oft_pipeline/papers/paper_combined.tex`](openvla_oft_pipeline/papers/paper_combined.tex)  
**Also:** separate papers on activations (`paper1_`), transform zoo (`paper2_`), reachability (`paper3_`).

Post-training quantization of an **OpenVLA-OFT policy** (Llama-2-7B + DINOv2+SigLIP) fine-tuned on LIBERO manipulation benchmarks. Core finding: the quantization bottleneck is **activation outliers, not weights** — a single orthogonal DCT rotation enables W4A4 at 98–100% task retention.

| Config | Success | Size | Reduction |
|---|---|---|---|
| bf16 baseline (A100) | 88.0% | 15.4 GB | — |
| Vision-INT8 + DCT-W3A8 + Head-INT8 (A100) | **89.5%** | **4.0 GB** | **74%** |
| W4A4+DCT **on real Jetson Orin** | **86.0%** | ~4.1 GB | ~73% |

**Real-hardware confirmed:** W4A4+DCT runs on Jetson AGX Orin at 86.0% (−2.2 pt vs A100 bf16, within ±4% CI — statistically indistinguishable). First closed-loop confirmation of a quantized L1-regression VLA on edge compute.

```bibtex
@article{williams2026quant,
  title   = {Activations, Not Weights: Orthogonal Preconditioning for Low-Bit Quantization of Vision-Language-Action Policies},
  author  = {Williams, Justin and Gupta, Kishor Datta and George, Roy and Sarkar, Mrinmoy},
  year    = {2026},
  note    = {Clark Atlanta University — manuscript}
}
```

---

## Repository Structure

```
offline-robotcs/
├── papers/                              # Papers 1 & 2
│   ├── paper_smolvla.tex
│   └── paper_satellite.tex
├── smolvlm_turtlebot_action_ft/        # SmolVLA: base fine-tune
├── smolvlm_turtlebot_vla/              # SmolVLA: VLA framing
├── smolvlm_turtlebot_vla_temporal/     # SmolVLA: temporal context
├── smolvlm_turtlebot_vla_chunked/      # SmolVLA: action chunking
├── smolvlm_turtlebot_action_balanced/  # SmolVLA: balanced dataset
├── smolvlm_satellite_captioning/       # Satellite captioning model
└── openvla_oft_pipeline/               # OpenVLA-OFT quantization
    └── papers/                         # Papers 3a–3d
        ├── paper_combined.tex
        ├── paper1_activations_not_weights.tex
        ├── paper2_transform_zoo_benchmark.tex
        └── paper3_reachability_short.tex
```

---

## Author

**Justin Williams**  
Ph.D. Candidate, Cyber-Physical Systems  
Clark Atlanta University  
[justinwilliamstech693@gmail.com](mailto:justinwilliamstech693@gmail.com)
