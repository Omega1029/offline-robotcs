---
base_model: HuggingFaceTB/SmolVLM-Base
library_name: peft
tags:
  - base_model:adapter:HuggingFaceTB/SmolVLM-Base
  - lora
  - remote-sensing
  - satellite
  - image-captioning
  - isr
license: apache-2.0
---

# SmolVLM Satellite Captioning — Offline ISR Model

**Paper:** [`../papers/paper_satellite.tex`](../papers/paper_satellite.tex)

Fine-tuned **SmolVLM-Base** (256M parameters) with LoRA for natural-language description of
satellite imagery, running entirely offline at < 512 MB RAM. Designed for ISR in GPS-denied
and communication-contested environments.

---

## Model Details

| Field | Value |
|---|---|
| **Base model** | HuggingFaceTB/SmolVLM-Base (256M params) |
| **Fine-tuning method** | LoRA (r=32, alpha=64) |
| **Trainable parameters** | ~12M (4.7% of total) |
| **Adapter size** | ~50 MB |
| **Task** | Satellite image captioning |
| **Developed by** | Justin Williams, Clark Atlanta University |
| **Funded by** | AFRL / Griffiss Institute |

---

## Training Data

| Split | Images |
|---|---|
| Training | 8,734 |
| Validation | 1,094 |
| **Total** | **9,828** |

- Geo-referenced overhead imagery spanning urban, agricultural, forested, coastal, and arid terrain
- Multiple human-authored reference captions per image
- **Caption strategy:** random multi-reference sampling per batch
- **Fixed instruction:** `"Describe this satellite image in detail."`

---

## Results

| Model | BLEU-4 | CIDEr | ROUGE-L | Memory | Latency (RPi5) |
|---|---|---|---|---|---|
| SmolVLM zero-shot | 3.1 | 18.7 | 22.8 | 510 MB | — |
| BLIP-2 zero-shot | 4.8 | 22.3 | 25.6 | 15 GB | GPU only |
| **This model (fine-tuned)** | **18.6** | **89.4** | **51.3** | **140 MB (Q4)** | **1.8 s/img** |

Fine-tuning delivers a **6x CIDEr improvement** over zero-shot while using **17x fewer parameters** than BLIP-2.

---

## Usage

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image

base = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Base")
model = PeftModel.from_pretrained(base, "path/to/smolvlm_satellite_captioning")
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Base")

image = Image.open("satellite_tile.jpg")
prompt = "USER: <image> Describe this satellite image in detail. ASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

---

## Deployment

| Mode | Memory | Latency | Power |
|---|---|---|---|
| bf16 (GPU) | 510 MB | 0.4 s | ~70 W |
| 8-bit (CPU, RPi5) | 260 MB | 2.8 s | 5.8 W |
| GGUF Q4 (CPU, RPi5) | 140 MB | 1.8 s | 5.2 W |

---

## Citation

```bibtex
@article{williams2026satellite,
  title   = {Offline Satellite Image Captioning for Contested ISR:
             Fine-Tuning SmolVLM on Multi-Reference Aerial Imagery},
  author  = {Williams, Justin and Gupta, Kishor Datta and
             George, Roy and Sarkar, Mrinmoy},
  year    = {2026},
  note    = {Clark Atlanta University -- manuscript}
}
```

**Author:** Justin Williams · Clark Atlanta University · [justinwilliamstech693@gmail.com](mailto:justinwilliamstech693@gmail.com)  
**Acknowledgments:** AFRL, Griffiss Institute, Dr. Kishor Datta Gupta
