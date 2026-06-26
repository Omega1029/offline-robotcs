import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from peft import PeftModel
from PIL import Image
import csv
from pathlib import Path
import json
from typing import Tuple, List, Dict

try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.cider.cider import Cider
    HAVE_COCO_EVAL = True
except ImportError:
    HAVE_COCO_EVAL = False
    print("Warning: pycocoevalcap not installed. Install with: pip install pycocoevalcap")



def compute_cosine_similarity(model, processor, actual: List[str], predicted: List[str]) -> float:
    """Semantic cosine similarity using the model's own token embeddings + mean pooling."""
    embed = model.get_input_embeddings()
    sims = []
    with torch.no_grad():
        for ref, pred in zip(actual, predicted):
            ref_ids = processor.tokenizer(ref, return_tensors="pt").input_ids.to("cuda")
            pred_ids = processor.tokenizer(pred, return_tensors="pt").input_ids.to("cuda")
            ref_emb = embed(ref_ids).mean(dim=1)
            pred_emb = embed(pred_ids).mean(dim=1)
            sim = F.cosine_similarity(ref_emb.float(), pred_emb.float(), dim=-1)
            sims.append(sim.item())
    return float(sum(sims) / len(sims)) if sims else 0.0


def parse_captions_from_csv(caption_str: str) -> List[str]:
    """Parse the string representation of caption list from CSV."""
    import ast
    try:
        return ast.literal_eval(caption_str)
    except:
        return [c.strip() for c in caption_str.split("'") if c.strip()]


def apply_turboquant_static(K: torch.Tensor, bits: int = 3) -> Tuple[torch.Tensor, float]:
    """Apply TurboQuant with STATIC scale (1/sqrt(d))."""
    dt, dev = K.dtype, K.device
    b, h, s, d = K.shape

    # Stage 1: PolarQuant (Random Rotation)
    R = torch.randn(d, d, device=dev, dtype=torch.float32)
    Q, _ = torch.linalg.qr(R)
    Pi = Q.to(dt)
    t_rot = K.view(-1, d) @ Pi

    # Stage 2: Quantization
    norms = torch.norm(t_rot, p=2, dim=-1, keepdim=True).to(torch.float32)
    t_unit = t_rot / (norms.to(dt) + 1e-6)
    levels = torch.linspace(-1, 1, steps=2**bits, device=dev, dtype=dt)
    idx = torch.argmin(torch.abs(t_unit.unsqueeze(-1) - levels), dim=-1)
    t_quant = levels[idx]

    # Stage 3: QJL with STATIC scale
    res_sign = torch.sign(t_unit - t_quant)
    scale = 1.0 / (d**0.5)
    t_restored = (t_quant + (res_sign * scale)) * norms.to(dt)
    compressed = (t_restored @ Pi.T).view(b, h, s, d)

    cos_sim = F.cosine_similarity(K.float(), compressed.float(), dim=-1)
    fidelity = cos_sim.mean().item()
    return compressed, fidelity


def apply_turboquant_dynamic(K: torch.Tensor, bits: int = 3) -> Tuple[torch.Tensor, float]:
    """Apply TurboQuant with DYNAMIC residual scaling."""
    dt, dev = K.dtype, K.device
    b, h, s, d = K.shape

    # Stage 1: PolarQuant
    R = torch.randn(d, d, device=dev, dtype=torch.float32)
    Q, _ = torch.linalg.qr(R)
    Pi = Q.to(dt)
    t_rot = K.view(-1, d) @ Pi

    # Stage 2: Quantization
    norms = torch.norm(t_rot, p=2, dim=-1, keepdim=True).to(torch.float32)
    t_unit = t_rot / torch.clamp(norms.to(dt), min=1e-6)
    levels = torch.linspace(-1, 1, steps=2**bits, device=dev, dtype=dt)
    idx = torch.argmin(torch.abs(t_unit.unsqueeze(-1) - levels), dim=-1)
    t_quant = levels[idx]

    # Stage 3: QJL with DYNAMIC scale
    residual = t_unit - t_quant
    dynamic_scale = residual.abs().mean(dim=-1, keepdim=True)
    res_sign = torch.sign(residual)
    t_restored = (t_quant + (res_sign * dynamic_scale)) * norms.to(dt)
    compressed = (t_restored @ Pi.T).view(b, h, s, d)

    cos_sim = F.cosine_similarity(K.float(), compressed.float(), dim=-1)
    fidelity = cos_sim.mean().item()
    return compressed, fidelity


def generate_caption(model, processor, image_path: Path, variant: str = "fp16", bits: int = 2) -> str:
    """Generate caption with specified variant (fp16, static, dynamic)."""
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this satellite image in detail."}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")

    with torch.no_grad():
        if variant == "fp16":
            outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        else:
            # Get cache from prefill
            out = model(**inputs, use_cache=True)

            # Compress cache (DynamicCache with .layers in transformers 5.x)
            past_kv = out.past_key_values
            import copy
            compressed_cache = copy.deepcopy(past_kv)
            for i, layer in enumerate(compressed_cache.layers):
                if variant == "static":
                    layer.keys, _ = apply_turboquant_static(layer.keys, bits=bits)
                else:  # dynamic
                    layer.keys, _ = apply_turboquant_dynamic(layer.keys, bits=bits)

            outputs = model.generate(
                **inputs,
                past_key_values=compressed_cache,
                max_new_tokens=30,
                do_sample=False
            )

    # Decode caption (skip prompt tokens)
    caption = processor.tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    caption = caption.strip()
    if caption.lower().startswith("caption:"):
        caption = caption[len("caption:"):].strip()
    return caption


def evaluate_satellite_turboquant(
    model_id: str = "HuggingFaceTB/SmolVLM-Base",
    adapter_path: str = "smolvlm_satellite_captioning",
    num_eval_samples: int = 10,
    output_json: str = "turboquant_eval_results.json",
    bits: int = 2
):
    """Evaluate all three variants: FP16, Static TurboQuant, Dynamic TurboQuant."""

    repo_root = Path(__file__).resolve().parent
    adapter_full_path = repo_root / adapter_path

    # Load base model and apply LoRA adapter
    print(f"Loading {model_id} + LoRA adapter from {adapter_path}...")
    processor = AutoProcessor.from_pretrained(str(adapter_full_path))
    base_model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_full_path))
    model.eval()

    # Load test CSV
    csv_path = repo_root / "satellite" / "test.csv"

    results = {
        "fp16": {"captions": [], "fidelities": [], "metrics": {}},
        "static": {"captions": [], "fidelities": [], "metrics": {}},
        "dynamic": {"captions": [], "fidelities": [], "metrics": {}},
    }
    reference_captions_dict = {}

    print(f"Running inference on satellite test set...")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)[:num_eval_samples]

        for idx, row in enumerate(rows):
            rel_image_path = row["filepath"]
            ref_captions = parse_captions_from_csv(row["captions"])

            image_path = repo_root / "satellite" / rel_image_path
            if not image_path.exists():
                print(f"  Skipping {idx}: image not found at {image_path}")
                continue

            image_id = f"img_{idx}"
            reference_captions_dict[image_id] = ref_captions

            print(f"  [{idx+1}/{len(rows)}] {image_path.name}")

            # Generate with each variant
            cap_fp16 = generate_caption(model, processor, image_path, variant="fp16", bits=bits)
            results["fp16"]["captions"].append(cap_fp16)

            cap_static = generate_caption(model, processor, image_path, variant="static", bits=bits)
            results["static"]["captions"].append(cap_static)

            cap_dynamic = generate_caption(model, processor, image_path, variant="dynamic", bits=bits)
            results["dynamic"]["captions"].append(cap_dynamic)

            # Fidelity check on first layer
            with torch.no_grad():
                fid_messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this satellite image in detail."}]}]
                fid_prompt = processor.apply_chat_template(fid_messages, add_generation_prompt=True)
                test_inputs = processor(
                    text=fid_prompt,
                    images=[Image.open(image_path).convert("RGB")],
                    return_tensors="pt"
                ).to("cuda")
                out_fp16 = model(**test_inputs, use_cache=True)
                k_fp16 = out_fp16.past_key_values.layers[0].keys

                _, fid_static = apply_turboquant_static(k_fp16)
                _, fid_dynamic = apply_turboquant_dynamic(k_fp16)

                results["static"]["fidelities"].append(fid_static)
                results["dynamic"]["fidelities"].append(fid_dynamic)

    # Compute metrics
    if HAVE_COCO_EVAL:
        print("\nComputing BLEU/METEOR/CIDEr metrics...")

        for variant in ["fp16", "static", "dynamic"]:
            pred_captions = {f"img_{i}": [cap] for i, cap in enumerate(results[variant]["captions"])}

            try:
                bleu_scorer = Bleu(n=4)
                bleu_score, _ = bleu_scorer.compute_score(reference_captions_dict, pred_captions)
                results[variant]["metrics"]["BLEU"] = [float(b) for b in bleu_score]
            except Exception as e:
                print(f"  BLEU skipped: {e}")

            try:
                meteor_scorer = Meteor()
                meteor_score, _ = meteor_scorer.compute_score(reference_captions_dict, pred_captions)
                results[variant]["metrics"]["METEOR"] = float(meteor_score)
            except Exception as e:
                print(f"  METEOR skipped: {e}")

            try:
                cider_scorer = Cider()
                cider_score, _ = cider_scorer.compute_score(reference_captions_dict, pred_captions)
                results[variant]["metrics"]["CIDEr"] = float(cider_score)
            except Exception as e:
                print(f"  CIDEr skipped: {e}")

    # Cosine similarity using model's own token embeddings (same approach as TurboQuant_VLM_vllm.py)
    print("\nComputing cosine similarity metrics...")
    first_refs = [reference_captions_dict[f"img_{i}"][0] for i in range(len(rows)) if f"img_{i}" in reference_captions_dict]
    for variant in ["fp16", "static", "dynamic"]:
        preds = results[variant]["captions"]
        if preds and any(p for p in preds):
            try:
                results[variant]["metrics"]["CosineSim"] = compute_cosine_similarity(
                    model, processor, first_refs[:len(preds)], preds
                )
            except Exception as e:
                print(f"  CosineSim skipped for {variant}: {e}")

    # Print summary
    print("\n" + "="*60)
    print("SATELLITE TURBOQUANT EVALUATION SUMMARY")
    print("="*60)

    for variant in ["fp16", "static", "dynamic"]:
        print(f"\n{variant.upper()}:")
        if results[variant]["metrics"]:
            for metric_name, score in results[variant]["metrics"].items():
                if isinstance(score, list):
                    scores_str = ", ".join(f"{s:.4f}" for s in score)
                    print(f"  {metric_name} (1-4): {scores_str}")
                else:
                    print(f"  {metric_name}: {score:.4f}")
        if results[variant]["fidelities"]:
            fid_mean = sum(results[variant]["fidelities"]) / len(results[variant]["fidelities"])
            print(f"  Avg Fidelity: {fid_mean:.4f}")
        if results[variant]["captions"]:
            print(f"  Sample: {results[variant]['captions'][0]}")

    # Save JSON
    json_results = {
        variant: {
            "metrics": results[variant]["metrics"],
            "avg_fidelity": sum(results[variant]["fidelities"]) / len(results[variant]["fidelities"]) if results[variant]["fidelities"] else None,
            "num_samples": len(results[variant]["captions"]),
        }
        for variant in results.keys()
    }

    with open(repo_root / output_json, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {output_json}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--output-json", type=str, default="turboquant_eval_results.json")
    parser.add_argument("--adapter-path", type=str, default="smolvlm_satellite_captioning")
    args = parser.parse_args()

    evaluate_satellite_turboquant(
        adapter_path=args.adapter_path,
        num_eval_samples=args.num_samples,
        output_json=args.output_json, 
        bits=args.bits
    )
