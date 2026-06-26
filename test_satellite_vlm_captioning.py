"""
Smoke-test / eval for fine_tune_satellite_vlm_captioning.py.

Loads SmolVLM base + LoRA safetensors (HF/PEFT only — no GGUF/mmproj).

Either pass image file paths as arguments, or omit them to sample rows from
satellite/{train,valid,test}.csv under DATA_ROOT (same prompts as training).
"""
import argparse
import ast
import csv
import io
import os
import random
import tokenize

import numpy as np
import torch
from PIL import Image

from peft import PeftModel
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

# Keep in sync with fine_tune_satellite_vlm_captioning.py
MODEL_ID = "HuggingFaceTB/SmolVLM-Base"
OUTPUT_DIR = "./smolvlm_satellite_captioning"
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "satellite"))
FIXED_INSTRUCTION = "Describe this satellite image in detail."
SEED = 42


def parse_caption_strings(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw.startswith("["):
        return []
    out: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(raw).readline):
            if tok.type == tokenize.STRING:
                out.append(ast.literal_eval(tok.string))
    except (tokenize.TokenError, SyntaxError, ValueError):
        return []
    return out


def load_split(csv_path: str, data_root: str):
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp = row["filepath"].strip()
            captions = parse_caption_strings(row["captions"])
            if not captions:
                continue
            img_path = os.path.join(data_root, fp)
            if not os.path.isfile(img_path):
                continue
            records.append(
                {
                    "image": os.path.abspath(img_path),
                    "captions": captions,
                    "instruction": FIXED_INSTRUCTION,
                }
            )
    return records


def build_messages(instruction: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Instruction: {instruction}"},
            ],
        },
    ]


# --- Caption similarity metrics (corpus / aggregate over a batch) -------------


def compute_bleu_score(
    references_per_sample: list[list[str]],
    predicted: list[str],
) -> float:
    """Corpus BLEU; each sample may have multiple reference strings (NLTK tokenization)."""
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    list_of_references = [[r.split() for r in refs] for refs in references_per_sample]
    hypotheses = [h.split() for h in predicted]
    smooth = SmoothingFunction().method4
    return float(corpus_bleu(list_of_references, hypotheses, smoothing_function=smooth))


def compute_rouge_score(actual: list[str], predicted: list[str]) -> float:
    """ROUGE-2 style bigram F1, averaged over aligned pairs (single ref string per row)."""

    def _get_ngrams(n, text):
        text_length = len(text)
        if text_length < n:
            return set()
        ngram_set = set()
        for i in range(text_length - n + 1):
            ngram_set.add(tuple(text[i : i + n]))
        return ngram_set

    def _rouge_n(evaluated_sentences, reference_sentences, n=2):
        evaluated_ngrams = _get_ngrams(n, evaluated_sentences)
        reference_ngrams = _get_ngrams(n, reference_sentences)
        reference_count = len(reference_ngrams)
        evaluated_count = len(evaluated_ngrams)

        overlapping_ngrams = evaluated_ngrams & reference_ngrams
        overlapping_count = len(overlapping_ngrams)

        precision = overlapping_count / evaluated_count if evaluated_count > 0 else 0.0
        recall = overlapping_count / reference_count if reference_count > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        return f1

    scores = [
        _rouge_n(pred.split(), act.split())
        for pred, act in zip(predicted, actual)
    ]
    return float(np.mean(scores))


def compute_cider_score(
    references_per_sample: list[list[str]],
    predicted: list[str],
) -> float:
    from pycocoevalcap.cider.cider import Cider

    n = len(predicted)
    gts = {i: list(references_per_sample[i]) for i in range(n)}
    res = {i: [predicted[i]] for i in range(n)}
    cider = Cider()
    score, _ = cider.compute_score(gts, res)
    return float(score)


def compute_cosine_similarity(
    actual: list[str],
    predicted: list[str],
    embedding_model_id: str = "all-MiniLM-L6-v2",
) -> float:
    """Semantic cosine similarity (Sentence-Transformers). SmolVLM has no get_text_features."""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer(embedding_model_id)
    with torch.no_grad():
        act_e = model.encode(actual, convert_to_numpy=True, show_progress_bar=False)
        pred_e = model.encode(
            predicted, convert_to_numpy=True, show_progress_bar=False
        )
    pairs = [
        float(cosine_similarity(a.reshape(1, -1), p.reshape(1, -1))[0, 0])
        for a, p in zip(act_e, pred_e)
    ]
    return float(np.mean(pairs))


def print_validation_metrics(
    samples: list[dict],
    predictions: list[str],
    embedding_model_id: str,
) -> None:
    """First reference caption per row for ROUGE/cosine; all references for BLEU/CIDEr."""
    references_per_sample = [ex["captions"] for ex in samples]
    first_refs = [refs[0] for refs in references_per_sample]

    print("--- Validation metrics (this batch) ---")

    try:
        bleu_score = compute_bleu_score(references_per_sample, predictions)
        print(f"Validation BLEU Score: {bleu_score:.4f}")
    except ImportError as e:
        print(f"Validation BLEU Score: (skipped — pip install nltk) {e}")
    except Exception as e:
        print(f"Validation BLEU Score: (error) {e}")

    try:
        rouge_score = compute_rouge_score(first_refs, predictions)
        print(f"Validation ROUGE Score: {rouge_score:.4f}")
    except Exception as e:
        print(f"Validation ROUGE Score: (error) {e}")

    try:
        cider_score = compute_cider_score(references_per_sample, predictions)
        print(f"Validation CIDEr Score: {cider_score:.4f}")
    except ImportError as e:
        print(f"Validation CIDEr Score: (skipped — pip install pycocoevalcap) {e}")
    except Exception as e:
        print(f"Validation CIDEr Score: (error) {e}")

    try:
        cosine_sim = compute_cosine_similarity(
            first_refs, predictions, embedding_model_id=embedding_model_id
        )
        print(f"Validation Cosine Similarity: {cosine_sim:.4f}")
    except ImportError as e:
        print(
            f"Validation Cosine Similarity: (skipped — pip install sentence-transformers scikit-learn) {e}"
        )
    except Exception as e:
        print(f"Validation Cosine Similarity: (error) {e}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run satellite captioning (SmolVLM + LoRA): optional image paths, "
            "or CSV sampling when no paths are given."
        )
    )
    parser.add_argument(
        "images",
        nargs="*",
        metavar="IMAGE",
        help=(
            "One or more image files to caption. If omitted, samples are taken "
            "from satellite CSVs (--split)."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default="valid",
        help="Which CSV to draw samples from (default: valid).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of images to run (default: 3).",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Fine-tuned adapter dir (default: {OUTPUT_DIR}). Must contain adapter_config.json.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Use base MODEL_ID only (ignore adapter).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Generation length (default: 256).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="RNG seed for shuffling which rows to test (default: 42).",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip BLEU, ROUGE, CIDEr, and embedding cosine similarity after inference.",
    )
    parser.add_argument(
        "--sentence-embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-Transformers checkpoint for cosine similarity (default: all-MiniLM-L6-v2).",
    )
    args = parser.parse_args()

    if args.images:
        samples = []
        for p in args.images:
            path = os.path.abspath(os.path.expanduser(p))
            if not os.path.isfile(path):
                raise SystemExit(f"Image not found: {path}")
            samples.append(
                {
                    "image": path,
                    "captions": [],
                    "instruction": FIXED_INSTRUCTION,
                }
            )
        csv_path = None
        records_count = None
    else:
        csv_name = f"{args.split}.csv"
        csv_path = os.path.join(DATA_ROOT, csv_name)
        if not os.path.isfile(csv_path):
            raise SystemExit(
                f"Missing {csv_path}. Expected satellite/{csv_name} next to this script."
            )

        random.seed(args.seed)
        records = load_split(csv_path, DATA_ROOT)
        if not records:
            raise SystemExit(
                f"No rows with existing image files under {DATA_ROOT}. "
                f"Unpack or sync images so paths in {csv_name} resolve (e.g. train/foo.jpg)."
            )

        random.shuffle(records)
        samples = records[: args.num_samples]
        records_count = len(records)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    dtype = torch.float32# if device != "cpu" else torch.float32

    adapter_path = os.path.abspath(args.adapter_dir)
    use_adapter = (
        not args.base_only
        and os.path.isdir(adapter_path)
        and os.path.isfile(os.path.join(adapter_path, "adapter_config.json"))
    )

    load_dir = adapter_path if use_adapter else MODEL_ID
    print(f"DATA_ROOT: {DATA_ROOT}")
    if csv_path is not None:
        print(f"CSV: {csv_path} ({records_count} rows with files on disk)")
    else:
        print(f"Images: {len(samples)} path(s) from CLI")
    print(f"Loading processor from: {load_dir}")
    processor = AutoProcessor.from_pretrained(load_dir)

    print(f"Loading base model: {MODEL_ID} ({device}, {dtype})")
    model = Idefics3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    if use_adapter:
        #print(f"Attaching LoRA from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        if not args.base_only:
            print(
                f"No adapter at {adapter_path} (adapter_config.json missing); using base weights only."
            )

    model.eval()

    mode = f"CLI paths" if args.images else f"'{args.split}' CSV"
    print(f"\nRunning inference on {len(samples)} sample(s) ({mode})...\n")

    predictions: list[str] = []
    for i, ex in enumerate(samples, start=1):
        img = Image.open(ex["image"]).convert("RGB")
        messages = build_messages(ex["instruction"])
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(images=[img], text=[prompt], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        new_tokens = out_ids[:, inputs["input_ids"].shape[1] :]
        pred = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        predictions.append(pred)

        print(f"--- Sample {i}/{len(samples)} ---")
        print(f"Image: {ex['image']}")
        if ex["captions"]:
            ref = ex["captions"][0][:200] + (
                "..." if len(ex["captions"][0]) > 200 else ""
            )
            print(f"Reference (first caption): {ref}")
        print(f"Predicted: {pred}\n")

    has_refs = all(len(ex["captions"]) > 0 for ex in samples)
    if not args.no_metrics and has_refs:
        print()
        print_validation_metrics(
            samples, predictions, args.sentence_embedding_model
        )
    elif not args.no_metrics and not has_refs:
        print(
            "\nSkipping metrics: no reference captions in this run "
            "(use CSV sampling instead of raw image paths only)."
        )


if __name__ == "__main__":
    main()
