#!/usr/bin/env python3
"""Convert the OpenVLA-OFT checkpoint's Llama-2-7B language_model backbone to GGUF.

The checkpoint is a prismatic `OpenVLAForActionPrediction`: a Llama-2-7B `language_model`
plus a `vision_backbone` + `projector` (+ a separate action head). llama.cpp's GGUF only
represents the LLM, so we keep the `language_model.*` tensors and drop the rest.

Reuses llama.cpp's own `convert_hf_to_gguf.py` (imported, not modified). `LlamaModel`
already strips the `language_model.` prefix and undoes the Q/K permute; we only add a
subclass that skips openvla's `vision_backbone.`/`projector.` tensors. Point this at the
symlink staging dir (full Llama hparams; the real config's text_config is minimal).

Run from the llama.cpp dir on PYTHONPATH; args pass through to convert_hf_to_gguf.main().
"""
import sys
from pathlib import Path

# llama.cpp dir (has convert_hf_to_gguf.py) + its gguf-py on the path
LLAMA = Path(__file__).resolve().parent.parent / "llama.cpp"
sys.path.insert(0, str(LLAMA))
sys.path.insert(0, str(LLAMA / "gguf-py"))

import gguf
import convert_hf_to_gguf as C


@C.ModelBase.register("OpenVLAForActionPrediction")
class OpenVLAModel(C.LlamaModel):
    """Llama-2 backbone of an OpenVLA-OFT checkpoint; non-LLM tensors are dropped."""
    model_arch = gguf.MODEL_ARCH.LLAMA

    def modify_tensors(self, data_torch, name, bid):
        # Keep only the Llama backbone; drop vision_backbone.*, projector.*,
        # action_head.* and any other non-LLM tensors (not representable in a Llama GGUF).
        if not name.startswith("language_model."):
            return []
        return super().modify_tensors(data_torch, name, bid)


if __name__ == "__main__":
    C.main()
