from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from peft import PeftModel
import torch

# base_model_id = "HuggingFaceTB/SmolVLM-Base"
# ft_dir = "smolvlm_turtlebot_action_balanced"
# output_dir = "smolvlm_merged_for_gguf"
#
# # load base + adapter
# model = Idefics3ForConditionalGeneration.from_pretrained(base_model_id, torch_dtype=torch.float16)
# model = PeftModel.from_pretrained(model, ft_dir)
#
# # merge and save
# model = model.merge_and_unload()
# model.save_pretrained(output_dir, safe_serialization=True)
# print(f"✅ merged weights saved to {output_dir}")

from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM-Base").save_pretrained("../smolvlm_merged_for_gguf")
print("✅ Tokenizer files saved inside ../smolvlm_merged_for_gguf")

