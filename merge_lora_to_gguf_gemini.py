#from transformers import AutoProcessor, Idefics3ForConditionalGeneration
#from peft import PeftModel
#import torch

#base_model_id = "HuggingFaceTB/SmolVLM-Base"
#ft_dir = "smolvlm_turtlebot_action_ft"
#output_dir = "smolvlm_merged_for_gguf"
#
# # load base + adapter
#model = Idefics3ForConditionalGeneration.from_pretrained(base_model_id, torch_dtype=torch.float16)
#model = PeftModel.from_pretrained(model, ft_dir)
#
# # merge and save
#model = model.merge_and_unload()
#model.save_pretrained(output_dir, safe_serialization=True)
#print(f"✅ merged weights saved to {output_dir}")

#from transformers import AutoTokenizer
#AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM-Base").save_pretrained("./smolvlm_merged_for_gguf")
#print("✅ Tokenizer files saved inside ./smolvlm_merged_for_gguf")

import torch
from transformers import AutoProcessor
from transformers import Idefics3ForConditionalGeneration
from peft import PeftModel

base_model_id = "HuggingFaceTB/SmolVLM-Instruct" # Or your specific base version
#adapter_model_path = "./smolvlm_turtlebot_vla"
adapter_model_path = "./smolvlm_satellite_captioning"
output_path = "./smolvlm-merged-satellite-captioning-model"

print("Loading base model...")
# Load the base model in FP16 to save memory during the merge
model = Idefics3ForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="cpu", # Merge on CPU if you lack VRAM; use "cuda" if available
    trust_remote_code=True
)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, adapter_model_path)

print("Merging weights... (This may take a moment)")
# This merges the LoRA weights into the main layers permanently
model = model.merge_and_unload()

print(f"Saving merged model to {output_path}...")
model.save_pretrained(output_path)

# Also save the processor (tokenizer + image processor) to the same folder
processor = AutoProcessor.from_pretrained(base_model_id)
processor.save_pretrained(output_path)

print("Done! You can now run the GGUF conversion on the 'smolvlm-merged-model' folder.")
