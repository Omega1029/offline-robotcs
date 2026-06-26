cd llama.cpp/

python convert_hf_to_gguf.py \
  ../smolvlm-merged-satellite-captioning-model \
  --outfile ../smolvlm-satellite-captioning-f16.gguf \
  --outtype f16

python convert_hf_to_gguf.py \
  ../smolvlm-merged-satellite-captioning-model \
  --outfile ../mmproj-smolvlm-satellite-captioning-f16.gguf \
  --outtype f16 \
  --mmproj

cd ..

llama-quantize smolvlm-satellite-captioning-f16.gguf smolvlm-satellite-captioning-Q4_K_M.gguf Q4_K_M