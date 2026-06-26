cd llama.cpp/

python convert_hf_to_gguf.py \
  ../openvla_oft_libero_bnb_int8 \
  --outfile ../openvla-oft-libero-bnb-int8-f16.gguf \
  --outtype f8

python convert_hf_to_gguf.py \
  ../openvla_oft_libero_bnb_int8 \
  --outfile ../mmproj-openvla-oft-libero-bnb-int8-f16.gguf \
  --outtype f16 \
  --mmproj

cd ..

llama-quantize smolvlm-satellite-captioning-f16.gguf smolvlm-satellite-captioning-Q4_K_M.gguf Q4_K_M