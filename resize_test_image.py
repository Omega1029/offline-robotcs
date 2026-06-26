from PIL import Image

# 1. Load your original robot camera frame
input_file = "frame_000000_20250604_142222_485_resized.jpg"
output_file = "test_robot_384.jpg"

print(f"Opening {input_file}...")
img = Image.open(input_file)

# 2. Resize to the native SmolVLM patch size (384x384)
# Using LANCZOS filter for best quality at small sizes
resized_img = img.resize((384, 384), Image.Resampling.LANCZOS)

# 3. Save as a new file
resized_img.save(output_file)
print(f"Saved {output_file}. Ready for testing!")
