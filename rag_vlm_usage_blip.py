import sys
import os
import traceback
from pathlib import Path
import torch
from typing import List
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers.image_utils import load_image
from huggingface_hub import snapshot_download


class BLIP2RAGPipeline:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.all_images = {}
        self._setup_model()

    def _setup_model(self):
        print("🔄 Loading BLIP-2 model...")
        try:
            # Load processor and model from the same path
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()
            print("✅ BLIP-2 model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading BLIP-2: {e}")
            traceback.print_exc()
            raise

    def load_images_from_directory(self, image_folder: str):
        print(f"🔄 Loading images from {image_folder}...")
        try:
            image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            all_images = {}
            for image_id, image_file in enumerate(image_files):
                image_path = os.path.join(image_folder, image_file)
                image = load_image(image_path)
                all_images[image_id] = image
            self.all_images = all_images
            print(f"✅ Loaded {len(all_images)} images")
        except Exception as e:
            print(f"❌ Error loading images: {e}")
            traceback.print_exc()

    def analyze_multiple_images(self, query: str) -> str:
        if not self.all_images:
            return "No images loaded."

        # BLIP-2 processes one image at a time
        all_imgs = list(self.all_images.values())
        responses = []

        try:
            for i, image in enumerate(all_imgs[:4]):  # Limit to 4 images
                # Process each image individually
                inputs = self.processor(images=image, text=query, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=100)

                response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                # Clean up the response (remove the input text)
                response = response.replace(query, "").strip()
                responses.append(f"Image {i + 1}: {response}")

            return "\n".join(responses)
        except Exception as e:
            traceback.print_exc()
            return f"Error: {e}"


def main():
    print("🚀 Starting BLIP-2 Multi-Image Pipeline...")
    try:
        # Use BLIP-2 which is well supported
        HF_REPO = "Salesforce/blip2-opt-2.7b"
        LOCAL_DIR = "./blip2-model"

        if not os.path.exists(os.path.join(LOCAL_DIR, "config.json")):
            print("📦 Downloading model from Hugging Face...")
            snapshot_download(
                repo_id=HF_REPO,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False
            )
            print("✅ Model downloaded to:", LOCAL_DIR)

        pipeline = BLIP2RAGPipeline(model_path=LOCAL_DIR)
        pipeline.load_images_from_directory("captured_frames/captured_frames")

        while True:
            try:
                query = input("\n❓ Enter your question about all images (or 'quit' to exit): ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue
                print(f"\n🔍 Processing query: {query}")
                response = pipeline.analyze_multiple_images(query)
                print(f"\n💡 Answer: {response}")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")


if __name__ == "__main__":
    main()
