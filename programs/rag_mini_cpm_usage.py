import os
import traceback
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
from transformers.image_utils import load_image
from huggingface_hub import snapshot_download


class PhiVisionRAGPipeline:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.all_images = {}
        self._setup_model()

    def _setup_model(self):
        print("🔄 Loading Phi-3 Vision model...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(self.device)
            self.model.eval()
            print("✅ Phi-3 Vision model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading Phi-3 Vision: {e}")
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

        MAX_IMGS = 4  # adjust for VRAM
        all_imgs = list(self.all_images.values())
        used_images = all_imgs[:MAX_IMGS]

        messages = [{
            "role": "user",
            "content": ([{"type": "image"}] * len(used_images)) + [{"type": "text", "text": query}],
        }]

        try:
            RESERVE = 512
            prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                truncation=True,
                max_length=max(getattr(self.model.config, "max_position_embeddings", 16384) - RESERVE, 32)
            )
            inputs = self.processor(
                text=prompt, images=used_images,
                return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=RESERVE)
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()
        except Exception as e:
            traceback.print_exc()
            return f"Error: {e}"


def main():
    print("🚀 Starting Phi-3 Vision Multi-Image Pipeline...")
    try:
        HF_REPO = "microsoft/Phi-3-vision-128k-instruct"
        LOCAL_DIR = "./Phi-3-vision-128k-instruct"

        if not os.path.exists(os.path.join(LOCAL_DIR, "config.json")):
            print("📦 Downloading Phi-3 Vision model from Hugging Face...")
            snapshot_download(
                repo_id=HF_REPO,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False
            )
            print("✅ Model downloaded to:", LOCAL_DIR)

        pipeline = PhiVisionRAGPipeline(model_path=LOCAL_DIR)
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
