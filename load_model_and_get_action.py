import subprocess, threading, queue, os, time
from PIL import Image

LLAMA_CMD = [
    "llama-mtmd-cli",
    "-m", "smol_vlm_balanced_iq4_xs.gguf",
    "--mmproj", "mmproj-smol_vlm_balanced.gguf",
    "--chat-template", "deepseek"
]

RESIZE_SIZE = 256


# ================================================================
# IMAGE RESIZE
# ================================================================
def resize_image(input_path, output_path=None, size=RESIZE_SIZE):
    """Resize image for faster inference."""
    try:
        img = Image.open(input_path).convert("RGB")
        img = img.resize((size, size))
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_resized{ext}"
        img.save(output_path, quality=90)
        return output_path
    except Exception as e:
        print(f"⚠️ Failed to resize {input_path}: {e}")
        return input_path


# ================================================================
# BACKGROUND MODEL WRAPPER
# ================================================================
class PersistentLlama:
    def __init__(self):
        self.proc = subprocess.Popen(
            LLAMA_CMD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        self.out_queue = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        self._wait_for_ready()

    def _reader(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                self.out_queue.put(line)

    def _wait_for_ready(self, timeout=120):
        """Wait until the model prints its 'Running in chat mode' banner."""
        print("⏳ Waiting for model to enter chat mode...")
        start = time.time()
        ready = False
        while time.time() - start < timeout:
            try:
                line = self.out_queue.get(timeout=5)
                print(line)
                if "exit the program" in line:
                    ready = True
                    print("✅ Model ready for inference.")
                    break
            except queue.Empty:
                pass

        if not ready:
            print("⚠️ Timeout waiting for chat mode; continuing anyway.")
        else:
            pass
            #input("💡 Press ENTER to begin inference...")

            # 🧹 Flush startup leftovers before real inference
            flushed = 0
            while not self.out_queue.empty():
                _ = self.out_queue.get_nowait()
                flushed += 1
            print(f"🧹 Cleared {flushed} startup lines before inference.")

    def infer_image(self, image_path, prompt):
        """Send an image and prompt, wait for the model’s response."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        resized_path = resize_image(image_path)

        # Load image
        self.proc.stdin.write(f"/image {resized_path}\n")
        self.proc.stdin.flush()
        time.sleep(1.0)

        # Send prompt
        self.proc.stdin.write(prompt.strip() + "\n")
        self.proc.stdin.flush()

        # Wait for model output
        prediction = None
        start_time = time.time()
        while time.time() - start_time < 300:
            try:
                line = self.out_queue.get(timeout=5).strip()

                # Skip all noise lines
                if not line or line.startswith((">", "main:", "llama_", "clip_", "load_", "alloc_")):
                    continue
                if any(bad in line for bad in (
                    "encoding image", "decoding", "image decoded", "image loaded",
                    "Running in chat mode", "available commands", "exit the program"
                )):
                    continue

                # ✅ Real prediction pattern: forward_0.2_3.0s
                if "_" in line and not line.startswith(">"):
                    prediction = line
                    break

            except queue.Empty:
                pass

        return prediction or "⚠️ No prediction received."


# ================================================================
# SINGLETON MANAGEMENT
# ================================================================
_persistent_llama = None

def get_persistent_model():
    global _persistent_llama
    if _persistent_llama is None:
        print("🚀 Launching persistent llama-mtmd-cli process...")
        _persistent_llama = PersistentLlama()
        print("✅ Model boot completed.")
    return _persistent_llama


def predict_from_image(image_path):
    model = get_persistent_model()
    return model.infer_image(
        image_path,
        "Predict the correct robot_action value, e.g., forward_0.2_3.0s."
    )


# ================================================================
# TEST ENTRY POINT
# ================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "captured_frames/captured_frames/frame_000000_20250604_142222_485.jpg"

    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        exit(1)

    print(f"🧠 Running inference on: {img_path}")
    start = time.time()
    result = predict_from_image(img_path)
    print(f"✅ Prediction: {result}")
    end = time.time()
    print(f"It Took {round(end-start, 2)}s")
