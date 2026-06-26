#!/usr/bin/env python3
import zmq
import base64
import io
import cv2
import numpy as np
from PIL import Image
from load_model_and_get_action import get_persistent_model  # <-- import your class file
import time

# ================================================================
# ZeroMQ Setup
# ================================================================
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://0.0.0.0:5555")
print("✅ GGUF Inference Server (ZMQ) started on port 5555")

model = get_persistent_model()

while True:
    try:
        msg = socket.recv_json()
        b64_img = msg["image"]
        img_bytes = base64.b64decode(b64_img)
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Save temp image for GGUF model
        tmp_path = "tmp_frame.jpg"
        cv2.imwrite(tmp_path, frame)

        start = time.time()
        action = model.infer_image(tmp_path, "Predict the correct robot_action value, e.g., forward_0.2_3.0s.")
        latency = round(time.time() - start, 2)

        print(f"[SERVER] Predicted: {action} ({latency}s)")
        socket.send_json({"action": action, "latency": latency})

    except Exception as e:
        print(f"[SERVER] Error: {e}")
        socket.send_json({"error": str(e)})
