
# 🤖 SmolVLA: Edge-Deployed Vision-Language-Action System

SmolVLA (Small Vision-Language-Action) is a lightweight **Edge AI framework** that enables mobile robots to **see, reason, and act** directly from visual input — entirely **on-device**.  
It fine-tunes a small Vision-Language Model (VLM) on paired RGB images and ROS action data, allowing the robot to map **visual scenes → structured motion commands** (e.g., `forward_0.2_3.0s`).

This project demonstrates one of the first **embedded VLA systems** capable of performing **real-time multimodal reasoning** and **autonomous mobility** on resource-constrained hardware such as the **Raspberry Pi 5** or **TurtleBot 4**.

---

## 🧠 Key Features
- **On-Device Reasoning:** Runs entirely on embedded hardware — no GPU or cloud connection required.  
- **Vision-to-Action Mapping:** Fine-tunes a small VLM to translate camera frames directly into robot commands.  
- **ROS 2 Integration:** Fully compatible with ROS 2 topics for motion control and telemetry feedback.  
- **LoRA Fine-Tuning:** Uses efficient LoRA adapters for rapid task adaptation with minimal trainable parameters.  
- **Edge AI Optimization:** Supports quantized GGUF or 8-bit/4-bit models for real-time inference on CPUs.  

---

## 🏗️ System Overview

```text
+---------------------------+
|     Camera (RGB Input)    |
+------------+--------------+
             |
             v
+---------------------------+
|   SmolVLA (VLM Inference) |
|  Image → Text → Action    |
+------------+--------------+
             |
             v
+---------------------------+
|   ROS 2 Node Integration  |
|   Publishes /cmd_vel etc. |
+---------------------------+
             |
             v
+---------------------------+
|    TurtleBot / CoDrone    |
|   Executes Motion Command |
+---------------------------+
````

---

## 🧩 Architecture

| Component               | Description                                    |
| ----------------------- | ---------------------------------------------- |
| **Model Backbone**      | SmolVLM-Base (small multimodal transformer)    |
| **Fine-Tuning Method**  | LoRA (Low-Rank Adaptation)                     |
| **Dataset**             | Paired RGB frames and ROS action JSON metadata |
| **Frameworks**          | PyTorch, Hugging Face Transformers, PEFT       |
| **Hardware**            | Raspberry Pi 5 / TurtleBot 4                   |
| **Runtime Environment** | Python 3.10+, ROS 2 Humble or newer            |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/SmolVLA.git
cd SmolVLA
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

Ensure you have paired RGB images and their corresponding JSON metadata:

```
captured_frames/
├── frame_000001_20250822_171811_098.jpg
├── frame_000001_20250822_171811_098.json
├── frame_000002_20250822_171813_730.jpg
├── frame_000002_20250822_171813_730.json
└── ...
```

Each JSON file should include:

```json
{
    "frame_number": 8,
    "timestamp": "20250822_182052_781",
    "robot_status": "idle",
    "robot_action": "forward_0.2_3.0s",
    "filename": "captured_frames/frame_000008_20250822_182052_781.jpg"
}
```

---

### 4. Fine-Tune the Model

Run the training script to fine-tune SmolVLM on your dataset:

```bash
python fine_tune_robot_actions.py
```

The model will be saved under:

```
./smolvlm_turtlebot_action_ft/
```

---

### 5. Test Inference

```bash
python infer_smolvla.py
```

Example output:

```
Predicted: Assistant: forward_0.2_3.0s
From image: frame_000008_20250822_182052_781.jpg
```

---

### 6. OpenVLA 7B Proof of Concept

This repository also includes a Proof of Concept (POC) for fine-tuning the full **OpenVLA-7B** model using LoRA adapters on the Bridge dataset.

To run the fine-tuning POC:
```bash
python openvla_poc.py
```

To evaluate the fine-tuned model frame-by-frame on a full trajectory:
```bash
python test_finetune_openvla.py
```

---

## ⚙️ ROS 2 Integration

SmolVLA integrates directly into ROS 2 through publishers and subscribers:

| Topic               | Type                | Description                |
| ------------------- | ------------------- | -------------------------- |
| `/camera/image_raw` | sensor_msgs/Image   | Incoming RGB frames        |
| `/smolvla/cmd`      | std_msgs/String     | Predicted action command   |
| `/cmd_vel`          | geometry_msgs/Twist | Robot motion control topic |

You can adapt the inference node to automatically publish commands:

```bash
ros2 run smolvla smolvla_node.py
```

---

## 📊 Experimental Summary

| Metric             |                                              Value |
| ------------------ | -------------------------------------------------: |
| Dataset Size       |                          15,883 image–action pairs |
| Model Size         |                    ~256M parameters (SmolVLM Base) |
| Fine-Tune Duration |                                         ~10 epochs |
| Hardware           | NVIDIA GPU (training), Raspberry Pi 5 (deployment) |
| Inference Latency  |                            ~1.2s/frame on RPi5 CPU |
| Power Draw         |                                      ~6.5W average |

---

## 🧩 Applications

* Mobile robots in GPS-denied environments
* Defense and search-and-rescue robotics
* Educational AI robotics research
* Embedded autonomous systems

---

## 📘 Citation

If you use SmolVLA or reference this work, please cite:

```bibtex
@article{williams2025smolvla,
  title={SmolVLA: A Lightweight On-Device Vision-Language-Action Prototype for Autonomous Robots},
  author={Williams, Justin},
  year={2025},
  journal={IEEE Edge AI Systems (submitted)}
}
```

---

## 🧑‍💻 Author

**Justin Williams**
Ph.D. Candidate in Cyber-Physical Systems, Clark Atlanta University
[LinkedIn](https://www.linkedin.com/in/justin-williams-a35581138/) | [Website](https://yourwebsite.com)

---

## 🪶 Acknowledgments

Supported by the **Air Force Research Laboratory (AFRL)** and the **Griffiss Institute**.
Special thanks to **Dr. Gupta** for advisory support and research mentorship.

---

