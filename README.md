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
