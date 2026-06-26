#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import socket, time, os
from PIL import Image as PILImage
import load_model_and_get_action   # <-- import the CLI module

# ================================================================
# CONFIGURATION
# ================================================================
TCP_HOST = "127.0.0.1"
TCP_PORT = 5000
INFER_INTERVAL = 1.5  # seconds
WATCH_DIR = "captured_frames"

# ================================================================
# ROS2 Node Definition
# ================================================================
class SmolVLMController(Node):
    def __init__(self):
        super().__init__('smolvlm_controller')
        self.bridge = CvBridge()
        self.latest_frame = None
        self.last_infer_time = 0.0
        self.sock = None
        self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self.image_callback, 10)
        self.connect_to_robot()
        self.get_logger().info("🧠 SmolVLM Controller node initialized.")

    def connect_to_robot(self):
        """Connect to the local TCP command server."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((TCP_HOST, TCP_PORT))
            self.get_logger().info(f"Connected to robot command server at {TCP_HOST}:{TCP_PORT}")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to robot command server: {e}")

    def image_callback(self, msg):
        """Periodically trigger inference on new frames."""
        try:
            now = self.get_clock().now().seconds_nanoseconds()[0]
            if now - self.last_infer_time < INFER_INTERVAL:
                return
            self.last_infer_time = now

            # Save current frame to disk for llama-mtmd-cli
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img_path = os.path.join(WATCH_DIR, f"frame_{int(time.time())}.jpg")
            PILImage.fromarray(self.latest_frame[:, :, ::-1]).save(img_path)

            # Run CLI prediction via first_script
            result = load_model_and_get_action.predict_from_image(img_path)
            self.handle_prediction(result)

        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")

    def handle_prediction(self, result):
        """Send parsed prediction over TCP."""
        if result is None:
            return

        if isinstance(result, tuple):
            command, distance, duration = result
            msg = f"{command}_{distance}_{duration}"
        else:
            msg = str(result)

        self.get_logger().info(f"Predicted action: {msg}")

        try:
            if self.sock:
                self.sock.sendall(msg.encode('utf-8') + b'\n')
                self.get_logger().info(f"✅ Sent to robot: {msg}")
            else:
                self.get_logger().warn("Socket not connected.")
        except Exception as e:
            self.get_logger().error(f"Socket send failed: {e}")
            self.connect_to_robot()


# ================================================================
# ENTRY POINT
# ================================================================
def main(args=None):
    rclpy.init(args=args)
    node = SmolVLMController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted, shutting down...")
    finally:
        if node.sock:
            node.sock.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
