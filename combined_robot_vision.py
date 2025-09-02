#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import socket
import os
import time
from datetime import datetime
import json

# Global variables for shared state
bridge = CvBridge()
latest_frame = None
frame_counter = 0
capture_enabled = False
capture_dir = "captured_frames"
robot_status = "idle"

# Ensure capture directory exists
os.makedirs(capture_dir, exist_ok=True)

class CombinedRobotVision(Node):
    def __init__(self):
        super().__init__('combined_robot_vision')
        
        # ROS2 publishers and subscribers
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            Image,
            '/oakd/rgb/preview/image_raw',
            self.listener_callback,
            10)
        
        # Initialize robot action parameters BEFORE starting threads
        self.action_sequence = [
            ("forward", 0.2, 3.0),    # (command, speed, duration)
            ("stop", 0.0, 1.0),
            ("left", 0.5, 2.0),
            ("stop", 0.0, 1.0),
            ("right", 0.5, 2.0),
            ("stop", 0.0, 1.0),
            ("backward", 0.2, 3.0),
            ("stop", 0.0, 1.0),
        ]
        self.current_action_index = 0
        self.action_start_time = None
        self.is_performing_sequence = False
        
        self.get_logger().info("CombinedRobotVision node started!")
        self.get_logger().info(f"Camera feed available at: http://localhost:8081")
        self.get_logger().info(f"Command server listening on port 5000")
        self.get_logger().info(f"Images being saved to: {capture_dir}")
        
        # Start background threads AFTER initializing attributes
        self.start_camera_server()
        self.start_command_server()
        self.start_robot_action_thread()

    def listener_callback(self, msg):
        """Handle incoming camera frames"""
        global latest_frame, frame_counter
        
        try:
            latest_frame = bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Save frame if capture is enabled
            if capture_enabled and latest_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"{capture_dir}/frame_{frame_counter:06d}_{timestamp}.jpg"
                
                # Save frame with metadata
                cv2.imwrite(filename, latest_frame)
                
                # Save metadata
                metadata = {
                    "frame_number": frame_counter,
                    "timestamp": timestamp,
                    "robot_status": robot_status,
                    "robot_action": self.get_current_action_name(),
                    "filename": filename
                }
                
                metadata_file = filename.replace('.jpg', '_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                frame_counter += 1
                
                if frame_counter % 30 == 0:  # Log every 30 frames
                    self.get_logger().info(f"Captured {frame_counter} frames")
                    
        except Exception as e:
            self.get_logger().error(f"Error processing camera frame: {e}")

    def start_camera_server(self):
        """Start the MJPEG camera server"""
        server_thread = threading.Thread(target=self.run_camera_server, daemon=True)
        server_thread.start()

    def run_camera_server(self):
        """Run the MJPEG camera server"""
        server = HTTPServer(('0.0.0.0', 8081), MJPEGHandler)
        self.get_logger().info("Camera server started on port 8081")
        server.serve_forever()

    def start_command_server(self):
        """Start the TCP command server"""
        server_thread = threading.Thread(target=self.run_command_server, daemon=True)
        server_thread.start()

    def run_command_server(self, host='0.0.0.0', port=5000):
        """Run the TCP command server"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(5)
        self.get_logger().info(f"Command server listening on {host}:{port}")

        while True:
            try:
                conn, addr = sock.accept()
                self.get_logger().info(f"Command connection from {addr}")
                threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()
            except Exception as e:
                self.get_logger().error(f"Command server error: {e}")

    def handle_client(self, conn):
        """Handle individual client connections"""
        with conn:
            while True:
                try:
                    data = conn.recv(1024)
                    if not data:
                        break
                    command = data.decode('utf-8').strip().lower()
                    self.get_logger().info(f"Received command: {command}")
                    self.handle_command(command)
                except Exception as e:
                    self.get_logger().error(f"Error handling client: {e}")
                    break

    def handle_command(self, command: str):
        """Handle movement commands"""
        global robot_status
        
        msg = Twist()
        
        if command == "start_capture":
            self.start_image_capture()
        elif command == "stop_capture":
            self.stop_image_capture()
        elif command == "start_sequence":
            self.start_action_sequence()
        elif command == "stop_sequence":
            self.stop_action_sequence()
        elif command.startswith("move_rotate_move"):
            try:
                _, d1, angle, d2 = command.split()
                self.move_rotate_move(float(d1), float(angle), float(d2))
            except Exception as e:
                self.get_logger().error(f"Invalid move_rotate_move command: {e}")
        elif command.startswith("forward"):
            speed = float(command.split()[1]) if len(command.split()) > 1 else 0.2
            msg.linear.x = speed
            robot_status = f"moving_forward_{speed}"
        elif command.startswith("backward"):
            speed = float(command.split()[1]) if len(command.split()) > 1 else 0.2
            msg.linear.x = -speed
            robot_status = f"moving_backward_{speed}"
        elif command.startswith("left"):
            speed = float(command.split()[1]) if len(command.split()) > 1 else 0.5
            msg.angular.z = speed
            robot_status = f"turning_left_{speed}"
        elif command.startswith("right"):
            speed = float(command.split()[1]) if len(command.split()) > 1 else 0.5
            msg.angular.z = -speed
            robot_status = f"turning_right_{speed}"
        elif command == "stop":
            robot_status = "stopped"
            pass  # zero velocity
        else:
            self.get_logger().warn(f"Unknown command: {command}")
            return
            
        self.publisher.publish(msg)

    def start_image_capture(self):
        """Enable image capture"""
        global capture_enabled
        capture_enabled = True
        self.get_logger().info("Image capture started!")

    def stop_image_capture(self):
        """Disable image capture"""
        global capture_enabled
        capture_enabled = False
        self.get_logger().info("Image capture stopped!")

    def start_action_sequence(self):
        """Start the automated action sequence"""
        self.is_performing_sequence = True
        self.current_action_index = 0
        self.action_start_time = time.time()
        self.get_logger().info("Starting automated action sequence!")

    def stop_action_sequence(self):
        """Stop the automated action sequence"""
        self.is_performing_sequence = False
        self.stop_robot()
        self.get_logger().info("Stopped automated action sequence!")

    def start_robot_action_thread(self):
        """Start thread for automated robot actions"""
        action_thread = threading.Thread(target=self.robot_action_loop, daemon=True)
        action_thread.start()

    def robot_action_loop(self):
        """Main loop for automated robot actions"""
        while True:
            if self.is_performing_sequence:
                self.execute_action_sequence()
            time.sleep(0.1)  # 10 Hz update rate

    def execute_action_sequence(self):
        """Execute the current action in the sequence"""
        if self.current_action_index >= len(self.action_sequence):
            # Sequence complete, restart
            self.current_action_index = 0
            self.action_start_time = time.time()
            self.get_logger().info("Action sequence completed, restarting...")
            return

        if self.action_start_time is None:
            self.action_start_time = time.time()

        command, speed, duration = self.action_sequence[self.current_action_index]
        elapsed_time = time.time() - self.action_start_time

        if elapsed_time >= duration:
            # Move to next action
            self.current_action_index += 1
            self.action_start_time = time.time()
            self.get_logger().info(f"Moving to next action: {self.get_current_action_name()}")
        else:
            # Execute current action
            self.execute_action(command, speed)

    def execute_action(self, command: str, speed: float):
        """Execute a single action"""
        msg = Twist()
        
        if command == "forward":
            msg.linear.x = speed
        elif command == "backward":
            msg.linear.x = -speed
        elif command == "left":
            msg.angular.z = speed
        elif command == "right":
            msg.angular.z = -speed
        elif command == "stop":
            pass  # zero velocity
            
        self.publisher.publish(msg)

    def get_current_action_name(self):
        """Get the name of the current action"""
        if self.current_action_index < len(self.action_sequence):
            command, speed, duration = self.action_sequence[self.current_action_index]
            return f"{command}_{speed}_{duration}s"
        return "sequence_complete"

    def stop_robot(self):
        """Stop the robot"""
        msg = Twist()
        self.publisher.publish(msg)

    def move_rotate_move(self, distance1: float, angle_deg: float, distance2: float):
        """
        Moves forward by distance1 (meters), rotates by angle_deg (degrees),
        then moves forward by distance2 (meters).
        """
        linear_speed = 0.2      # m/s
        angular_speed = 0.5     # rad/s

        # Convert degrees to radians
        angle_rad = angle_deg * (3.14159265 / 180.0)

        # Durations
        duration1 = abs(distance1 / linear_speed)
        duration2 = abs(angle_rad / angular_speed)
        duration3 = abs(distance2 / linear_speed)

        # Determine direction
        linear_sign1 = 1.0 if distance1 >= 0 else -1.0
        angular_sign = 1.0 if angle_rad >= 0 else -1.0
        linear_sign2 = 1.0 if distance2 >= 0 else -1.0

        self.get_logger().info(f"Moving {distance1}m, rotating {angle_deg}°, then moving {distance2}m")

        # Move forward
        self.move_for_duration(linear=linear_speed * linear_sign1, angular=0.0, duration=duration1)
        # Rotate
        self.move_for_duration(linear=0.0, angular=angular_speed * angular_sign, duration=duration2)
        # Move forward again
        self.move_for_duration(linear=linear_speed * linear_sign2, angular=0.0, duration=duration3)

    def move_for_duration(self, linear: float, angular: float, duration: float):
        """Publishes a twist command for a specific duration (in seconds)."""
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular

        end_time = time.time() + duration
        while time.time() < end_time:
            self.publisher.publish(msg)
            time.sleep(0.1)  # 10 Hz

        # Stop after movement
        stop_msg = Twist()
        self.publisher.publish(stop_msg)

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        
        while True:
            if latest_frame is None:
                time.sleep(0.1)
                continue
                
            try:
                _, jpeg = cv2.imencode('.jpg', latest_frame)
                self.wfile.write(b"--frame\r\n")
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b"\r\n")
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                break

def main(args=None):
    rclpy.init(args=args)
    node = CombinedRobotVision()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
        node.stop_robot()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()



