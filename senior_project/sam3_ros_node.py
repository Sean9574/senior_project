#!/usr/bin/env python3
"""
SAM3 ROS2 Client Node
Run this in your ROS2 environment (Python 3.10)

This node:
  - Subscribes to camera images
  - Sends them to SAM3 server for segmentation
  - Publishes segmentation masks and visualization
  - Provides services to change prompts dynamically

Usage:
    # Make sure sam3_server.py is running first!
    ros2 run senior_project sam3_ros_node --ros-args -p prompt:="chair"
    
    # Or launch with parameters
    ros2 run senior_project sam3_ros_node --ros-args \
        -p prompt:="furniture" \
        -p camera_topic:="/camera/color/image_raw" \
        -p rate:=10.0
"""

import base64
import io
import json
import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
import requests
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from std_srvs.srv import SetBool

# For bounding box visualization
try:
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
    HAS_VISION_MSGS = True
except ImportError:
    HAS_VISION_MSGS = False
    print("[WARN] vision_msgs not installed, detection publishing disabled")


class SAM3ROSNode(Node):
    """ROS2 node that interfaces with SAM3 segmentation server"""

    def __init__(self):
        super().__init__("sam3_segmentation_node")

        # Parameters
        self.declare_parameter("server_url", "http://localhost:8100")
        self.declare_parameter("prompt", "object")
        self.declare_parameter("camera_topic", "/camera/color/image_raw")
        self.declare_parameter("rate", 5.0)  # Hz - limit inference rate
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("publish_visualization", True)
        self.declare_parameter("enabled", True)

        self.server_url = self.get_parameter("server_url").value
        self.prompt = self.get_parameter("prompt").value
        self.camera_topic = self.get_parameter("camera_topic").value
        self.rate_hz = self.get_parameter("rate").value
        self.confidence_threshold = self.get_parameter("confidence_threshold").value
        self.publish_viz = self.get_parameter("publish_visualization").value
        self.enabled = self.get_parameter("enabled").value

        # Parameter callback for dynamic updates
        self.add_on_set_parameters_callback(self.parameter_callback)

        # CV Bridge
        self.bridge = CvBridge()

        # Latest image storage
        self.latest_image: Optional[np.ndarray] = None
        self.latest_image_msg: Optional[Image] = None
        self.image_lock = threading.Lock()

        # Rate limiting
        self.min_interval = 1.0 / self.rate_hz
        self.last_inference_time = 0

        # Publishers
        self.mask_pub = self.create_publisher(Image, "~/segmentation_mask", 10)
        self.viz_pub = self.create_publisher(Image, "~/visualization", 10)
        self.prompt_pub = self.create_publisher(String, "~/current_prompt", 10)
        self.status_pub = self.create_publisher(String, "~/status", 10)
        
        if HAS_VISION_MSGS:
            self.detection_pub = self.create_publisher(Detection2DArray, "~/detections", 10)

        # QoS profile to match camera
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            qos_profile
        )

        # Also fix compressed subscriber
        self.compressed_sub = self.create_subscription(
            CompressedImage,
            self.camera_topic + "/compressed",
            self.compressed_callback,
            qos_profile
        )

        # Also try compressed images (common in real robots)
        self.compressed_sub = self.create_subscription(
            CompressedImage,
            self.camera_topic + "/compressed",
            self.compressed_callback,
            10
        )

        # Prompt subscriber (change prompt via topic)
        self.prompt_sub = self.create_subscription(
            String,
            "~/set_prompt",
            self.prompt_callback,
            10
        )

        # Services
        self.enable_srv = self.create_service(
            SetBool,
            "~/enable",
            self.enable_callback
        )

        # Timer for inference (decoupled from image callback)
        timer_period = self.min_interval
        self.inference_timer = self.create_timer(timer_period, self.inference_loop)

        # Check server connection
        self.check_server_connection()

        self.get_logger().info(f"SAM3 ROS Node started")
        self.get_logger().info(f"  Server: {self.server_url}")
        self.get_logger().info(f"  Prompt: '{self.prompt}'")
        self.get_logger().info(f"  Camera: {self.camera_topic}")
        self.get_logger().info(f"  Rate: {self.rate_hz} Hz")

    def check_server_connection(self):
        """Check if SAM3 server is reachable"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.get_logger().info(f"✓ SAM3 server connected")
                self.get_logger().info(f"  CUDA: {data.get('cuda_device', 'N/A')}")
                self.publish_status("connected")
            else:
                self.get_logger().warn(f"SAM3 server returned status {response.status_code}")
                self.publish_status("error")
        except requests.exceptions.ConnectionError:
            self.get_logger().error(
                f"Cannot connect to SAM3 server at {self.server_url}\n"
                "Make sure sam3_server.py is running in your sam3 conda environment!"
            )
            self.publish_status("disconnected")
        except Exception as e:
            self.get_logger().error(f"Server check failed: {e}")
            self.publish_status("error")

    def parameter_callback(self, params):
        """Handle parameter updates"""
        for param in params:
            if param.name == "prompt":
                self.prompt = param.value
                self.get_logger().info(f"Prompt changed to: '{self.prompt}'")
            elif param.name == "rate":
                self.rate_hz = param.value
                self.min_interval = 1.0 / self.rate_hz
            elif param.name == "confidence_threshold":
                self.confidence_threshold = param.value
            elif param.name == "enabled":
                self.enabled = param.value
            elif param.name == "publish_visualization":
                self.publish_viz = param.value
        return SetParametersResult(successful=True)

    def image_callback(self, msg: Image):
        """Store latest image for processing"""
        with self.image_lock:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            self.latest_image_msg = msg

    def compressed_callback(self, msg: CompressedImage):
        """Handle compressed images"""
        with self.image_lock:
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.latest_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.latest_image = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB)
            # Create a fake Image msg for header info
            self.latest_image_msg = Image()
            self.latest_image_msg.header = msg.header

    def prompt_callback(self, msg: String):
        """Update prompt from topic"""
        self.prompt = msg.data
        self.get_logger().info(f"Prompt updated to: '{self.prompt}'")

    def enable_callback(self, request, response):
        """Enable/disable segmentation"""
        self.enabled = request.data
        response.success = True
        response.message = f"Segmentation {'enabled' if self.enabled else 'disabled'}"
        self.get_logger().info(response.message)
        return response

    def publish_status(self, status: str):
        """Publish node status"""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def inference_loop(self):
        """Main inference loop - runs at configured rate"""
        if not self.enabled:
            return

        # Check rate limiting
        now = time.time()
        if now - self.last_inference_time < self.min_interval:
            return

        # Get latest image
        with self.image_lock:
            if self.latest_image is None:
                return
            image = self.latest_image.copy()
            header = self.latest_image_msg.header if self.latest_image_msg else None

        self.last_inference_time = now

        # Publish current prompt
        prompt_msg = String()
        prompt_msg.data = self.prompt
        self.prompt_pub.publish(prompt_msg)

        # Encode image
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        image_b64 = base64.b64encode(buffer).decode("utf-8")

        # Call SAM3 server
        try:
            response = requests.post(
                f"{self.server_url}/segment",
                json={
                    "image_base64": image_b64,
                    "prompt": self.prompt,
                    "confidence_threshold": self.confidence_threshold,
                    "return_visualization": self.publish_viz,
                },
                timeout=10,
            )

            if response.status_code != 200:
                self.get_logger().warn(f"Server error: {response.status_code}")
                self.publish_status("error")
                return

            result = response.json()

            if not result.get("success", False):
                self.get_logger().warn(f"Segmentation failed: {result.get('error')}")
                return

            num_objects = result.get("num_objects", 0)
            inference_ms = result.get("inference_time_ms", 0)

            self.get_logger().debug(
                f"Found {num_objects} '{self.prompt}' objects in {inference_ms:.0f}ms"
            )

            # Publish combined mask
            if num_objects > 0:
                masks_b64 = result.get("masks_base64", [])
                combined_mask = None

                for mask_b64 in masks_b64:
                    mask_bytes = base64.b64decode(mask_b64)
                    mask_np = np.array(
                        cv2.imdecode(
                            np.frombuffer(mask_bytes, np.uint8),
                            cv2.IMREAD_GRAYSCALE
                        )
                    )

                    if combined_mask is None:
                        combined_mask = mask_np
                    else:
                        combined_mask = np.maximum(combined_mask, mask_np)

                # Publish mask
                if combined_mask is not None:
                    mask_msg = self.bridge.cv2_to_imgmsg(combined_mask, "mono8")
                    if header:
                        mask_msg.header = header
                    self.mask_pub.publish(mask_msg)

                # Publish detections
                if HAS_VISION_MSGS:
                    self.publish_detections(result, header)

            # Publish visualization
            if self.publish_viz and result.get("visualization_base64"):
                viz_bytes = base64.b64decode(result["visualization_base64"])
                viz_np = cv2.imdecode(
                    np.frombuffer(viz_bytes, np.uint8),
                    cv2.IMREAD_COLOR
                )
                viz_np = cv2.cvtColor(viz_np, cv2.COLOR_BGR2RGB)
                viz_msg = self.bridge.cv2_to_imgmsg(viz_np, "rgb8")
                if header:
                    viz_msg.header = header
                self.viz_pub.publish(viz_msg)

            self.publish_status("running")

        except requests.exceptions.Timeout:
            self.get_logger().warn("SAM3 server timeout")
            self.publish_status("timeout")
        except requests.exceptions.ConnectionError:
            self.get_logger().error("Lost connection to SAM3 server")
            self.publish_status("disconnected")
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            self.publish_status("error")

    def publish_detections(self, result, header):
        """Publish detection results as Detection2DArray"""
        if not HAS_VISION_MSGS:
            return

        det_array = Detection2DArray()
        if header:
            det_array.header = header

        boxes = result.get("boxes", [])
        scores = result.get("scores", [])

        for box, score in zip(boxes, scores):
            det = Detection2D()
            
            # Bounding box (x1, y1, x2, y2) -> center + size
            x1, y1, x2, y2 = box
            det.bbox.center.position.x = (x1 + x2) / 2
            det.bbox.center.position.y = (y1 + y2) / 2
            det.bbox.size_x = x2 - x1
            det.bbox.size_y = y2 - y1

            # Hypothesis
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = self.prompt
            hyp.hypothesis.score = float(score)
            det.results.append(hyp)

            det_array.detections.append(det)

        self.detection_pub.publish(det_array)


def main(args=None):
    rclpy.init(args=args)
    node = SAM3ROSNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()