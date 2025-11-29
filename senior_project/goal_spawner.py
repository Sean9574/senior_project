#!/usr/bin/env python3
import math
import random

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from visualization_msgs.msg import Marker


class GoalSpawner(Node):
    """
    Spawns a random goal and publishes it continuously on:
      /stretch/goal         (PointStamped)
      /stretch/goal_marker  (Marker, sphere) for RViz visualization
    """

    def __init__(self):
        super().__init__("goal_spawner")

        # Parameters
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("min_radius", 1.0)
        self.declare_parameter("max_radius", 3.0)
        self.declare_parameter("min_angle_deg", -90.0)
        self.declare_parameter("max_angle_deg", 90.0)
        # publish_period controls how often we republish (seconds)
        self.declare_parameter("publish_period", 0.5)

        self.frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )
        self.min_radius = (
            self.get_parameter("min_radius").get_parameter_value().double_value
        )
        self.max_radius = (
            self.get_parameter("max_radius").get_parameter_value().double_value
        )
        self.min_angle_deg = (
            self.get_parameter("min_angle_deg").get_parameter_value().double_value
        )
        self.max_angle_deg = (
            self.get_parameter("max_angle_deg").get_parameter_value().double_value
        )
        self.publish_period = (
            self.get_parameter("publish_period").get_parameter_value().double_value
        )

        # Publishers: absolute topics so they always match PPO learner
        self.goal_pub = self.create_publisher(PointStamped, "/stretch/goal", 10)
        self.marker_pub = self.create_publisher(Marker, "/stretch/goal_marker", 10)

        # Sample one goal
        self.goal_x, self.goal_y = self._sample_goal()
      

        # Pre-create msgs that we’ll republish continuously
        self.goal_msg = self._make_goal_msg(self.goal_x, self.goal_y)
        self.marker_msg = self._make_marker_msg(self.goal_x, self.goal_y)

        # Timer: publish forever at publish_period
        self.timer = self.create_timer(self.publish_period, self._timer_cb)

    def _sample_goal(self):
        r = random.uniform(self.min_radius, self.max_radius)
        angle_deg = random.uniform(self.min_angle_deg, self.max_angle_deg)
        angle_rad = math.radians(angle_deg)
        x = r * math.cos(angle_rad)
        y = r * math.sin(angle_rad)
        return x, y

    def _make_goal_msg(self, x, y):
        msg = PointStamped()
        msg.header.frame_id = self.frame_id
        msg.point.x = float(x)
        msg.point.y = float(y)
        msg.point.z = 0.0
        return msg

    def _make_marker_msg(self, x, y):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.ns = "goal"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD

        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.4   # a bit above the floor
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        # Big and obvious
        m.scale.x = 0.6
        m.scale.y = 0.6
        m.scale.z = 0.6

        # Bright red
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0

        # 0 lifetime = keep it around, but we still republish
        m.lifetime.sec = 0
        m.lifetime.nanosec = 0
        return m

    def _timer_cb(self):
        now = self.get_clock().now().to_msg()
        self.goal_msg.header.stamp = now
        self.marker_msg.header.stamp = now

        self.goal_pub.publish(self.goal_msg)
        self.marker_pub.publish(self.marker_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GoalSpawner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down goal_spawner")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
