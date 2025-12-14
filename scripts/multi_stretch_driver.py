#!/usr/bin/env python3
"""
Multi-Stretch MuJoCo Driver for ROS2

Runs N Stretch robots in parallel in MuJoCo and publishes/subscribes
to namespaced ROS2 topics for each robot.

Topics per robot (namespace /robot{i}/):
  - /robot{i}/odom (nav_msgs/Odometry)
  - /robot{i}/scan (sensor_msgs/LaserScan)
  - /robot{i}/cmd_vel (geometry_msgs/Twist) [subscriber]
  - /robot{i}/joint_states (sensor_msgs/JointState)
  - /robot{i}/goal (geometry_msgs/PointStamped)

Usage:
    ros2 run senior_project multi_stretch_driver --ros-args -p num_robots:=4
"""

import math
import threading
import time
from typing import Dict, List, Optional

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Twist, TransformStamped, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, JointState
from std_msgs.msg import Header, String
from tf2_ros import TransformBroadcaster

from multi_stretch_arena import MultiStretchArena


class RobotInterface:
    """Interface for a single robot's ROS2 topics."""
    
    def __init__(self, node: Node, robot_id: int, callback_group):
        self.robot_id = robot_id
        self.ns = f"/robot{robot_id}"
        
        # Publishers
        self.odom_pub = node.create_publisher(Odometry, f"{self.ns}/odom", 10)
        self.scan_pub = node.create_publisher(LaserScan, f"{self.ns}/scan", 10)
        self.joint_pub = node.create_publisher(JointState, f"{self.ns}/joint_states", 10)
        self.goal_pub = node.create_publisher(PointStamped, f"{self.ns}/goal", 10)
        
        # Subscriber
        self.cmd_vel_sub = node.create_subscription(
            Twist, f"{self.ns}/cmd_vel", 
            self._cmd_vel_callback, 10,
            callback_group=callback_group
        )
        
        # Command state
        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        self.last_cmd_time = time.time()
        
    def _cmd_vel_callback(self, msg: Twist):
        self.cmd_linear = msg.linear.x
        self.cmd_angular = msg.angular.z
        self.last_cmd_time = time.time()
    
    def get_cmd_vel(self) -> tuple:
        """Get current velocity command, zero if stale."""
        if time.time() - self.last_cmd_time > 0.5:
            return 0.0, 0.0
        return self.cmd_linear, self.cmd_angular


class MultiStretchDriver(Node):
    """
    ROS2 node that runs multiple Stretch robots in MuJoCo.
    """
    
    def __init__(self):
        super().__init__('multi_stretch_driver')
        
        # Parameters
        self.declare_parameter('num_robots', 4)
        self.declare_parameter('room_size', 6.0)
        self.declare_parameter('use_viewer', True)
        self.declare_parameter('sim_rate', 100.0)  # Hz
        self.declare_parameter('publish_rate', 30.0)  # Hz
        self.declare_parameter('randomize_goals', True)
        
        self.num_robots = self.get_parameter('num_robots').value
        self.room_size = self.get_parameter('room_size').value
        self.use_viewer = self.get_parameter('use_viewer').value
        self.sim_rate = self.get_parameter('sim_rate').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.randomize_goals = self.get_parameter('randomize_goals').value
        
        self.get_logger().info(f"Starting Multi-Stretch Driver with {self.num_robots} robots")
        
        # Create arena and load model
        self.arena = MultiStretchArena(
            num_robots=self.num_robots,
            room_size=self.room_size,
            randomize_goals=self.randomize_goals,
        )
        
        xml_string = self.arena.generate_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        
        # Save XML for debugging
        with open("/tmp/multi_stretch_ros.xml", "w") as f:
            f.write(xml_string)
        self.get_logger().info("Saved debug XML to /tmp/multi_stretch_ros.xml")
        
        # Cache robot IDs
        self._cache_robot_ids()
        
        # Create robot interfaces
        self.callback_group = ReentrantCallbackGroup()
        self.robots: Dict[int, RobotInterface] = {}
        for i in range(self.num_robots):
            self.robots[i] = RobotInterface(self, i, self.callback_group)
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Viewer (optional)
        self.viewer = None
        self.viewer_thread = None
        if self.use_viewer:
            self._start_viewer()
        
        # Simulation thread
        self.sim_running = True
        self.sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.sim_thread.start()
        
        # Publisher timer
        self.create_timer(1.0 / self.publish_rate, self._publish_all)
        
        # Goal publisher timer (slower rate)
        self.create_timer(1.0, self._publish_goals)
        
        self.get_logger().info(f"Multi-Stretch Driver ready with {self.num_robots} robots")
    
    def _cache_robot_ids(self):
        """Cache MuJoCo IDs for all robots."""
        self.robot_ids = {}
        
        for i in range(self.num_robots):
            prefix = f"robot{i}_"
            room = self.arena.rooms[i]
            
            # Get body ID
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}base_link")
            
            # Get actuator IDs
            left_wheel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}left_wheel_vel")
            right_wheel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}right_wheel_vel")
            
            # Get lidar sensor
            lidar_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"{prefix}base_lidar")
            
            self.robot_ids[i] = {
                'body_id': body_id,
                'left_wheel': left_wheel_id,
                'right_wheel': right_wheel_id,
                'lidar_id': lidar_id,
                'goal_x': room.goal_x,
                'goal_y': room.goal_y,
                'spawn_x': room.spawn_x,
                'spawn_y': room.spawn_y,
            }
            
            self.get_logger().info(f"Robot {i}: body={body_id}, wheels=({left_wheel_id}, {right_wheel_id}), lidar={lidar_id}")
    
    def _start_viewer(self):
        """Start MuJoCo viewer in separate thread."""
        def viewer_thread_fn():
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            while self.sim_running and self.viewer.is_running():
                time.sleep(0.01)
        
        self.viewer_thread = threading.Thread(target=viewer_thread_fn, daemon=True)
        self.viewer_thread.start()
        time.sleep(1.0)  # Wait for viewer to initialize
    
    def _sim_loop(self):
        """Main simulation loop."""
        dt = 1.0 / self.sim_rate
        
        while self.sim_running:
            start = time.time()
            
            # Apply commands from all robots
            for i in range(self.num_robots):
                self._apply_cmd_vel(i)
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            # Sync viewer
            if self.viewer is not None:
                self.viewer.sync()
            
            # Sleep to maintain rate
            elapsed = time.time() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    def _apply_cmd_vel(self, robot_id: int):
        """Apply velocity command to robot wheels."""
        v_lin, v_ang = self.robots[robot_id].get_cmd_vel()
        
        # Stretch wheel parameters
        wheel_base = 0.34
        wheel_radius = 0.05
        
        # Differential drive
        v_left = (v_lin - v_ang * wheel_base / 2) / wheel_radius
        v_right = (v_lin + v_ang * wheel_base / 2) / wheel_radius
        
        # Scale and clamp
        v_left = np.clip(v_left * 2.0, -6.0, 6.0)
        v_right = np.clip(v_right * 2.0, -6.0, 6.0)
        
        left_id = self.robot_ids[robot_id]['left_wheel']
        right_id = self.robot_ids[robot_id]['right_wheel']
        
        if left_id >= 0:
            self.data.ctrl[left_id] = v_left
        if right_id >= 0:
            self.data.ctrl[right_id] = v_right
    
    def _get_robot_pose(self, robot_id: int) -> tuple:
        """Get robot position and orientation."""
        body_id = self.robot_ids[robot_id]['body_id']
        
        pos = self.data.xpos[body_id]
        quat = self.data.xquat[body_id]  # [w, x, y, z]
        
        # Extract yaw
        yaw = math.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                         1 - 2 * (quat[2]**2 + quat[3]**2))
        
        return pos[0], pos[1], pos[2], yaw, quat
    
    def _get_robot_velocity(self, robot_id: int) -> tuple:
        """Get robot velocity."""
        body_id = self.robot_ids[robot_id]['body_id']
        vel = self.data.cvel[body_id]
        
        x, y, z, yaw, _ = self._get_robot_pose(robot_id)
        
        # Project world velocity to robot frame
        v_lin = vel[3] * math.cos(yaw) + vel[4] * math.sin(yaw)
        v_ang = vel[2]
        
        return v_lin, v_ang
    
    def _get_lidar(self, robot_id: int) -> np.ndarray:
        """Get lidar readings."""
        lidar_id = self.robot_ids[robot_id]['lidar_id']
        
        if lidar_id < 0:
            return np.ones(360) * 10.0
        
        # Stretch has 360 lidar rays
        readings = []
        sensor_adr = self.model.sensor_adr[lidar_id]
        
        # The lidar is replicated 360 times
        for j in range(360):
            if sensor_adr + j < len(self.data.sensordata):
                val = self.data.sensordata[sensor_adr + j]
                if val < 0:
                    val = 10.0  # No hit
                readings.append(val)
            else:
                readings.append(10.0)
        
        return np.array(readings)
    
    def _publish_all(self):
        """Publish data for all robots."""
        now = self.get_clock().now().to_msg()
        
        for i in range(self.num_robots):
            self._publish_odom(i, now)
            self._publish_scan(i, now)
            self._publish_tf(i, now)
    
    def _publish_odom(self, robot_id: int, stamp):
        """Publish odometry for a robot."""
        x, y, z, yaw, quat = self._get_robot_pose(robot_id)
        v_lin, v_ang = self._get_robot_velocity(robot_id)
        
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = f"robot{robot_id}/odom"
        msg.child_frame_id = f"robot{robot_id}/base_link"
        
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z
        msg.pose.pose.orientation.w = quat[0]
        msg.pose.pose.orientation.x = quat[1]
        msg.pose.pose.orientation.y = quat[2]
        msg.pose.pose.orientation.z = quat[3]
        
        msg.twist.twist.linear.x = v_lin
        msg.twist.twist.angular.z = v_ang
        
        self.robots[robot_id].odom_pub.publish(msg)
    
    def _publish_scan(self, robot_id: int, stamp):
        """Publish laser scan for a robot."""
        ranges = self._get_lidar(robot_id)
        
        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = f"robot{robot_id}/laser"
        msg.angle_min = 0.0
        msg.angle_max = 2 * math.pi
        msg.angle_increment = 2 * math.pi / len(ranges)
        msg.range_min = 0.1
        msg.range_max = 10.0
        msg.ranges = ranges.tolist()
        
        self.robots[robot_id].scan_pub.publish(msg)
    
    def _publish_tf(self, robot_id: int, stamp):
        """Publish TF for a robot."""
        x, y, z, yaw, quat = self._get_robot_pose(robot_id)
        
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = f"robot{robot_id}/odom"
        t.child_frame_id = f"robot{robot_id}/base_link"
        
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.w = quat[0]
        t.transform.rotation.x = quat[1]
        t.transform.rotation.y = quat[2]
        t.transform.rotation.z = quat[3]
        
        self.tf_broadcaster.sendTransform(t)
    
    def _publish_goals(self):
        """Publish goal positions for all robots."""
        now = self.get_clock().now().to_msg()
        
        for i in range(self.num_robots):
            msg = PointStamped()
            msg.header.stamp = now
            msg.header.frame_id = f"robot{i}/odom"
            msg.point.x = self.robot_ids[i]['goal_x']
            msg.point.y = self.robot_ids[i]['goal_y']
            msg.point.z = 0.0
            
            self.robots[i].goal_pub.publish(msg)
    
    def reset_robot(self, robot_id: int, new_goal: bool = True):
        """Reset a single robot to spawn position."""
        body_id = self.robot_ids[robot_id]['body_id']
        joint_id = self.model.body_jntadr[body_id]
        
        if joint_id >= 0:
            qpos_adr = self.model.jnt_qposadr[joint_id]
            
            spawn_x = self.robot_ids[robot_id]['spawn_x']
            spawn_y = self.robot_ids[robot_id]['spawn_y']
            
            self.data.qpos[qpos_adr:qpos_adr+3] = [spawn_x, spawn_y, 0.15]
            self.data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]
            
            # Zero velocities
            qvel_adr = self.model.jnt_dofadr[joint_id]
            self.data.qvel[qvel_adr:qvel_adr+6] = 0
        
        # Randomize goal if requested
        if new_goal and self.randomize_goals:
            room = self.arena.rooms[robot_id]
            self.robot_ids[robot_id]['goal_x'] = room.center_x + np.random.uniform(-1.5, 1.5)
            self.robot_ids[robot_id]['goal_y'] = room.center_y + np.random.uniform(-1.5, 1.5)
        
        mujoco.mj_forward(self.model, self.data)
    
    def reset_all(self):
        """Reset all robots."""
        for i in range(self.num_robots):
            self.reset_robot(i)
    
    def destroy_node(self):
        """Clean shutdown."""
        self.sim_running = False
        if self.viewer is not None:
            self.viewer.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = MultiStretchDriver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
