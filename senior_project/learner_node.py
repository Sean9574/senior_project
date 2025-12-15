#!/usr/bin/env python3
"""
Stretch Robot RL Environment + Learner (NO SKRL, TD3 PURE PYTORCH, LAUNCH COMPAT)

Adds requested behavior:
- When goal reached: log SUCCESS, command STOP, hold stop briefly, terminate episode (no spinning).

UPDATED (Obstacle-aware avoidance):
- Uses LiDAR direction (front/left/right + repulsive vector) to steer around obstacles toward goal.
- Blends/overrides policy action with avoidance when obstacles are inside SAFE distance.
"""

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import rclpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from geometry_msgs.msg import PointStamped, Twist
from gymnasium import spaces
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32
from std_msgs.msg import String as StringMsg
from visualization_msgs.msg import Marker

# =============================================================================
# CONFIG (EDIT THESE — NO CLI REQUIRED)
# =============================================================================

BOOTSTRAP_MODE = True

AUTO_LOAD_CHECKPOINT_FOR_TRAINING = True
AUTO_LOAD_CHECKPOINT_FOR_INFERENCE = True
CHECKPOINT_FILENAME = "td3_agent.pt"

GOAL_MODE = "curriculum"  # "fixed" or "curriculum"
FIXED_GOAL_X = 3.0
FIXED_GOAL_Y = -4.0
EPISODE_SECONDS = 45.0

CURR_START_RADIUS = 1.5
CURR_MAX_RADIUS = 6.0
CURR_PROMOTE_THRESHOLD = 0.70
CURR_DEMOTE_THRESHOLD = 0.20
CURR_WINDOW = 50
CURR_STEP_UP = 0.5
CURR_STEP_DOWN = 0.5

DEFAULT_START_STEPS = 8000
DEFAULT_EXPL_NOISE = 0.35

# --------------------------
# IMPORTANT: make avoidance kick in EARLY enough to "care"
# --------------------------
# Hard collision stop (meters)
DEFAULT_COLLISION_DIST = 0.40
# Start avoiding earlier (meters)
DEFAULT_SAFE_DIST = 0.8

R_GOAL = 2500.0
R_COLLISION = -2000.0
R_TIMEOUT_BOOTSTRAP = -50.0
R_TIMEOUT_NORMAL = -500.0

GOAL_RADIUS = 0.45

# Success stop behavior
STOP_ON_SUCCESS = True
SUCCESS_BRAKE_HOLD_SECONDS = 1.0   # hold cmd_vel = 0 for this long on success

# --------------------------
# Reward shaping (core)
# --------------------------
PROGRESS_SCALE = 650.0
FORWARD_HEADING_GATE_DEG = 70.0
FORWARD_WHEN_ALIGNED_BONUS = 5.0
FORWARD_WHEN_MISALIGNED_PENALTY = 2.5
ALIGN_W = 4.00
DIST_SHAPING_K = 0.20
STEP_COST = -0.03

FINISH_DIST = 2.75
FINISH_ALIGN_W = 5.0
FINISH_FWD_W = 1.5
FINISH_TURN_PENALTY_W = 0.45
FINISH_FORWARD_GATING_DEG = 35.0
TURN_TO_GOAL_BONUS = 1.50

OBSTACLE_W = 6.0

# =============================================================================
# LIDAR-AWARE AVOIDANCE LAYER (directional)
# =============================================================================

AVOID_ENABLE = True

# How to apply avoidance:
# - "blend": blend policy (TD3) action with avoidance when obstacles are within SAFE distance
# - "override": fully override policy when obstacles are within SAFE distance
AVOID_MODE = "override"  # "blend" or "override"

# Start avoiding at SAFE; hard stop at COLLISION
AVOID_SAFE_DIST = DEFAULT_SAFE_DIST
AVOID_COLLISION_DIST = DEFAULT_COLLISION_DIST

# --- Linear speed policy ---
AVOID_MIN_FWD_V = 0.40        # HARD minimum to overcome friction (while avoiding)
CLEAR_SPEED_BOOST = 1.00      # keep 1.0 to respect v_max; raise to >1.0 only if you also raise v_max
CLEAR_MARGIN = 0.25           # meters beyond SAFE where we allow full speed

# Sector widths (radians). Front centered at "forward"
FRONT_HALF_ANGLE_RAD = math.radians(25.0)
SIDE_HALF_ANGLE_RAD  = math.radians(35.0)
LEFT_CENTER_RAD      = math.radians(-90.0)
RIGHT_CENTER_RAD     = math.radians(90.0)

# If your scan "front" is actually the robot rear, set this to math.pi
LIDAR_FORWARD_OFFSET_RAD = math.pi

# Repulsion settings
REPULSE_GAIN = 1.65
REPULSE_DECAY = 0.55

# Desired clearance target for speed scaling
AVOID_CLEARANCE_TARGET = 0.55

# Turn limit scaling
AVOID_W_MAX_FRACTION = 1.0  # fraction of env.w_max

# Allow small reverse when front is blocked
AVOID_ALLOW_REVERSE = True
AVOID_REVERSE_V = -0.04  # m/s when front is very close

# Debug (prints min_front/left/right/all + commands)
AVOID_DEBUG = True
AVOID_DEBUG_EVERY_N_STEPS = 10


# =============================================================================
# Utils
# =============================================================================

def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    sin_y_cosp = 2.0 * (qw * qz + qx * qy)
    cos_y_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(sin_y_cosp, cos_y_cosp)

def wrap_to_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class CurriculumConfig:
    start_radius: float = CURR_START_RADIUS
    max_radius: float = CURR_MAX_RADIUS
    promote_threshold: float = CURR_PROMOTE_THRESHOLD
    demote_threshold: float = CURR_DEMOTE_THRESHOLD
    window: int = CURR_WINDOW
    step_up: float = CURR_STEP_UP
    step_down: float = CURR_STEP_DOWN


# =============================================================================
# ROS Interface
# =============================================================================

class StretchRosInterfaceNode(Node):
    def __init__(self, ns: str = "stretch", odom_topic="odom", scan_topic="scan", imu_topic="imu",
                 goal_topic="goal", cmd_topic="cmd_vel"):
        super().__init__("learner_node_no_skrl")

        self.ns = ns
        self.last_odom: Optional[Odometry] = None
        self.last_scan: Optional[LaserScan] = None
        self.last_goal: Optional[PointStamped] = None
        self.last_imu: Optional[Imu] = None

        odom_name = f"/{ns}/{odom_topic}"
        scan_name = f"/{ns}/{scan_topic}"
        imu_name  = f"/{ns}/{imu_topic}"
        goal_name = f"/{ns}/{goal_topic}"
        cmd_name  = f"/{ns}/{cmd_topic}"

        self.create_subscription(Odometry, odom_name, self.odom_cb, 10)
        self.create_subscription(LaserScan, scan_name, self.scan_cb, 10)
        self.create_subscription(Imu, imu_name, self.imu_cb, 10)
        self.create_subscription(PointStamped, goal_name, self.goal_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, cmd_name, 10)
        self.goal_pub = self.create_publisher(PointStamped, goal_name, 10)
        self.goal_marker_pub = self.create_publisher(Marker, f"/{ns}/goal_marker", 10)

        self.reward_pub = self.create_publisher(Float32, f"/{ns}/reward", 10)
        self.reward_breakdown_pub = self.create_publisher(StringMsg, "/reward_breakdown", 10)

    def odom_cb(self, msg: Odometry):
        self.last_odom = msg

    def scan_cb(self, msg: LaserScan):
        self.last_scan = msg

    def imu_cb(self, msg: Imu):
        self.last_imu = msg

    def goal_cb(self, msg: PointStamped):
        self.last_goal = msg

    def wait_for_sensors(self, timeout_sec: float = 10.0) -> bool:
        start = time.time()
        while time.time() - start < timeout_sec:
            if self.last_odom is not None and self.last_scan is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().warn("[ENV] Timeout waiting for initial sensors")
        return False

    def send_cmd(self, v: float, w: float):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def brake_hold(self, hold_seconds: float):
        """Publish zero cmd_vel for hold_seconds to ensure robot stops."""
        t_end = time.time() + max(0.0, float(hold_seconds))
        while time.time() < t_end:
            self.send_cmd(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.02)

    def publish_goal(self, goal_x: float, goal_y: float, frame_id: str = "odom"):
        goal_msg = PointStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = frame_id
        goal_msg.point.x = float(goal_x)
        goal_msg.point.y = float(goal_y)
        goal_msg.point.z = 0.0
        self.goal_pub.publish(goal_msg)
        self.last_goal = goal_msg
        self.publish_goal_marker(goal_x, goal_y, frame_id=frame_id)

    def publish_goal_marker(self, goal_x: float, goal_y: float, frame_id: str = "odom"):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = frame_id
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(goal_x)
        marker.pose.position.y = float(goal_y)
        marker.pose.position.z = 0.3
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0
        self.goal_marker_pub.publish(marker)


# =============================================================================
# Gym Env
# =============================================================================

class StretchRosEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        ros: StretchRosInterfaceNode,
        control_dt: float = 0.1,
        num_lidar_bins: int = 60,
        lidar_max_range: float = 20.0,
        v_max: float = 3.50,
        w_max: float = 2.84,
        v_min_reverse: float = -0.05,
        episode_time_seconds: float = EPISODE_SECONDS,
        goal_radius: float = GOAL_RADIUS,
        collision_dist: float = DEFAULT_COLLISION_DIST,
        safe_dist: float = DEFAULT_SAFE_DIST,
        use_curriculum: bool = True,
        curriculum: CurriculumConfig = CurriculumConfig(),
        fixed_goal_x: float = 2.0,
        fixed_goal_y: float = 0.0,
        max_goal_radius_for_norm: float = 6.0,
        r_goal: float = R_GOAL,
        r_collision: float = R_COLLISION,
        r_timeout: float = (R_TIMEOUT_BOOTSTRAP if BOOTSTRAP_MODE else R_TIMEOUT_NORMAL),
    ):
        super().__init__()
        self.ros = ros

        self.control_dt = float(control_dt)
        self.num_lidar_bins = int(num_lidar_bins)
        self.lidar_max_range = float(lidar_max_range)

        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.v_min_reverse = float(v_min_reverse)

        self.max_steps = int(float(episode_time_seconds) / self.control_dt)
        self.goal_radius = float(goal_radius)
        self.collision_dist = float(collision_dist)
        self.safe_dist = float(safe_dist)

        self.use_curriculum = bool(use_curriculum)
        self.curriculum = curriculum
        self.fixed_goal_x = float(fixed_goal_x)
        self.fixed_goal_y = float(fixed_goal_y)
        self.max_goal_radius_for_norm = float(max_goal_radius_for_norm)

        self.r_goal_terminal = float(r_goal)
        self.r_collision_terminal = float(r_collision)
        self.r_timeout_terminal = float(r_timeout)

        self.recent_success = deque(maxlen=self.curriculum.window)
        self.curr_radius = self.curriculum.start_radius

        self.step_count = 0
        self.episode_index = int(os.environ.get("RL_EPISODE_NUM", 1))
        self.episode_return = 0.0
        self.prev_goal_dist = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)

        self._cached_bins = None
        self._cached_state = None

        self._success_latched = False

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        obs_dim = self.num_lidar_bins + 9
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.ros.get_logger().info("[ENV] Waiting for sensors...")
        self.ros.wait_for_sensors(timeout_sec=10.0)

        self._set_new_goal(first_time=True)
        self.prev_goal_dist = self._goal_distance()

        mode = "CURRICULUM" if self.use_curriculum else "FIXED"
        self.ros.get_logger().info(f"[ENV] Goal mode: {mode}")
        if not self.use_curriculum:
            self.ros.get_logger().info(f"[ENV] Fixed goal: ({self.fixed_goal_x:.2f}, {self.fixed_goal_y:.2f})")
        self.ros.get_logger().info(f"[ENV] goal_radius={self.goal_radius:.2f} max_steps={self.max_steps}")
        self.ros.get_logger().info(f"[ENV] BOOTSTRAP_MODE={BOOTSTRAP_MODE} timeout_penalty={self.r_timeout_terminal}")
        self.ros.get_logger().info(
            f"[AVOID] ENABLE={AVOID_ENABLE} MODE={AVOID_MODE} SAFE={AVOID_SAFE_DIST:.2f} COLL={AVOID_COLLISION_DIST:.2f} "
            f"(reward safe_dist={self.safe_dist:.2f} collision_dist={self.collision_dist:.2f})"
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.episode_return = 0.0
        self.prev_action[:] = 0.0
        self._success_latched = False

        self._set_new_goal(first_time=False)
        self.prev_goal_dist = self._goal_distance()

        obs = self._build_observation()
        info = {"goal_distance": float(self.prev_goal_dist), "curr_radius": float(self.curr_radius)}
        return obs, info

    # ---------------------------
    # Directional LiDAR helpers
    # ---------------------------

    def _sanitize_ranges(self, scan: LaserScan) -> np.ndarray:
        r = np.array(scan.ranges, dtype=np.float32)
        max_r = float(scan.range_max if scan.range_max > 0 else self.lidar_max_range)
        min_r = float(scan.range_min if scan.range_min > 0 else 0.01)
        if r.size == 0:
            return np.full((1,), max_r, dtype=np.float32)
        bad = np.isnan(r) | np.isinf(r) | (r < min_r) | (r > max_r)
        r[bad] = max_r
        return r

    def _lidar_sectors(self) -> Tuple[float, float, float, float]:
        """
        Returns (min_front, min_left, min_right, min_all) using actual scan angles.
        Assumes scan angle 0 is forward; adjust with LIDAR_FORWARD_OFFSET_RAD if needed.
        """
        scan = self.ros.last_scan
        if scan is None:
            return self.lidar_max_range, self.lidar_max_range, self.lidar_max_range, self.lidar_max_range

        ranges = self._sanitize_ranges(scan)
        n = ranges.size
        a0 = float(scan.angle_min)
        da = float(scan.angle_increment if scan.angle_increment != 0 else (scan.angle_max - scan.angle_min) / max(1, n - 1))

        angles = a0 + da * np.arange(n, dtype=np.float32)
        angles = angles + float(LIDAR_FORWARD_OFFSET_RAD)
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        def sector_min(center: float, half: float) -> float:
            d = np.abs(np.arctan2(np.sin(angles - center), np.cos(angles - center)))
            m = d <= half
            if not np.any(m):
                return float(np.min(ranges))
            return float(np.min(ranges[m]))

        min_front = sector_min(0.0, float(FRONT_HALF_ANGLE_RAD))
        min_left  = sector_min(float(LEFT_CENTER_RAD), float(SIDE_HALF_ANGLE_RAD))
        min_right = sector_min(float(RIGHT_CENTER_RAD), float(SIDE_HALF_ANGLE_RAD))
        min_all   = float(max(0.01, np.min(ranges)))
        return min_front, min_left, min_right, min_all

    def _repulsive_vector(self) -> Tuple[float, float, float]:
        """
        Compute a simple 2D repulsion vector in the robot frame from the scan:
        - Each ray pushes opposite its direction, weighted by distance (closer -> stronger).
        Returns (rx, ry, min_all).
        """
        scan = self.ros.last_scan
        if scan is None:
            return 0.0, 0.0, self.lidar_max_range

        ranges = self._sanitize_ranges(scan)
        n = ranges.size
        a0 = float(scan.angle_min)
        da = float(scan.angle_increment if scan.angle_increment != 0 else (scan.angle_max - scan.angle_min) / max(1, n - 1))
        angles = a0 + da * np.arange(n, dtype=np.float32)
        angles = angles + float(LIDAR_FORWARD_OFFSET_RAD)
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        min_all = float(max(0.01, np.min(ranges)))

        influence = np.clip((AVOID_CLEARANCE_TARGET - ranges) / max(1e-3, AVOID_CLEARANCE_TARGET), 0.0, 1.0)
        w = (influence ** 2) * float(REPULSE_GAIN)

        rx = float(np.sum(-np.cos(angles) * w))
        ry = float(np.sum(-np.sin(angles) * w))

        norm = math.hypot(rx, ry)
        if norm > 1e-6:
            rx *= float(REPULSE_DECAY)
            ry *= float(REPULSE_DECAY)

        return rx, ry, min_all

    def _avoidance_action(self, policy_v: float, policy_w: float) -> Tuple[float, float, float]:
        """
        Compute an avoidance (v_cmd, w_cmd) and a blend factor alpha in [0,1]
        where alpha=0 means no avoidance, alpha=1 means full avoidance.
        """
        min_front, min_left, min_right, min_all = self._lidar_sectors()

        if min_all >= AVOID_SAFE_DIST:
            return policy_v, policy_w, 0.0

        alpha = float(np.clip((AVOID_SAFE_DIST - min_all) / max(1e-3, (AVOID_SAFE_DIST - AVOID_COLLISION_DIST)), 0.0, 1.0))

        if min_all <= AVOID_COLLISION_DIST:
            return 0.0, 0.0, 1.0

        goal_ang = float(self._goal_angle_relative())

        rx, ry, _ = self._repulsive_vector()

        gx = float(math.cos(goal_ang))
        gy = float(math.sin(goal_ang))

        vx = gx + rx
        vy = gy + ry

        if math.hypot(vx, vy) < 1e-4:
            if min_left < min_right:
                desired_ang = -math.radians(75.0)
            else:
                desired_ang = math.radians(75.0)
        else:
            desired_ang = math.atan2(vy, vx)

        k_w = 2.2
        w_cmd = float(np.clip(k_w * wrap_to_pi(desired_ang), -self.w_max * AVOID_W_MAX_FRACTION, self.w_max * AVOID_W_MAX_FRACTION))

        v_base = float(np.clip(policy_v, 0.0, self.v_max))
        slow = float(np.clip(min_all / max(1e-3, AVOID_CLEARANCE_TARGET), 0.0, 1.0))
        v_cmd = v_base * slow

        if min_front < (AVOID_SAFE_DIST * 0.95):
            v_cmd = min(v_cmd, 0.06 * slow)
            if AVOID_ALLOW_REVERSE and min_front < (AVOID_COLLISION_DIST + 0.10):
                v_cmd = float(AVOID_REVERSE_V)

        return v_cmd, w_cmd, alpha

    def step(self, action: np.ndarray):
        self._cached_bins = None
        self._cached_state = None

        if self._success_latched and STOP_ON_SUCCESS:
            self.ros.send_cmd(0.0, 0.0)
            obs = self._build_observation()
            terms = {"progress": 0.0, "alignment": 0.0, "movement": 0.0, "obstacle": 0.0,
                     "step_cost": 0.0, "goal": 0.0, "fail": 0.0}
            return obs, 0.0, True, False, {"reward_terms": terms, "collision": False, "goal_dist": self._goal_distance()}

        a = np.array(action, dtype=np.float32).reshape(2,)
        a = np.clip(a, -1.0, 1.0)

        v_cmd = float(a[0]) * self.v_max
        w_cmd = float(a[1]) * self.w_max

        if v_cmd < self.v_min_reverse:
            v_cmd = self.v_min_reverse

        # Lidar-aware avoidance (directional)
        alpha = 0.0
        min_front = min_left = min_right = min_all_dir = None

        if AVOID_ENABLE:
            v_avoid, w_avoid, alpha = self._avoidance_action(v_cmd, w_cmd)

            if AVOID_MODE == "override" and alpha > 0.0:
                v_cmd, w_cmd = v_avoid, w_avoid
            elif AVOID_MODE == "blend" and alpha > 0.0:
                v_cmd = (1.0 - alpha) * v_cmd + alpha * v_avoid
                w_cmd = (1.0 - alpha) * w_cmd + alpha * w_avoid

            if AVOID_DEBUG and (self.step_count % max(1, int(AVOID_DEBUG_EVERY_N_STEPS)) == 0):
                min_front, min_left, min_right, min_all_dir = self._lidar_sectors()
                self.ros.get_logger().info(
                    f"[AVOIDDBG] a={alpha:.2f} front={min_front:.2f} left={min_left:.2f} right={min_right:.2f} "
                    f"all={min_all_dir:.2f} cmd(v={v_cmd:.2f}, w={w_cmd:.2f}) goal_ang={self._goal_angle_relative():+.2f}"
                )

        # Extra hard safety: if we're inside collision_dist, stop now
        min_all_bins = self._min_all_lidar()
        if min_all_bins < self.collision_dist:
            v_cmd = 0.0
            w_cmd = 0.0

        self.ros.send_cmd(v_cmd, w_cmd)

        t_end = time.time() + self.control_dt
        while time.time() < t_end:
            rclpy.spin_once(self.ros, timeout_sec=0.01)

        obs = self._build_observation()
        reward, terminated, collided, terms = self._compute_reward_and_done(v_cmd, w_cmd)

        if STOP_ON_SUCCESS and terms.get("goal", 0.0) > 0.0:
            self._success_latched = True
            self.ros.send_cmd(0.0, 0.0)
            self.ros.brake_hold(SUCCESS_BRAKE_HOLD_SECONDS)

        self.episode_return += float(reward)
        self.ros.reward_pub.publish(Float32(data=float(reward)))

        st = self._get_robot_state()
        goal = self.ros.last_goal
        breakdown = {
            "rewards": {
                "progress": float(terms["progress"]),
                "movement": float(terms["movement"]),
                "alignment": float(terms["alignment"]),
                "slowness": float(terms["step_cost"]),
                "obstacle": float(terms["obstacle"]),
                "goal": float(terms["goal"]),
                "fail": float(terms["fail"]),
                "total": float(reward),
            },
            "state": {
                "x": float(st["x"]),
                "y": float(st["y"]),
                "yaw_deg": float(st["yaw"] * 180.0 / np.pi),
                "v_lin": float(st["v_lin"]),
                "v_ang": float(st["v_ang"]),
                "goal_x": float(goal.point.x) if goal else 0.0,
                "goal_y": float(goal.point.y) if goal else 0.0,
                "goal_distance": float(self._goal_distance()),
                "goal_angle": float(self._goal_angle_relative() * 180.0 / np.pi),
                "min_all": float(self._min_all_lidar()),
            },
            "episode": int(self.episode_index),
            "step": int(self.step_count),
            "episode_return": float(self.episode_return),
            "terminated": bool(terminated),
            "collided": bool(collided),
        }
        msg = StringMsg()
        msg.data = json.dumps(breakdown)
        self.ros.reward_breakdown_pub.publish(msg)

        self.prev_action[:] = a
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        if terminated or truncated:
            success = bool(terms["goal"] > 0.0)
            self.recent_success.append(1 if success else 0)
            self._update_curriculum()

            status = "✓ SUCCESS" if success else ("✗ COLLISION" if collided else "⊗ TIMEOUT")
            sr = float(np.mean(self.recent_success)) if len(self.recent_success) else 0.0
            self.ros.get_logger().info(
                f"[EP {self.episode_index:04d}] {status} | "
                f"Return {self.episode_return:+8.1f} | Steps {self.step_count:4d} | "
                f"Dist {self._goal_distance():.2f}m | SR({len(self.recent_success)}): {sr:.0%} | "
                f"CurrR {self.curr_radius:.1f}m"
            )
            self.episode_index += 1

        info = {"collision": bool(collided), "goal_dist": float(self._goal_distance()), "reward_terms": terms}
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _update_curriculum(self):
        if not self.use_curriculum:
            return
        if len(self.recent_success) < max(10, self.curriculum.window // 5):
            return
        sr = float(np.mean(self.recent_success))
        if sr >= self.curriculum.promote_threshold and self.curr_radius < self.curriculum.max_radius:
            self.curr_radius = min(self.curriculum.max_radius, self.curr_radius + self.curriculum.step_up)
        elif sr <= self.curriculum.demote_threshold and self.curr_radius > self.curriculum.start_radius:
            self.curr_radius = max(self.curriculum.start_radius, self.curr_radius - self.curriculum.step_down)

    def _set_new_goal(self, first_time: bool):
        if self.use_curriculum:
            r_max = self.curr_radius
            r_min = max(0.5, 0.7 * r_max)
            ang = np.random.uniform(0.0, 2.0 * np.pi)
            dist = np.random.uniform(r_min, r_max)
            gx = dist * math.cos(ang)
            gy = dist * math.sin(ang)
        else:
            gx, gy = self.fixed_goal_x, self.fixed_goal_y

        self.ros.publish_goal(gx, gy, frame_id="odom")
        rclpy.spin_once(self.ros, timeout_sec=0.05)
        self.ros.get_logger().info(f"[GOAL] {'Init ' if first_time else ''}Goal ({gx:.2f}, {gy:.2f})")

    def _get_robot_state(self) -> Dict[str, float]:
        odom = self.ros.last_odom
        if odom is None:
            return {"x": 0.0, "y": 0.0, "yaw": 0.0, "v_lin": 0.0, "v_ang": 0.0}
        q = odom.pose.pose.orientation
        return {
            "x": float(odom.pose.pose.position.x),
            "y": float(odom.pose.pose.position.y),
            "yaw": float(yaw_from_quat(q.x, q.y, q.z, q.w)),
            "v_lin": float(odom.twist.twist.linear.x),
            "v_ang": float(odom.twist.twist.angular.z),
        }

    def _lidar_to_bins(self, scan: LaserScan) -> np.ndarray:
        ranges = np.array(scan.ranges, dtype=np.float32)
        max_r = scan.range_max if scan.range_max > 0 else self.lidar_max_range
        min_r = scan.range_min if scan.range_min > 0 else 0.01
        if ranges.size == 0:
            return np.full(self.num_lidar_bins, max_r, dtype=np.float32)
        bad = np.isnan(ranges) | np.isinf(ranges) | (ranges < min_r) | (ranges > max_r)
        ranges[bad] = max_r
        n = ranges.size
        bin_idx = (np.arange(n) * self.num_lidar_bins // n).astype(int)
        bins = np.full(self.num_lidar_bins, max_r, dtype=np.float32)
        for i in range(self.num_lidar_bins):
            m = bin_idx == i
            if np.any(m):
                bins[i] = float(np.min(ranges[m]))
        return bins

    def _get_lidar_bins(self) -> np.ndarray:
        scan = self.ros.last_scan
        if scan is None:
            return np.full(self.num_lidar_bins, self.lidar_max_range, dtype=np.float32)
        return self._lidar_to_bins(scan)

    def _goal_distance(self) -> float:
        goal = self.ros.last_goal
        if goal is None:
            return 0.0
        st = self._get_robot_state()
        dx = float(goal.point.x) - st["x"]
        dy = float(goal.point.y) - st["y"]
        return float(math.hypot(dx, dy))

    def _goal_angle_relative(self) -> float:
        goal = self.ros.last_goal
        if goal is None:
            return 0.0
        st = self._get_robot_state()
        dx = float(goal.point.x) - st["x"]
        dy = float(goal.point.y) - st["y"]
        ang_world = math.atan2(dy, dx)
        return wrap_to_pi(ang_world - st["yaw"])

    def _min_all_lidar(self) -> float:
        bins = self._get_lidar_bins()
        return float(max(0.01, np.min(bins)))

    def _build_observation(self) -> np.ndarray:
        goal = self.ros.last_goal
        odom = self.ros.last_odom
        if goal is None or odom is None:
            rclpy.spin_once(self.ros, timeout_sec=0.05)
            goal = self.ros.last_goal
            odom = self.ros.last_odom
        if goal is None or odom is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        bins = self._get_lidar_bins()
        lidar_norm = np.clip(bins / self.lidar_max_range, 0.0, 1.0)

        st = self._get_robot_state()
        dx = float(goal.point.x) - st["x"]
        dy = float(goal.point.y) - st["y"]
        dist = math.hypot(dx, dy)
        ang = self._goal_angle_relative()

        cap = 6.0
        dx_n = float(np.clip(dx / cap, -1.0, 1.0))
        dy_n = float(np.clip(dy / cap, -1.0, 1.0))
        dist_n = float(np.clip(dist / cap, 0.0, 1.0))

        sin_a = math.sin(ang)
        cos_a = math.cos(ang)

        v_n = float(np.clip(st["v_lin"] / max(1e-3, self.v_max), -1.0, 1.0))
        w_n = float(np.clip(st["v_ang"] / max(1e-3, self.w_max), -1.0, 1.0))

        obs = np.concatenate(
            [
                lidar_norm.astype(np.float32),
                np.array([dx_n, dy_n, dist_n, sin_a, cos_a, v_n, w_n,
                          float(self.prev_action[0]), float(self.prev_action[1])], dtype=np.float32),
            ],
            axis=0,
        )
        return obs.astype(np.float32)

    def _compute_reward_and_done(self, v_cmd: float, w_cmd: float) -> Tuple[float, bool, bool, dict]:
        terminated = False
        collided = False

        d_goal = self._goal_distance()
        ang = self._goal_angle_relative()
        min_all = self._min_all_lidar()
        st = self._get_robot_state()

        raw_progress = self.prev_goal_dist - d_goal
        r_progress = PROGRESS_SCALE * float(raw_progress)
        r_dist = -DIST_SHAPING_K * float(d_goal)

        r_align = ALIGN_W * float(math.cos(ang))

        v_norm = float(np.clip(v_cmd / max(1e-3, self.v_max), -1.0, 1.0))
        heading_gate = float(math.cos(ang))

        r_move = 0.0
        if abs(ang) <= math.radians(FORWARD_HEADING_GATE_DEG):
            r_move += FORWARD_WHEN_ALIGNED_BONUS * max(0.0, v_norm) * max(0.0, heading_gate)
        else:
            r_move -= FORWARD_WHEN_MISALIGNED_PENALTY * max(0.0, v_norm) * max(0.0, -heading_gate)

        r_finish = 0.0
        r_turn_to_goal = 0.0
        w_abs_norm = float(np.clip(abs(st["v_ang"]) / max(1e-3, self.w_max), 0.0, 1.0))

        if d_goal <= FINISH_DIST:
            r_finish += FINISH_ALIGN_W * float(math.cos(ang))
            if abs(ang) <= math.radians(FINISH_FORWARD_GATING_DEG):
                r_finish += FINISH_FWD_W * max(0.0, v_norm)
            else:
                desired_sign = 1.0 if ang > 0.0 else -1.0
                yaw_sign = 1.0 if st["v_ang"] > 0.0 else -1.0
                if abs(st["v_ang"]) > 0.20:
                    r_turn_to_goal = TURN_TO_GOAL_BONUS * (1.0 if yaw_sign == desired_sign else -1.0)
            r_finish -= FINISH_TURN_PENALTY_W * w_abs_norm

        r_obstacle = 0.0
        if min_all < self.safe_dist:
            closeness = float((self.safe_dist - min_all) / max(1e-3, self.safe_dist))
            closeness = float(np.clip(closeness, 0.0, 1.0))
            r_obstacle = -OBSTACLE_W * (closeness ** 2)

        r_step = float(STEP_COST)

        r_goal = 0.0
        r_fail = 0.0

        if d_goal <= self.goal_radius:
            terminated = True
            r_goal = self.r_goal_terminal

        if min_all < self.collision_dist:
            terminated = True
            collided = True
            r_fail = self.r_collision_terminal

        timeout = (self.step_count + 1) >= self.max_steps
        if not terminated and timeout:
            terminated = True
            r_fail = self.r_timeout_terminal

        reward = r_progress + r_dist + r_align + r_move + r_finish + r_turn_to_goal + r_obstacle + r_step + r_goal + r_fail
        self.prev_goal_dist = d_goal

        terms = {
            "progress": float(r_progress + r_dist),
            "alignment": float(r_align + r_finish + r_turn_to_goal),
            "movement": float(r_move),
            "obstacle": float(r_obstacle),
            "step_cost": float(r_step),
            "goal": float(r_goal),
            "fail": float(r_fail),
        }
        return float(reward), bool(terminated), bool(collided), terms


# =============================================================================
# TD3 (Pure PyTorch)
# =============================================================================

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))

class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, device: torch.device):
        self.device = device
        self.size = int(size)
        self.ptr = 0
        self.count = 0
        self.obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((self.size, act_dim), dtype=np.float32)
        self.rews = np.zeros((self.size, 1), dtype=np.float32)
        self.done = np.zeros((self.size, 1), dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.count, size=batch_size)
        obs = torch.as_tensor(self.obs[idx], device=self.device)
        acts = torch.as_tensor(self.acts[idx], device=self.device)
        rews = torch.as_tensor(self.rews[idx], device=self.device)
        next_obs = torch.as_tensor(self.next_obs[idx], device=self.device)
        done = torch.as_tensor(self.done[idx], device=self.device)
        return obs, acts, rews, next_obs, done

class TD3Agent:
    def __init__(self, obs_dim: int, act_dim: int, device: torch.device,
                 gamma: float = 0.99, tau: float = 0.005,
                 actor_lr: float = 3e-4, critic_lr: float = 3e-4,
                 policy_noise: float = 0.20, noise_clip: float = 0.50, policy_delay: int = 2):
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_delay = int(policy_delay)

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.actor_targ = Actor(obs_dim, act_dim).to(device)
        self.actor_targ.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(obs_dim, act_dim).to(device)
        self.critic2 = Critic(obs_dim, act_dim).to(device)
        self.critic1_targ = Critic(obs_dim, act_dim).to(device)
        self.critic2_targ = Critic(obs_dim, act_dim).to(device)
        self.critic1_targ.load_state_dict(self.critic1.state_dict())
        self.critic2_targ.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)
        self.total_updates = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        o = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        a = self.actor(o).squeeze(0).cpu().numpy()
        if noise_std > 0:
            a = a + np.random.normal(0.0, noise_std, size=a.shape).astype(np.float32)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def update(self, replay: ReplayBuffer, batch_size: int):
        self.total_updates += 1
        obs, act, rew, next_obs, done = replay.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_act = (self.actor_targ(next_obs) + noise).clamp(-1.0, 1.0)

            q1_t = self.critic1_targ(next_obs, next_act)
            q2_t = self.critic2_targ(next_obs, next_act)
            q_t = torch.min(q1_t, q2_t)
            target = rew + (1.0 - done) * self.gamma * q_t

        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()), 1.0)
        self.critic_opt.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        if self.total_updates % self.policy_delay == 0:
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            self._soft_update(self.actor_targ, self.actor)
            self._soft_update(self.critic1_targ, self.critic1)
            self._soft_update(self.critic2_targ, self.critic2)

        return float(critic_loss.item()), float(actor_loss.item())

    def _soft_update(self, target: nn.Module, source: nn.Module):
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * sp.data)

    def save(self, path: str):
        payload = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "actor_targ": self.actor_targ.state_dict(),
            "critic1_targ": self.critic1_targ.state_dict(),
            "critic2_targ": self.critic2_targ.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "total_updates": self.total_updates,
        }
        torch.save(payload, path)

    def load(self, path: str, strict: bool = True):
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"], strict=strict)
        self.critic1.load_state_dict(payload["critic1"], strict=strict)
        self.critic2.load_state_dict(payload["critic2"], strict=strict)
        self.actor_targ.load_state_dict(payload["actor_targ"], strict=strict)
        self.critic1_targ.load_state_dict(payload["critic1_targ"], strict=strict)
        self.critic2_targ.load_state_dict(payload["critic2_targ"], strict=strict)
        try:
            self.actor_opt.load_state_dict(payload["actor_opt"])
            self.critic_opt.load_state_dict(payload["critic_opt"])
            self.total_updates = int(payload.get("total_updates", 0))
        except Exception:
            pass


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stretch learner (no skrl, compat args)")

    parser.add_argument("--ns", type=str, default="stretch")
    parser.add_argument("--odom-topic", type=str, default="odom")
    parser.add_argument("--lidar-topic", type=str, default="scan")
    parser.add_argument("--imu-topic", type=str, default="imu")
    parser.add_argument("--goal-topic", type=str, default="goal")
    parser.add_argument("--cmd-topic", type=str, default="cmd_vel")

    # legacy args (accepted for RL.launch.py)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--ckpt-dir", type=str, default="")
    parser.add_argument("--load-ckpt", nargs="?", const="", default="")
    parser.add_argument("--use-obstacle", type=int, default=1)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--episode-num", type=int, default=1)

    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--start-steps", type=int, default=DEFAULT_START_STEPS)
    parser.add_argument("--update-after", type=int, default=1_000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=200_000)
    parser.add_argument("--expl-noise", type=float, default=DEFAULT_EXPL_NOISE)
    parser.add_argument("--save-every", type=int, default=10_000)

    args = parser.parse_args()
    set_seed(args.seed)

    mode_infer = bool(args.inference) and (not bool(args.train))
    mode_train = not mode_infer

    if args.checkpoint.strip():
        ckpt_path = os.path.abspath(args.checkpoint.strip())
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    else:
        base_dir = args.ckpt_dir.strip() if args.ckpt_dir.strip() else os.path.join(os.path.abspath(args.models_dir), "current")
        os.makedirs(base_dir, exist_ok=True)
        ckpt_path = os.path.join(base_dir, CHECKPOINT_FILENAME)

    rclpy.init(args=None)
    ros = StretchRosInterfaceNode(
        ns=args.ns,
        odom_topic=args.odom_topic,
        scan_topic=args.lidar_topic,
        imu_topic=args.imu_topic,
        goal_topic=args.goal_topic,
        cmd_topic=args.cmd_topic,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(ros)

    ros.get_logger().info(f"[CKPT] resolved_path={ckpt_path}")

    use_curriculum = (GOAL_MODE.strip().lower() == "curriculum")

    env = StretchRosEnv(
        ros=ros,
        control_dt=0.1,
        episode_time_seconds=EPISODE_SECONDS,
        num_lidar_bins=60,
        lidar_max_range=20.0,
        v_max=1.25,
        w_max=7.0,
        use_curriculum=use_curriculum,
        fixed_goal_x=FIXED_GOAL_X,
        fixed_goal_y=FIXED_GOAL_Y,
        max_goal_radius_for_norm=6.0,
        goal_radius=GOAL_RADIUS,
        collision_dist=DEFAULT_COLLISION_DIST,
        safe_dist=DEFAULT_SAFE_DIST,
        r_goal=R_GOAL,
        r_collision=R_COLLISION,
        r_timeout=(R_TIMEOUT_BOOTSTRAP if BOOTSTRAP_MODE else R_TIMEOUT_NORMAL),
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ros.get_logger().info(f"[AGENT] device={device} obs_dim={obs_dim} act_dim={act_dim}")

    agent = TD3Agent(obs_dim, act_dim, device=device)
    replay = ReplayBuffer(obs_dim, act_dim, size=args.replay_size, device=device)

    should_load = (mode_train and AUTO_LOAD_CHECKPOINT_FOR_TRAINING) or (mode_infer and AUTO_LOAD_CHECKPOINT_FOR_INFERENCE)
    if should_load and os.path.exists(ckpt_path):
        ros.get_logger().info("[CKPT] attempting load...")
        try:
            agent.load(ckpt_path, strict=False)
            ros.get_logger().info("[CKPT] load SUCCESS")
        except Exception as e:
            ros.get_logger().warn(f"[CKPT] load FAILED: {e}")
    else:
        ros.get_logger().info(f"[CKPT] not loading (should_load={should_load})")

    def shutdown_and_save(signum=None, frame=None):
        try:
            ros.send_cmd(0.0, 0.0)
        except Exception:
            pass
        ros.get_logger().info(f"[CKPT] saving to {ckpt_path}")
        try:
            tmp = ckpt_path + ".tmp"
            agent.save(tmp)
            os.replace(tmp, ckpt_path)
            ros.get_logger().info("[CKPT] save SUCCESS")
        except Exception as e:
            ros.get_logger().error(f"[CKPT] save FAILED: {e}")
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            ros.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_and_save)
    signal.signal(signal.SIGTERM, shutdown_and_save)

    if mode_infer:
        ros.get_logger().info("[MODE] INFERENCE")
        while True:
            obs, _ = env.reset()
            done = False
            while not done:
                act = agent.act(obs, noise_std=0.0)
                obs, r, term, trunc, info = env.step(act)
                done = term or trunc

    ros.get_logger().info("[MODE] TRAINING")
    obs, _ = env.reset()
    last_save = 0
    critic_l = 0.0
    actor_l = 0.0

    for t in range(1, int(args.total_steps) + 1):
        if t < args.start_steps:
            ang = env._goal_angle_relative()
            w = np.clip(ang / math.pi, -1.0, 1.0)
            w += np.random.uniform(-0.25, 0.25)
            if abs(ang) > math.radians(90.0):
                v = np.random.uniform(0.0, 0.25)
            elif abs(ang) > math.radians(40.0):
                v = np.random.uniform(0.10, 0.60)
            else:
                v = np.random.uniform(0.40, 1.0)
            act = np.array([v, w], dtype=np.float32)
        else:
            act = agent.act(obs, noise_std=args.expl_noise)

        next_obs, reward, terminated, truncated, info = env.step(act)
        done = terminated or truncated

        replay.add(
            obs, act,
            np.array([reward], dtype=np.float32),
            next_obs,
            np.array([1.0 if done else 0.0], dtype=np.float32),
        )
        obs = next_obs

        if done:
            obs, _ = env.reset()

        if t >= args.update_after and replay.count >= args.batch_size:
            for _ in range(int(args.update_every)):
                critic_l, actor_l = agent.update(replay, int(args.batch_size))

        if t - last_save >= int(args.save_every):
            last_save = t
            ros.get_logger().info(f"[CKPT] periodic save step={t} critic_loss={critic_l:.4f} actor_loss={actor_l:.4f}")
            try:
                tmp = ckpt_path + ".tmp"
                agent.save(tmp)
                os.replace(tmp, ckpt_path)
                ros.get_logger().info("[CKPT] periodic save SUCCESS")
            except Exception as e:
                ros.get_logger().warn(f"[CKPT] periodic save FAILED: {e}")

    shutdown_and_save()


if __name__ == "__main__":
    main()
