#!/usr/bin/env python3

import math
import time
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import rclpy
import torch
import torch.nn as nn
from geometry_msgs.msg import PointStamped, Twist
from gymnasium import spaces
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan

# skrl imports
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from std_msgs.msg import Float32
from std_srvs.srv import Empty

# ----------------------------
# Helpers
# ----------------------------

def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Compute yaw from quaternion (ROS convention)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# ----------------------------
# ROS node wrapper
# ----------------------------

class StretchRosInterface(Node):
    """ROS2 node that holds the latest sensor messages and publishes cmd_vel."""

    def __init__(self):
        super().__init__("stretch_skrl_env_node")

        # Subscriptions
        self.last_odom: Optional[Odometry] = None
        self.last_scan: Optional[LaserScan] = None
        self.last_goal: Optional[PointStamped] = None
        self.last_imu: Optional[Imu] = None

        self.create_subscription(
            Odometry, "/stretch/odom", self._odom_cb, 10
        )
        self.create_subscription(
            LaserScan, "/stretch/scan", self._scan_cb, 10
        )
        self.create_subscription(
            PointStamped, "/stretch/goal", self._goal_cb, 10
        )
        self.create_subscription(
            Imu, "/stretch/imu", self._imu_cb, 10
        )

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, "/stretch/cmd_vel", 10)

        # Reset service client (optional – currently unused)
        self.reset_client = self.create_client(Empty, "reset_sim")
        if not self.reset_client.service_is_ready():
            self.get_logger().warn(
                "[SKRL_ENV] reset_sim service not ready (soft resets only)"
            )

    # --- callbacks ---
    def _odom_cb(self, msg: Odometry):
        self.last_odom = msg

    def _scan_cb(self, msg: LaserScan):
        self.last_scan = msg

    def _goal_cb(self, msg: PointStamped):
        self.last_goal = msg

    def _imu_cb(self, msg: Imu):
        self.last_imu = msg

    # --- helper methods ---
    def wait_for_sensors(self, timeout_sec: float = 5.0) -> bool:
        """Wait for at least one message on odom, scan, goal."""
        start = time.time()
        while time.time() - start < timeout_sec:
            if (
                self.last_odom is not None
                and self.last_scan is not None
                and self.last_goal is not None
            ):
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().warn("[SKRL_ENV] Timeout waiting for odom/scan/goal")
        return False

    def send_cmd(self, v: float, w: float):
        """Publish cmd_vel."""
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def call_reset(self, timeout_sec: float = 2.0) -> bool:
        """Call /reset_sim if available."""
        if not self.reset_client.service_is_ready():
            self.get_logger().warn(
                "[SKRL_ENV] reset_sim service not ready, skipping"
            )
            return False
        req = Empty.Request()
        future = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if future.done():
            self.get_logger().info("[SKRL_ENV] reset_sim service call returned")
            return True
        self.get_logger().warn("[SKRL_ENV] reset_sim service call timed out")
        return False


# ----------------------------
# Gym environment
# ----------------------------

class StretchRosEnv(gym.Env):
    """Gym-style environment that uses ROS2 topics to control the Stretch robot in MuJoCo.

    Observation:
      - 24 lidar bins (min range in each angular sector)
      - goal_dx, goal_dy in odom frame
      - robot linear velocity (x) and angular velocity (z)

    Action:
      - 2D continuous vector in [-1, 1]: [linear_cmd, angular_cmd]
        mapped to:
          linear.x in [-max_lin_vel, max_lin_vel]
          angular.z in [-max_ang_vel, max_ang_vel]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_lin_vel: float = 1.0,
        max_ang_vel: float = 1.0,
        control_dt: float = 0.05,
        num_lidar_bins: int = 24,
        collision_dist: float = 0.3,
        goal_radius: float = 0.3,
    ):
        super().__init__()

        # ROS setup (one node + executor)
        rclpy.init(args=None)
        self.ros_node = StretchRosInterface()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.ros_node)

        # Reward publisher
        self.reward_pub = self.ros_node.create_publisher(
            Float32, "/stretch/reward", 10
        )

        # Env parameters
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.control_dt = control_dt
        self.num_lidar_bins = num_lidar_bins
        self.collision_dist = collision_dist
        self.goal_radius = goal_radius

        # Internal state
        self.prev_goal_dist: float = 0.0
        self.step_count = 0
        self.max_steps_per_episode = int(
            30.0 / control_dt
        )  # ~30 seconds per episode

        # Action: [linear_cmd, angular_cmd] in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: 24 lidar bins + 2 goal + 2 velocities = 28
        obs_dim = self.num_lidar_bins + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    # -------- Gym API methods --------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        # Hard reset removed (no reset_sim service)
        # self.ros_node.call_reset()

        # Stop robot
        self.ros_node.send_cmd(0.0, 0.0)

        # Clear internal counters
        self.step_count = 0

        # Wait for sensors
        self.ros_node.wait_for_sensors(timeout_sec=5.0)

        # Build initial obs
        obs = self._build_observation()
        # Initialize prev_goal_dist for reward shaping
        self.prev_goal_dist = self._goal_distance()

        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        # Clip action
        action = np.clip(
            action, self.action_space.low, self.action_space.high
        )
        v_cmd = float(action[0]) * self.max_lin_vel
        w_cmd = float(action[1]) * self.max_ang_vel

        # Send cmd to ROS
        self.ros_node.send_cmd(v_cmd, w_cmd)

        # Spin ROS a bit to get new data
        t_end = time.time() + self.control_dt
        while time.time() < t_end:
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        # Build obs and compute reward
        obs = self._build_observation()
        reward, terminated, collided = self._compute_reward_done()

        # Publish reward to ROS
        self.reward_pub.publish(Float32(data=float(reward)))

        self.step_count += 1
        truncated = self.step_count >= self.max_steps_per_episode

        info = {
            "collision": collided,
            "goal_dist": self._goal_distance(),
            "heading_error": self._heading_error(),
            "min_front": self._min_front_lidar(),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        self.ros_node.send_cmd(0.0, 0.0)
        self.executor.shutdown()
        self.ros_node.destroy_node()
        rclpy.shutdown()

    # -------- Observation & reward helpers --------

    def _build_observation(self) -> np.ndarray:
        """Build observation vector from latest ROS messages."""
        odom = self.ros_node.last_odom
        scan = self.ros_node.last_scan
        goal = self.ros_node.last_goal

        # Fallbacks if something is missing
        if odom is None or scan is None or goal is None:
            # Try to spin once to update
            rclpy.spin_once(self.ros_node, timeout_sec=0.1)
            odom = self.ros_node.last_odom
            scan = self.ros_node.last_scan
            goal = self.ros_node.last_goal
            if odom is None or scan is None or goal is None:
                # Return zeros if still missing
                return np.zeros(
                    self.observation_space.shape, dtype=np.float32
                )

        # --- lidar bins ---
        lidar_bins = self._lidar_to_bins(scan)

        # --- goal delta in odom frame ---
        gx = goal.point.x
        gy = goal.point.y
        rx = odom.pose.pose.position.x
        ry = odom.pose.pose.position.y
        goal_dx = gx - rx
        goal_dy = gy - ry

        # --- robot velocities (from odom twist) ---
        v_lin = odom.twist.twist.linear.x
        v_ang = odom.twist.twist.angular.z

        obs = np.concatenate(
            [
                lidar_bins,
                np.array(
                    [goal_dx, goal_dy, v_lin, v_ang], dtype=np.float32
                ),
            ],
            axis=0,
        ).astype(np.float32)
        return obs

    def _lidar_to_bins(self, scan: LaserScan) -> np.ndarray:
        """Convert full LaserScan to num_lidar_bins using min range per sector."""
        ranges = np.array(scan.ranges, dtype=np.float32)
        # Replace inf / nan with max range
        max_range = scan.range_max
        ranges = np.nan_to_num(
            ranges, nan=max_range, posinf=max_range, neginf=0.0
        )

        n = len(ranges)
        bins = np.zeros(self.num_lidar_bins, dtype=np.float32)
        # Split indices into equal bins and take min range in each
        for i in range(self.num_lidar_bins):
            start = int(i * n / self.num_lidar_bins)
            end = int((i + 1) * n / self.num_lidar_bins)
            if start >= end:
                bins[i] = max_range
            else:
                bins[i] = float(np.min(ranges[start:end]))
        return bins

    def _goal_distance(self) -> float:
        """Euclidean distance from robot base to goal in odom frame."""
        odom = self.ros_node.last_odom
        goal = self.ros_node.last_goal
        if odom is None or goal is None:
            return 0.0
        gx = goal.point.x
        gy = goal.point.y
        rx = odom.pose.pose.position.x
        ry = odom.pose.pose.position.y
        return float(math.hypot(gx - rx, gy - ry))

    def _heading_error(self) -> float:
        """Heading error (goal heading - robot yaw), wrapped to [-pi, pi]."""
        odom = self.ros_node.last_odom
        goal = self.ros_node.last_goal
        if odom is None or goal is None:
            return 0.0

        # Robot pose
        rx = odom.pose.pose.position.x
        ry = odom.pose.pose.position.y
        q = odom.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        # Goal vector
        gx = goal.point.x
        gy = goal.point.y
        dx = gx - rx
        dy = gy - ry
        goal_yaw = math.atan2(dy, dx)

        # Wrap angle difference to [-pi, pi]
        raw = goal_yaw - yaw
        heading_err = math.atan2(math.sin(raw), math.cos(raw))
        return heading_err

    def _min_front_lidar(self, num_front_bins: int = 4) -> float:
        """Min lidar range in the front sector."""
        scan = self.ros_node.last_scan
        if scan is None:
            return float("inf")
        bins = self._lidar_to_bins(scan)
        center = self.num_lidar_bins // 2
        start = max(0, center - num_front_bins // 2)
        end = min(self.num_lidar_bins, center + num_front_bins // 2)
        if start >= end:
            return float("inf")
        return float(np.min(bins[start:end]))

    def _compute_reward_done(self) -> Tuple[float, bool, bool]:
        """Compute reward, done, and collision flag.

        Shaping terms:
          - progress toward goal (distance decrease)
          - time penalty (encourage faster solutions)
          - heading alignment (face the goal)
          - forward motion toward goal
          - obstacle clearance (safer navigation)
          - collision penalty + goal bonus
        """
        reward = 0.0
        terminated = False
        collided = False

        # --- base terms: distance & progress ---
        d_goal = self._goal_distance()
        progress = self.prev_goal_dist - d_goal  # >0 when moving closer

        # Reward progress
        reward += 2.0 * progress

        # Small time penalty
        reward -= 0.001

        # --- heading alignment (encourage facing the goal) ---
        heading_err = self._heading_error()  # [-pi, pi]
        heading_cos = math.cos(heading_err)  # 1 when perfectly aligned
        # Scale to a modest reward range ~[-0.2, 0.2]
        reward += 0.2 * heading_cos

        # --- forward motion (encourage moving forward TOWARD the goal) ---
        odom = self.ros_node.last_odom
        v_lin = 0.0
        if odom is not None:
            v_lin = odom.twist.twist.linear.x

        # Only reward positive forward speed, scaled by how well we face the goal
        if v_lin > 0.0 and heading_cos > 0.0:
            # Normalize by max_lin_vel so max speed gives ~0.1 reward per step
            forward_term = (v_lin / max(self.max_lin_vel, 1e-6)) * heading_cos
            reward += 0.1 * forward_term

        # --- obstacle clearance (safer navigation) ---
        min_front = self._min_front_lidar()
        # Soft penalty when too close, soft reward for staying comfortably away
        safe_margin = 0.2  # meters beyond collision_dist considered "comfortable"
        if min_front < self.collision_dist + safe_margin:
            # Linearly penalize closeness before collision
            closeness = max(0.0, (self.collision_dist + safe_margin) - min_front)
            reward -= 0.2 * closeness  # up to ~-0.04 if barely out of collision
        else:
            # Small reward for keeping good clearance
            reward += 0.01

        # --- collision penalty and termination ---
        if min_front < self.collision_dist:
            collided = True
            reward -= 5.0
            terminated = True

        # --- goal reached bonus ---
        if d_goal < self.goal_radius:
            reward += 10.0
            terminated = True

        # Update stored distance
        self.prev_goal_dist = d_goal

        return float(reward), terminated, collided


# ----------------------------
# skrl Models (policy & value)
# ----------------------------

class Policy(GaussianMixin, Model):
    """Gaussian policy model for continuous actions."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: str = "sum",
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions,
            clip_log_std,
            min_log_std,
            max_log_std,
            reduction,
        )

        # Simple MLP over observations -> action mean
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
        )
        # Learnable log-std for each action dimension
        self.log_std_parameter = nn.Parameter(
            torch.zeros(self.num_actions)
        )

    def compute(self, inputs, role):
        # inputs["states"] is a tensor of observations
        mean = self.net(inputs["states"])
        # GaussianMixin expects (mean, log_std, extra_dict)
        return mean, self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    """Deterministic value function model."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        # Return value prediction and empty dict
        value = self.net(inputs["states"])
        return value, {}


# ----------------------------
# Simple skrl PPO setup
# ----------------------------

def main():
    # Create ROS-based env and wrap for skrl
    base_env = StretchRosEnv()
    env = wrap_env(base_env)  # adds num_envs, device helpers, etc.

    device = env.device  # use skrl env's device

    # Models for PPO (proper skrl Models)
    policy_model = Policy(
        env.observation_space, env.action_space, device=device
    ).to(device)
    value_model = Value(
        env.observation_space, env.action_space, device=device
    ).to(device)

    models = {
        "policy": policy_model,
        "value": value_model,
    }

    # Memory
    memory = RandomMemory(
        memory_size=2048, num_envs=env.num_envs, device=device
    )

    # PPO config
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo["rollouts"] = 2048
    cfg_ppo["learning_epochs"] = 10
    cfg_ppo["mini_batches"] = 32
    cfg_ppo["discount_factor"] = 0.99
    cfg_ppo["learning_rate"] = 3e-4
    cfg_ppo["grad_norm_clip"] = 0.5
    cfg_ppo["ratio_clip"] = 0.2
    cfg_ppo["entropy_loss_scale"] = 0.0
    cfg_ppo["value_loss_scale"] = 0.5
    cfg_ppo["state_preprocessor"] = None
    cfg_ppo["value_preprocessor"] = None

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg_ppo,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # Trainer config
    cfg_trainer = {
        "timesteps": 200_000,
        "headless": True,
    }

    trainer = SequentialTrainer(
        cfg=cfg_trainer, env=env, agents=agent
    )

    try:
        trainer.train()
    finally:
        env.close()


if __name__ == "__main__":
    main()
