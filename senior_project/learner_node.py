#!/usr/bin/env python3

import math
import subprocess
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

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

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
        super().__init__("skrl")

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
        """Publish cmd_vel.

        NOTE: in this setup, forward is -x on the robot.
        So v > 0 => cmd_vel.linear.x < 0 (forward motion).
        """
        msg = Twist()
        msg.linear.x = -float(v)
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
      - num_lidar_bins lidar bins (min range in each angular sector)
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
        num_lidar_bins: int = 60,
        collision_dist: float = 0.3,
        goal_radius: float = 0.3,
        max_goal_distance: float = 5.0,
        min_safe_distance: float = 0.5,
        writer: Optional[SummaryWriter] = None,
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

        # TensorBoard writer (can be None)
        self.writer = writer

        # Env parameters
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.control_dt = control_dt
        self.num_lidar_bins = num_lidar_bins
        self.collision_dist = collision_dist
        self.goal_radius = goal_radius

        # Distance / shaping
        self.max_goal_distance = max_goal_distance
        self.min_safe_distance = max(min_safe_distance, collision_dist + 0.05)

        # RL episode settings
        self.prev_goal_dist: float = 0.0
        self.step_count = 0
        self.max_steps_per_episode = int(
            30.0 / control_dt
        )  # ~30 seconds per episode

        # Progress-based reward parameters (stronger now)
        self.alpha_target = 20.0        # weight for progress term (was 10.0)
        self.progress_deadband = 0.005  # meters

        # Movement reward scaling (forward-only; stronger now)
        self.movement_scale = 0.1       # was 0.05

        # Obstacle gradient punishment
        self.obstacle_scale = 3.0       # per-step penalty when inside min_safe_distance

        # Terminal rewards
        self.mu_goal = 80.0             # success reward (was 50.0)
        self.mu_fail = -60.0            # failure penalty (slightly stronger than before)

        # Episode statistics for summaries
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_index = 0

        # Action: [linear_cmd, angular_cmd] in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: lidar bins + 2 goal + 2 velocities
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

        # Stop robot
        self.ros_node.send_cmd(0.0, 0.0)

        # Reset step counter and episode stats for new episode
        self.step_count = 0
        self.episode_return = 0.0
        self.episode_length = 0

        # Wait for sensors
        self.ros_node.wait_for_sensors(timeout_sec=5.0)

        # Build initial obs and distance
        obs = self._build_observation()
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

        # Build obs and compute reward + breakdown
        obs = self._build_observation()
        reward, terminated, collided, terms = self._compute_reward_done()

        # Update per-episode stats
        self.episode_return += float(reward)
        self.episode_length += 1

        # Publish reward for rqt_plot / logging
        self.reward_pub.publish(Float32(data=float(reward)))

        # Per-step reward log (compact)
        self.ros_node.get_logger().info(
            "[REWARD] total={:+6.2f} | prog={:+6.2f} move={:+6.2f} "
            "obs={:+6.2f} goal={:+6.2f} fail={:+6.2f}"
            .format(
                reward,
                terms["progress"],
                terms["movement"],
                terms["obstacle"],
                terms["goal"],
                terms["fail"],
            )
        )

        self.step_count += 1
        truncated = self.step_count >= self.max_steps_per_episode

        # If episode ended, log a summary and send to TensorBoard
        if terminated or truncated:
            success = (terms["goal"] != 0.0)

            self.ros_node.get_logger().info(
                "[EPISODE] idx={} return={:+7.2f} length={} success={}".format(
                    self.episode_index,
                    self.episode_return,
                    self.episode_length,
                    success,
                )
            )

            if self.writer is not None:
                self.writer.add_scalar(
                    "Episode/Return", self.episode_return, self.episode_index
                )
                self.writer.add_scalar(
                    "Episode/Length", self.episode_length, self.episode_index
                )
                self.writer.add_scalar(
                    "Episode/Success",
                    1.0 if success else 0.0,
                    self.episode_index,
                )

            # Prepare for next episode stats
            self.episode_index += 1
            self.episode_return = 0.0
            self.episode_length = 0

        info = {
            "collision": collided,
            "goal_dist": self._goal_distance(),
            "min_front": self._min_front_lidar(),
            "reward_terms": terms,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        self.ros_node.send_cmd(0.0, 0.0)
        self.executor.shutdown()
        self.ros_node.destroy_node()
        rclpy.shutdown()
        if self.writer is not None:
            self.writer.close()

    # -------- Observation helpers --------

    def _build_observation(self) -> np.ndarray:
        """Build observation vector from latest ROS messages."""
        odom = self.ros_node.last_odom
        scan = self.ros_node.last_scan
        goal = self.ros_node.last_goal

        if odom is None or scan is None or goal is None:
            rclpy.spin_once(self.ros_node, timeout_sec=0.1)
            odom = self.ros_node.last_odom
            scan = self.ros_node.last_scan
            goal = self.ros_node.last_goal
            if odom is None or scan is None or goal is None:
                return np.zeros(
                    self.observation_space.shape, dtype=np.float32
                )

        lidar_bins = self._lidar_to_bins(scan)

        gx = goal.point.x
        gy = goal.point.y
        rx = odom.pose.pose.position.x
        ry = odom.pose.pose.position.y
        goal_dx = gx - rx
        goal_dy = gy - ry

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
        """Convert full LaserScan to num_lidar_bins using min range per sector.

        IMPORTANT: treat 0 / NaN / inf as 'no return', not as collision.
        """
        ranges = np.array(scan.ranges, dtype=np.float32)
        max_range = scan.range_max

        bad = np.isnan(ranges) | np.isinf(ranges) | (ranges <= 0.0)
        ranges[bad] = max_range

        n = len(ranges)
        bins = np.zeros(self.num_lidar_bins, dtype=np.float32)
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

    # -------- Progress + movement + obstacle reward --------

    def _compute_reward_done(self) -> Tuple[float, bool, bool, dict]:
        """Reward based on:
           - positive progress toward goal
           - positive reward for forward motion
           - gradient negative reward near obstacles
           - big bonus on goal
           - big penalty on collision or timeout
        """
        terminated = False
        collided = False

        d_goal = self._goal_distance()
        min_front = self._min_front_lidar()
        odom = self.ros_node.last_odom
        goal = self.ros_node.last_goal

        # Debug: print robot and goal pose + distance
        if odom is not None and goal is not None:
            rx = odom.pose.pose.position.x
            ry = odom.pose.pose.position.y
            gx = goal.point.x
            gy = goal.point.y
            self.ros_node.get_logger().info(
                f"[GOAL_DEBUG] robot=({rx:.2f}, {ry:.2f})  "
                f"goal=({gx:.2f}, {gy:.2f})  d_goal={d_goal:.3f}"
            )

        # --- PROGRESS reward ---
        raw_progress = self.prev_goal_dist - d_goal  # >0 closer, <0 farther
        if abs(raw_progress) > self.progress_deadband:
            r_progress = self.alpha_target * raw_progress
        else:
            r_progress = 0.0

        # --- MOVEMENT reward (encourage forward motion; forward is -x) ---
        r_movement = 0.0
        if odom is not None:
            v_lin = odom.twist.twist.linear.x  # forward is negative
            if v_lin < -0.01:  # moving forward
                r_movement = self.movement_scale * (-v_lin)

        # --- OBSTACLE gradient punishment ---
        r_obstacle = 0.0
        if min_front < self.min_safe_distance:
            # closeness in [0, min_safe_distance], normalized to [0,1]
            closeness = max(0.0, self.min_safe_distance - min_front)
            frac = closeness / max(self.min_safe_distance, 1e-6)
            r_obstacle = -self.obstacle_scale * frac

        # --- FAILURE conditions ---
        collision = min_front < self.collision_dist
        timeout = (self.step_count + 1) >= self.max_steps_per_episode

        r_goal = 0.0
        r_fail = 0.0

        if d_goal < self.goal_radius:
            # SUCCESS
            r_goal = self.mu_goal
            terminated = True

        elif collision or timeout:
            # FAILURE
            r_fail = self.mu_fail
            terminated = True
            collided = collision
            if collision:
                self.ros_node.get_logger().info(
                    f"[FAIL] collision: min_front={min_front:.3f} < "
                    f"{self.collision_dist:.3f}"
                )
            elif timeout:
                self.ros_node.get_logger().info(
                    f"[FAIL] timeout at step {self.step_count + 1}"
                )

        # Total reward
        reward = r_progress + r_movement + r_obstacle + r_goal + r_fail

        # Update distance for next step
        self.prev_goal_dist = d_goal

        terms = {
            "progress": r_progress,
            "movement": r_movement,
            "obstacle": r_obstacle,
            "goal": r_goal,
            "fail": r_fail,
        }

        return float(reward), terminated, collided, terms


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

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(
            torch.zeros(self.num_actions)
        )

    def compute(self, inputs, role):
        mean = self.net(inputs["states"])
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
        value = self.net(inputs["states"])
        return value, {}


# ----------------------------
# Simple skrl PPO setup
# ----------------------------

def main():
    # TensorBoard writer with a fresh logdir (new run)
    writer = SummaryWriter(log_dir="runs/stretch_nav_v3")

    # Optionally auto-start TensorBoard on port 6006
    subprocess.Popen(
        ["tensorboard", "--logdir", "runs", "--port", "6006"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print("TensorBoard started at http://localhost:6006 (using runs/stretch_nav_v3)")

    # Create ROS-based env and wrap for skrl
    # Increase speed here via max_lin_vel if you want even more, e.g. 1.5
    base_env = StretchRosEnv(writer=writer, max_lin_vel=1.0)
    env = wrap_env(base_env)

    device = env.device

    # Models for PPO
    policy_model = Policy(
        env.observation_space, env.action_space, device=device
    ).to(device)
    value_model = Value(
        env.observation_space, env.action_space, device=device
    ).to(device)

    models = {"policy": policy_model, "value": value_model}

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
    cfg_ppo["learning_rate"] = 3e-7
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
