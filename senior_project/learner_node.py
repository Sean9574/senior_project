import math
import os
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
from rclpy.time import Time
from sensor_msgs.msg import Imu, LaserScan
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Compute yaw from quaternion (ROS convention)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class StretchRosInterface(Node):
    """ROS2 node that holds the latest sensor messages and publishes cmd_vel."""

    def __init__(self, fixed_goal_x: float = 3.0, fixed_goal_y: float = 0.0):
        super().__init__("skrl")

        self.last_odom: Optional[Odometry] = None
        self.last_scan: Optional[LaserScan] = None
        self.last_goal: Optional[PointStamped] = None
        self.last_imu: Optional[Imu] = None

        self.create_subscription(Odometry, "/stretch/odom", self._odom_cb, 10)
        self.create_subscription(LaserScan, "/stretch/scan", self._scan_cb, 10)
        self.create_subscription(PointStamped, "/stretch/goal", self._goal_cb, 10)
        self.create_subscription(Imu, "/stretch/imu", self._imu_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, "/stretch/cmd_vel", 10)
        self.goal_pub = self.create_publisher(PointStamped, "/stretch/goal", 10)
        self.goal_marker_pub = self.create_publisher(Marker, "/stretch/goal_marker", 10)

        self.reset_client = self.create_client(Trigger, "reset_sim")

        # FIXED GOAL - same position every episode
        self.FIXED_GOAL_X = fixed_goal_x
        self.FIXED_GOAL_Y = fixed_goal_y
        
        self.get_logger().info(f"[GOAL] Using FIXED goal at ({self.FIXED_GOAL_X}, {self.FIXED_GOAL_Y})")

    def _odom_cb(self, msg: Odometry):
        self.last_odom = msg

    def _scan_cb(self, msg: LaserScan):
        self.last_scan = msg

    def _goal_cb(self, msg: PointStamped):
        self.last_goal = msg

    def _imu_cb(self, msg: Imu):
        self.last_imu = msg

    def wait_for_sensors(self, timeout_sec: float = 5.0) -> bool:
        """Initial startup wait."""
        start = time.time()
        while time.time() - start < timeout_sec:
            if (
                self.last_odom is not None
                and self.last_scan is not None
            ):
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().warn("[SKRL_ENV] Timeout waiting for initial sensors")
        return False

    def wait_for_fresh_sensors(self, reset_time_ns: int, timeout_sec: float = 5.0) -> bool:
        """
        CRITICAL FIX: Spin until we get odom and scan messages 
        strictly NEWER than reset_time_ns.
        """
        start = time.time()
        fresh_odom = False
        fresh_scan = False

        while time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # Check Odom freshness
            if self.last_odom is not None:
                msg_time = Time.from_msg(self.last_odom.header.stamp).nanoseconds
                if msg_time > reset_time_ns:
                    fresh_odom = True
            
            # Check Scan freshness
            if self.last_scan is not None:
                msg_time = Time.from_msg(self.last_scan.header.stamp).nanoseconds
                if msg_time > reset_time_ns:
                    fresh_scan = True

            if fresh_odom and fresh_scan:
                return True
                
        self.get_logger().warn(f"[RESET] Timeout waiting for FRESH sensors (>{reset_time_ns})")
        return False

    def send_cmd(self, v: float, w: float):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _get_robot_pose(self) -> Tuple[float, float, float]:
        if self.last_odom is None:
            return 0.0, 0.0, 0.0
        rx = self.last_odom.pose.pose.position.x
        ry = self.last_odom.pose.pose.position.y
        q = self.last_odom.pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        return rx, ry, yaw

    def publish_fixed_goal(self):
        """Publish the FIXED goal position (same every episode)."""
        rx, ry, _ = self._get_robot_pose()
        
        goal_x = self.FIXED_GOAL_X
        goal_y = self.FIXED_GOAL_Y

        goal_msg = PointStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "odom"
        goal_msg.point.x = goal_x
        goal_msg.point.y = goal_y
        goal_msg.point.z = 0.0

        self.goal_pub.publish(goal_msg)
        self.last_goal = goal_msg
        self._publish_goal_marker(goal_x, goal_y)

    def _publish_goal_marker(self, goal_x: float, goal_y: float):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "odom"
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = goal_x
        marker.pose.position.y = goal_y
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

    def call_reset(self, timeout_sec: float = 3.0) -> bool:
        if not self.reset_client.service_is_ready():
            self.get_logger().warn("[SKRL_ENV] reset_sim service not ready")
            return False

        try:
            req = Trigger.Request()
            future = self.reset_client.call_async(req)

            start_time = time.time()
            while not future.done():
                if time.time() - start_time > timeout_sec:
                    self.get_logger().warn("[SKRL_ENV] reset_sim service call timed out")
                    return False
                rclpy.spin_once(self, timeout_sec=0.1)

            if future.exception():
                self.get_logger().error(f"[SKRL_ENV] reset_sim failed: {future.exception()}")
                return False

            response = future.result()
            return response.success

        except Exception as e:
            self.get_logger().error(f"[SKRL_ENV] Exception calling reset_sim: {e}")
            return False


class StretchRosEnv(gym.Env):
    """Gym-style environment that uses ROS2 topics to control the Stretch robot."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_lin_vel: float = 1.0,
        max_ang_vel: float = 1.0,
        control_dt: float = 0.05,
        num_lidar_bins: int = 60,
        collision_dist: float = 0.3,
        goal_radius: float = 0.3,
        episode_time_seconds: float = 30.0,
        fixed_goal_x: float = 3.0,
        fixed_goal_y: float = 0.0,
    ):
        super().__init__()

        rclpy.init(args=None)
        self.ros_node = StretchRosInterface(fixed_goal_x=fixed_goal_x, fixed_goal_y=fixed_goal_y)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.ros_node)

        self.reward_pub = self.ros_node.create_publisher(Float32, "/stretch/reward", 10)

        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.control_dt = control_dt
        self.num_lidar_bins = num_lidar_bins
        self.collision_dist = collision_dist
        self.goal_radius = goal_radius

        self.prev_goal_dist: float = 0.0
        self.step_count = 0
        self.max_steps_per_episode = int(episode_time_seconds / control_dt)

        # Reward parameters
        self.alpha_target = 20.0
        self.progress_deadband = 0.001
        self.movement_scale = 0.2
        self.obstacle_scale = 3.0
        self.obstacle_gradient_start = 1.0
        self.mu_goal = 60
        self.mu_fail = -30

        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_index = 0
        self.total_successes = 0
        self.total_collisions = 0
        self.total_timeouts = 0
        self._initialized = False

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        obs_dim = self.num_lidar_bins + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if not self._initialized:
            self.ros_node.get_logger().info("[RESET] First reset - waiting for sensors...")
            self.ros_node.wait_for_sensors(timeout_sec=10.0)
            self._initialized = True

        # 1. Stop Robot before reset (prevents physics glitches)
        self.ros_node.send_cmd(0.0, 0.0)
        for _ in range(5):
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        # 2. Capture Time BEFORE calling reset service
        reset_req_time = self.ros_node.get_clock().now().nanoseconds

        # 3. Call Reset
        reset_success = self.ros_node.call_reset(timeout_sec=5.0)
        if not reset_success:
            self.ros_node.get_logger().warn("[RESET] Service failed. Retrying soft reset...")

        # 4. Wait for FRESH sensors (Timestamp Handshake)
        sensors_ready = self.ros_node.wait_for_fresh_sensors(reset_req_time, timeout_sec=5.0)
        
        if not sensors_ready:
             self.ros_node.get_logger().error("[RESET] FAILED to sync fresh sensors! Observation may be stale.")

        # 5. Publish Goal and set state
        self.ros_node.publish_fixed_goal()
        
        rclpy.spin_once(self.ros_node, timeout_sec=0.05)

        self.step_count = 0
        self.episode_return = 0.0
        self.episode_length = 0

        obs = self._build_observation()
        self.prev_goal_dist = self._goal_distance()

        return obs, {"goal_distance": self.prev_goal_dist}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        v_cmd = float(action[0]) * self.max_lin_vel
        w_cmd = float(action[1]) * self.max_ang_vel

        self.ros_node.send_cmd(v_cmd, w_cmd)

        t_end = time.time() + self.control_dt
        while time.time() < t_end:
            rclpy.spin_once(self.ros_node, timeout_sec=0.01)

        obs = self._build_observation()
        reward, terminated, collided, terms = self._compute_reward_done()

        self.episode_return += float(reward)
        self.episode_length += 1

        self.reward_pub.publish(Float32(data=float(reward)))

        self.step_count += 1
        truncated = self.step_count >= self.max_steps_per_episode

        if terminated or truncated:
            success = (terms["goal"] != 0.0)
            status = "✓ SUCCESS" if success else ("✗ COLLISION" if collided else "⊗ TIMEOUT")
            self.ros_node.get_logger().info(
                f"[EP {self.episode_index:04d}] {status} | "
                f"Return: {self.episode_return:+7.1f} | "
                f"Steps: {self.episode_length:3d} | "
                f"Final Dist: {self._goal_distance():.2f}m | "
                f"S/C/T: {self.total_successes}/{self.total_collisions}/{self.total_timeouts}"
            )

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
        try:
            if rclpy.ok():
                self.ros_node.send_cmd(0.0, 0.0)
        except Exception:
            pass
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown()
        except Exception:
            pass
        try:
            if hasattr(self, 'ros_node'):
                self.ros_node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    def _build_observation(self) -> np.ndarray:
        """Build observation with ENHANCED goal representation."""
        odom = self.ros_node.last_odom
        scan = self.ros_node.last_scan
        goal = self.ros_node.last_goal

        if odom is None or scan is None or goal is None:
            rclpy.spin_once(self.ros_node, timeout_sec=0.1)
            odom = self.ros_node.last_odom
            scan = self.ros_node.last_scan
            goal = self.ros_node.last_goal
            if odom is None or scan is None or goal is None:
                return np.zeros(self.observation_space.shape, dtype=np.float32)

        lidar_bins = self._lidar_to_bins(scan)

        # Goal position
        gx = goal.point.x
        gy = goal.point.y
        rx = odom.pose.pose.position.x
        ry = odom.pose.pose.position.y
        
        # Robot orientation
        q = odom.pose.pose.orientation
        robot_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        
        # Goal in Cartesian coordinates (relative to robot)
        goal_dx = gx - rx
        goal_dy = gy - ry
        
        # Goal in Polar coordinates
        goal_distance = math.hypot(goal_dx, goal_dy)
        goal_angle_world = math.atan2(goal_dy, goal_dx)
        goal_angle_relative = goal_angle_world - robot_yaw
        # Normalize angle to [-pi, pi]
        goal_angle_relative = math.atan2(math.sin(goal_angle_relative), math.cos(goal_angle_relative))

        # Velocities
        v_lin = odom.twist.twist.linear.x
        v_ang = odom.twist.twist.angular.z

        # ENHANCED OBSERVATION: 60 lidar + 6 goal features
        obs = np.concatenate([
            lidar_bins,  # 60 values
            np.array([
                goal_dx,              # Cartesian X offset
                goal_dy,              # Cartesian Y offset  
                goal_distance,        # Polar distance
                goal_angle_relative,  # Polar angle (relative to robot heading)
                v_lin,                # Linear velocity
                v_ang                 # Angular velocity
            ], dtype=np.float32),
        ], axis=0).astype(np.float32)
        return obs

    def _lidar_to_bins(self, scan: LaserScan) -> np.ndarray:
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

    def _min_all_lidar(self) -> float:
        scan = self.ros_node.last_scan
        if scan is None:
            return float("inf")
        bins = self._lidar_to_bins(scan)
        return float(np.min(bins))

    def _compute_reward_done(self) -> Tuple[float, bool, bool, dict]:
        terminated = False
        collided = False

        d_goal = self._goal_distance()
        min_front = self._min_front_lidar()
        min_all = self._min_all_lidar()
        odom = self.ros_node.last_odom

        # Progress reward - main driving signal
        raw_progress = self.prev_goal_dist - d_goal
        if abs(raw_progress) > self.progress_deadband:
            r_progress = self.alpha_target * raw_progress
        else:
            r_progress = 0.0

        # Movement reward - encourage forward motion
        r_movement = 0.0
        if odom is not None:
            v_lin = odom.twist.twist.linear.x
            if v_lin > 0.01:
                r_movement = self.movement_scale * v_lin

        # Obstacle avoidance - gradient penalty
        r_obstacle = 0.0
        if min_all < self.obstacle_gradient_start:
            closeness = (self.obstacle_gradient_start - min_all) / (self.obstacle_gradient_start - self.collision_dist)
            closeness = np.clip(closeness, 0.0, 1.0)
            r_obstacle = -self.obstacle_scale * (closeness ** 2)

        # Terminal conditions
        collision = min_front < self.collision_dist
        timeout = (self.step_count + 1) >= self.max_steps_per_episode

        r_goal = 0.0
        r_fail = 0.0

        if d_goal < self.goal_radius:
            r_goal = self.mu_goal
            terminated = True
            self.total_successes += 1
            self.ros_node.get_logger().info(f"[SUCCESS] Reached goal! Distance: {d_goal:.3f}m")

        elif collision or timeout:
            r_fail = self.mu_fail
            terminated = True
            collided = collision

            if collision:
                self.total_collisions += 1
                self.ros_node.get_logger().info(
                    f"[COLLISION] min_front={min_front:.3f} < {self.collision_dist:.3f}"
                )
            elif timeout:
                self.total_timeouts += 1
                self.ros_node.get_logger().info(f"[TIMEOUT] at step {self.step_count + 1}")

        # Total reward - REMOVED conflicting distance reward
        reward = r_progress + r_movement + r_obstacle + r_goal + r_fail
        self.prev_goal_dist = d_goal

        terms = {
            "progress": r_progress,
            "movement": r_movement,
            "obstacle": r_obstacle,
            "goal": r_goal,
            "fail": r_fail,
        }

        return float(reward), terminated, collided, terms


class Policy(GaussianMixin, Model):
    """Policy network - 66 inputs (60 lidar + 6 goal features)."""
    
    def __init__(
        self, observation_space, action_space, device,
        clip_actions: bool = False, clip_log_std: bool = True,
        min_log_std: float = -20, max_log_std: float = 2,
        reduction: str = "sum",
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
        )
        
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        mean = self.net(inputs["states"])
        return mean, self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    """Value network - 66 inputs (60 lidar + 6 goal features)."""
    
    def __init__(self, observation_space, action_space, device, clip_actions: bool = False):
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


def main():
  
    print("\nInitializing environment...")
    base_env = StretchRosEnv(
        max_lin_vel=1.0, 
        episode_time_seconds=15,
        fixed_goal_x=3.0,
        fixed_goal_y=0.0,
    )
    env = wrap_env(base_env)
    device = env.device

    print("Creating neural networks...")
    policy_model = Policy(env.observation_space, env.action_space, device=device).to(device)
    value_model = Value(env.observation_space, env.action_space, device=device).to(device)
    models = {"policy": policy_model, "value": value_model}

    memory = RandomMemory(memory_size=2048, num_envs=env.num_envs, device=device)

    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo["rollouts"] = 2048
    cfg_ppo["learning_epochs"] = 10
    cfg_ppo["mini_batches"] = 32
    cfg_ppo["discount_factor"] = 0.99
    cfg_ppo["learning_rate"] = 3e-4
    cfg_ppo["grad_norm_clip"] = 0.5
    cfg_ppo["ratio_clip"] = 0.2
    cfg_ppo["entropy_loss_scale"] = 0.02
    cfg_ppo["value_loss_scale"] = 0.5
    cfg_ppo["state_preprocessor"] = None
    cfg_ppo["value_preprocessor"] = None
    cfg_ppo["learning_rate_scheduler"] = None
    cfg_ppo["learning_rate_scheduler_kwargs"] = {}

    # FIXED: Same directory for all training runs
    cfg_ppo["experiment"] = {
        "directory": "runs",
        "experiment_name": "stretch_continuous",  # Same name = continuous training
        "write_interval": 0,  # Disable TensorBoard logging
        "checkpoint_interval": 20_000,
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {},
    }

    agent = PPO(
        models=models, memory=memory, cfg=cfg_ppo,
        observation_space=env.observation_space,
        action_space=env.action_space, device=device,
    )

    # Always use the same checkpoint directory
    checkpoints_dir = os.path.join("runs", "stretch_continuous", "checkpoints")
    best_ckpt = os.path.join(checkpoints_dir, "best_agent.pt")
    
    if os.path.exists(best_ckpt):
        print(f"\n✓ Loading checkpoint: {best_ckpt}")
        print("  Continuing previous training...")
        agent.load(best_ckpt)
    else:
        print("\n✓ No checkpoint found")
        print("  Starting fresh training...")

    cfg_trainer = {"timesteps": 200_000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    print("\n" + "="*70)
    print("TRAINING STARTED")
    print("Checkpoints save to: runs/stretch_continuous/checkpoints/")
    print("="*70 + "\n")

    try:
        trainer.train()
    finally:
        env.close()
        print("\n✓ Training complete! Progress saved to checkpoint.")


if __name__ == "__main__":
    main()