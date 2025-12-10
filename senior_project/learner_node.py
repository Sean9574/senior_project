import math
import os
import signal
import sys
import time
from collections import deque
from typing import Optional, Tuple

import gymnasium as gym
import mujoco as mj
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
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Compute yaw from quaternion (ROS convention)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class StretchRosInterface(Node):
    """ROS2 node that holds the latest sensor messages and publishes cmd_vel."""

    def __init__(self, use_curriculum: bool = True):
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

        # Curriculum learning settings
        self.use_curriculum = use_curriculum
        self.get_logger().info(
            f"[GOAL] Using {'CURRICULUM (random goals)' if use_curriculum else 'FIXED goal'}"
        )

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
            if self.last_odom is not None and self.last_scan is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().warn("[SKRL_ENV] Timeout waiting for initial sensors")
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

    def sample_random_goal(self) -> Tuple[float, float]:
        """Sample random goal position for curriculum learning."""
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(2.0, 5.0)
        goal_x = distance * np.cos(angle)
        goal_y = distance * np.sin(angle)
        return goal_x, goal_y

    def publish_goal(self, goal_x: float, goal_y: float):
        """Publish goal position (can be fixed or random)."""
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


class StretchRosEnv(gym.Env):
    """
    Gym-style environment for Stretch robot training.
    
    NOTE: This environment does NOT handle resets!
    The hard_reset.sh bash script will kill and restart this entire process,
    giving us a guaranteed fresh simulation state every episode.
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
        episode_time_seconds: float = 30.0,
        use_curriculum: bool = True,
        fixed_goal_x: float = 3.0,
        fixed_goal_y: float = 0.0,
    ):
        super().__init__()

        rclpy.init(args=None)
        self.ros_node = StretchRosInterface(use_curriculum=use_curriculum)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.ros_node)

        self.reward_pub = self.ros_node.create_publisher(
            Float32, "/stretch/reward", 10
        )

        # Get episode number from environment variable (set by hard_reset.sh)
        self.episode_num = int(os.environ.get('RL_EPISODE_NUM', 1))
        self.models_dir = os.environ.get('RL_MODELS_DIR', './models')
        self.checkpoint_path = os.path.join(self.models_dir, 'current')

        self.ros_node.get_logger().info(
            f"[INIT] Starting Episode {self.episode_num}"
        )
        self.ros_node.get_logger().info(
            f"[INIT] Checkpoint directory: {self.models_dir}"
        )

        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.control_dt = control_dt
        self.num_lidar_bins = num_lidar_bins
        self.collision_dist = collision_dist
        self.goal_radius = goal_radius
        self.use_curriculum = use_curriculum
        self.fixed_goal_x = fixed_goal_x
        self.fixed_goal_y = fixed_goal_y

        self.prev_goal_dist: float = 0.0
        self.step_count = 0
        self.max_steps_per_episode = int(episode_time_seconds / control_dt)

        # Reward parameters (REBALANCED)
        self.alpha_target = 5.0              # Reduced from 20 - progress reward
        self.progress_deadband = 0.01         # Ignore tiny movements
        self.movement_scale = 1.0             # Increased - reward forward motion
        self.obstacle_scale = 5.0             # Reduced from 10 - less harsh
        self.obstacle_gradient_start = 2.0    # Increased from 1.5 - more space
        self.alignment_scale = 2.0            # Increased from 0.5 - face goal!
        self.mu_goal = 200                    # Increased - big success reward
        self.mu_fail = -50                    # Less harsh than -100

        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_index = self.episode_num
        self.total_successes = 0
        self.total_collisions = 0
        self.total_timeouts = 0

        # Track recent outcomes for success rate
        self.recent_outcomes = deque(maxlen=100)

        self.action_space = spaces.Box(
            low=np.array([-max_lin_vel, -max_ang_vel], dtype=np.float32),  # Allow reverse
            high=np.array([max_lin_vel, max_ang_vel], dtype=np.float32),
            dtype=np.float32,
        )

        obs_dim = self.num_lidar_bins + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Initialize on startup
        self.ros_node.get_logger().info("[INIT] Waiting for sensors...")
        self.ros_node.wait_for_sensors(timeout_sec=10.0)

        # Set initial goal
        if self.use_curriculum:
            goal_x, goal_y = self.ros_node.sample_random_goal()
            self.ros_node.get_logger().info(
                f"[INIT] Random goal: ({goal_x:.2f}, {goal_y:.2f})"
            )
        else:
            goal_x, goal_y = self.fixed_goal_x, self.fixed_goal_y
            self.ros_node.get_logger().info(
                f"[INIT] Fixed goal: ({goal_x:.2f}, {goal_y:.2f})"
            )

        self.ros_node.publish_goal(goal_x, goal_y)
        rclpy.spin_once(self.ros_node, timeout_sec=0.05)
        
        self.prev_goal_dist = self._goal_distance()
        self.ros_node.get_logger().info(
            f"[INIT] ✓ Complete - Initial distance to goal: {self.prev_goal_dist:.2f}m"
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Minimal reset for skrl compatibility.
        
        The bash script handles actual resets by killing/restarting the process.
        This just returns the current observation state and resets counters.
        """
        super().reset(seed=seed)
        
        # Reset episode counters
        self.step_count = 0
        self.episode_return = 0.0
        self.episode_length = 0
        
        # Just return current state - no actual reset performed
        obs = self._build_observation()
        self.prev_goal_dist = self._goal_distance()
        info = {"goal_distance": self.prev_goal_dist}
        
        return obs, info

    def step(self, action: np.ndarray):
        # Clip to action space bounds (safety, but RL learns the velocities directly)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        v_cmd = float(action[0])  # Direct velocity (no scaling!)
        w_cmd = float(action[1])  # Direct angular velocity (no scaling!)

        # Debug: Log actions periodically
        if self.step_count % 100 == 0:
            self.ros_node.get_logger().info(
                f"[DEBUG] Action: v={v_cmd:.3f} m/s, w={w_cmd:.3f} rad/s"
            )

        # Emergency stop check
        if self._check_emergency_stop():
            self.ros_node.get_logger().error("[EMERGENCY] Stop triggered!")
            v_cmd, w_cmd = 0.0, 0.0

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
            success = terms["goal"] != 0.0
            self.recent_outcomes.append(1 if success else 0)
            
            success_rate = (
                np.mean(self.recent_outcomes) 
                if len(self.recent_outcomes) > 0 
                else 0.0
            )
            
            status = (
                "✓ SUCCESS"
                if success
                else ("✗ COLLISION" if collided else "⊗ TIMEOUT")
            )
            self.ros_node.get_logger().info(
                f"[EP {self.episode_index:04d}] {status} | "
                f"Return: {self.episode_return:+7.1f} | "
                f"Steps: {self.episode_length:3d} | "
                f"Final Dist: {self._goal_distance():.2f}m | "
                f"S/C/T: {self.total_successes}/"
                f"{self.total_collisions}/{self.total_timeouts} | "
                f"Success Rate: {success_rate:.1%}"
            )

            self.episode_index += 1
            self.episode_return = 0.0
            self.episode_length = 0

        info = {
            "collision": collided,
            "goal_dist": self._goal_distance(),
            "min_front": self._min_front_lidar(),
            "reward_terms": terms,
            "success_rate": (
                np.mean(self.recent_outcomes) 
                if len(self.recent_outcomes) > 0 
                else 0.0
            ),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        """Save checkpoint before being killed by bash script."""
        try:
            self.ros_node.get_logger().info("[CLOSE] Saving checkpoint...")
            # The agent will save its checkpoint in main()
            if rclpy.ok():
                self.ros_node.send_cmd(0.0, 0.0)
        except Exception:
            pass
        try:
            if hasattr(self, "executor"):
                self.executor.shutdown()
        except Exception:
            pass
        try:
            if hasattr(self, "ros_node"):
                self.ros_node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    def _check_emergency_stop(self) -> bool:
        """Check for emergency conditions (e.g., tip-over)."""
        imu = self.ros_node.last_imu
        if imu and abs(imu.linear_acceleration.z) > 20.0:
            return True
        return False

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
        goal_angle_relative = math.atan2(
            math.sin(goal_angle_relative), math.cos(goal_angle_relative)
        )

        # Velocities
        v_lin = odom.twist.twist.linear.x
        v_ang = odom.twist.twist.angular.z

        obs = np.concatenate(
            [
                lidar_bins,
                np.array(
                    [
                        goal_dx,
                        goal_dy,
                        goal_distance,
                        goal_angle_relative,
                        v_lin,
                        v_ang,
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        ).astype(np.float32)
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

    def _goal_angle_relative(self) -> float:
        """Get relative angle to goal."""
        odom = self.ros_node.last_odom
        goal = self.ros_node.last_goal
        if odom is None or goal is None:
            return 0.0
        
        gx = goal.point.x
        gy = goal.point.y
        rx = odom.pose.pose.position.x
        ry = odom.pose.pose.position.y
        
        q = odom.pose.pose.orientation
        robot_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        
        goal_angle_world = math.atan2(gy - ry, gx - rx)
        goal_angle_relative = goal_angle_world - robot_yaw
        goal_angle_relative = math.atan2(
            math.sin(goal_angle_relative), math.cos(goal_angle_relative)
        )
        
        return goal_angle_relative

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

    def _get_front_lidar_bins(self, num_front_bins: int = 8) -> str:
        """Get string representation of front lidar bins for debugging."""
        scan = self.ros_node.last_scan
        if scan is None:
            return "no_scan"
        bins = self._lidar_to_bins(scan)
        center = self.num_lidar_bins // 2
        start = max(0, center - num_front_bins // 2)
        end = min(self.num_lidar_bins, center + num_front_bins // 2)
        front_bins = bins[start:end]
        return "[" + ", ".join([f"{x:.2f}" for x in front_bins]) + "]"

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

        # Progress reward
        raw_progress = self.prev_goal_dist - d_goal
        if abs(raw_progress) > self.progress_deadband:
            r_progress = self.alpha_target * raw_progress
        else:
            r_progress = 0.0

        # Movement reward
        r_movement = 0.0
        if odom is not None:
            v_lin = odom.twist.twist.linear.x
            # Strongly reward forward motion, heavily penalize backward
            if v_lin > 0.01:  # Moving forward
                if min_front > 0.5:
                    r_movement = self.movement_scale * v_lin
                elif min_front < 0.4:
                    r_movement = -self.movement_scale * v_lin
            elif v_lin < -0.01:  # Moving backward - VERY STRONGLY DISCOURAGE THIS
                r_movement = -10.0 * self.movement_scale * abs(v_lin)  # Increased from -5.0
                # Debug: Log when going backward
                if self.step_count % 50 == 0:  # Every 50 steps
                    self.ros_node.get_logger().info(
                        f"[DEBUG] Going BACKWARD: v_lin={v_lin:.3f}, penalty={r_movement:.2f}"
                    )

        # Alignment reward - IMPORTANT: face the goal!
        r_alignment = 0.0
        goal_angle = self._goal_angle_relative()
        r_alignment = -self.alignment_scale * abs(goal_angle) / np.pi

        # Distance-based shaping reward - encourage getting closer
        r_distance_shaping = 0.0
        if d_goal > 0:
            # Reduced exponential reward (was 10.0, now 3.0)
            r_distance_shaping = 3.0 * math.exp(-d_goal / 3.0)

        # Obstacle avoidance
        r_obstacle = 0.0
        if min_all < self.obstacle_gradient_start:
            closeness = (self.obstacle_gradient_start - min_all) / self.obstacle_gradient_start
            closeness = np.clip(closeness, 0.0, 1.0)
            r_obstacle = -self.obstacle_scale * (closeness ** 3)
        
        if min_front < 0.4:
            r_obstacle -= 5.0 * (0.4 - min_front)

        collision = min_front < self.collision_dist
        timeout = (self.step_count + 1) >= self.max_steps_per_episode

        r_goal = 0.0
        r_fail = 0.0

        if d_goal < self.goal_radius:
            r_goal = self.mu_goal
            terminated = True
            self.total_successes += 1
            self.ros_node.get_logger().info(
                f"[SUCCESS] Reached goal! Distance: {d_goal:.3f}m"
            )

        elif collision or timeout:
            r_fail = self.mu_fail
            terminated = True
            collided = collision

            if collision:
                self.total_collisions += 1
                self.ros_node.get_logger().info(
                    f"[COLLISION] min_front={min_front:.3f} < {self.collision_dist:.3f} | "
                    f"min_all={min_all:.3f} | "
                    f"front_bins: {self._get_front_lidar_bins()}"
                )
            elif timeout:
                self.total_timeouts += 1
                self.ros_node.get_logger().info(
                    f"[TIMEOUT] at step {self.step_count + 1} | "
                    f"min_front={min_front:.3f}"
                )

        reward = r_progress + r_movement + r_alignment + r_distance_shaping + r_obstacle + r_goal + r_fail
        self.prev_goal_dist = d_goal

        terms = {
            "progress": r_progress,
            "movement": r_movement,
            "alignment": r_alignment,
            "distance_shaping": r_distance_shaping,
            "obstacle": r_obstacle,
            "goal": r_goal,
            "fail": r_fail,
        }

        return float(reward), terminated, collided, terms


class Policy(GaussianMixin, Model):
    """Policy network with LayerNorm."""

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
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction
        )

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
        )

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        mean = self.net(inputs["states"])
        return mean, self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    """Value network with LayerNorm."""

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
            nn.Linear(self.num_observations, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        value = self.net(inputs["states"])
        return value, {}


def main():
    # Get episode info from environment (set by hard_reset.sh)
    episode_num = int(os.environ.get('RL_EPISODE_NUM', 1))
    models_dir = os.environ.get('RL_MODELS_DIR', './models')
    checkpoint_path = os.path.join(models_dir, 'current', 'agent.pt')

    print(f"\n{'='*70}")
    print(f"EPISODE {episode_num} - HARD RESET MODE")
    print(f"{'='*70}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"This process will be killed after episode completes!")
    print(f"Neural network weights will persist via checkpoints.")
    print(f"{'='*70}\n")

    print("Initializing environment...")
    base_env = StretchRosEnv(
        max_lin_vel=1.0,
        max_ang_vel=1.0,
        episode_time_seconds=30,
        collision_dist=0.25,
        use_curriculum=False,  # Fixed goal
        fixed_goal_x=2.5,      # Closer goal for learning (was 10.0!)
        fixed_goal_y=0.0,      # Straight ahead
    )
    
    env = wrap_env(base_env)
    device = env.device

    print("Creating neural networks...")
    policy_model = Policy(env.observation_space, env.action_space, device=device).to(
        device
    )
    value_model = Value(env.observation_space, env.action_space, device=device).to(
        device
    )
    models = {"policy": policy_model, "value": value_model}

    memory = RandomMemory(memory_size=2048, num_envs=env.num_envs, device=device)

    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo["rollouts"] = 5000
    cfg_ppo["learning_epochs"] = 15
    cfg_ppo["mini_batches"] = 32
    cfg_ppo["discount_factor"] = 0.99
    cfg_ppo["lambda"] = 0.95
    cfg_ppo["learning_rate"] = 3e-4
    cfg_ppo["grad_norm_clip"] = 0.5
    cfg_ppo["ratio_clip"] = 0.2
    cfg_ppo["entropy_loss_scale"] = 0.1
    cfg_ppo["value_loss_scale"] = 1.0
    cfg_ppo["state_preprocessor"] = None
    cfg_ppo["value_preprocessor"] = None
    
    cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
    cfg_ppo["learning_rate_scheduler_kwargs"] = {
        "kl_threshold": 0.008,
        "kl_factor": 2.0,
    }

    cfg_ppo["experiment"] = {
        "directory": "runs",
        "experiment_name": "stretch_hard_reset",
        "write_interval": 250,
        "checkpoint_interval": 20_000,
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {},
    }

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg_ppo,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # Load checkpoint if it exists (from previous episode)
    checkpoint_dir = os.path.join(models_dir, 'current')
    if os.path.exists(checkpoint_path):
        print(f"\n✓ Found checkpoint: {checkpoint_path}")
        try:
            agent.load(checkpoint_path)
            print(f"✓ Loaded checkpoint from episode {episode_num - 1}")
            print("  Continuing training from previous episode...")
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            print("  Checkpoint may be corrupted, starting fresh...")
            # Delete corrupted checkpoint
            try:
                os.remove(checkpoint_path)
                print(f"  Deleted corrupted checkpoint")
            except:
                pass
    else:
        print(f"\n✓ No checkpoint found at: {checkpoint_path}")
        print("  Starting fresh training...")

    # Signal handler to save before being killed
    def save_and_exit(signum, frame):
        print(f"\n[SIGNAL {signum}] Received kill signal, saving checkpoint...")
        checkpoint_dir = os.path.join(models_dir, 'current')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save to temp file first, then rename (atomic operation)
        temp_path = checkpoint_path + '.tmp'
        try:
            agent.save(temp_path)
            os.rename(temp_path, checkpoint_path)
            print(f"[SAVED] ✓ Checkpoint saved to: {checkpoint_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        env.close()
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, save_and_exit)
    signal.signal(signal.SIGINT, save_and_exit)

    # Train for a reasonable amount given episode will be cut short
    # Episode length is ~180s, with 30s per episode = ~6 episodes
    # Each episode is ~600 steps (30s / 0.05s control_dt)
    cfg_trainer = {"timesteps": 10_000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    print("\n" + "=" * 70)
    print("TRAINING STARTED - HARD RESET MODE")
    print("Key features:")
    print("  • Bash script handles kill/restart (guaranteed clean reset)")
    print("  • Neural network checkpoints loaded/saved each episode")
    print("  • Training progress preserved across episodes")
    print("  • This process will be killed when episode time expires")
    print(f"Checkpoint save location: {checkpoint_path}")
    print("=" * 70 + "\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training stopped by user")
    finally:
        # Save checkpoint before being killed
        checkpoint_dir = os.path.join(models_dir, 'current')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"\n[SAVING] Checkpoint to: {checkpoint_path}")
        # Save to temp file first, then rename (atomic operation)
        temp_path = checkpoint_path + '.tmp'
        try:
            agent.save(temp_path)
            os.rename(temp_path, checkpoint_path)
            print(f"[SAVED] ✓ Checkpoint saved successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        env.close()
        print("\n✓ Episode complete! Bash script will handle restart.")


if __name__ == "__main__":
    main()