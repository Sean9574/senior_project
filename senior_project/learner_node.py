#!/usr/bin/env python3
import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
import torch
import torch.nn as nn
import torch.optim as optim
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, Imu, LaserScan
from std_msgs.msg import Float32
from std_srvs.srv import Empty  # <-- needed for reset services


def yaw_from_quat(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class Config:
    # Namespacing & topics
    ns: str = "stretch"
    odom_topic: str = "odom"
    cmd_topic: str = "cmd_vel"          # will be used as /ns/cmd_topic
    lidar_topic: str = "scan"           # will be used as /ns/lidar_topic
    imu_topic: str = "imu"
    obstacle_topic: str = "obstacle_score"
    use_obstacle: bool = False
    goal_topic: str = "goal"

    # Optional camera input
    use_camera: bool = False
    camera_topic: str = "camera/image_raw"
    cam_width: int = 32
    cam_height: int = 24

    # Control
    control_dt: float = 0.05
    max_lin: float = 0.25
    max_ang: float = 0.8

    # LiDAR
    lidar_min_range: float = 0.05
    lidar_max_range: float = 6.0
    lidar_bins: int = 24
    lidar_front_fov_deg: float = 120.0
    lidar_front_weight: float = 2.0

    # Goal
    goal_x: float = 2.0
    goal_y: float = 0.0
    goal_tol: float = 0.25

    # Reward weights
    w_progress: float = 5.0
    w_change: float = 0.2
    w_speed_pen: float = 0.05
    w_action_pen: float = 0.01
    w_goal_bonus: float = 10.0
    w_lidar_pen: float = 2.5
    w_obstacle_pen: float = 2.0

    # Smoothness
    w_smooth_delta: float = 0.02    # penalty on Δu
    w_smooth_jerk: float = 0.01     # penalty on Δ²u
    w_yaw_accel: float = 0.01       # penalty on Δgz (IMU yaw-rate acceleration)

    # Collision / near-collision
    collision_dist: float = 0.25    # used for "near collision" logging/penalty
    robot_radius_for_lidar: float = 0.30  # (unused now, but kept for compatibility)
    reset_timeout: float = 3.0

    # Exploration reward (reward for visiting new areas)
    explore_grid_size: float = 0.5    # meters per grid cell
    w_explore: float = 0.2            # reward for first time visiting a cell

    # PPO
    total_steps: int = 200_000
    rollout_steps: int = 2048
    gamma: float = 0.995
    lam: float = 0.95
    lr: float = 3e-4
    clip_eps: float = 0.2
    epochs: int = 10
    minibatch_size: int = 256
    vf_coef: float = 0.5
    ent_coef: float = 0.0

    # Checkpoints
    ckpt_dir: str = os.path.expanduser("~/rl_checkpoints")
    load_ckpt: Optional[str] = None
    save_every: int = 50_000  # env steps

    # Episode length
    episode_len: int = 800

    # Evaluation
    eval_every_steps: int = 1200   # ~1 minute at 20 Hz
    eval_episodes: int = 10


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden=(256, 256), act=nn.Tanh):
        super().__init__()
        layers = []
        last = in_size
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(act())
            last = h
        layers.append(nn.Linear(last, out_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.body = MLP(obs_dim, act_dim * 2, hidden=hidden)
        self.act_dim = act_dim

    def forward(self, x):
        out = self.body(x)
        mean, log_std = out[..., : self.act_dim], out[..., self.act_dim :]
        log_std = torch.clamp(log_std, -4.0, 1.0)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, x):
        mean, std = self(x)
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True), mean


class LearnerNode(Node):
    def __init__(self, cfg: Config):
        super().__init__("learner_node")
        self.cfg = cfg

        qos = QoSProfile(depth=10)

        self.reset_client = self.create_client(Empty, f"/{cfg.ns}/reset_sim")
        if not self.reset_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                "reset_sim service not available; will only soft-reset episode"
            )

        self.pub_cmd = self.create_publisher(
            Twist, f"/{cfg.ns}/{cfg.cmd_topic}", qos
        )
        self.sub_odom = self.create_subscription(
            Odometry, f"/{cfg.ns}/{cfg.odom_topic}", self._odom_cb, qos
        )
        self.sub_lidar = self.create_subscription(
            LaserScan, f"/{cfg.ns}/{cfg.lidar_topic}", self._lidar_cb, qos
        )
        self.sub_imu = self.create_subscription(
            Imu, f"/{cfg.ns}/{cfg.imu_topic}", self._imu_cb, qos
        )
        if cfg.use_obstacle:
            self.sub_obs = self.create_subscription(
                Float32, f"/{cfg.ns}/{cfg.obstacle_topic}", self._obstacle_cb, qos
            )

        # Optional camera subscription
        self.bridge: Optional[CvBridge] = None
        cam_dim = 0
        if cfg.use_camera:
            self.bridge = CvBridge()
            self.cam_feat = np.zeros(
                cfg.cam_width * cfg.cam_height, dtype=np.float32
            )
            self.sub_cam = self.create_subscription(
                Image,
                f"/{cfg.ns}/{cfg.camera_topic}",
                self._cam_cb,
                qos,
            )
            cam_dim = cfg.cam_width * cfg.cam_height
        else:
            self.cam_feat = np.zeros(0, dtype=np.float32)

        # Path visualization
        self.path_pub = self.create_publisher(Path, f"/{cfg.ns}/rl_path", qos)
        self.path_msg = Path()

        # Desired straight-line path to goal for visualization
        self.desired_path_pub = self.create_publisher(
            Path, f"/{cfg.ns}/desired_path", qos
        )
        self.desired_path_msg = Path()

        # Goal subscription
        self.sub_goal = self.create_subscription(
            PointStamped,
            f"/{cfg.ns}/{cfg.goal_topic}",
            self._goal_cb,
            qos,
        )

        # State
        self.x = self.y = self.yaw = 0.0
        self.vx = self.vy = self.wz = 0.0
        self.ax = self.ay = self.az = 0.0
        self.gz = 0.0
        self.goal_x = cfg.goal_x
        self.goal_y = cfg.goal_y

        self.lidar_feat = np.zeros(cfg.lidar_bins, dtype=np.float32)
        self.min_range_front = cfg.lidar_max_range

        # Episode accounting
        self.step_in_ep = 0
        self.last_goal_dist = None
        self.last_obs_vec = None

        # Exploration tracking: set of visited grid cells (per episode)
        self.visited_cells = set()

        # Smoothness history
        self.prev_act = np.zeros(2, dtype=np.float32)
        self.prev2_act = np.zeros(2, dtype=np.float32)
        self.prev_gz = 0.0

        # PPO buffers & networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = (
            6  # base state [x,y,yaw,vx,vy,wz]
            + 4  # IMU [ax,ay,az,gz]
            + cfg.lidar_bins  # lidar bins
            + (1 if cfg.use_obstacle else 0)  # obstacle score
            + cam_dim  # flattened camera image (if used)
            + 3  # goal vec [dx,dy,d]
        )
        self.act_dim = 2

        self.policy = GaussianPolicy(self.obs_dim, self.act_dim).to(self.device)
        self.value = MLP(self.obs_dim, 1).to(self.device)

        self.opt_pi = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.opt_v = optim.Adam(self.value.parameters(), lr=cfg.lr)

        self.obs_buf = np.zeros((cfg.rollout_steps, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((cfg.rollout_steps, self.act_dim), dtype=np.float32)
        self.logp_buf = np.zeros((cfg.rollout_steps, 1), dtype=np.float32)
        self.ret_buf = np.zeros((cfg.rollout_steps, 1), dtype=np.float32)
        self.adv_buf = np.zeros((cfg.rollout_steps, 1), dtype=np.float32)
        self.val_buf = np.zeros((cfg.rollout_steps, 1), dtype=np.float32)
        self.done_buf = np.zeros((cfg.rollout_steps, 1), dtype=np.float32)
        self.rew_buf = np.zeros((cfg.rollout_steps, 1), dtype=np.float32)

        self.buf_idx = 0
        self.total_env_steps = 0

        self.in_eval = False
        self.eval_episode = 0
        self.eval_return = 0.0
        self.best_eval_return = -1e9

        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        if cfg.load_ckpt:
            self._load_checkpoint(cfg.load_ckpt)

        self._reset_env()

        self.timer = self.create_timer(cfg.control_dt, self._control_step)
        self.get_logger().info(
            f"LearnerNode initialized: obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
            f"control_dt={self.cfg.control_dt}, total_steps={self.cfg.total_steps}, "
            f"use_camera={self.cfg.use_camera}"
        )

    # --- Subscribers ---
    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        v = msg.twist.twist

        self.x = p.x
        self.y = p.y
        self.yaw = yaw_from_quat(o.x, o.y, o.z, o.w)
        self.vx = v.linear.x
        self.vy = v.linear.y
        self.wz = v.angular.z

        # RL path
        if not self.path_msg.header.frame_id:
            self.path_msg.header.frame_id = msg.header.frame_id
        self.path_msg.header.stamp = msg.header.stamp

        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

    def _imu_cb(self, msg: Imu):
        self.ax = msg.linear_acceleration.x
        self.ay = msg.linear_acceleration.y
        self.az = msg.linear_acceleration.z
        self.gz = msg.angular_velocity.z

    def _obstacle_cb(self, msg: Float32):
        self.obstacle_score = msg.data

    def _lidar_cb(self, msg: LaserScan):
        """
        Process raw LaserScan into:
          - self.lidar_feat: binned + normalized scan for the policy
          - self.min_range_front: min range in a true "front" sector.
        """
        rng = np.array(msg.ranges, dtype=np.float32)

        # Clean NaNs / infs
        rng = np.nan_to_num(
            rng,
            nan=self.cfg.lidar_max_range,
            posinf=self.cfg.lidar_max_range,
            neginf=self.cfg.lidar_max_range,
        )
        rng = np.clip(rng, self.cfg.lidar_min_range, self.cfg.lidar_max_range)

        n = rng.shape[0]
        bins = self.cfg.lidar_bins

        # --- 1) True front distance using scan angles ---
        angles = msg.angle_min + np.arange(n, dtype=np.float32) * msg.angle_increment
        half_fov_rad = math.radians(self.cfg.lidar_front_fov_deg) / 4.0
        front_mask = np.abs(angles) <= half_fov_rad
        if np.any(front_mask):
            self.min_range_front = float(rng[front_mask].min())
        else:
            self.min_range_front = self.cfg.lidar_max_range

        # --- 2) Binned lidar_feat for the RL observation ---
        pad = (bins - (n % bins)) % bins
        feat_rng = rng
        if pad:
            feat_rng = np.pad(
                feat_rng, (0, pad), constant_values=self.cfg.lidar_max_range
            )
        feat_rng = feat_rng.reshape(bins, -1).min(axis=1)

        rng_norm = (feat_rng - self.cfg.lidar_min_range) / (
            self.cfg.lidar_max_range - self.cfg.lidar_min_range
        )
        rng_norm = np.clip(rng_norm, 0.0, 1.0)
        self.lidar_feat = rng_norm.astype(np.float32)

    def _cam_cb(self, msg: Image):
        """
        Convert incoming camera image to a small grayscale feature vector.
        """
        if self.bridge is None:
            return
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"[CAMERA] cv_bridge conversion failed: {e}")
            return

        # Convert to grayscale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        # Resize to configured resolution
        resized = cv2.resize(
            gray,
            (self.cfg.cam_width, self.cfg.cam_height),
            interpolation=cv2.INTER_AREA,
        )
        # Normalize to [0, 1] and flatten
        feat = resized.astype(np.float32) / 255.0
        self.cam_feat = feat.flatten()

    def _goal_cb(self, msg: PointStamped):
        self.goal_x = msg.point.x
        self.goal_y = msg.point.y
        self.get_logger().info(
            f"[GOAL] Updated goal to ({self.goal_x:.2f}, {self.goal_y:.2f})"
        )
        # update straight-line desired path for visualization
        self._update_desired_path()

    # --- Desired path (visual only) ---
    def _update_desired_path(self):
        frame_id = "odom"
        if self.path_msg.header.frame_id:
            frame_id = self.path_msg.header.frame_id

        path = Path()
        path.header.frame_id = frame_id
        path.header.stamp = self.get_clock().now().to_msg()

        num_points = 50
        x0, y0 = self.x, self.y
        x1, y1 = self.goal_x, self.goal_y

        yaw = math.atan2(y1 - y0, x1 - x0)
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)

        for i in range(num_points + 1):
            t = i / num_points
            px = x0 + t * (x1 - x0)
            py = y0 + t * (y1 - y0)

            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = px
            pose.pose.position.y = py
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path.poses.append(pose)

        self.desired_path_msg = path
        self.desired_path_pub.publish(path)

    # --- Core helpers ---
    def _goal_dist(self):
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        return math.hypot(dx, dy)

    def _obs(self):
        base_state = np.array(
            [self.x, self.y, self.yaw, self.vx, self.vy, self.wz],
            dtype=np.float32,
        )
        imu = np.array(
            [self.ax, self.ay, self.az, self.gz], dtype=np.float32
        )

        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        d = math.hypot(dx, dy)
        goal_vec = np.array([dx, dy, d], dtype=np.float32)

        parts = [base_state, imu, self.lidar_feat]
        if self.cfg.use_obstacle:
            score = float(getattr(self, "obstacle_score", 0.0))
            parts.append(np.array([score], dtype=np.float32))
        if self.cfg.use_camera:
            parts.append(self.cam_feat.astype(np.float32))
        parts.append(goal_vec)

        obs = np.concatenate(parts, axis=0)

        if obs.shape[0] != self.obs_dim:
            self.get_logger().warn(
                f"Obs dim mismatch: got {obs.shape[0]}, expected {self.obs_dim}"
            )
        return obs

    def _reset_env(self):
        try:
            if self.reset_client.service_is_ready():
                self.reset_client.call_async(Empty.Request())
                self.get_logger().info("[RESET] Called reset_sim service")
            else:
                self.get_logger().warn(
                    "[RESET] reset_sim service not ready; doing soft reset only"
                )
        except Exception as e:
            self.get_logger().warn(f"[RESET] reset_sim call failed: {e}")

        self.x = self.y = self.yaw = 0.0
        self.vx = self.vy = self.wz = 0.0
        self.ax = self.ay = self.az = 0.0
        self.gz = 0.0

        self.step_in_ep = 0
        self.last_goal_dist = None
        self.last_obs_vec = None
        self.prev_act[:] = 0.0
        self.prev2_act[:] = 0.0
        self.prev_gz = 0.0

        self.path_msg = Path()
        self.desired_path_msg = Path()

        # reset exploration memory for this episode
        self.visited_cells = set()

        self.get_logger().info("[RESET] Episode reset")

    def _compute_action(self, obs_np):
        """
        Pure RL policy action (with smoothing). No hard-coded avoidance.
        """
        obs = torch.from_numpy(obs_np).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            mu, std = self.policy(obs)
            dist = torch.distributions.Normal(mu, std)
            z = dist.sample()
            action = torch.tanh(z)
        act = action.cpu().numpy().squeeze(0)

        # --- Smooth actions: low-pass filter with previous action + deadzone ---
        alpha = 0.8  # closer to 1.0 = smoother, slower to change
        act = alpha * self.prev_act + (1.0 - alpha) * act
        for i in range(2):
            if abs(act[i]) < 0.05:
                act[i] = 0.0

        return act

    def _reward_done(self, obs, act):
        d_now = self._goal_dist()
        d_prev = self.last_goal_dist if self.last_goal_dist is not None else d_now
        progress = d_prev - d_now
        self.last_goal_dist = d_now

        if self.last_obs_vec is None:
            change = 0.0
        else:
            delta = obs[:3] - self.last_obs_vec[:3]
            change = float(
                np.tanh(
                    np.linalg.norm(delta)
                    + 0.5 * abs(obs[5] - self.last_obs_vec[5])
                )
            )
        self.last_obs_vec = obs.copy()

        vx = float(obs[3])
        wz = float(obs[5])
        speed_pen = -self.cfg.w_speed_pen * (abs(vx) + abs(wz))
        action_pen = -self.cfg.w_action_pen * (abs(act[0]) + abs(act[1]))

        lidar_max = self.cfg.lidar_max_range
        min_range_front = self.min_range_front
        near = (lidar_max - min_range_front) / lidar_max
        lidar_pen = -self.cfg.w_lidar_pen * (
            self.cfg.lidar_front_weight * max(0.0, near)
        )

        obstacle_pen = 0.0
        if self.cfg.use_obstacle:
            score = float(getattr(self, "obstacle_score", 0.0))
            obstacle_pen = -self.cfg.w_obstacle_pen * score

        du = act - self.prev_act
        jerk = act - 2.0 * self.prev_act + self.prev2_act
        smooth_pen = (
            -self.cfg.w_smooth_delta * float(np.sum(np.abs(du)))
            - self.cfg.w_smooth_jerk * float(np.sum(np.abs(jerk)))
        )

        gz_now = float(obs[9])  # imu gz index (unchanged by camera features)
        yaw_accel_pen = -self.cfg.w_yaw_accel * float((gz_now - self.prev_gz) ** 2)

        # Exploration reward: reward visiting new cells in odom space
        cell_size = self.cfg.explore_grid_size
        cell_x = int(math.floor(self.x / cell_size))
        cell_y = int(math.floor(self.y / cell_size))
        cell = (cell_x, cell_y)
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            explore_reward = self.cfg.w_explore
        else:
            explore_reward = 0.0

        reward = (
            self.cfg.w_progress * progress
            + self.cfg.w_change * change
            + speed_pen
            + action_pen
            + lidar_pen
            + obstacle_pen
            + smooth_pen
            + yaw_accel_pen
            + explore_reward
        )

        done = False
        collided = False

        if d_now < self.cfg.goal_tol:
            reward += self.cfg.w_goal_bonus
            done = True

        # Near collision -> strong penalty + end episode
        if self.min_range_front < self.cfg.collision_dist:
            collided = True
            reward -= 5.0  # strong penalty for getting too close
            done = True
            self.get_logger().warn(
                f"[COLLISION] front min={self.min_range_front:.2f} m < "
                f"{self.cfg.collision_dist} m -> ending episode"
            )

        # Episode length limit
        if self.step_in_ep >= self.cfg.episode_len:
            done = True

        return float(reward), bool(done), bool(collided)

    # --- Control loop (train + eval) ---
    def _control_step(self):
        obs_np = self._obs()
        obs = torch.from_numpy(obs_np).float().to(self.device).unsqueeze(0)

        if self.in_eval:
            with torch.no_grad():
                mu, _std = self.policy(obs)
                action = torch.tanh(mu)
            act = action.cpu().numpy().squeeze(0)
        else:
            act = self._compute_action(obs_np)

        v = float(act[0]) * self.cfg.max_lin
        w = float(act[1]) * self.cfg.max_ang

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.pub_cmd.publish(cmd)

        reward, done, collided = self._reward_done(obs_np, act)

        if not self.in_eval:
            with torch.no_grad():
                val = self.value(obs).cpu().numpy().squeeze(0)

            if self.buf_idx < self.cfg.rollout_steps:
                self.obs_buf[self.buf_idx] = obs_np
                self.act_buf[self.buf_idx] = act
                self.val_buf[self.buf_idx, 0] = val
                self.rew_buf[self.buf_idx, 0] = reward
                self.done_buf[self.buf_idx, 0] = float(done)

                mu, std = self.policy(obs)
                dist = torch.distributions.Normal(mu, std)
                z = torch.atanh(
                    torch.clamp(torch.from_numpy(act).to(self.device), -0.999, 0.999)
                )
                log_prob = dist.log_prob(z) - torch.log(
                    1 - torch.tanh(z).pow(2) + 1e-6
                )
                log_prob = log_prob.sum(-1, keepdim=True)
                self.logp_buf[self.buf_idx, 0] = log_prob.detach().cpu().numpy()[0, 0]

            self.buf_idx += 1
            self.total_env_steps += 1

        self.prev2_act = self.prev_act.copy()
        self.prev_act = act.copy()
        self.prev_gz = float(obs_np[9])
        self.step_in_ep += 1

        if done:
            self.get_logger().info(
                f"[EPISODE DONE] steps={self.step_in_ep}, reward={reward:.3f}, "
                f"collided={collided}, d_goal={self._goal_dist():.3f}"
            )
            self._reset_env()

        if (not self.in_eval) and self.buf_idx >= self.cfg.rollout_steps:
            self._finish_rollout()

        if (not self.in_eval) and (
            self.total_env_steps >= self.cfg.total_steps
            or (self.total_env_steps % self.cfg.eval_every_steps == 0)
        ):
            if not self.in_eval:
                self.in_eval = True
                self.eval_episode = 0
                self.eval_return = 0.0
                self.get_logger().info("[EVAL] Starting evaluation mode")
            else:
                if self.eval_episode >= self.cfg.eval_episodes:
                    avg_ret = self.eval_return / max(1, self.eval_episode)
                    self.get_logger().info(
                        f"[EVAL DONE] episodes={self.eval_episode}, avg_return={avg_ret:.3f}"
                    )
                    if avg_ret > self.best_eval_return:
                        self.best_eval_return = avg_ret
                        self._save_checkpoint("best")
                    self.in_eval = False
                else:
                    self.eval_episode += 1
                    self.get_logger().info(
                        f"[EVAL] Episode {self.eval_episode}/{self.cfg.eval_episodes}"
                    )

        if (not self.in_eval) and (
            self.total_env_steps > 0
            and self.total_env_steps % self.cfg.save_every == 0
        ):
            self._save_checkpoint(f"step_{self.total_env_steps}")

    def _finish_rollout(self):
        self.get_logger().info(
            f"[PPO] Finishing rollout at step {self.total_env_steps}, buf_idx={self.buf_idx}"
        )
        last_val = 0.0
        if self.done_buf[self.buf_idx - 1, 0] == 0.0:
            obs_np = self.obs_buf[self.buf_idx - 1]
            obs = torch.from_numpy(obs_np).float().to(self.device).unsqueeze(0)
            with torch.no_grad():
                last_val = (
                    self.value(obs).cpu().numpy().squeeze(0).astype(np.float32)
                )

        rews = self.rew_buf[: self.buf_idx]
        vals = self.val_buf[: self.buf_idx]
        dones = self.done_buf[: self.buf_idx]

        adv = np.zeros_like(rews, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.buf_idx)):
            if t == self.buf_idx - 1:
                next_nonterminal = 1.0 - dones[t, 0]
                next_values = last_val
            else:
                next_nonterminal = 1.0 - dones[t + 1, 0]
                next_values = vals[t + 1, 0]
            delta = (
                rews[t, 0]
                + self.cfg.gamma * next_values * next_nonterminal
                - vals[t, 0]
            )
            last_gae = (
                delta
                + self.cfg.gamma * self.cfg.lam * next_nonterminal * last_gae
            )
            adv[t, 0] = last_gae

        ret = adv + vals[: self.buf_idx]

        self.adv_buf[: self.buf_idx] = adv
        self.ret_buf[: self.buf_idx] = ret

        obs_t = torch.from_numpy(self.obs_buf[: self.buf_idx]).float().to(
            self.device
        )
        act_t = torch.from_numpy(self.act_buf[: self.buf_idx]).float().to(
            self.device
        )
        logp_old_t = torch.from_numpy(self.logp_buf[
            : self.buf_idx
        ]).float().to(self.device)
        adv_t = torch.from_numpy(self.adv_buf[: self.buf_idx]).float().to(
            self.device
        )
        ret_t = torch.from_numpy(self.ret_buf[: self.buf_idx]).float().to(
            self.device
        )

        adv_mean = adv_t.mean()
        adv_std = adv_t.std() + 1e-8
        adv_t = (adv_t - adv_mean) / adv_std

        num_samples = self.buf_idx
        batch_size = self.cfg.minibatch_size
        idxs = np.arange(num_samples)

        for _ in range(self.cfg.epochs):
            np.random.shuffle(idxs)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]

                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_logp_old = logp_old_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                mu, std = self.policy(mb_obs)
                dist = torch.distributions.Normal(mu, std)
                z = torch.atanh(
                    torch.clamp(mb_act, -0.999, 0.999)
                )
                logp = dist.log_prob(z) - torch.log(
                    1 - torch.tanh(z).pow(2) + 1e-6
                )
                logp = logp.sum(-1, keepdim=True)

                ratio = torch.exp(logp - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.cfg.clip_eps,
                        1.0 + self.cfg.clip_eps,
                    )
                    * mb_adv
                )
                loss_pi = -torch.min(surr1, surr2).mean()

                v_pred = self.value(mb_obs)
                loss_v = ((v_pred - mb_ret) ** 2).mean()

                entropy = dist.entropy().sum(-1).mean()
                loss = (
                    loss_pi
                    + self.cfg.vf_coef * loss_v
                    - self.cfg.ent_coef * entropy
                )

                self.opt_pi.zero_grad()
                self.opt_v.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())
                    + list(self.value.parameters()),
                    max_norm=0.5,
                )
                self.opt_pi.step()
                self.opt_v.step()

        self.get_logger().info(
            f"[PPO] Updated policy at steps={self.total_env_steps}"
        )
        self.buf_idx = 0

    def _save_checkpoint(self, tag: str):
        ckpt_path = os.path.join(self.cfg.ckpt_dir, f"ppo_{tag}.pt")
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            ckpt_path,
        )
        self.get_logger().info(f"[CKPT] Saved checkpoint to {ckpt_path}")

    def _load_checkpoint(self, tag_or_path: str):
        if os.path.isfile(tag_or_path):
            ckpt_path = tag_or_path
        else:
            ckpt_path = os.path.join(self.cfg.ckpt_dir, f"ppo_{tag_or_path}.pt")

        if not os.path.isfile(ckpt_path):
            self.get_logger().warn(f"[CKPT] No checkpoint file at {ckpt_path}")
            return

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.value.load_state_dict(ckpt["value"])
        self.get_logger().info(f"[CKPT] Loaded checkpoint from {ckpt_path}")

    def destroy_node(self):
        zero = Twist()
        self.pub_cmd.publish(zero)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=str, default="stretch")
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--rollout_steps", type=int, default=2048)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--load_ckpt", action="store_true")
    parser.add_argument("--odom_topic", type=str, default="odom")
    parser.add_argument("--cmd_topic", type=str, default="cmd_vel")
    parser.add_argument("--lidar_topic", type=str, default="scan")
    parser.add_argument("--imu_topic", type=str, default="imu")
    parser.add_argument("--use_obstacle", type=int, default=0)
    parser.add_argument("--obstacle_topic", type=str, default="obstacle_score")
    parser.add_argument("--goal_topic", type=str, default="goal")
    parser.add_argument("--eval_every_steps", type=int, default=None)
    parser.add_argument("--eval_episodes", type=int, default=None)
    parser.add_argument("--collision_dist", type=float, default=None)

    # Camera-related CLI args
    parser.add_argument("--use_camera", type=int, default=0)
    parser.add_argument("--camera_topic", type=str, default="camera/image_raw")
    parser.add_argument("--cam_width", type=int, default=32)
    parser.add_argument("--cam_height", type=int, default=24)

    cli_args, unknown = parser.parse_known_args()

    cfg = Config()
    cfg.ns = cli_args.ns
    cfg.total_steps = cli_args.total_steps
    cfg.rollout_steps = cli_args.rollout_steps
    if cli_args.ckpt_dir is not None:
        cfg.ckpt_dir = cli_args.ckpt_dir
    if cli_args.load_ckpt:
        cfg.load_ckpt = "best"
    cfg.odom_topic = cli_args.odom_topic
    cfg.cmd_topic = cli_args.cmd_topic
    cfg.lidar_topic = cli_args.lidar_topic
    cfg.imu_topic = cli_args.imu_topic
    cfg.use_obstacle = bool(cli_args.use_obstacle)
    cfg.obstacle_topic = cli_args.obstacle_topic
    cfg.goal_topic = cli_args.goal_topic
    if cli_args.eval_every_steps is not None:
        cfg.eval_every_steps = cli_args.eval_every_steps
    if cli_args.eval_episodes is not None:
        cfg.eval_episodes = cli_args.eval_episodes
    if args and "--collision_dist" in unknown:
        idx = unknown.index("--collision_dist")
        if idx + 1 < len(unknown):
            cfg.collision_dist = float(unknown[idx + 1])
    if cli_args.collision_dist is not None:
        cfg.collision_dist = cli_args.collision_dist

    # Camera config
    cfg.use_camera = bool(cli_args.use_camera)
    cfg.camera_topic = cli_args.camera_topic
    cfg.cam_width = cli_args.cam_width
    cfg.cam_height = cli_args.cam_height

    node = LearnerNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down learner_node")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
