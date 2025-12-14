#!/usr/bin/env python3
"""
Multi-Robot RL Learner Node

Trains a single PPO policy using data from N parallel Stretch robots.
Each robot runs in its own namespace and contributes to the shared policy.

This provides N times faster training than single-robot training.
"""

import argparse
import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy

import torch
import torch.nn as nn
from torch.distributions import Normal

from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


@dataclass
class RobotState:
    """Current state of a robot."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v_lin: float = 0.0
    v_ang: float = 0.0
    goal_x: float = 0.0
    goal_y: float = 0.0
    lidar: np.ndarray = None
    last_update: float = 0.0
    
    def __post_init__(self):
        if self.lidar is None:
            self.lidar = np.ones(60) * 10.0


class PolicyNetwork(nn.Module):
    """Shared policy network."""
    
    def __init__(self, obs_dim: int, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
    
    def forward(self, obs):
        features = self.net(obs)
        mean = torch.tanh(self.mean(features))
        std = self.log_std.exp().expand_as(mean)
        return mean, std
    
    def get_action(self, obs, deterministic=False):
        mean, std = self.forward(obs)
        if deterministic:
            return mean, None
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
    
    def evaluate(self, obs, actions):
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Shared value network."""
    
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class RobotInterface:
    """ROS2 interface for a single robot."""
    
    def __init__(self, node: Node, robot_id: int, num_lidar_bins: int, callback_group):
        self.robot_id = robot_id
        self.ns = f"/robot{robot_id}"
        self.num_lidar_bins = num_lidar_bins
        
        self.state = RobotState()
        self.state_lock = threading.Lock()
        
        # Publishers
        self.cmd_pub = node.create_publisher(Twist, f"{self.ns}/cmd_vel", 10)
        
        # Subscribers
        self.odom_sub = node.create_subscription(
            Odometry, f"{self.ns}/odom",
            self._odom_callback, 10,
            callback_group=callback_group
        )
        self.scan_sub = node.create_subscription(
            LaserScan, f"{self.ns}/scan",
            self._scan_callback, 10,
            callback_group=callback_group
        )
        self.goal_sub = node.create_subscription(
            PointStamped, f"{self.ns}/goal",
            self._goal_callback, 10,
            callback_group=callback_group
        )
    
    def _odom_callback(self, msg: Odometry):
        with self.state_lock:
            self.state.x = msg.pose.pose.position.x
            self.state.y = msg.pose.pose.position.y
            
            # Extract yaw from quaternion
            q = msg.pose.pose.orientation
            self.state.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                         1 - 2 * (q.y**2 + q.z**2))
            
            self.state.v_lin = msg.twist.twist.linear.x
            self.state.v_ang = msg.twist.twist.angular.z
            self.state.last_update = time.time()
    
    def _scan_callback(self, msg: LaserScan):
        with self.state_lock:
            ranges = np.array(msg.ranges)
            ranges = np.clip(ranges, msg.range_min, msg.range_max)
            ranges[~np.isfinite(ranges)] = msg.range_max
            
            # Bin to num_lidar_bins
            n = len(ranges)
            if n > 0:
                bin_indices = (np.arange(n) * self.num_lidar_bins // n).astype(int)
                binned = np.ones(self.num_lidar_bins) * msg.range_max
                for i in range(self.num_lidar_bins):
                    mask = bin_indices == i
                    if mask.any():
                        binned[i] = ranges[mask].min()
                self.state.lidar = binned
    
    def _goal_callback(self, msg: PointStamped):
        with self.state_lock:
            self.state.goal_x = msg.point.x
            self.state.goal_y = msg.point.y
    
    def get_state(self) -> RobotState:
        with self.state_lock:
            return RobotState(
                x=self.state.x,
                y=self.state.y,
                yaw=self.state.yaw,
                v_lin=self.state.v_lin,
                v_ang=self.state.v_ang,
                goal_x=self.state.goal_x,
                goal_y=self.state.goal_y,
                lidar=self.state.lidar.copy(),
                last_update=self.state.last_update,
            )
    
    def send_cmd(self, v_lin: float, v_ang: float):
        msg = Twist()
        msg.linear.x = float(v_lin)
        msg.angular.z = float(v_ang)
        self.cmd_pub.publish(msg)


class MultiRobotLearner(Node):
    """
    Multi-robot PPO learner node.
    
    Collects experience from all robots in parallel and trains a shared policy.
    """
    
    def __init__(self, args):
        super().__init__('multi_robot_learner')
        
        self.num_robots = args.num_robots
        self.num_lidar_bins = 60
        self.obs_dim = self.num_lidar_bins + 4  # lidar + [goal_dist, goal_angle, v_lin, v_ang]
        self.action_dim = 2
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")
        
        # Networks
        self.policy = PolicyNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.value = ValueNetwork(self.obs_dim).to(self.device)
        
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': 3e-4},
            {'params': self.value.parameters(), 'lr': 3e-4},
        ])
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.entropy_coef = 0.05
        self.value_coef = 0.5
        
        # Experience buffer (per robot)
        self.buffers = {i: {
            'obs': [], 'actions': [], 'log_probs': [], 'values': [], 'rewards': [], 'dones': []
        } for i in range(self.num_robots)}
        
        # Robot interfaces
        self.callback_group = ReentrantCallbackGroup()
        self.robots: Dict[int, RobotInterface] = {}
        for i in range(self.num_robots):
            self.robots[i] = RobotInterface(self, i, self.num_lidar_bins, self.callback_group)
        
        # Episode tracking
        self.episode_steps = np.zeros(self.num_robots, dtype=int)
        self.episode_rewards = np.zeros(self.num_robots)
        self.total_episodes = np.zeros(self.num_robots, dtype=int)
        self.max_steps = args.max_steps
        
        # Training state
        self.total_steps = 0
        self.rollout_steps = args.rollout_steps
        
        # Reward publisher (for monitoring)
        self.reward_pub = self.create_publisher(String, '/training_stats', 10)
        
        # Control timer
        self.control_dt = 0.1  # 10 Hz
        self.create_timer(self.control_dt, self._control_step)
        
        # Checkpoint
        self.checkpoint_dir = args.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        if args.load_checkpoint and os.path.exists(args.load_checkpoint):
            self._load_checkpoint(args.load_checkpoint)
        
        self.get_logger().info(f"Multi-Robot Learner ready: {self.num_robots} robots, {self.obs_dim}D obs")
    
    def _get_observation(self, robot_id: int) -> np.ndarray:
        """Get observation for a robot."""
        state = self.robots[robot_id].get_state()
        
        # Goal relative
        dx = state.goal_x - state.x
        dy = state.goal_y - state.y
        goal_dist = math.sqrt(dx**2 + dy**2)
        
        goal_angle_world = math.atan2(dy, dx)
        goal_angle = goal_angle_world - state.yaw
        goal_angle = math.atan2(math.sin(goal_angle), math.cos(goal_angle))
        
        obs = np.concatenate([
            state.lidar,
            [goal_dist, goal_angle, state.v_lin, state.v_ang]
        ]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self, robot_id: int) -> tuple:
        """Compute reward for a robot."""
        state = self.robots[robot_id].get_state()
        
        dx = state.goal_x - state.x
        dy = state.goal_y - state.y
        goal_dist = math.sqrt(dx**2 + dy**2)
        
        goal_angle_world = math.atan2(dy, dx)
        goal_angle = goal_angle_world - state.yaw
        goal_angle = math.atan2(math.sin(goal_angle), math.cos(goal_angle))
        
        # Alignment reward
        angle_deg = abs(goal_angle) * 180.0 / math.pi
        r_alignment = -0.1 * angle_deg
        
        # Turning reward
        r_turning = 2.0 * goal_angle * state.v_ang
        
        # Goal bonus
        r_goal = 100.0 if goal_dist < 0.5 else 0.0
        
        # Collision penalty
        r_collision = -50.0 if state.lidar.min() < 0.15 else 0.0
        
        total = r_alignment + r_turning + r_goal + r_collision
        
        done = False
        outcome = ""
        if goal_dist < 0.5:
            done = True
            outcome = "success"
        elif state.lidar.min() < 0.15:
            done = True
            outcome = "collision"
        
        return total, done, outcome, {
            'alignment': r_alignment,
            'turning': r_turning,
            'goal': r_goal,
            'collision': r_collision,
            'goal_dist': goal_dist,
            'goal_angle_deg': angle_deg,
        }
    
    def _control_step(self):
        """Main control loop - runs for all robots."""
        # Check if data is available
        all_ready = all(
            self.robots[i].get_state().last_update > 0 
            for i in range(self.num_robots)
        )
        
        if not all_ready:
            return
        
        # Get observations
        obs_batch = np.array([self._get_observation(i) for i in range(self.num_robots)])
        obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
        
        # Get actions
        with torch.no_grad():
            actions, log_probs = self.policy.get_action(obs_tensor)
            values = self.value(obs_tensor)
        
        actions_np = actions.cpu().numpy()
        log_probs_np = log_probs.cpu().numpy()
        values_np = values.cpu().numpy()
        
        # Send commands
        for i in range(self.num_robots):
            v_lin = float(actions_np[i, 0]) * 0.5  # Scale to 0.5 m/s max
            v_ang = float(actions_np[i, 1]) * 1.0  # Scale to 1.0 rad/s max
            self.robots[i].send_cmd(v_lin, v_ang)
        
        # Compute rewards and check done
        for i in range(self.num_robots):
            reward, done, outcome, info = self._compute_reward(i)
            
            # Timeout
            if self.episode_steps[i] >= self.max_steps:
                done = True
                outcome = "timeout"
            
            # Store experience
            self.buffers[i]['obs'].append(obs_batch[i])
            self.buffers[i]['actions'].append(actions_np[i])
            self.buffers[i]['log_probs'].append(log_probs_np[i])
            self.buffers[i]['values'].append(values_np[i])
            self.buffers[i]['rewards'].append(reward)
            self.buffers[i]['dones'].append(done)
            
            self.episode_rewards[i] += reward
            self.episode_steps[i] += 1
            self.total_steps += 1
            
            # Episode end
            if done:
                self.get_logger().info(
                    f"Robot {i} episode {self.total_episodes[i]}: "
                    f"steps={self.episode_steps[i]}, "
                    f"return={self.episode_rewards[i]:.1f}, "
                    f"outcome={outcome}"
                )
                
                # Publish stats
                stats = {
                    'robot_id': i,
                    'episode': int(self.total_episodes[i]),
                    'steps': int(self.episode_steps[i]),
                    'return': float(self.episode_rewards[i]),
                    'outcome': outcome,
                    'total_steps': self.total_steps,
                }
                self.reward_pub.publish(String(data=json.dumps(stats)))
                
                self.total_episodes[i] += 1
                self.episode_rewards[i] = 0
                self.episode_steps[i] = 0
        
        # Update policy when enough steps collected
        total_buffer_steps = sum(len(self.buffers[i]['obs']) for i in range(self.num_robots))
        if total_buffer_steps >= self.rollout_steps:
            self._update_policy()
            
            # Save checkpoint periodically
            if self.total_steps % 10000 < self.rollout_steps:
                self._save_checkpoint()
    
    def _update_policy(self):
        """Update policy using collected experience from all robots."""
        # Combine buffers from all robots
        all_obs = []
        all_actions = []
        all_log_probs = []
        all_values = []
        all_rewards = []
        all_dones = []
        
        for i in range(self.num_robots):
            if len(self.buffers[i]['obs']) > 0:
                all_obs.extend(self.buffers[i]['obs'])
                all_actions.extend(self.buffers[i]['actions'])
                all_log_probs.extend(self.buffers[i]['log_probs'])
                all_values.extend(self.buffers[i]['values'])
                all_rewards.extend(self.buffers[i]['rewards'])
                all_dones.extend(self.buffers[i]['dones'])
        
        if len(all_obs) == 0:
            return
        
        obs = torch.FloatTensor(np.array(all_obs)).to(self.device)
        actions = torch.FloatTensor(np.array(all_actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(all_log_probs)).to(self.device)
        
        rewards = np.array(all_rewards)
        values = np.array(all_values)
        dones = np.array(all_dones)
        
        # Compute returns and advantages with GAE
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        
        last_gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        batch_size = 64
        epochs = 10
        
        for _ in range(epochs):
            indices = np.random.permutation(T)
            for start in range(0, T, batch_size):
                idx = indices[start:start + batch_size]
                
                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                # Policy loss
                log_probs, entropy = self.policy.evaluate(batch_obs, batch_actions)
                ratio = (log_probs - batch_old_log_probs).exp()
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_pred = self.value(batch_obs)
                value_loss = 0.5 * (value_pred - batch_returns).pow(2).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.optimizer.step()
        
        # Clear buffers
        for i in range(self.num_robots):
            for key in self.buffers[i]:
                self.buffers[i][key] = []
        
        self.get_logger().info(f"Policy updated at step {self.total_steps}")
    
    def _save_checkpoint(self):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{self.total_steps}.pt")
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }, path)
        self.get_logger().info(f"Saved checkpoint: {path}")
        
        # Also save as latest
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }, latest_path)
    
    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy'])
        self.value.load_state_dict(ckpt['value'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_steps = ckpt.get('total_steps', 0)
        self.get_logger().info(f"Loaded checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-robots', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=600)
    parser.add_argument('--rollout-steps', type=int, default=2048)
    parser.add_argument('--checkpoint-dir', type=str, default='./multi_robot_checkpoints')
    parser.add_argument('--load-checkpoint', type=str, default='')
    
    # Parse known args to allow ROS2 args
    args, _ = parser.parse_known_args()
    
    rclpy.init()
    node = MultiRobotLearner(args)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node._save_checkpoint()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
