# Stretch2 RL with Change Detection

A reinforcement learning system for the Hello Robot Stretch2 that detects environmental changes and adapts navigation/manipulation behavior accordingly.

## Overview

This project implements a closed-loop system where:
1. **Perception** - Cameras and sensors detect the environment and changes
2. **RL Agent** - A trained neural network decides robot actions
3. **Change Detection** - Identifies when the environment has been modified
4. **Robot Control** - Executes actions on the Stretch2 mobile manipulator

The system enables the robot to learn manipulation and navigation tasks while dynamically responding to changes in its environment.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           STRETCH2 HARDWARE                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ D435i Camera │  │    LiDAR     │  │  Joint       │  │   Mobile    │ │
│  │              │  │              │  │  Encoders    │  │   Base      │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
└─────────┼──────────────────┼──────────────────┼──────────────────┼──────┘
          │                  │                  │                  │
          │ /camera/color/   │ /scan            │ /joint_states    │ /cmd_vel
          │ image_raw        │                  │                  │
          ▼                  ▼                  ▼                  ▲
┌─────────────────────────────────────────────────────────────────┴──────┐
│                         ROS 2 MIDDLEWARE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────┐         ┌─────────────────────┐                │
│  │ PERCEPTION NODE    │         │ CHANGE DETECTION    │                │
│  │                    │         │ NODE                │                │
│  │ - Object Detection │────────▶│                     │                │
│  │ - Depth Processing │         │ - Background Sub    │                │
│  │ - Feature Extract  │         │ - Temporal Diff     │                │
│  └─────────┬──────────┘         │ - Change Mask       │                │
│            │                     └──────────┬──────────┘                │
│            │ /perception/               │ /change_detection/           │
│            │ objects                    │ changes                      │
│            │                            │                              │
│            └────────────┬───────────────┘                              │
│                         ▼                                               │
│              ┌─────────────────────┐                                    │
│              │   RL AGENT NODE     │                                    │
│              │                     │                                    │
│              │ - State Aggregation │                                    │
│              │ - Policy Network    │                                    │
│              │ - Action Selection  │                                    │
│              │ - Reward Calc       │                                    │
│              └──────────┬──────────┘                                    │
│                         │ /rl/action                                    │
│                         ▼                                               │
│              ┌─────────────────────┐                                    │
│              │  CONTROL NODE       │                                    │
│              │                     │                                    │
│              │ - Action→Commands   │                                    │
│              │ - Safety Checks     │                                    │
│              │ - Joint Control     │                                    │
│              └──────────┬──────────┘                                    │
│                         │ /stretch/cmd_vel                              │
│                         │ /stretch/joint_trajectory                     │
└─────────────────────────┼───────────────────────────────────────────────┘
                          │
                          ▼
                    [Robot Actuators]
```

## Key ROS 2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/color/image_raw` | sensor_msgs/Image | RGB camera feed |
| `/camera/depth/image_rect_raw` | sensor_msgs/Image | Depth image |
| `/scan` | sensor_msgs/LaserScan | LiDAR scans |
| `/joint_states` | sensor_msgs/JointState | Robot joint positions |
| `/perception/objects` | vision_msgs/Detection2DArray | Detected objects |
| `/change_detection/changes` | sensor_msgs/Image | Change visualization |
| `/change_detection/mask` | sensor_msgs/Image | Binary change mask |
| `/rl/state` | std_msgs/Float32MultiArray | Current state vector |
| `/rl/action` | std_msgs/Float32MultiArray | Selected action |
| `/rl/reward` | std_msgs/Float32 | Current reward |
| `/cmd_vel` | geometry_msgs/Twist | Velocity commands |
| `/stretch/joint_trajectory` | trajectory_msgs/JointTrajectory | Arm commands |

## Installation

### Prerequisites
- Ubuntu 22.04
- ROS 2 Humble (installed)
- Python 3.10+
- Stretch ROS 2 packages

### Quick Setup

```bash
# 1. Create workspace and clone
mkdir -p ~/stretch_ws/src
cd ~/stretch_ws/src
git clone https://github.com/Sean9574/senior_project.git

# 2. Install Python dependencies
cd senior_project
pip install -r requirements.txt

# 3. Build workspace
cd ~/stretch_ws
colcon build --packages-select senior_project
source install/setup.bash
```

## Running the System

### Launch Full System

```bash
# Launch the complete RL navigation system with change detection
ros2 launch senior_project sam3_navigation.launch.py
```

This single command starts:
- Stretch robot driver (or simulation)
- Perception node
- Change detection node
- RL agent node
- Control node

### Training Mode (Optional)

```bash
# Train a new model using the training script
python3 scripts/train_agent.py --config config/rl_params.yaml
```

### Evaluation Mode (Optional)

```bash
# Evaluate a trained model
python3 scripts/evaluate_agent.py \
    --model models/best_model.pth \
    --episodes 50
```

## Configuration

Edit `config/rl_params.yaml` to adjust:

```yaml
# RL Algorithm Selection
algorithm: "TD3"  # Options: DQN, DDPG, TD3, PPO, SAC

# Training Parameters
learning_rate: 0.0003
batch_size: 256
episodes: 5000
max_steps_per_episode: 500

# Change Detection
detection_threshold: 30
background_method: "MOG2"  # Options: MOG2, KNN, FrameDiff
```

## Project Structure

```
senior_project/
├── config/               # YAML configuration files
├── launch/              # ROS 2 launch files
├── scripts/             # Training and evaluation scripts
├── src/                 # Core Python modules
│   ├── rl_agent/       # RL algorithms
│   ├── change_detection/
│   └── perception/
├── senior_project/      # ROS 2 node implementations
├── models/              # Saved model checkpoints
└── package.xml
```

## Quick Commands Reference

```bash
# List active nodes
ros2 node list

# Monitor RL metrics
ros2 topic echo /rl/reward

# View camera feed
ros2 run rqt_image_view rqt_image_view /camera/color/image_raw

# Visualize in RViz
ros2 launch senior_project visualization.launch.py
```
## License

MIT License - See LICENSE file

## Contact

GitHub: [@Sean9574](https://github.com/Sean9574)
