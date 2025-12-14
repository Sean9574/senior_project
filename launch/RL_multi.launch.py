#!/usr/bin/env python3
"""
Multi-Robot RL Training Launch File

Launches N Stretch robots in MuJoCo and trains a shared policy.

Usage:
    ros2 launch senior_project RL_multi.launch.py num_robots:=4
    ros2 launch senior_project RL_multi.launch.py num_robots:=8 use_viewer:=true
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    
    # ============================================================
    # ARGUMENTS
    # ============================================================
    
    ld.add_action(DeclareLaunchArgument(
        'num_robots', default_value='4',
        description='Number of parallel robots'
    ))
    
    ld.add_action(DeclareLaunchArgument(
        'room_size', default_value='6.0',
        description='Size of each training room in meters'
    ))
    
    ld.add_action(DeclareLaunchArgument(
        'use_viewer', default_value='true',
        choices=['true', 'false'],
        description='Show MuJoCo viewer'
    ))
    
    ld.add_action(DeclareLaunchArgument(
        'use_rviz', default_value='false',
        choices=['true', 'false'],
        description='Launch RViz'
    ))
    
    ld.add_action(DeclareLaunchArgument(
        'max_steps', default_value='600',
        description='Maximum steps per episode'
    ))
    
    ld.add_action(DeclareLaunchArgument(
        'rollout_steps', default_value='2048',
        description='Steps between policy updates'
    ))
    
    ld.add_action(DeclareLaunchArgument(
        'checkpoint_dir', default_value='./multi_robot_checkpoints',
        description='Checkpoint directory'
    ))
    
    ld.add_action(DeclareLaunchArgument(
        'load_checkpoint', default_value='',
        description='Path to checkpoint to load'
    ))
    
    ld.add_action(DeclareLaunchArgument(
        'use_reward_monitor', default_value='true',
        choices=['true', 'false'],
        description='Launch reward monitor web interface'
    ))
    
    # ============================================================
    # MULTI-STRETCH DRIVER
    # ============================================================
    
    # Get package path
    # Assuming this launch file is in senior_project/launch/
    launch_file_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(launch_file_dir)
    scripts_dir = os.path.join(package_dir, 'scripts')
    
    # Multi-robot MuJoCo driver
    driver_proc = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(scripts_dir, 'multi_stretch_driver.py'),
            '--ros-args',
            '-p', ['num_robots:=', LaunchConfiguration('num_robots')],
            '-p', ['room_size:=', LaunchConfiguration('room_size')],
            '-p', ['use_viewer:=', LaunchConfiguration('use_viewer')],
        ],
        output='screen',
    )
    
    ld.add_action(driver_proc)
    
    # ============================================================
    # MULTI-ROBOT LEARNER
    # ============================================================
    
    learner_proc = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(scripts_dir, 'multi_robot_learner.py'),
            '--num-robots', LaunchConfiguration('num_robots'),
            '--max-steps', LaunchConfiguration('max_steps'),
            '--rollout-steps', LaunchConfiguration('rollout_steps'),
            '--checkpoint-dir', LaunchConfiguration('checkpoint_dir'),
            '--load-checkpoint', LaunchConfiguration('load_checkpoint'),
        ],
        output='screen',
    )
    
    # Start learner after driver has time to initialize
    ld.add_action(TimerAction(period=3.0, actions=[learner_proc]))
    
    # ============================================================
    # REWARD MONITOR (optional)
    # ============================================================
    
    reward_monitor_path = os.path.join(scripts_dir, 'multi_robot_monitor.py')
    
    # Create a simple multi-robot monitor if needed
    monitor_proc = ExecuteProcess(
        cmd=['python3', reward_monitor_path],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_reward_monitor')),
    )
    
    ld.add_action(TimerAction(period=5.0, actions=[monitor_proc]))
    
    # ============================================================
    # RVIZ (optional)
    # ============================================================
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )
    
    ld.add_action(rviz_node)
    
    # ============================================================
    # INFO
    # ============================================================
    
    ld.add_action(LogInfo(msg=[
        '\n',
        '=' * 60, '\n',
        '  Multi-Robot RL Training\n',
        '  Robots: ', LaunchConfiguration('num_robots'), '\n',
        '  Room size: ', LaunchConfiguration('room_size'), 'm\n',
        '  Viewer: ', LaunchConfiguration('use_viewer'), '\n',
        '  Monitor: http://localhost:5000\n',
        '=' * 60, '\n',
    ]))
    
    return ld
