#!/bin/bash

# Quick Setup Script for Docker on Linux with ROS 2 Humble
# Usage: ./setup_docker.sh

set -e

echo "=== Docker Setup for senior_project ==="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed!"
    echo "Please install Docker first: https://docs.docker.com/engine/install/ubuntu/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Error: Docker Compose is not installed!"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✓ Docker and Docker Compose are installed"
echo ""

# Enable GUI support (X11)
echo "Enabling GUI support for RViz/Gazebo..."
xhost +local:docker
echo "✓ X11 access granted to Docker"
echo ""

# Build the Docker image
echo "Building Docker image (this may take a few minutes)..."
docker-compose build
echo "✓ Docker image built successfully"
echo ""

# Start the container
echo "Starting Docker container..."
docker-compose up -d
echo "✓ Container is running"
echo ""

echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Enter the container: make shell"
echo "   Or: docker exec -it senior_project_ros2 bash"
echo ""
echo "2. First time: Build your workspace inside the container"
echo "   cd /ament_ws"
echo "   colcon build --symlink-install"
echo "   source install/setup.bash"
echo ""
echo "3. Your workspace structure:"
echo "   /ament_ws/src/senior_project/  - Your RL project"
echo "   /ament_ws/src/stretch_ros2/    - Your Stretch packages"
echo "   (All packages from your host src/ directory are available)"
echo ""
echo "4. Run your launch files:"
echo "   ros2 launch senior_project <your_launch_file>.launch.py"
echo ""
echo "Useful commands:"
echo "  make shell  - Enter container"
echo "  make down   - Stop container"
echo "  make logs   - View container logs"
echo "  make help   - See all available commands"
echo ""
