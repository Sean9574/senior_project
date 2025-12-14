#!/usr/bin/env python3
"""
Multi-Stretch Training Arena Generator

Generates a MuJoCo scene with N Stretch robots in N isolated rooms.
Each robot is a full copy of the Stretch model with prefixed names.

Usage:
    from multi_stretch_arena import MultiStretchArena
    
    arena = MultiStretchArena(num_robots=4)
    arena.save_xml("/path/to/scene.xml")
    
    # Or get the XML string directly
    xml_string = arena.generate_xml()
"""

import copy
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class RoomConfig:
    """Configuration for a single training room."""
    robot_id: int
    center_x: float
    center_y: float
    size: float
    spawn_x: float
    spawn_y: float
    spawn_yaw: float  # radians
    goal_x: float
    goal_y: float


def get_stretch_model_path() -> Path:
    """Find the Stretch MuJoCo model path."""
    # Try common locations
    possible_paths = [
        Path.home() / "ament_ws/src/stretch_ros2/stretch_simulation/stretch_mujoco_driver/dependencies/stretch_mujoco/stretch_mujoco/models/stretch.xml",
        Path("/home/sean/ament_ws/src/stretch_ros2/stretch_simulation/stretch_mujoco_driver/dependencies/stretch_mujoco/stretch_mujoco/models/stretch.xml"),
    ]
    
    # Try to find via Python import
    try:
        import stretch_mujoco
        pkg_dir = Path(stretch_mujoco.__file__).parent
        possible_paths.insert(0, pkg_dir / "models" / "stretch.xml")
    except ImportError:
        pass
    
    for path in possible_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError(
        f"Could not find stretch.xml model. Searched:\n" + 
        "\n".join(f"  - {p}" for p in possible_paths)
    )


def get_stretch_assets_path() -> Path:
    """Get path to Stretch model assets."""
    model_path = get_stretch_model_path()
    return model_path.parent / "assets"


class MultiStretchArena:
    """
    Generates a multi-robot Stretch training arena.
    
    Creates N isolated rooms, each containing a full Stretch robot.
    All robot elements are prefixed with robot{i}_ to avoid name collisions.
    """
    
    def __init__(
        self,
        num_robots: int = 4,
        room_size: float = 6.0,
        wall_height: float = 0.8,
        randomize_goals: bool = True,
        randomize_spawns: bool = False,
    ):
        """
        Args:
            num_robots: Number of robots/rooms
            room_size: Size of each room in meters
            wall_height: Height of walls
            randomize_goals: Randomize goal positions within rooms
            randomize_spawns: Randomize spawn positions/orientations
        """
        self.num_robots = num_robots
        self.room_size = room_size
        self.wall_height = wall_height
        self.randomize_goals = randomize_goals
        self.randomize_spawns = randomize_spawns
        
        # Grid layout
        self.grid_cols = math.ceil(math.sqrt(num_robots))
        self.grid_rows = math.ceil(num_robots / self.grid_cols)
        self.room_spacing = room_size + 0.5  # Small gap between rooms
        
        # Load Stretch model
        self.stretch_model_path = get_stretch_model_path()
        self.assets_path = get_stretch_assets_path()
        
        # Generate room configs
        self.rooms = self._generate_room_configs()
        
    def _generate_room_configs(self) -> List[RoomConfig]:
        """Generate configuration for each room."""
        rooms = []
        
        # Center the grid
        total_width = self.grid_cols * self.room_spacing
        total_height = self.grid_rows * self.room_spacing
        offset_x = -total_width / 2 + self.room_spacing / 2
        offset_y = -total_height / 2 + self.room_spacing / 2
        
        for i in range(self.num_robots):
            col = i % self.grid_cols
            row = i // self.grid_cols
            
            cx = offset_x + col * self.room_spacing
            cy = offset_y + row * self.room_spacing
            
            # Default spawn on left, goal on right
            spawn_x = cx - self.room_size / 3
            spawn_y = cy
            spawn_yaw = 0.0  # Facing +X (toward goal)
            
            goal_x = cx + self.room_size / 3
            goal_y = cy
            
            if self.randomize_spawns:
                # Random position in left half
                spawn_x = cx - self.room_size / 4 + np.random.uniform(-0.5, 0.5)
                spawn_y = cy + np.random.uniform(-self.room_size / 4, self.room_size / 4)
                spawn_yaw = np.random.uniform(-np.pi, np.pi)
            
            if self.randomize_goals:
                # Random position in right half
                goal_x = cx + self.room_size / 4 + np.random.uniform(-0.5, 0.5)
                goal_y = cy + np.random.uniform(-self.room_size / 4, self.room_size / 4)
            
            rooms.append(RoomConfig(
                robot_id=i,
                center_x=cx,
                center_y=cy,
                size=self.room_size,
                spawn_x=spawn_x,
                spawn_y=spawn_y,
                spawn_yaw=spawn_yaw,
                goal_x=goal_x,
                goal_y=goal_y,
            ))
        
        return rooms
    
    def _prefix_names(self, element: ET.Element, prefix: str, parent_tag: str = ""):
        """
        Recursively prefix all name/joint/body/site/etc attributes.
        """
        # Attributes that should be prefixed
        name_attrs = ['name', 'body1', 'body2', 'joint', 'joint1', 'joint2', 
                      'tendon', 'site', 'objname', 'body', 'geom', 'mesh',
                      'material', 'texture', 'class', 'childclass', 'target']
        
        for attr in name_attrs:
            if attr in element.attrib:
                old_val = element.attrib[attr]
                # Don't prefix if it's a class reference to default classes
                if attr in ['class', 'childclass'] and old_val in ['stretch', 'wheel', 'lift', 
                    'telescope', 'head', 'head_pan', 'head_tilt', 'wrist', 'wrist_yaw',
                    'wrist_pitch', 'wrist_roll', 'finger_open', 'finger_slide', 'visual',
                    'collision', 'caster', 'rubber']:
                    continue
                # Don't prefix material/texture references (shared assets)
                if attr in ['material', 'texture', 'mesh']:
                    continue
                element.attrib[attr] = f"{prefix}{old_val}"
        
        # Handle joint references in equality constraints
        if element.tag == 'joint' and parent_tag == 'equality':
            for attr in ['joint1', 'joint2']:
                if attr in element.attrib:
                    element.attrib[attr] = f"{prefix}{element.attrib[attr]}"
        
        # Recurse
        for child in element:
            self._prefix_names(child, prefix, element.tag)
    
    def _parse_stretch_model(self) -> ET.Element:
        """Parse the Stretch XML model."""
        tree = ET.parse(self.stretch_model_path)
        return tree.getroot()
    
    def _extract_robot_body(self, root: ET.Element, robot_id: int, room: RoomConfig) -> ET.Element:
        """
        Extract the robot body from Stretch model and prefix all names.
        """
        prefix = f"robot{robot_id}_"
        
        # Find worldbody
        worldbody = root.find('worldbody')
        if worldbody is None:
            raise ValueError("No worldbody found in Stretch model")
        
        # Find base_link (the robot root body)
        base_link = worldbody.find("body[@name='base_link']")
        if base_link is None:
            raise ValueError("No base_link body found in Stretch model")
        
        # Deep copy
        robot_body = copy.deepcopy(base_link)
        
        # Prefix all names
        self._prefix_names(robot_body, prefix)
        
        # Set position and orientation
        # Quaternion for yaw rotation: [cos(θ/2), 0, 0, sin(θ/2)]
        qw = math.cos(room.spawn_yaw / 2)
        qz = math.sin(room.spawn_yaw / 2)
        
        robot_body.attrib['pos'] = f"{room.spawn_x} {room.spawn_y} 0.1"
        robot_body.attrib['quat'] = f"{qw} 0 0 {qz}"
        
        return robot_body
    
    def _extract_contacts(self, root: ET.Element, robot_id: int) -> List[ET.Element]:
        """Extract and prefix contact exclusions."""
        prefix = f"robot{robot_id}_"
        contacts = []
        
        contact_elem = root.find('contact')
        if contact_elem is not None:
            for exclude in contact_elem.findall('exclude'):
                new_exclude = copy.deepcopy(exclude)
                if 'body1' in new_exclude.attrib:
                    new_exclude.attrib['body1'] = f"{prefix}{new_exclude.attrib['body1']}"
                if 'body2' in new_exclude.attrib:
                    new_exclude.attrib['body2'] = f"{prefix}{new_exclude.attrib['body2']}"
                contacts.append(new_exclude)
        
        return contacts
    
    def _extract_tendons(self, root: ET.Element, robot_id: int) -> List[ET.Element]:
        """Extract and prefix tendons."""
        prefix = f"robot{robot_id}_"
        tendons = []
        
        tendon_elem = root.find('tendon')
        if tendon_elem is not None:
            for tendon in tendon_elem:
                new_tendon = copy.deepcopy(tendon)
                if 'name' in new_tendon.attrib:
                    new_tendon.attrib['name'] = f"{prefix}{new_tendon.attrib['name']}"
                for joint_ref in new_tendon.findall('joint'):
                    if 'joint' in joint_ref.attrib:
                        joint_ref.attrib['joint'] = f"{prefix}{joint_ref.attrib['joint']}"
                tendons.append(new_tendon)
        
        return tendons
    
    def _extract_equality(self, root: ET.Element, robot_id: int) -> List[ET.Element]:
        """Extract and prefix equality constraints."""
        prefix = f"robot{robot_id}_"
        equalities = []
        
        equality_elem = root.find('equality')
        if equality_elem is not None:
            for eq in equality_elem:
                new_eq = copy.deepcopy(eq)
                for attr in ['joint1', 'joint2', 'joint', 'body1', 'body2', 'name']:
                    if attr in new_eq.attrib:
                        new_eq.attrib[attr] = f"{prefix}{new_eq.attrib[attr]}"
                equalities.append(new_eq)
        
        return equalities
    
    def _extract_actuators(self, root: ET.Element, robot_id: int) -> List[ET.Element]:
        """Extract and prefix actuators."""
        prefix = f"robot{robot_id}_"
        actuators = []
        
        actuator_elem = root.find('actuator')
        if actuator_elem is not None:
            for act in actuator_elem:
                new_act = copy.deepcopy(act)
                if 'name' in new_act.attrib:
                    new_act.attrib['name'] = f"{prefix}{new_act.attrib['name']}"
                if 'joint' in new_act.attrib:
                    new_act.attrib['joint'] = f"{prefix}{new_act.attrib['joint']}"
                if 'tendon' in new_act.attrib:
                    new_act.attrib['tendon'] = f"{prefix}{new_act.attrib['tendon']}"
                actuators.append(new_act)
        
        return actuators
    
    def _extract_sensors(self, root: ET.Element, robot_id: int) -> List[ET.Element]:
        """Extract and prefix sensors."""
        prefix = f"robot{robot_id}_"
        sensors = []
        
        sensor_elem = root.find('sensor')
        if sensor_elem is not None:
            for sensor in sensor_elem:
                new_sensor = copy.deepcopy(sensor)
                if 'name' in new_sensor.attrib:
                    new_sensor.attrib['name'] = f"{prefix}{new_sensor.attrib['name']}"
                if 'site' in new_sensor.attrib:
                    new_sensor.attrib['site'] = f"{prefix}{new_sensor.attrib['site']}"
                if 'joint' in new_sensor.attrib:
                    new_sensor.attrib['joint'] = f"{prefix}{new_sensor.attrib['joint']}"
                sensors.append(new_sensor)
        
        return sensors
    
    def _create_room_geometry(self, room: RoomConfig) -> ET.Element:
        """Create walls for a room."""
        room_body = ET.Element('body')
        room_body.attrib['name'] = f"room{room.robot_id}"
        room_body.attrib['pos'] = f"{room.center_x} {room.center_y} 0"
        
        half = room.size / 2
        wh = self.wall_height
        wt = 0.05  # Wall thickness
        
        # North wall
        wall_n = ET.SubElement(room_body, 'geom')
        wall_n.attrib.update({
            'name': f"room{room.robot_id}_wall_n",
            'type': 'box',
            'pos': f"0 {half} {wh/2}",
            'size': f"{half + wt} {wt} {wh/2}",
            'rgba': '0.5 0.5 0.55 1',
        })
        
        # South wall
        wall_s = ET.SubElement(room_body, 'geom')
        wall_s.attrib.update({
            'name': f"room{room.robot_id}_wall_s",
            'type': 'box',
            'pos': f"0 {-half} {wh/2}",
            'size': f"{half + wt} {wt} {wh/2}",
            'rgba': '0.5 0.5 0.55 1',
        })
        
        # East wall
        wall_e = ET.SubElement(room_body, 'geom')
        wall_e.attrib.update({
            'name': f"room{room.robot_id}_wall_e",
            'type': 'box',
            'pos': f"{half} 0 {wh/2}",
            'size': f"{wt} {half} {wh/2}",
            'rgba': '0.5 0.5 0.55 1',
        })
        
        # West wall
        wall_w = ET.SubElement(room_body, 'geom')
        wall_w.attrib.update({
            'name': f"room{room.robot_id}_wall_w",
            'type': 'box',
            'pos': f"{-half} 0 {wh/2}",
            'size': f"{wt} {half} {wh/2}",
            'rgba': '0.5 0.5 0.55 1',
        })
        
        # Goal marker
        goal = ET.SubElement(room_body, 'geom')
        goal.attrib.update({
            'name': f"room{room.robot_id}_goal",
            'type': 'cylinder',
            'pos': f"{room.goal_x - room.center_x} {room.goal_y - room.center_y} 0.005",
            'size': '0.3 0.005',
            'rgba': '0.2 0.8 0.2 0.7',
            'contype': '0',
            'conaffinity': '0',
        })
        
        # Goal site (for distance calculations)
        goal_site = ET.SubElement(room_body, 'site')
        goal_site.attrib.update({
            'name': f"room{room.robot_id}_goal_site",
            'pos': f"{room.goal_x - room.center_x} {room.goal_y - room.center_y} 0.01",
            'size': '0.1',
            'rgba': '0 1 0 0.5',
        })
        
        return room_body
    
    def generate_xml(self) -> str:
        """Generate the complete multi-robot arena XML."""
        
        # Parse original Stretch model
        stretch_root = self._parse_stretch_model()
        
        # Create new root
        root = ET.Element('mujoco')
        root.attrib['model'] = 'multi_stretch_arena'
        
        # Copy compiler with updated asset path
        compiler = ET.SubElement(root, 'compiler')
        compiler.attrib['angle'] = 'radian'
        compiler.attrib['assetdir'] = str(self.assets_path)
        
        # Copy option
        orig_option = stretch_root.find('option')
        if orig_option is not None:
            root.append(copy.deepcopy(orig_option))
        
        # Copy size
        orig_size = stretch_root.find('size')
        if orig_size is not None:
            # Increase limits for multi-robot
            size_elem = copy.deepcopy(orig_size)
            size_elem.attrib['njmax'] = str(int(size_elem.attrib.get('njmax', 5000)) * self.num_robots)
            size_elem.attrib['nconmax'] = str(int(size_elem.attrib.get('nconmax', 5000)) * self.num_robots)
            root.append(size_elem)
        
        # Copy default classes
        orig_default = stretch_root.find('default')
        if orig_default is not None:
            root.append(copy.deepcopy(orig_default))
        
        # Copy assets (shared between all robots)
        orig_asset = stretch_root.find('asset')
        if orig_asset is not None:
            asset = copy.deepcopy(orig_asset)
            
            # Add floor texture/material
            floor_tex = ET.SubElement(asset, 'texture')
            floor_tex.attrib.update({
                'type': '2d',
                'name': 'floor_tex',
                'builtin': 'checker',
                'width': '512',
                'height': '512',
                'rgb1': '0.15 0.15 0.15',
                'rgb2': '0.25 0.25 0.25',
            })
            
            floor_mat = ET.SubElement(asset, 'material')
            floor_mat.attrib.update({
                'name': 'floor_mat',
                'texture': 'floor_tex',
                'texrepeat': '20 20',
            })
            
            root.append(asset)
        
        # Create worldbody
        worldbody = ET.SubElement(root, 'worldbody')
        
        # Add floor
        arena_size = max(self.grid_cols, self.grid_rows) * self.room_spacing + 5
        floor = ET.SubElement(worldbody, 'geom')
        floor.attrib.update({
            'name': 'floor',
            'type': 'plane',
            'size': f'{arena_size} {arena_size} 0.1',
            'material': 'floor_mat',
        })
        
        # Add light
        light = ET.SubElement(worldbody, 'light')
        light.attrib.update({
            'pos': '0 0 15',
            'dir': '0 0 -1',
            'diffuse': '0.8 0.8 0.8',
            'specular': '0.2 0.2 0.2',
            'castshadow': 'true',
        })
        
        # Add rooms and robots
        all_contacts = []
        all_tendons = []
        all_equalities = []
        all_actuators = []
        all_sensors = []
        
        for room in self.rooms:
            # Add room geometry
            room_geom = self._create_room_geometry(room)
            worldbody.append(room_geom)
            
            # Add robot
            robot_body = self._extract_robot_body(stretch_root, room.robot_id, room)
            worldbody.append(robot_body)
            
            # Collect other elements
            all_contacts.extend(self._extract_contacts(stretch_root, room.robot_id))
            all_tendons.extend(self._extract_tendons(stretch_root, room.robot_id))
            all_equalities.extend(self._extract_equality(stretch_root, room.robot_id))
            all_actuators.extend(self._extract_actuators(stretch_root, room.robot_id))
            all_sensors.extend(self._extract_sensors(stretch_root, room.robot_id))
        
        # Add contact exclusions
        if all_contacts:
            contact = ET.SubElement(root, 'contact')
            for c in all_contacts:
                contact.append(c)
        
        # Add tendons
        if all_tendons:
            tendon = ET.SubElement(root, 'tendon')
            for t in all_tendons:
                tendon.append(t)
        
        # Add equality constraints
        if all_equalities:
            equality = ET.SubElement(root, 'equality')
            for e in all_equalities:
                equality.append(e)
        
        # Add actuators
        if all_actuators:
            actuator = ET.SubElement(root, 'actuator')
            for a in all_actuators:
                actuator.append(a)
        
        # Add sensors
        if all_sensors:
            sensor = ET.SubElement(root, 'sensor')
            for s in all_sensors:
                sensor.append(s)
        
        # Convert to string with proper formatting
        self._indent(root)
        return ET.tostring(root, encoding='unicode')
    
    def _indent(self, elem: ET.Element, level: int = 0):
        """Add pretty-print indentation to XML."""
        indent = "\n" + "  " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent
    
    def save_xml(self, filepath: str):
        """Save generated XML to file."""
        xml_string = self.generate_xml()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write(xml_string)
        
        print(f"Saved multi-robot arena to: {filepath}")
        print(f"  - {self.num_robots} robots in {self.grid_cols}x{self.grid_rows} grid")
        print(f"  - Room size: {self.room_size}m x {self.room_size}m")
        print(f"  - Assets path: {self.assets_path}")
    
    def get_room_configs(self) -> List[RoomConfig]:
        """Get room configurations."""
        return self.rooms
    
    def get_robot_actuator_names(self, robot_id: int) -> dict:
        """Get actuator names for a specific robot."""
        prefix = f"robot{robot_id}_"
        return {
            'left_wheel': f"{prefix}left_wheel_vel",
            'right_wheel': f"{prefix}right_wheel_vel",
            'lift': f"{prefix}lift",
            'arm': f"{prefix}arm",
            'wrist_yaw': f"{prefix}wrist_yaw",
            'wrist_pitch': f"{prefix}wrist_pitch",
            'wrist_roll': f"{prefix}wrist_roll",
            'gripper': f"{prefix}gripper",
            'head_pan': f"{prefix}head_pan",
            'head_tilt': f"{prefix}head_tilt",
        }
    
    def get_robot_sensor_names(self, robot_id: int) -> dict:
        """Get sensor names for a specific robot."""
        prefix = f"robot{robot_id}_"
        return {
            'gyro': f"{prefix}base_gyro",
            'accel': f"{prefix}base_accel",
            'lidar': f"{prefix}base_lidar",
        }


def test_arena():
    """Test the arena generation."""
    print("Testing MultiStretchArena...")
    
    for n in [1, 2, 4, 8]:
        print(f"\n--- {n} robots ---")
        try:
            arena = MultiStretchArena(num_robots=n, room_size=6.0)
            
            for room in arena.rooms:
                print(f"  Room {room.robot_id}: "
                      f"center=({room.center_x:.1f}, {room.center_y:.1f}), "
                      f"spawn=({room.spawn_x:.1f}, {room.spawn_y:.1f}), "
                      f"goal=({room.goal_x:.1f}, {room.goal_y:.1f})")
            
            # Generate but don't save
            xml = arena.generate_xml()
            print(f"  Generated XML length: {len(xml)} chars")
            
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
    
    # Try to save one
    try:
        arena = MultiStretchArena(num_robots=4)
        arena.save_xml("/tmp/multi_stretch_arena.xml")
    except Exception as e:
        print(f"Save test: {e}")


if __name__ == "__main__":
    test_arena()
