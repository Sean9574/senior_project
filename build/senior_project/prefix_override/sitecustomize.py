import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/sean/ament_ws/src/stretch_ros2/senior_project/install/senior_project'
