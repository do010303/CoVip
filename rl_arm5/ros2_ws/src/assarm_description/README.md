# assarm_description - ROS 2 Humble + Gazebo Harmonic

## 1. Cai dat (Ubuntu 22.04)
    sudo apt install gz-harmonic ros-humble-ros-gzharmonic ros-humble-ros-gzharmonic-bridge \
         ros-humble-xacro ros-humble-controller-manager ros-humble-joint-state-broadcaster \
         ros-humble-joint-trajectory-controller ros-humble-robot-state-publisher
    # gz_ros2_control cho Humble + Harmonic phai build tu source:
    mkdir -p ~/gz_ws/src && cd ~/gz_ws/src
    git clone https://github.com/ros-controls/gz_ros2_control -b humble
    export GZ_VERSION=harmonic
    cd ~/gz_ws && rosdep install -r --from-paths src --ignore-src --rosdistro humble -y
    colcon build && source install/setup.bash

## 2. Copy file vao package do exporter tao
    urdf/   -> assarm_description/urdf/     (XOA assarm.trans cu)
    config/ -> assarm_description/config/
    launch/ -> assarm_description/launch/   (xoa gazebo.launch.py / display.launch.py kieu Classic neu co)

Ten file dung: launch/gazebo.launch.py, urdf/assarm_ros2_control.xacro.
Neu trinh duyet doi dau cham thanh gach duoi (gazebo_launch.py), hay doi lai ten
thanh gazebo.launch.py, hoac sua setup.py glob thanh 'launch/*.py'.

## 3. Cau truc package Python (ament_python)
    assarm_description/            <- thu muc goc package
      package.xml  setup.py  setup.cfg
      resource/assarm_description  <- file RONG, ten trung ten package
      assarm_description/__init__.py   <- thu muc Python + file rong (setup.py co packages=[package_name])
      launch/  urdf/  config/  meshes/

## 4. setup.py: data_files (chi cai FILE, bo qua thu muc con)
    import os
    from glob import glob

    def files(pattern):
        return [f for f in glob(pattern) if os.path.isfile(f)]

    data_files = [
        ('share/ament_index/resource_index/packages', ['resource/assarm_description']),
        ('share/assarm_description', ['package.xml']),
        ('share/assarm_description/launch', files('launch/*.launch.py')),
        ('share/assarm_description/urdf',   files('urdf/*')),
        ('share/assarm_description/config', files('config/*')),
        ('share/assarm_description/meshes', files('meshes/*')),
    ]
Neu meshes/ co thu muc con thi them dong rieng cho tung thu muc con.

## 5. package.xml: them dependency chay
    <exec_depend>xacro</exec_depend>
    <exec_depend>robot_state_publisher</exec_depend>
    <exec_depend>ros_gz_sim</exec_depend>
    <exec_depend>ros_gz_bridge</exec_depend>
    <exec_depend>gz_ros2_control</exec_depend>
    <exec_depend>controller_manager</exec_depend>
    <exec_depend>joint_state_broadcaster</exec_depend>
    <exec_depend>joint_trajectory_controller</exec_depend>

## 6. Chay
    cd ~/ws && colcon build --symlink-install && source install/setup.bash
    ros2 launch assarm_description gazebo.launch.py

## 7. Thu dieu khien
    ros2 topic pub --once /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
    "{joint_names: [Revolute_Joint_1, Revolute_Joint_2, Revolute_Joint_3, Revolute_Joint_4], \
      points: [{positions: [0.5, 0.4, -0.4, 0.0], time_from_start: {sec: 3}}]}"
    ros2 topic pub --once /gripper_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
    "{joint_names: [Revolute_Joint_Active], points: [{positions: [1.0], time_from_start: {sec: 2}}]}"
    ros2 topic echo /joint_states   # 4 khop kep (Active, Passive, 5, 8) phai cung chuyen dong

## 8. Ghep vao UAV (canh tay treo duoi, gripper huong xuong)
Huong CAD hien tai da dung, khong can xoay. Khi ghep:
- Xoa link `world` va `world_to_base_joint` trong assarm.xacro, them khop fixed tu link than UAV toi `base_link`.
- base_link co goc toa do nam o goc CAD, mat gan nam o z_top (do tren Fusion, don vi m).
  Neu mat duoi UAV nam o z = -h so voi frame UAV thi origin khop la:  xyz = "0 0 ${-h - z_top}"  rpy = "0 0 0"
- Canh tay nang ~0.51 kg: kiem tra lai khoi luong/quan tinh/tam khoi luong cua UAV va bo dieu khien bay.
