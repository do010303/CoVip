import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            RegisterEventHandler, SetEnvironmentVariable)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = 'assarm_description'
    share = get_package_share_directory(pkg)
    xacro_file = os.path.join(share, 'urdf', 'assarm.xacro')

    # Gazebo Sim can phan giai package://assarm_description/... qua GZ_SIM_RESOURCE_PATH / IGN_GAZEBO_RESOURCE_PATH
    res = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    resource_path = os.path.dirname(share) + (os.pathsep + res if res else '')
    ign_res = os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    ign_resource_path = os.path.dirname(share) + (os.pathsep + ign_res if ign_res else '')

    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', xacro_file]), value_type=str)

    world = LaunchConfiguration('world')
    gz_version = LaunchConfiguration('gz_version')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': ['-r ', world],
            'gz_version': gz_version
        }.items())

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               output='screen',
               parameters=[{'robot_description': robot_description, 'use_sim_time': True}])

    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-topic', 'robot_description', '-name', 'assarm'])

    clock_bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
                        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
                        output='screen')

    def spawner(name):
        return Node(package='controller_manager', executable='spawner',
                    arguments=[name, '--controller-manager', '/controller_manager',
                               '--controller-manager-timeout', '60'],
                    output='screen')

    jsb = spawner('joint_state_broadcaster')
    arm = spawner('arm_controller')
    gripper = spawner('gripper_controller')

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='empty.sdf', description='Gazebo world file'),
        DeclareLaunchArgument('gz_version', default_value='6', description='Gazebo major version (8 for Harmonic, 6 for Fortress)'),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_path),
        SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', ign_resource_path),
        gz_sim, clock_bridge, rsp, spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[arm, gripper])),
    ])
