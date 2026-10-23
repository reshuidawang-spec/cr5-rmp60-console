"""Launch the CR5/RMP60 MoveIt stack and optional visualization processes."""

from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


PROJECT_DIR = Path(__file__).resolve().parents[1]
CR5_MOVEIT_CONFIG = Path(get_package_share_directory("cr5_moveit")) / "config"


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("adapter_length", default_value="0.0494"),
        DeclareLaunchArgument("adapter_radius", default_value="0.030"),
        DeclareLaunchArgument("probe_body_length", default_value="0.076"),
        DeclareLaunchArgument("probe_body_radius", default_value="0.0315"),
        DeclareLaunchArgument("stylus_length", default_value="0.083870898"),
        DeclareLaunchArgument("stylus_radius", default_value="0.0008"),
        DeclareLaunchArgument("center_stem_length", default_value="0.0490"),
        DeclareLaunchArgument("center_stem_radius", default_value="0.0035"),
        DeclareLaunchArgument("stylus_ball_radius", default_value="0.001"),
        DeclareLaunchArgument("x_pos_x", default_value="-0.021926126"),
        DeclareLaunchArgument("x_pos_y", default_value="0.029921686"),
        DeclareLaunchArgument("x_pos_z", default_value="0.174438871"),
        DeclareLaunchArgument("x_neg_x", default_value="0.022126096"),
        DeclareLaunchArgument("x_neg_y", default_value="-0.031316519"),
        DeclareLaunchArgument("x_neg_z", default_value="0.173763882"),
        DeclareLaunchArgument("y_pos_x", default_value="0.031888504"),
        DeclareLaunchArgument("y_pos_y", default_value="0.022272196"),
        DeclareLaunchArgument("y_pos_z", default_value="0.172149056"),
        DeclareLaunchArgument("y_neg_x", default_value="-0.030650710"),
        DeclareLaunchArgument("y_neg_y", default_value="-0.022242713"),
        DeclareLaunchArgument("y_neg_z", default_value="0.174850003"),
        DeclareLaunchArgument("tool_axis_sign", default_value="1"),
        DeclareLaunchArgument("mount_xyz", default_value="0 0 0"),
        DeclareLaunchArgument("mount_rpy", default_value="0 0 0"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("use_fake_hardware", default_value="false"),
    ]

    xacro_mappings = {
        "initial_positions_file": str(PROJECT_DIR / "config/moveit/rmp60_initial_positions.yaml"),
        "adapter_length": LaunchConfiguration("adapter_length"),
        "adapter_radius": LaunchConfiguration("adapter_radius"),
        "probe_body_length": LaunchConfiguration("probe_body_length"),
        "probe_body_radius": LaunchConfiguration("probe_body_radius"),
        "stylus_length": LaunchConfiguration("stylus_length"),
        "stylus_radius": LaunchConfiguration("stylus_radius"),
        "center_stem_length": LaunchConfiguration("center_stem_length"),
        "center_stem_radius": LaunchConfiguration("center_stem_radius"),
        "stylus_ball_radius": LaunchConfiguration("stylus_ball_radius"),
        "x_pos_x": LaunchConfiguration("x_pos_x"),
        "x_pos_y": LaunchConfiguration("x_pos_y"),
        "x_pos_z": LaunchConfiguration("x_pos_z"),
        "x_neg_x": LaunchConfiguration("x_neg_x"),
        "x_neg_y": LaunchConfiguration("x_neg_y"),
        "x_neg_z": LaunchConfiguration("x_neg_z"),
        "y_pos_x": LaunchConfiguration("y_pos_x"),
        "y_pos_y": LaunchConfiguration("y_pos_y"),
        "y_pos_z": LaunchConfiguration("y_pos_z"),
        "y_neg_x": LaunchConfiguration("y_neg_x"),
        "y_neg_y": LaunchConfiguration("y_neg_y"),
        "y_neg_z": LaunchConfiguration("y_neg_z"),
        "tool_axis_sign": LaunchConfiguration("tool_axis_sign"),
        "mount_xyz": LaunchConfiguration("mount_xyz"),
        "mount_rpy": LaunchConfiguration("mount_rpy"),
    }

    moveit_config = (
        MoveItConfigsBuilder("cr5_robot", package_name="cr5_moveit")
        .robot_description(PROJECT_DIR / "urdf/cr5_moveit_with_rmp60.urdf.xacro", mappings=xacro_mappings)
        .robot_description_semantic(PROJECT_DIR / "config/moveit/cr5_rmp60.srdf")
        .robot_description_kinematics(CR5_MOVEIT_CONFIG / "kinematics.yaml")
        .joint_limits(CR5_MOVEIT_CONFIG / "joint_limits.yaml")
        .trajectory_execution(CR5_MOVEIT_CONFIG / "moveit_controllers.yaml")
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"], load_all=False)
        .to_moveit_configs()
    )

    move_group_configuration = {
        "publish_robot_description_semantic": True,
        "allow_trajectory_execution": True,
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "monitor_dynamics": False,
    }

    return LaunchDescription(
        declared_arguments
        + [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[moveit_config.robot_description],
                remappings=[("/joint_states", "/joint_states_robot")],
                output="screen",
            ),
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                parameters=[moveit_config.to_dict(), move_group_configuration],
                remappings=[("/joint_states", "/joint_states_robot")],
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", str(PROJECT_DIR / "config/rviz/rmp60_measurement_markers.rviz")],
                parameters=[
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.planning_pipelines,
                    moveit_config.robot_description_kinematics,
                    moveit_config.joint_limits,
                ],
                output="log",
                condition=IfCondition(LaunchConfiguration("use_rviz")),
            ),
            Node(
                package="controller_manager",
                executable="ros2_control_node",
                parameters=[
                    moveit_config.robot_description,
                    str(CR5_MOVEIT_CONFIG / "ros2_controllers.yaml"),
                ],
                condition=IfCondition(LaunchConfiguration("use_fake_hardware")),
                output="screen",
            ),
            TimerAction(
                period=2.0,
                actions=[
                    ExecuteProcess(
                        cmd=["ros2", "run", "controller_manager", "spawner", "joint_state_broadcaster"],
                        condition=IfCondition(LaunchConfiguration("use_fake_hardware")),
                        output="screen",
                    )
                ],
            ),
            TimerAction(
                period=4.0,
                actions=[
                    ExecuteProcess(
                        cmd=["ros2", "run", "controller_manager", "spawner", "cr5_group_controller"],
                        condition=IfCondition(LaunchConfiguration("use_fake_hardware")),
                        output="screen",
                    )
                ],
            ),
        ]
    )
