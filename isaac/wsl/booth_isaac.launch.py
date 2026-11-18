# Isaac Sim(Windows)과 함께 쓰는 WSL 쪽 launch: 시뮬레이터만 빼고 MuJoCo 실습과 같은 ROS 2 스택(Nav2 · MoveIt · 스킬)을 띄운다
"""warehouse.launch.py 에서 MuJoCo 브리지 프로세스만 뺀 구성이다. 시뮬레이터는 Windows 의 isaac/booth_sim.py 가 맡는다.

    isaac/wsl/robotworld-isaac start            # = ros2 launch isaac/wsl/booth_isaac.launch.py mode:=nav
    isaac/wsl/robotworld-isaac drive | slam | moveit

원본 launch · 설정 파일을 그대로 쓴다: navigation.launch.py, nav2.yaml, slam.yaml, MoveIt config.parameters(),
warehouse.launch.py 의 resolve_spawn · rviz_config · booth_centre. 바꾼 것은 둘이다.
  - MoveIt 실행 감시 여유 (allowed_execution_duration_scaling 4.0, goal margin 8.0): 감시는 벽시계 기준이고 Isaac 은 실시간보다 느리다
  - 카메라 핸들러 대신 vision_bridge.py (/isaac/<카메라>/image_raw → /vision/markers, FMS JPEG)
"""
import importlib.util
import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from mobile_openarm_description.model import expand_urdf

HERE = Path(__file__).resolve().parent


def original():
    """원본 warehouse.launch.py 를 모듈로 읽는다 (수정하지 않고 함수만 빌려 쓴다)."""
    path = Path(get_package_share_directory("mobile_openarm_bringup")) / "launch/warehouse.launch.py"
    spec = importlib.util.spec_from_file_location("robotworld_warehouse_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup(context):
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    base = original()
    mode, moveit, rviz = arg("mode"), arg("moveit") == "true", arg("rviz") == "true"
    root = Path(os.environ.get("ROBOTWORLD_ISAAC_RUN", Path.home() / ".cache/robotworld_isaac/run"))
    root.mkdir(parents=True, exist_ok=True)
    urdf = expand_urdf()
    (root / "robot.urdf").write_text(urdf)
    spawn = base.resolve_spawn(arg("spawn"))
    print(f"[isaac-launch] mode={mode} moveit={moveit} spawn={spawn} domain={os.environ.get('ROS_DOMAIN_ID')} "
          f"rmw={os.environ.get('RMW_IMPLEMENTATION')} (Isaac Sim 쪽 --spawn 도 같게)", flush=True)
    sim_time = {"use_sim_time": True}
    actions = [Node(package="robot_state_publisher", executable="robot_state_publisher",
                    parameters=[{"robot_description": urdf}, sim_time], output="screen")]
    nav = Path(get_package_share_directory("mobile_openarm_navigation"))
    if mode != "drive":
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(nav / "launch/navigation.launch.py")),
            launch_arguments={"mode": mode, "map": arg("map"),
                              "initial_pose": ",".join(f"{v:.4f}" for v in spawn) if spawn else ""}.items()))
    fps = arg("rviz_fps")
    if moveit:
        from mobile_openarm_moveit_config.config import parameters

        m = Path(get_package_share_directory("mobile_openarm_moveit_config"))
        params = {**parameters(urdf), "trajectory_execution.allowed_execution_duration_scaling": 4.0,
                  "trajectory_execution.allowed_goal_duration_margin": 8.0}
        actions += [Node(package="moveit_ros_move_group", executable="move_group", parameters=[params], output="screen"),
                    Node(package="mobile_openarm_moveit_config", executable="warehouse_scene", parameters=[sim_time], output="screen")]
        if rviz:
            actions.append(Node(package="rviz2", executable="rviz2", output="screen", parameters=[params],
                                arguments=["-d", base.rviz_config(m / "config/moveit.rviz", fps, root)]))
        if mode != "drive" and arg("skills") == "true":
            actions += [Node(package="warehouse_skills", executable="pick_place_server", output="screen",
                             parameters=[{"use_sim_time": True, "locate": arg("locate")}]),
                        Node(package="warehouse_lecture", executable="task_manager", output="screen", parameters=[sim_time])]
    elif rviz:
        actions.append(Node(package="rviz2", executable="rviz2", output="screen", parameters=[sim_time],
                            arguments=["-d", base.rviz_config(nav / "config/navigation.rviz", fps, root, base.booth_centre(mode)),
                                       "-f", "odom" if mode == "drive" else "map"]))
    if arg("vision") == "true":
        actions.append(ExecuteProcess(cmd=[sys.executable, str(HERE / "vision_bridge.py"), "--cameras", arg("cameras")],
                                      output="screen", name="isaac_vision_bridge"))
    return actions


def generate_launch_description():
    nav_map = str(Path(get_package_share_directory("mobile_openarm_navigation")) / "maps/booth.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="nav", choices=["drive", "slam", "nav"]),
        DeclareLaunchArgument("map", default_value=nav_map, description="map:='' 이면 SLAM + Nav2"),
        DeclareLaunchArgument("moveit", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("rviz", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("rviz_fps", default_value="15"),
        DeclareLaunchArgument("skills", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("locate", default_value="truth", choices=["truth", "vision"]),
        DeclareLaunchArgument("spawn", default_value="", description="locations.yaml 이름 또는 X,Y,YAW. Isaac 쪽 --spawn 과 같게"),
        DeclareLaunchArgument("vision", default_value="true", choices=["true", "false"],
                              description="vision_bridge.py: Isaac 카메라 → /vision/markers (도킹에 필요)"),
        DeclareLaunchArgument("cameras", default_value="base_camera,head_camera"),
        OpaqueFunction(function=setup),
    ])
