# isaac_physics.IsaacPhysics 의 계산과 원본 actions.Controllers 와의 호환을 Isaac 없이 확인하는 자체 점검 (WSL venv 에서 실행)
"""python isaac/test_isaac_physics.py   →  마지막 줄 "ok" 면 통과. rclpy 가 있으면 Controllers 연결도 본다."""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from isaac_physics import IsaacPhysics  # noqa: E402

ARMS = [f"openarm_{s}_joint{i}" for i in range(1, 8) for s in ("left", "right")]
DOFS = ["left_wheel_joint", "right_wheel_joint"] + ARMS + [f"openarm_{s}_finger_joint{j}" for s in ("left", "right") for j in (1, 2)]


def fake_sim():
    init = {n: (1.2 if n.endswith("joint4") else 0.0) for n in ARMS}
    init.update({"openarm_left_finger_joint1": 0.025, "openarm_right_finger_joint1": 0.025})
    settings = {"control": {"acceleration": 0.45, "angular_acceleration": 1.0, "transport_tolerance": 0.15,
                            "max_linear": 0.35, "max_angular": 0.7, "command_timeout": 0.5},
                "motors": {"max_speed": 8.0}, "gripper": {"max_force": 30.0, "goal_tolerance": 0.002},
                "arm": {"goal_tolerance": 0.035, "goal_time_tolerance": 3.0}}
    return {"settings": settings, "initial_positions": init, "wheel": {"radius": 0.0825, "track": 0.4288},
            "drive_poses": {"transport": {n: init[n] for n in ARMS}}}


def make():
    n = len(DOFS)
    p = IsaacPhysics(DOFS, -np.full(n, 3.0), np.full(n, 3.0), fake_sim())
    q = np.array([p.targets.get(d, 0.0) for d in DOFS])
    p.update_state(0.0, q, np.zeros(n))
    return p, q


def main():
    p, q = make()
    assert p.transport_ready(), "initial pose must be a drive pose"
    assert p.targets["openarm_left_finger_joint2"] == 0.025, "mimic finger follows joint1"
    p.set_targets(["openarm_right_finger_joint1"], [0.01])
    assert p.targets["openarm_right_finger_joint2"] == 0.01
    # acceleration limit: 0.45 m/s^2 for 0.1 s -> 0.045 m/s
    w = p.wheel_targets(1.0, 0.0, 0.1)
    assert abs(p.command[0] - 0.045) < 1e-9 and abs(w[0] - 0.045 / 0.0825) < 1e-9
    # odometry: both wheels +1 rad -> 0.0825 m forward; opposite -> yaw = 2 r / track
    p.integrate_odom(0.01)
    q[0] += 1.0; q[1] += 1.0
    p.update_state(0.01, q, np.zeros(len(q)))
    p.integrate_odom(0.01)
    assert abs(p.odom[0] - 0.0825) < 1e-9 and abs(p.odom[2]) < 1e-12
    q[0] -= 0.5; q[1] += 0.5
    p.update_state(0.02, q, np.zeros(len(q)))
    p.integrate_odom(0.01)
    assert abs(p.odom[2] - 2 * 0.5 * 0.0825 / 0.4288) < 1e-9
    # leaving the drive pose blocks driving
    q[DOFS.index("openarm_left_joint4")] = 0.5
    p.update_state(0.03, q, np.zeros(len(q)))
    assert not p.transport_ready()
    assert p.finger_max_efforts() == [30.0] * 2
    assert not any("finger_joint2" in n for n in p.position_names), "mimic finger is not driven"
    controllers()
    print("ok")


def controllers():
    """원본 Controllers 가 어댑터를 받아들이는지 (액션 서버 4개 생성 · 그리퍼 goal 검증 · 팔 goal 검증)."""
    try:
        import rclpy
        from control_msgs.action import FollowJointTrajectory, GripperCommand
        from rclpy.action import GoalResponse
        from rclpy.node import Node
    except ImportError:
        print("rclpy not available: Controllers check skipped")
        return
    sys.path.insert(0, str(HERE.parent / "src/mobile_openarm_mujoco"))
    from mobile_openarm_mujoco.actions import Controllers

    rclpy.init()
    node = Node("test_isaac_physics")
    node.physics, _ = make()
    node.command, node.command_time = (0.0, 0.0), -math.inf
    node.sim_stamp = lambda: node.get_clock().now().to_msg()
    c = Controllers(node)
    assert len(c.servers) == 4
    goal = GripperCommand.Goal()
    goal.command.position, goal.command.max_effort = 0.012, 25.0
    assert c.goal("left_gripper_controller", goal) == GoalResponse.ACCEPT
    assert "left_gripper_controller" in c.reserved
    bad = FollowJointTrajectory.Goal()
    bad.trajectory.joint_names = ["nope"]
    assert c.goal("right_joint_trajectory_controller", bad) == GoalResponse.REJECT   # wrong joint names
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
