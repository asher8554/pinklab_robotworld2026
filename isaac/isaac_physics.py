# Isaac Sim 관절 상태를 MuJoCo Physics 처럼 보이게 하는 어댑터. 원본 actions.Controllers 와 브리지 주행 계산을 그대로 쓰게 한다
"""MuJoCo 브리지(mobile_openarm_mujoco)의 Physics 가 actions.Controllers 에 주는 것만 같은 이름으로 흉내 낸다.

Isaac 모듈을 import 하지 않는다 (numpy 만). booth_sim.py 가 물리 스텝마다 update_state() 로 관절 값을 넣고,
position_targets() · wheel_targets() 를 읽어 Isaac 관절 구동기에 넣는다. 그래서 이 파일은 Isaac 없이 시험할 수 있다
(isaac/test_isaac_physics.py).

Controllers 가 읽는 MuJoCo API → 여기서 흉내 내는 것
  p.data.time · p.data.qvel[dofadr] · p.data.actuator_force[aid]
  p.model.joint(name).id / .dofadr[0] · p.model.jnt_range[id] · p.model.actuator(name).id · p.model.actuator_forcerange[aid]
  p.positions · p.set_targets · p.hold · p.settings · p.twist · p.command
"""
import math
from types import SimpleNamespace

import numpy as np


class _Model:
    """관절 번호 = Isaac DOF 번호. 액추에이터는 MuJoCo 이름 규칙(<관절>_position)을 따른다."""

    def __init__(self, dof_names, lower, upper, actuators):
        self._joint = {n: SimpleNamespace(id=i, dofadr=[i], qposadr=[i]) for i, n in enumerate(dof_names)}
        self._act = {n: SimpleNamespace(id=i) for i, n in enumerate(actuators)}
        self.jnt_range = np.column_stack([lower, upper]).astype(float)
        self.actuator_forcerange = np.zeros((len(actuators), 2))

    def joint(self, name):
        return self._joint[name]

    def actuator(self, name):
        return self._act[name]


class IsaacPhysics:
    def __init__(self, dof_names, lower, upper, sim):
        """dof_names · lower · upper: Isaac 관절 순서. sim: export_assets.py 가 만든 sim.json."""
        self.settings = sim["settings"]
        self.dof_names = list(dof_names)
        self.index = {n: i for i, n in enumerate(self.dof_names)}
        # 위치 제어 관절 = 팔 + 손가락 joint1. joint2 는 mimic(joint1 을 따라감)이라 구동하지 않는다 (MuJoCo 도 equality 로만 묶는다)
        self.position_names = [n for n in self.dof_names if "wheel" not in n and "finger_joint2" not in n]
        self.wheel_names = ["left_wheel_joint", "right_wheel_joint"]
        # MuJoCo 처럼 mimic 이 아닌 관절에만 액추에이터가 있다 (finger_joint2 제외)
        self.actuators = [n + "_position" for n in self.position_names if "finger_joint2" not in n]
        self.model = _Model(self.dof_names, lower, upper, self.actuators)
        n = len(self.dof_names)
        self.data = SimpleNamespace(time=0.0, qpos=np.zeros(n), qvel=np.zeros(n), actuator_force=np.zeros(len(self.actuators)))
        gripper = self.settings["gripper"]["max_force"]
        for i, name in enumerate(self.actuators):
            if "finger" in name:
                self.model.actuator_forcerange[i] = [-gripper, gripper]
        self.targets = {k: float(v) for k, v in sim["initial_positions"].items()}
        for name in list(self.targets):                       # mimic 손가락은 joint1 과 같은 목표
            if "finger_joint1" in name:
                self.targets[name.replace("joint1", "joint2")] = self.targets[name]
        self.transport_names = [k for k in sim["initial_positions"] if "finger" not in k]
        self.drive_poses = {name: np.array([pose[k] for k in self.transport_names]) for name, pose in sim["drive_poses"].items()}
        self.radius, self.track = float(sim["wheel"]["radius"]), float(sim["wheel"]["track"])
        self.odom, self.twist, self.command = np.zeros(3), np.zeros(2), np.zeros(2)
        self._wheel_q = None

    # --- Controllers 가 쓰는 MuJoCo Physics 인터페이스 ----------------------------------------
    def positions(self, names):
        return np.array([self.data.qpos[self.index[n]] for n in names])

    def set_targets(self, names, values):
        for n, v in zip(names, values):
            self.targets[n] = float(v)
            if "finger_joint1" in n:
                self.targets[n.replace("joint1", "joint2")] = float(v)

    def hold(self, names):
        self.set_targets(names, self.positions(names))

    def transport_ready(self):
        """양팔이 각각 주행 자세(transport · ready · carry ...) 중 하나에 있을 때 True. MuJoCo Physics.transport_ready 와 같다."""
        q = self.positions(self.transport_names)
        tol = self.settings["control"]["transport_tolerance"]
        for side in ("left", "right"):
            idx = [i for i, n in enumerate(self.transport_names) if f"_{side}_" in n]
            if not any(np.max(np.abs(q[idx] - pose[idx])) < tol for pose in self.drive_poses.values()):
                return False
        return True

    # --- booth_sim.py 가 물리 스텝마다 부르는 것 ----------------------------------------------
    def update_state(self, t, q, v, efforts=None):
        """Isaac 에서 읽은 관절 위치 · 속도 [rad, m] 를 넣는다. efforts: 액추에이터 순서의 힘 (없으면 0)."""
        self.data.time = float(t)
        self.data.qpos[:] = q
        self.data.qvel[:] = v
        if efforts is not None:
            self.data.actuator_force[:] = efforts

    def position_targets(self):
        """위치 제어 관절(position_names 순서)의 목표값."""
        return np.array([self.targets.get(n, self.data.qpos[self.index[n]]) for n in self.position_names])

    def finger_max_efforts(self):
        """Controllers 가 GripperCommand 의 max_effort 로 바꾼 손가락 힘 한계 [N], position_names 중 손가락 순서."""
        return [float(self.model.actuator_forcerange[self.model.actuator(n + "_position").id][1])
                for n in self.position_names if "finger" in n]

    def wheel_targets(self, linear, angular, dt):
        """가속 제한을 건 (v, w) 명령 → 바퀴 각속도 목표 [rad/s]. MuJoCo Physics.step 의 앞부분과 같다."""
        cfg = self.settings["control"]
        change = np.array([cfg["acceleration"], cfg["angular_acceleration"]]) * dt
        self.command += np.clip(np.array([linear, angular]) - self.command, -change, change)
        v, w = self.command
        wheel = np.array([v - w * self.track / 2, v + w * self.track / 2]) / self.radius
        limit = self.settings["motors"]["max_speed"]
        return np.clip(wheel, -limit, limit)

    def integrate_odom(self, dt):
        """바퀴 각도 변화로 오도메트리를 적분한다 (엔코더 방식, MuJoCo Physics.step 의 뒷부분과 같다)."""
        wq = self.positions(self.wheel_names)
        if self._wheel_q is None:
            self._wheel_q = wq.copy()
            return
        dl, dr = (wq - self._wheel_q) * self.radius
        self._wheel_q = wq.copy()
        ds, da = (dl + dr) / 2, (dr - dl) / self.track
        theta = self.odom[2] + da / 2
        self.odom += [ds * math.cos(theta), ds * math.sin(theta), da]
        self.twist[:] = [ds / dt, da / dt]
