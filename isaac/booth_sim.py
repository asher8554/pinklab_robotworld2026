# PinkLAB 부스와 이동형 양팔 로봇을 Isaac Sim 6.2에서 돌리는 시뮬레이터 노드. MuJoCo 브리지와 같은 ROS 2 토픽 · 액션을 낸다 (Windows)
r"""실행 (Windows, 저장소 폴더에서. 먼저 WSL 에서 isaac/wsl/robotworld-isaac export):
    C:\isaacsim-6.2\python.bat isaac\booth_sim.py                         # 창 모드
    C:\isaacsim-6.2\python.bat isaac\booth_sim.py --headless --seconds 60   # 60 s 돌리고 자체 점검 후 종료
    C:\isaacsim-6.2\python.bat isaac\booth_sim.py --spawn drinks --cameras none

MuJoCo 실습의 브리지(src/mobile_openarm_mujoco/bridge.py) 자리를 대신한다. WSL 쪽 Nav2 · MoveIt · PickPlace · Task Manager 는 그대로다.
  발행  /clock /joint_states /odom /tf(odom→base_footprint) /scan(best effort) /ground_truth /warehouse/object_poses
        /isaac/<카메라>/image_raw (rgb8). MuJoCo 의 프로세스 안 카메라 핸들러 대신 WSL 의 isaac/wsl/vision_bridge.py 가 받는다
  구독  /cmd_vel /cmd_vel_stamped  (양팔이 주행 자세일 때만, 0.5 s 끊기면 정지: 브리지와 같은 인터록)
  액션  /{left,right}_joint_trajectory_controller/follow_joint_trajectory · /{left,right}_gripper_controller/gripper_cmd
        원본 mobile_openarm_mujoco.actions.Controllers 를 수정 없이 import 한다 (isaac_physics.IsaacPhysics 가 MuJoCo Physics 를 흉내)
  통신  Fast DDS, ROS_DOMAIN_ID = %ROBOTWORLD_ISAAC_DOMAIN_ID% (기본 45. MuJoCo 실습 44 · 강좌 43 과 섞이지 않는다)
라이다는 RTX Lidar 대신 PhysX 레이캐스트 360개다 (MuJoCo 브리지의 mj_multiRay 와 같은 방식, 로봇 자신은 제외).
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# SimulationApp 보다 먼저 정해야 한다. Windows Isaac Sim 의 기본 RMW(Zenoh)는 WSL 과 통하지 않는다.
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
os.environ["ROS_DOMAIN_ID"] = os.environ.get("ROBOTWORLD_ISAAC_DOMAIN_ID", "45")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--headless", action="store_true", help="창 없이 실행")
parser.add_argument("--seconds", type=float, default=0.0, help="이 시뮬레이션 시간만 돌리고 자체 점검 후 종료 (0 = 창을 닫을 때까지)")
parser.add_argument("--export", default=str(REPO / "artifacts/isaac"), help="export_assets.py 출력 폴더")
parser.add_argument("--reimport", action="store_true", help="URDF · MJCF 를 다시 USD 로 변환 (내보내기를 다시 했을 때)")
parser.add_argument("--spawn", default="", help="시작 자세: X,Y,YAW 또는 장소 이름(drinks ...). 비우면 booth.yaml spawn. WSL 쪽 spawn:= 과 같게")
parser.add_argument("--physics-hz", type=float, default=120.0)
parser.add_argument("--render-hz", type=float, default=30.0, help="화면 · 카메라를 그리는 주기 (물리 여러 스텝마다 한 번)")
parser.add_argument("--cameras", default="base_camera,head_camera", help="쉼표로 구분, none = 카메라 없음 (주행만 볼 때 가볍다)")
parser.add_argument("--camera-size", default="320x240")
parser.add_argument("--camera-fps", type=float, default=2.0, help="시뮬레이션 시간 기준 발행 주기")
parser.add_argument("--no-realtime", action="store_true", help="실시간보다 빨라도 늦추지 않는다")
parser.add_argument("--no-booth", action="store_true", help="부스 없이 바닥과 로봇만 (로봇 단독 시험)")
parser.add_argument("--debug", action="store_true", help="처음 5 s 동안 0.1 s 마다 차체 높이 · 가장 빠른 관절을 찍는다 (발산 원인 찾기)")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402

for _ext in ("isaacsim.ros2.bridge", "isaacsim.asset.importer.urdf", "isaacsim.asset.importer.mjcf"):
    app_utils.enable_extension(_ext)
simulation_app.update()

import carb  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig  # noqa: E402
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402
from isaacsim.core.experimental.objects import DistantLight  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.core.rendering_manager import RenderingManager, ViewportManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents  # noqa: E402
from omni.physx import get_physx_scene_query_interface  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

import rclpy  # noqa: E402  (ros2.bridge 확장이 켜져야 Isaac Sim 안의 Jazzy rclpy 를 import 할 수 있다)
from builtin_interfaces.msg import Time  # noqa: E402
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, TransformStamped, Twist, TwistStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from sensor_msgs.msg import Image, JointState, LaserScan  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src/mobile_openarm_mujoco"))
from isaac_physics import IsaacPhysics  # noqa: E402
from mobile_openarm_mujoco.actions import Controllers  # noqa: E402  원본 그대로 (mujoco 를 import 하지 않는 모듈)

EXPORT = Path(args.export).resolve()
SIM = json.loads((EXPORT / "sim.json").read_text(encoding="utf-8"))
SETTINGS, WORLD = SIM["settings"], SIM["world"]
RATES = SETTINGS["rates"]
ROBOT, BOOTH = "/World/Robot", "/World/Booth"
LOG = "[ISAAC-BOOTH]"


def say(text):
    print(f"{LOG} {text}", flush=True)


def stamp(seconds):
    ns = round(seconds * 1e9)
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def quat_matrix(wxyz):
    w, x, y, z = wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def pose_matrix(position, wxyz):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = quat_matrix(wxyz), position
    return T


def yaw_of(T):
    return math.atan2(T[1, 0], T[0, 0])


def planar(pose, x, y, yaw):
    pose.position.x, pose.position.y = float(x), float(y)
    pose.orientation.z, pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)


def resolve_spawn(text):
    """'' → booth.yaml spawn, 'X,Y,YAW' → 그대로, 장소 이름 → locations.yaml base_goal (warehouse.launch.py 와 같은 규칙)."""
    text = text.strip()
    if not text:
        return tuple(float(v) for v in WORLD["spawn"])
    try:
        x, y, yaw = (float(v) for v in text.split(","))
        return x, y, yaw
    except ValueError:
        if text not in SIM["locations"]:
            raise SystemExit(f"--spawn {text!r}: X,Y,YAW 도 장소 이름도 아니다 ({', '.join(SIM['locations'])})")
        return tuple(float(v) for v in SIM["locations"][text])


# ------------------------------------------------------------------ 1. URDF · MJCF → USD (한 번 변환해 두고 다시 쓴다)
def convert():
    record = EXPORT / "usd/import.json"
    sources = [EXPORT / "robot.urdf", EXPORT / "world/booth_world.xml"]
    if record.exists() and not args.reimport and record.stat().st_mtime > max(s.stat().st_mtime for s in sources):
        paths = json.loads(record.read_text())
        if all(Path(p).exists() for p in paths.values()):
            return paths
    for sub in ("robot", "world"):
        shutil.rmtree(EXPORT / "usd" / sub, ignore_errors=True)
    t0 = time.time()
    packages = [{"name": n, "path": str(REPO / "src" / n)} for n in SIM["packages"]]
    # 구동기는 변환할 때 만든다 (URDF 에는 게인이 없어서 그냥 변환하면 관절에 구동기가 없다).
    # MuJoCo 와 같은 값: 팔 position kp · kv, 손가락 joint1 position (gripper kp · kv), 바퀴 velocity (motors.velocity_gain).
    # 손가락 joint2 는 mimic 이라 구동기를 두지 않는다. 단위는 SI (Nm/rad, N/m).
    arm, grip, motor = SETTINGS["arm"], SETTINGS["gripper"], SETTINGS["motors"]
    ARM_J, FINGER1, FINGER2, WHEEL = r".*openarm_(left|right)_joint[1-7]$", r".*_finger_joint1$", r".*_finger_joint2$", r".*_wheel_joint$"
    robot = URDFImporter(URDFImporterConfig(
        urdf_path=str(EXPORT / "robot.urdf"), usd_path=str(EXPORT / "usd/robot"), ros_package_paths=packages,
        joint_drive_type="force",
        joint_target_type={ARM_J: "position", FINGER1: "position", FINGER2: "none", WHEEL: "velocity"},
        override_joint_stiffness={ARM_J: float(arm["kp"]), FINGER1: float(grip["kp"]), WHEEL: 0.0},
        override_joint_damping={ARM_J: float(arm["kv"]), FINGER1: float(grip["kv"]), WHEEL: float(motor["velocity_gain"])},
    )).import_urdf()
    world = MJCFImporter(MJCFImporterConfig(mjcf_path=str(EXPORT / "world/booth_world.xml"), usd_path=str(EXPORT / "usd/world"),
                                            import_scene=False)).import_mjcf()
    paths = {"robot": robot, "world": world}
    record.write_text(json.dumps(paths, indent=1))
    say(f"converted URDF + MJCF to USD in {time.time() - t0:.0f} s")
    return paths


usd = convert()
stage_utils.create_new_stage()
stage_utils.add_reference_to_stage(usd["robot"], ROBOT)
if args.no_booth:
    from isaacsim.core.experimental.objects import GroundPlane

    GroundPlane("/World/GroundPlane", positions=[0.0, 0.0, 0.0])
else:
    stage_utils.add_reference_to_stage(usd["world"], BOOTH)
stage = omni.usd.get_context().get_stage()
# 변환기는 물리 설정을 변형(variant)으로 나눠 둔다. 고르지 않으면 관절 · 강체가 없다.
for path, variant in ((ROBOT, "physx"), (BOOTH, "physics")):
    prim = stage.GetPrimAtPath(path)
    sets = prim.GetVariantSets() if prim.IsValid() else None
    if sets and sets.HasVariantSet("Physics"):
        sets.GetVariantSet("Physics").SetVariantSelection(variant)
SPAWN = resolve_spawn(args.spawn)
# 시작 위치 아래 바닥 윗면 (부스 바닥판 0.02 m). 바닥판 속에 박힌 채 시작하면 PhysX 가 밀어내며 발산한다.
FLOOR_Z = 0.0 if args.no_booth else max([b[4] for b in SIM.get("floor_boxes", []) if b[0] <= SPAWN[0] <= b[1] and b[2] <= SPAWN[1] <= b[3]] + [0.0])
UsdGeom.XformCommonAPI(stage.GetPrimAtPath(ROBOT)).SetTranslate(Gf.Vec3d(SPAWN[0], SPAWN[1], FLOOR_Z + 0.002))
UsdGeom.XformCommonAPI(stage.GetPrimAtPath(ROBOT)).SetRotate(Gf.Vec3f(0.0, 0.0, math.degrees(SPAWN[2])))
# MJCF 변환기는 조명을 옮기지 않는다: 부스 위 해와 하늘빛만 둔다.
DistantLight("/World/Sun").set_intensities(2500)
UsdLux.DomeLight.Define(stage, "/World/Sky").CreateIntensityAttr(600)
while stage_utils.is_stage_loading():
    simulation_app.update()


def prims_named(root):
    """root 아래 prim 이름 → 경로 (같은 이름이면 처음 것)."""
    out = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        out.setdefault(prim.GetName(), prim.GetPath().pathString)
    return out


def world_matrix(path):
    m = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m).T          # Gf 는 행벡터 규약 → 열벡터 4x4


ROBOT_PRIMS = prims_named(ROBOT)
# URDF 에 질량이 없는 링크(카메라 하우징 · 지지대 · 로고)도 PhysX 에서는 관절체 링크가 된다. 질량 0 이면 음수 질량으로 들어가
# 물리가 발산한다 (MuJoCo 는 자유도 없는 무질량 몸체를 허용한다). 작은 질량을 넣는다.
for _prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
    if _prim.HasAPI(UsdPhysics.RigidBodyAPI):
        _mass = UsdPhysics.MassAPI(_prim).GetMassAttr().Get() if _prim.HasAPI(UsdPhysics.MassAPI) else None
        if not _mass or _mass <= 0:
            _api = UsdPhysics.MassAPI.Apply(_prim)
            _api.CreateMassAttr(0.02)
            _api.CreateDiagonalInertiaAttr(Gf.Vec3f(2e-5, 2e-5, 2e-5))
            _api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
            print(f"{LOG} massless link {_prim.GetName()} -> 0.02 kg", flush=True)
ARTICULATION = next(p.GetPath().pathString for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)) if p.HasAPI(UsdPhysics.ArticulationRootAPI))
T_W_BL = world_matrix(ROBOT_PRIMS["base_link"])
T_FP_BL = np.linalg.inv(world_matrix(ROBOT_PRIMS[SIM["robot_root"]])) @ T_W_BL   # base_footprint → base_link (고정)
T_BL_LASER = np.linalg.inv(T_W_BL) @ world_matrix(ROBOT_PRIMS["laser_link"])     # base_link → laser_link (고정)
BOOTH_PRIMS = {} if args.no_booth else prims_named(BOOTH)
CUPS = [] if args.no_booth else [o["name"] for o in WORLD["objects"]]
missing = [c for c in CUPS if c not in BOOTH_PRIMS]
if missing:
    raise SystemExit(f"부스 USD 에 컵이 없다: {missing}")


# ------------------------------------------------------------------ 2. 마찰 재질 (MuJoCo 의 마찰 값을 PhysX 재질로)
def material(name, mu, combine="average"):
    mat = UsdShade.Material.Define(stage, f"/World/PhysicsMaterials/{name}")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(mu)
    api.CreateDynamicFrictionAttr(mu)
    api.CreateRestitutionAttr(0.0)
    mat.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(combine)
    return mat


def bind(path, mat):
    count = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
            count += 1
    return count


caster, wheel, finger, cup = material("caster", 0.0, "min"), material("wheel", 1.0), material("finger", 1.5, "max"), material("cup", 1.2)
for name, path in ROBOT_PRIMS.items():
    if name.endswith("_caster"):
        bind(path, caster)          # 캐스터는 고정 링크라서 마찰이 있으면 회전을 막는다 (MuJoCo 는 바퀴 접촉만 본다)
    elif name in ("left_wheel", "right_wheel"):
        bind(path, wheel)
    elif name.endswith("_finger"):
        bind(path, finger)
for c in CUPS:
    bind(BOOTH_PRIMS[c], cup)


# ------------------------------------------------------------------ 3. 카메라 (MuJoCo 의 CameraRig 대신 RTX 렌더)
CAMERAS = [] if args.cameras.strip().lower() == "none" else [c.strip() for c in args.cameras.split(",") if c.strip()]
CAM_W, CAM_H = (int(v) for v in args.camera_size.lower().split("x"))
annotators = {}
if CAMERAS:
    import omni.replicator.core as rep

    for name in CAMERAS:
        frame = ROBOT_PRIMS.get(f"{name}_optical_frame")
        if frame is None or name not in SIM["cameras"]["cameras"]:
            raise SystemExit(f"--cameras: {name} 이 로봇에 없다")
        cam = UsdGeom.Camera.Define(stage, f"{frame}/isaac_camera")
        fovy = math.radians(SIM["cameras"]["cameras"][name]["fovy"])
        aperture = 2 * 10.0 * math.tan(fovy / 2)           # 초점 거리 10 기준 세로 화각 = fovy
        cam.CreateFocalLengthAttr(10.0)
        cam.CreateVerticalApertureAttr(aperture)
        cam.CreateHorizontalApertureAttr(aperture * CAM_W / CAM_H)
        render = SIM["cameras"].get("render", {})
        cam.CreateClippingRangeAttr(Gf.Vec2f(render.get("near_m", 0.01), render.get("far_m", 50.0)))
        UsdGeom.XformCommonAPI(cam.GetPrim()).SetRotate(Gf.Vec3f(180.0, 0.0, 0.0))  # optical(+Z 앞, +Y 아래) → USD 카메라(-Z 앞, +Y 위)
        rp = rep.create.render_product(cam.GetPath().pathString, (CAM_W, CAM_H))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        annotators[name] = (ann, f"{name}_optical_frame")

SimulationManager.set_physics_dt(1.0 / args.physics_hz)
RenderingManager.set_dt(1.0 / args.render_hz)
if not args.headless:
    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[SPAWN[0] - 3.0, SPAWN[1] - 3.5, 3.2], target=[SPAWN[0], SPAWN[1], 0.6])


# ------------------------------------------------------------------ 4. ROS 2 노드 (bridge.py 의 Bridge 를 옮김)
class IsaacBridge(Node):
    """MuJoCo bridge.Bridge 와 같은 토픽 · 주기 · 인터록. Controllers 가 쓰는 physics · command · sim_stamp 를 가진다."""

    def __init__(self, physics):
        super().__init__("mobile_openarm_isaac")
        self.physics = physics
        self.command = (0.0, 0.0)
        self.command_time = -math.inf
        self.create_subscription(Twist, "/cmd_vel", self.receive_command, 10)
        self.create_subscription(TwistStamped, "/cmd_vel_stamped", lambda m: self.receive_command(m.twist), 10)
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_pub = self.create_publisher(TFMessage, "/tf", 100)
        self.truth_pub = self.create_publisher(PoseStamped, "/ground_truth", 10)
        self.objects_pub = self.create_publisher(PoseArray, "/warehouse/object_poses", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.image_pubs = {n: self.create_publisher(Image, f"/isaac/{n}/image_raw", 2) for n in CAMERAS}
        self.controllers = Controllers(self)

    def sim_stamp(self):
        return stamp(self.physics.data.time)

    def receive_command(self, msg):
        """bridge.Bridge.receive_command 와 같다: 유한값만, 인터록 중이면 0, 한계로 자른다."""
        if not all(math.isfinite(v) for v in [msg.linear.x, msg.angular.z]):
            return
        if self.controllers.reserved or not self.physics.transport_ready():
            self.command = (0.0, 0.0)
            return
        cfg = SETTINGS["control"]
        self.command = (max(-cfg["max_linear"], min(cfg["max_linear"], msg.linear.x)),
                        max(-cfg["max_angular"], min(cfg["max_angular"], msg.angular.z)))
        self.command_time = time.monotonic()

    def allowed(self):
        return (not self.controllers.reserved and self.physics.transport_ready()
                and time.monotonic() - self.command_time < SETTINGS["control"]["command_timeout"])

    def publish_state(self, now, T_w_fp, q, v, cups):
        p = self.physics
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = now, "odom", "base_footprint"
        planar(odom.pose.pose, *p.odom)
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = float(p.twist[0]), float(p.twist[1])
        for i, c in enumerate([0.0025, 0.0025, 1e6, 1e6, 1e6, 0.01]):
            odom.pose.covariance[7 * i] = c
            odom.twist.covariance[7 * i] = c
        self.odom_pub.publish(odom)
        tf = TransformStamped()
        tf.header, tf.child_frame_id = odom.header, odom.child_frame_id
        tf.transform.translation.x, tf.transform.translation.y = odom.pose.pose.position.x, odom.pose.pose.position.y
        tf.transform.rotation = odom.pose.pose.orientation
        self.tf_pub.publish(TFMessage(transforms=[tf]))
        js = JointState(name=p.dof_names, position=[float(x) for x in q], velocity=[float(x) for x in v])
        js.header.stamp = now
        self.joint_pub.publish(js)
        truth = PoseStamped()
        truth.header.stamp, truth.header.frame_id = now, "world"
        planar(truth.pose, T_w_fp[0, 3], T_w_fp[1, 3], yaw_of(T_w_fp))
        self.truth_pub.publish(truth)
        if cups is not None:
            poses = PoseArray()
            poses.header = truth.header
            for pos, quat in zip(*cups):
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = map(float, pos)
                pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = map(float, quat)
                poses.poses.append(pose)
            self.objects_pub.publish(poses)

    def publish_scan(self, now, ranges):
        lidar = SETTINGS["lidar"]
        scan = LaserScan()
        scan.header.stamp, scan.header.frame_id = now, "laser_link"
        scan.angle_min, scan.angle_max = float(ANGLES[0]), float(ANGLES[-1])
        scan.angle_increment = float(ANGLES[1] - ANGLES[0])
        scan.scan_time = 1.0 / RATES["scan_hz"]
        scan.range_min, scan.range_max = float(lidar["range_min"]), float(lidar["range_max"])
        scan.ranges = [float(r) for r in ranges]
        self.scan_pub.publish(scan)

    def publish_image(self, name, now, frame_id, rgba):
        img = Image()
        img.header.stamp, img.header.frame_id = now, frame_id
        img.height, img.width, img.encoding = rgba.shape[0], rgba.shape[1], "rgb8"
        img.step = rgba.shape[1] * 3
        img.data = np.ascontiguousarray(rgba[:, :, :3]).tobytes()
        self.image_pubs[name].publish(img)


# ------------------------------------------------------------------ 5. 라이다 (PhysX 레이캐스트, 로봇 자신은 건너뛴다)
LIDAR = SETTINGS["lidar"]
ANGLES = np.linspace(-math.pi, math.pi, LIDAR["samples"], endpoint=False)
RAYS = np.column_stack([np.cos(ANGLES), np.sin(ANGLES), np.zeros(len(ANGLES))])
query = get_physx_scene_query_interface()


def scan(T_w_bl):
    T = T_w_bl @ T_BL_LASER
    origin = tuple(float(v) for v in T[:3, 3])
    out = np.full(len(RAYS), np.inf)
    for i, d in enumerate(RAYS @ T[:3, :3].T):
        best = [math.inf]

        def hit(h, best=best):
            if not h.collision.startswith(ROBOT) and h.distance < best[0]:
                best[0] = h.distance
            return True

        query.raycast_all(origin, tuple(float(v) for v in d), float(LIDAR["range_max"]), hit)
        if LIDAR["range_min"] <= best[0] <= LIDAR["range_max"]:
            out[i] = best[0]
    return out


# ------------------------------------------------------------------ 6. 물리 스텝 콜백
timeline = omni.timeline.get_timeline_interface()
timeline.play()
simulation_app.update()
robot = Articulation(ARTICULATION)
cups_view = RigidPrim([BOOTH_PRIMS[c] for c in CUPS]) if CUPS else None
DOF = list(robot.dof_names)
lower, upper = (np.asarray(a.numpy() if hasattr(a, "numpy") else a)[0] for a in robot.get_dof_limits())
physics = IsaacPhysics(DOF, lower, upper, SIM)
POS_IDX = [DOF.index(n) for n in physics.position_names]
WHEEL_IDX = [DOF.index(n) for n in physics.wheel_names]
FINGER_IDX = [DOF.index(n) for n in physics.position_names if "finger" in n]
ACT_FINGER = [(i, DOF.index(a[: -len("_position")])) for i, a in enumerate(physics.actuators) if "finger" in a]

# 구동기 게인은 변환 때 넣었다 (convert). 여기서는 힘 한계만 MuJoCo 와 맞춘다: 바퀴 max_torque, 손가락 gripper.max_force.
_kp, _kd = (g.numpy()[0] for g in robot.get_dof_gains())
say("dof gains kp " + ", ".join(f"{n.replace('openarm_', '')}={_kp[i]:g}/{_kd[i]:g}" for i, n in enumerate(DOF)))
robot.set_dof_max_efforts([SETTINGS["motors"]["max_torque"]] * 2, dof_indices=WHEEL_IDX)
robot.set_dof_max_efforts(physics.finger_max_efforts(), dof_indices=FINGER_IDX)
# 팔 중력 보상 (MuJoCo arm.bias_compensation 과 같은 효과): 팔 링크 · 손 · 손가락의 중력을 끈다. 든 물체의 무게는 그대로다.
ARM_LINKS = [i for i, n in enumerate(robot.link_names) if n.startswith(("openarm_left_", "openarm_right_")) and "link0" not in n]
if SETTINGS["arm"].get("bias_compensation", True):
    robot.set_link_enabled_gravities(False, link_indices=ARM_LINKS)
q0 = np.zeros(len(DOF))
for n, value in physics.targets.items():
    q0[DOF.index(n)] = value
robot.set_default_state(dof_positions=q0)
robot.reset_to_default_state()
_last_finger_effort = physics.finger_max_efforts()

ros = None
executor = None
state = {"t": 0.0, "error": None, "tick": 0, "next_state": 0.0, "next_scan": 0.0, "next_cam": 0.0, "count": {"js": 0, "scan": 0, "img": 0},
         "T_w_bl": T_W_BL, "z0": None, "nan_at": None, "cb_s": 0.0, "scan_s": 0.0}
STATE_DT, SCAN_DT = 1.0 / RATES["state_hz"], 1.0 / RATES["scan_hz"]


def on_physics_step(dt, _context):
    global _last_finger_effort
    if state["error"] or ros is None:
        return
    try:
        cb0 = time.perf_counter()
        state["t"] += dt
        t = state["t"]
        q = robot.get_dof_positions().numpy()[0]
        if state["nan_at"] is None and not np.all(np.isfinite(q)):
            state["nan_at"] = t
            say(f"physics diverged at t={t:.3f} s (NaN joint positions)")
        v = robot.get_dof_velocities().numpy()[0]
        efforts = None
        if any("gripper" in k for k in ros.controllers.active):
            f = robot.get_dof_projected_joint_forces().numpy()[0]
            efforts = np.zeros(len(physics.actuators))
            for a, d in ACT_FINGER:
                efforts[a] = f[d]
        physics.update_state(t, q, v, efforts)
        if args.debug and t <= 5.0 and (int(t * 10) != int((t - dt) * 10) or t < 2.5 * dt):
            pos, quat = (a.numpy()[0] for a in robot.get_world_poses())
            k = int(np.nanargmax(np.abs(v))) if np.any(np.isfinite(v)) else 0
            err = np.abs(q[POS_IDX] - physics.position_targets())
            e = int(np.nanargmax(err)) if np.any(np.isfinite(err)) else 0
            say(f"debug t={t:.3f} base xyz={np.round(pos, 4).tolist()} quat={np.round(quat, 3).tolist()} fastest {DOF[k]} v={v[k]:.3f} "
                f"worst {physics.position_names[e]} err={err[e]:.4f} wheels q={np.round(q[WHEEL_IDX], 3).tolist()} "
                f"v={np.round(v[WHEEL_IDX], 3).tolist()} odom={np.round(physics.odom, 4).tolist()}")
            if t < 2.5 * dt:
                say("debug q " + " ".join(f"{n.replace('openarm_', '')}={q[i]:.3f}" for i, n in enumerate(DOF)))
                say("debug target " + " ".join(f"{n.replace('openarm_', '')}={x:.3f}" for n, x in zip(physics.position_names, physics.position_targets())))
        for _ in range(int(RATES.get("ros_callbacks_per_tick", 8))):
            executor.spin_once(timeout_sec=0.0)
        ros.controllers.update()                                # 원본 궤적 · 그리퍼 액션 진행
        linear, angular = ros.command if ros.allowed() else (0.0, 0.0)
        robot.set_dof_velocity_targets(physics.wheel_targets(linear, angular, dt), dof_indices=WHEEL_IDX)
        robot.set_dof_position_targets(physics.position_targets(), dof_indices=POS_IDX)
        effort = physics.finger_max_efforts()
        if effort != _last_finger_effort:                       # GripperCommand 의 max_effort 반영
            robot.set_dof_max_efforts(effort, dof_indices=FINGER_IDX)
            _last_finger_effort = effort
        physics.integrate_odom(dt)
        now = stamp(t)
        ros.clock_pub.publish(Clock(clock=now))
        if t >= state["next_state"] or t >= state["next_scan"]:
            pos, quat = (a.numpy()[0] for a in robot.get_world_poses())
            state["T_w_bl"] = pose_matrix(pos, quat)
        if t >= state["next_state"]:
            state["tick"] += 1
            cups = None
            if cups_view is not None and state["tick"] % 10 == 0:
                cp, cq = cups_view.get_world_poses()
                cups = (cp.numpy(), cq.numpy())
            ros.publish_state(now, state["T_w_bl"] @ np.linalg.inv(T_FP_BL), q, v, cups)
            state["count"]["js"] += 1
            state["next_state"] += STATE_DT
        if t >= state["next_scan"]:
            s0 = time.perf_counter()
            ros.publish_scan(now, scan(state["T_w_bl"]))
            state["scan_s"] += time.perf_counter() - s0
            state["count"]["scan"] += 1
            state["next_scan"] += SCAN_DT
        state["cb_s"] += time.perf_counter() - cb0
    except Exception:  # noqa: BLE001 - 물리 콜백은 예외를 밖으로 던지면 안 된다
        state["error"] = traceback.format_exc()
        carb.log_error(f"booth_sim: physics callback failed\n{state['error']}")


rclpy.init()
ros = IsaacBridge(physics)
executor = rclpy.executors.SingleThreadedExecutor()
executor.add_node(ros)
callback_id = SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)
for _ in range(3):
    simulation_app.update()
state["z0"] = float(state["T_w_bl"][2, 3])
say(f"ready | domain {os.environ['ROS_DOMAIN_ID']} | spawn {SPAWN} | {len(DOF)} dofs | cameras {CAMERAS or 'none'} | "
    f"physics {args.physics_hz:g} Hz | actions left/right_joint_trajectory_controller, left/right_gripper_controller")

# ------------------------------------------------------------------ 7. 메인 루프 (렌더 · 카메라 발행 · 실시간 맞추기)
CAM_DT = 1.0 / args.camera_fps
wall0, sim0 = time.monotonic(), state["t"]
while simulation_app.is_running() and not state["error"]:
    simulation_app.update()
    t = state["t"]
    if annotators and t >= state["next_cam"]:
        for name, (ann, frame_id) in annotators.items():
            data = ann.get_data()
            if data is not None and getattr(data, "size", 0):
                ros.publish_image(name, stamp(t), frame_id, np.asarray(data))
                state["count"]["img"] += 1
        state["next_cam"] = t + CAM_DT
    if not args.no_realtime:
        ahead = (t - sim0) - (time.monotonic() - wall0)
        if ahead > 0:
            time.sleep(min(ahead, 0.1))
    if args.seconds and t >= args.seconds:
        break

# ------------------------------------------------------------------ 8. 자체 점검 (--seconds 를 줬을 때)
if args.seconds:
    sim_s, wall_s = state["t"] - sim0, time.monotonic() - wall0
    z = float(state["T_w_bl"][2, 3])
    checks = [
        ("physics callback without error", state["error"] is None, (state["error"] or "ok").strip().splitlines()[-1]),
        ("/joint_states at state_hz", state["count"]["js"] >= 0.9 * sim_s * RATES["state_hz"], f"{state['count']['js']} msgs"),
        ("/scan at scan_hz", state["count"]["scan"] >= 0.9 * sim_s * RATES["scan_hz"], f"{state['count']['scan']} msgs"),
        ("base stays on the floor", abs(z - state["z0"]) < 0.03, f"base_link z {state['z0']:.3f} -> {z:.3f} m"),
        ("arms hold a drive pose", physics.transport_ready(), "transport_ready"),
        ("camera images", not CAMERAS or state["count"]["img"] > 0, f"{state['count']['img']} images"),
    ]
    for name, ok, detail in checks:
        print(f"CHECK {'PASS' if ok else 'FAIL'} | {name} | {detail}", flush=True)
    qf = robot.get_dof_positions().numpy()[0]
    print("INFO fingers joint1/joint2 " + ", ".join(f"{s} {qf[DOF.index(f'openarm_{s}_finger_joint1')]:.4f}/{qf[DOF.index(f'openarm_{s}_finger_joint2')]:.4f}"
                                                for s in ("left", "right")), flush=True)
    print(f"REALTIME {sim_s / max(wall_s, 1e-6):.2f}x ({sim_s:.1f} s sim in {wall_s:.1f} s wall; physics callback "
          f"{state['cb_s']:.1f} s of which lidar {state['scan_s']:.1f} s; diverged at {state['nan_at']})", flush=True)
    print("SELF-CHECK", "PASS" if all(ok for _, ok, _ in checks) else "FAIL", flush=True)

timeline.stop()
SimulationManager.deregister_callback(callback_id)
ros.destroy_node()
rclpy.shutdown()
simulation_app.close()
