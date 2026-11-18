#!/usr/bin/env python3
# MuJoCo 쪽 원본 코드로 Isaac Sim에 넘길 로봇 URDF · 로봇 없는 부스 월드 MJCF · 설정 JSON을 만드는 내보내기 스크립트 (WSL)
"""Isaac Sim 경로의 입력 파일을 만든다. MuJoCo 실습과 같은 함수로 만들어서 두 시뮬레이터의 로봇 · 부스가 같다.

    isaac/wsl/robotworld-isaac export                       # 기본 출력: 저장소의 artifacts/isaac
    python isaac/export_assets.py --out /mnt/e/.../artifacts/isaac

만드는 것 (--out 아래)
  robot.urdf          expand_urdf() 결과. 주석을 뺀 URDF. 메시는 package:// 그대로 (Isaac 쪽이 src/<패키지>로 푼다)
  world/booth_world.xml   build_model()이 만든 MuJoCo 모델에서 로봇 몸체 · 로봇 전용 항목을 뺀 부스 월드
  world/...           월드가 쓰는 텍스처 · 메시 사본 (상대 경로로 바꿔 둔다)
  sim.json            mujoco.yaml · booth.yaml · 초기 자세 · 주행 자세 · 카메라 · 바퀴 치수
원본 파일은 읽기만 한다. 빌드 중간 파일은 --build-dir(WSL 로컬)에 둔다.
"""
import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from mobile_openarm_description.model import expand_urdf
from mobile_openarm_mujoco.model import (
    build_model,
    camera_settings,
    drive_poses,
    initial_positions,
    load_settings,
    load_world,
)

ROBOT_ROOT = "base_footprint"           # build_model()이 worldbody 바로 아래에 두는 로봇 몸체
ROBOT_SECTIONS = ("contact", "equality", "actuator")   # 로봇 관절 · 링크만 가리키는 MJCF 구역


def strip_robot(xml_path):
    """MuJoCo 모델에서 로봇 몸체와 로봇 전용 구역을 빼고, 남은 지오메트리가 쓰지 않는 메시 자산도 뺀다."""
    root = ET.parse(xml_path).getroot()
    world = root.find("worldbody")
    robot = world.find(f"body[@name='{ROBOT_ROOT}']")
    if robot is None:
        raise SystemExit(f"{xml_path}: worldbody 에 {ROBOT_ROOT} 몸체가 없다 (build_model 구조가 바뀌었는지 확인)")
    world.remove(robot)
    for tag in ROBOT_SECTIONS:
        for element in root.findall(tag):
            root.remove(element)
    used = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    asset = root.find("asset")
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") not in used:
            asset.remove(mesh)
    return root


def copy_files(root, out_dir, roots):
    """자산의 절대 경로 file= 을 out_dir 아래 사본의 상대 경로로 바꾼다. roots: [(접두 디렉터리, 사본 하위 폴더)]."""
    copied = 0
    for element in root.find("asset"):
        src = element.get("file")
        if not src:
            continue
        path = Path(src)
        for base, sub in roots:
            try:
                rel = Path(sub) / path.relative_to(base)
                break
            except ValueError:
                continue
        else:
            rel = Path("ext") / path.name
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_mtime < path.stat().st_mtime or dst.stat().st_size != path.stat().st_size:
            shutil.copy2(path, dst)
            copied += 1
        element.set("file", rel.as_posix())
    return copied


def inline_meshes_to_obj(root, out_dir):
    """vertex= · face= 로 MJCF 안에 직접 적힌 메시(부스의 물결 간판 등)를 OBJ 파일로 바꾼다. Isaac 변환기는 파일 메시만 읽는다."""
    count = 0
    for mesh in root.find("asset").findall("mesh"):
        if mesh.get("file") or not mesh.get("vertex"):
            continue
        v = [float(x) for x in mesh.get("vertex").split()]
        f = [int(x) + 1 for x in mesh.get("face").split()]
        uv = [float(x) for x in mesh.get("texcoord", "").split()]
        lines = [f"v {v[i]} {v[i + 1]} {v[i + 2]}" for i in range(0, len(v), 3)]
        lines += [f"vt {uv[i]} {uv[i + 1]}" for i in range(0, len(uv), 2)]
        tri = [f[i:i + 3] for i in range(0, len(f), 3)]
        lines += [("f " + " ".join(f"{a}/{a}" for a in t)) if uv else ("f " + " ".join(map(str, t))) for t in tri]
        rel = Path("inline") / f"{mesh.get('name')}.obj"
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / rel).write_text("\n".join(lines) + "\n")
        for key in ("vertex", "face", "texcoord"):
            mesh.attrib.pop(key, None)
        mesh.set("file", rel.as_posix())
        count += 1
    return count


def visual_only_geoms(root):
    """MuJoCo 에서 충돌하지 않는 지오메트리 이름 (contype = conaffinity = 0: 벽 그래픽 · 바닥 글씨 · 머리 위 간판).
    Isaac MJCF 변환기는 이 필터를 옮기지 않으므로 Isaac 쪽이 이 목록의 충돌을 끈다."""
    return sorted(g.get("name") for g in root.iter("geom")
                  if g.get("name") and g.get("contype", "1") == "0" and g.get("conaffinity", "1") == "0")


def floor_boxes(root):
    """로봇이 올라설 수 있는 낮은 충돌 상자 [x0, x1, y0, y1, 윗면 z] (부스 바닥판 2 cm 등). 회전 없는 worldbody 직속 상자만.
    Isaac 은 겹친 채 시작하면 순간적으로 밀어내며 발산하므로, 시작 위치 아래 바닥 윗면에 로봇을 올려놓는 데 쓴다."""
    out = []
    for g in root.find("worldbody").findall("geom"):
        if g.get("type") != "box" or g.get("quat") or g.get("euler") or g.get("contype", "1") == "0":
            continue
        (x, y, z), (hx, hy, hz) = ([float(v) for v in g.get(k).split()] for k in ("pos", "size"))
        if z + hz < 0.1:
            out.append([x - hx, x + hx, y - hy, y + hy, z + hz])
    return out


def wheel_geometry(urdf_root):
    """바퀴 반지름(충돌 원기둥)과 좌우 바퀴 간격 [m]. MuJoCo Physics 가 모델에서 읽는 값과 같다."""
    radius = float(urdf_root.find("link[@name='left_wheel']/collision/geometry/cylinder").get("radius"))
    y = {}
    for side in ("left", "right"):
        joint = urdf_root.find(f"joint[@name='{side}_wheel_joint']")
        y[side] = float(joint.find("origin").get("xyz").split()[1])
    return {"radius": radius, "track": abs(y["left"] - y["right"])}


def packages(urdf_text):
    """URDF 메시가 쓰는 ROS 패키지 이름 (package://이름/...)."""
    names = set()
    for part in urdf_text.split("package://")[1:]:
        names.add(part.split("/", 1)[0])
    return sorted(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="출력 폴더 (Windows Isaac Sim이 읽을 수 있는 곳, 예: /mnt/e/.../artifacts/isaac)")
    ap.add_argument("--build-dir", default=str(Path.home() / ".cache/robotworld_isaac/build"),
                    help="build_model 중간 파일 (메시 변환 캐시, WSL 로컬 디스크가 빠르다)")
    args = ap.parse_args()
    out, build = Path(args.out).expanduser().resolve(), Path(args.build_dir).expanduser().resolve()
    (out / "world").mkdir(parents=True, exist_ok=True)

    urdf = expand_urdf()
    urdf_root = ET.fromstring(urdf)                 # 주석은 여기서 빠진다 (xacro 가 남긴 주석 속 ${...} 줄)
    ET.ElementTree(urdf_root).write(out / "robot.urdf", encoding="unicode")

    xml_path = build_model(build, urdf)             # MuJoCo 실습과 같은 월드 조립 (부스 · 컵 · 마커 · 사람)
    world = strip_robot(xml_path)
    copied = copy_files(world, out / "world", [(build, "build")])
    inline = inline_meshes_to_obj(world, out / "world")
    ET.indent(world)
    ET.ElementTree(world).write(out / "world/booth_world.xml", encoding="unicode")

    sim = {
        "robot_root": ROBOT_ROOT,
        "visual_only_geoms": visual_only_geoms(world),
        "floor_boxes": floor_boxes(world),
        "packages": packages(urdf),
        "wheel": wheel_geometry(urdf_root),
        "settings": load_settings(),
        "world": load_world(),
        "initial_positions": initial_positions(),
        "drive_poses": drive_poses(),
        "cameras": camera_settings(),
        "locations": {k: v["base_goal"] for k, v in yaml.safe_load(
            (Path(get_package_share_directory("warehouse_lecture")) / "worlds/locations.yaml").read_text())["locations"].items()},
    }
    (out / "sim.json").write_text(json.dumps(sim, indent=1, ensure_ascii=False))
    geoms = sum(1 for _ in world.iter("geom"))
    print(f"[export] {out}: robot.urdf, world/booth_world.xml ({geoms} geoms, {copied} files copied, {inline} inline meshes "
          f"-> OBJ, {len(sim['visual_only_geoms'])} visual-only), sim.json "
          f"(packages {sim['packages']}, wheel r={sim['wheel']['radius']} track={sim['wheel']['track']:.4f})", flush=True)


if __name__ == "__main__":
    main()
