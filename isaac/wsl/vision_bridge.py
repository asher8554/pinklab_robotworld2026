#!/usr/bin/env python3
# Isaac Sim 카메라 영상을 받아 원본 ArUco 검출기로 /vision/markers 를 내고, 웹 FMS 용 JPEG 를 남기는 노드 (WSL)
"""MuJoCo 실습에서는 카메라 핸들러(fms/frame_tap.py:FmsPipeline)가 시뮬레이터 프로세스 안에서 돈다.
Isaac Sim(Windows) 프로세스에는 이 저장소의 커스텀 메시지가 없으므로, Isaac 은 표준 sensor_msgs/Image 만 내고
이 노드가 WSL 에서 같은 일을 한다.

  구독  /isaac/<카메라>/image_raw (rgb8, booth_sim.py)
  발행  /vision/markers (warehouse_interfaces/ArucoMarkerArray, base_footprint) — 원본 ArucoDetector 그대로
  파일  /dev/shm/robotworld_fms/<카메라>.jpg · meta.json — FMS 가 카메라를 켰을 때만 (fms/frame_tap.py 와 같은 규칙)

카메라 내부 행렬은 MuJoCo CameraRig 와 같은 식이다: f = H / (2 tan(fovy/2)), 주점 ((W-1)/2, (H-1)/2).
카메라 → base_footprint 변환은 robot_state_publisher 의 TF 에서 읽는다 (MuJoCo 판은 시뮬레이터 모델에서 읽는다).
"""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener

from mobile_openarm_mujoco.model import camera_settings, load_world
from warehouse_lecture.vision.aruco import ArucoDetector

FRAMES = Path(os.environ.get("ROBOTWORLD_FRAMES", "/dev/shm/robotworld_fms"))


def intrinsics(fovy_deg, width, height):
    focal = height / (2 * math.tan(math.radians(fovy_deg) / 2))
    return np.array([[focal, 0.0, (width - 1) / 2], [0.0, focal, (height - 1) / 2], [0.0, 0.0, 1.0]])


def matrix(transform):
    t, q = transform.translation, transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [t.x, t.y, t.z]
    return T


class VisionBridge(Node):
    def __init__(self, cameras, aruco_camera):
        super().__init__("isaac_vision_bridge", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.fovy = {n: camera_settings()["cameras"][n]["fovy"] for n in cameras}
        self.tf = Buffer()
        TransformListener(self.tf, self)
        # 원본 검출기를 그대로 쓴다: on_setup 이 /vision/markers 발행자 · 마커 크기 · ArUco 사전을 만든다.
        self.aruco = ArucoDetector(camera=aruco_camera)
        self.aruco.on_setup(SimpleNamespace(node=self, world=load_world(), camera_names=tuple(cameras)))
        self.meta = {}
        for name in cameras:
            self.create_subscription(Image, f"/isaac/{name}/image_raw", lambda m, n=name: self.on_image(n, m), 2)
        self.get_logger().info(f"vision bridge: {cameras} -> /vision/markers ({aruco_camera}), FMS frames in {FRAMES}")

    def on_image(self, name, msg):
        rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if name == self.aruco.camera:
            try:
                T_bo = matrix(self.tf.lookup_transform("base_footprint", msg.header.frame_id, Time()).transform)
            except TransformException as error:
                self.get_logger().warning(f"no TF base_footprint <- {msg.header.frame_id}: {error}", throttle_duration_sec=5.0)
            else:
                K = intrinsics(self.fovy[name], msg.width, msg.height)
                self.aruco.publish(self.aruco.process((rgb, K, T_bo, sim_time)))
        if (FRAMES / "enabled").exists():          # FMS 가 카메라를 켰을 때만 인코딩한다 (fms/frame_tap.py 와 같음)
            ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                tmp = FRAMES / f".{name}.jpg.tmp"
                tmp.write_bytes(jpg.tobytes())
                os.replace(tmp, FRAMES / f"{name}.jpg")
                self.meta[name] = dict(sim_time=sim_time, width=msg.width, height=msg.height)
                tmp = FRAMES / ".meta.json.tmp"
                tmp.write_text(json.dumps(self.meta))
                os.replace(tmp, FRAMES / "meta.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", default="base_camera,head_camera")
    ap.add_argument("--aruco-camera", default="base_camera")
    a = ap.parse_args()
    FRAMES.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = VisionBridge([c.strip() for c in a.cameras.split(",") if c.strip()], a.aruco_camera)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
