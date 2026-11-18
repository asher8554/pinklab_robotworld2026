# PID · LQR 예제가 같이 쓰는 카트-폴 시뮬레이션 루프: 제어 함수 호출 → mj_step → 기록 · 외란 · 뷰어
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

MJCF = Path(__file__).with_name("cartpole.xml")
FALL_RAD = 0.8                                   # 막대가 이만큼 기울면 넘어졌다고 보고 멈춘다


def load():
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    return model, mujoco.MjData(model)


def add_args(parser):
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--angle0", type=float, default=0.15, help="처음 막대 기울기 [rad]")
    parser.add_argument("--push", type=float, default=3.0, help="t=--push-at 에 막대 끝을 0.1 s 미는 힘 [N], 0 이면 없음")
    parser.add_argument("--push-at", type=float, default=4.0)
    parser.add_argument("--viewer", action="store_true", help="MuJoCo 창으로 실시간 보기")


def simulate(model, data, control, args):
    """control(data, dt) -> 카트 힘 [N]. 매 스텝(2 ms) 호출. 결과 요약 dict 반환."""
    dt = model.opt.timestep
    pole = model.body("pole").id
    data.qpos[1] = args.angle0
    mujoco.mj_forward(model, data)
    lo, hi = model.actuator_ctrlrange[0]
    viewer = mujoco.viewer.launch_passive(model, data) if args.viewer else None
    max_angle, fell = 0.0, False

    print(f"{'t[s]':>5} {'x[m]':>7} {'angle[deg]':>10} {'u[N]':>7}")
    while data.time < args.seconds:
        start = time.perf_counter()
        data.ctrl[0] = np.clip(control(data, dt), lo, hi)
        pushing = args.push and args.push_at <= data.time < args.push_at + 0.1
        data.xfrc_applied[pole, 0] = args.push if pushing else 0.0    # 외란: 막대에 +x 방향 힘
        mujoco.mj_step(model, data)

        x, angle = data.qpos
        max_angle = max(max_angle, abs(angle))
        if round(data.time / dt) % 250 == 0:                        # 0.5 s 마다 출력
            print(f"{data.time:5.1f} {x:7.3f} {np.degrees(angle):10.2f} {data.ctrl[0]:7.2f}")
        if abs(angle) > FALL_RAD:
            fell = True
            print(f"넘어짐: t={data.time:.2f} s")
            break
        if viewer:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - start)))
    if viewer:
        viewer.close()

    result = {"fell": fell, "x": float(data.qpos[0]), "angle_deg": float(np.degrees(data.qpos[1])),
              "max_angle_deg": float(np.degrees(max_angle))}
    print(f"결과: 넘어짐 {fell}  최종 x {result['x']:.3f} m  최종 각도 {result['angle_deg']:.2f} deg  "
          f"최대 각도 {result['max_angle_deg']:.2f} deg")
    return result

