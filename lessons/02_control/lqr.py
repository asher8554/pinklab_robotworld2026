# 카트-폴을 MuJoCo 로 선형화하고 LQR 로 세우면서 카트를 목표 위치로 보내는 예제
"""
  python lessons/02_control/lqr.py                        # 막대를 세운 채 카트를 x=1 m 로
  python lessons/02_control/lqr.py --q 1 1 1 1 --r 10     # 힘을 아끼면?
  python lessons/02_control/lqr.py --viewer

상태 x = [카트 위치, 막대 각도, 카트 속도, 막대 각속도], 입력 u = 카트 힘 [N].
1. 똑바로 선 자세에서 mjd_transitionFD 로 x[k+1] ≈ A x[k] + B u[k] 를 구한다 (수치 미분, 식 유도 없음).
2. 비용 Σ (xᵀQx + uᵀRu) 를 최소로 하는 K 를 이산 리카티 방정식으로 구한다.
3. u = -K (x - x_goal).  K 하나가 4개 상태를 한꺼번에 본다 → 각도만 보는 PID 와 달리 카트 위치도 잡는다.
"""
import argparse

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

import sim


def linearize(model, data):
    """똑바로 선 평형점(qpos=0, qvel=0, ctrl=0) 근처의 이산 시간 A, B."""
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    nx = 2 * model.nv
    A, B = np.zeros((nx, nx)), np.zeros((nx, model.nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    return A, B


def lqr_gain(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--q", type=float, nargs=4, default=[10.0, 100.0, 1.0, 1.0],
                        help="Q 대각: 위치 각도 속도 각속도 오차에 주는 벌점")
    parser.add_argument("--r", type=float, default=0.1, help="R: 힘을 쓰는 데 주는 벌점")
    parser.add_argument("--target", type=float, default=1.0, help="카트 목표 위치 [m]")
    sim.add_args(parser)
    args = parser.parse_args(argv)

    model, data = sim.load()
    A, B = linearize(model, data)
    K = lqr_gain(A, B, np.diag(args.q), np.array([[args.r]]))
    np.set_printoptions(precision=3, suppress=True)
    print(f"A =\n{A}\nB = {B.ravel()}\nK = {K.ravel()}")
    print("폐루프 고윳값 |λ| (모두 1 미만이면 안정):", np.abs(np.linalg.eigvals(A - B @ K)))

    goal = np.array([args.target, 0.0, 0.0, 0.0])
    mujoco.mj_resetData(model, data)
    return sim.simulate(model, data, lambda d, dt: (-K @ (np.concatenate([d.qpos, d.qvel]) - goal))[0], args)


if __name__ == "__main__":
    main()
