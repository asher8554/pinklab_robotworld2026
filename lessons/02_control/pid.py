# 카트-폴 막대 각도를 PID 로 세우는 예제: --kp --ki --kd 를 바꿔 가며 응답을 본다
"""
  python lessons/02_control/pid.py                    # 기본 이득
  python lessons/02_control/pid.py --kd 0             # D 를 빼면?
  python lessons/02_control/pid.py --kx 0.05 --kv 0.1 --target 1   # 바깥 루프로 카트 위치까지
  python lessons/02_control/pid.py --viewer           # MuJoCo 창으로

오차 e = 막대 각도 - 목표(0 rad = 똑바로 위).  u = Kp e + Ki ∫e dt + Kd de/dt  (카트에 주는 힘 [N])
막대가 +x 로 기울면(e>0) 카트를 +x 로 밀어 막대 밑을 받친다. 각도만 보므로 카트 위치는 제어하지 않는다.

--kx --kv 를 주면 바깥 루프(캐스케이드)가 켜진다. 카트를 목표로 보내려면 막대를 먼저 그쪽으로 기울여야 하므로
바깥 루프가 각도 목표를 정한다: 목표 각도 = -(Kx (x - target) + Kv ẋ).  부호가 직관과 반대인 점에 주의.
"""
import argparse

import sim


class PID:
    def __init__(self, kp, ki, kd, setpoint=0.0):
        self.kp, self.ki, self.kd, self.setpoint = kp, ki, kd, setpoint
        self.integral = 0.0
        self.prev_error = None

    def __call__(self, measured, dt):
        error = measured - self.setpoint
        self.integral += error * dt
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--ki", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=8.0)
    parser.add_argument("--kx", type=float, default=0.0, help="바깥 루프 위치 이득 [rad/m], 0 이면 끔")
    parser.add_argument("--kv", type=float, default=0.0, help="바깥 루프 속도 이득 [rad/(m/s)]")
    parser.add_argument("--target", type=float, default=0.0, help="카트 목표 위치 [m] (바깥 루프가 켜졌을 때)")
    sim.add_args(parser)
    args = parser.parse_args(argv)

    model, data = sim.load()
    pid = PID(args.kp, args.ki, args.kd)
    print(f"PID  Kp {args.kp}  Ki {args.ki}  Kd {args.kd}  바깥 루프 Kx {args.kx}  Kv {args.kv}  target {args.target}")

    def control(d, dt):
        pid.setpoint = -(args.kx * (d.qpos[0] - args.target) + args.kv * d.qvel[0])
        return pid(d.qpos[1], dt)

    return sim.simulate(model, data, control, args)


if __name__ == "__main__":
    main()
