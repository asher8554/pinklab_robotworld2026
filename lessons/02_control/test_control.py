# PID · LQR 예제의 기대 동작을 확인하는 자체 점검: python lessons/02_control/test_control.py
import contextlib
import io

import lqr
import pid

COMMON = ["--seconds", "12"]


def run(main, argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return main(argv + COMMON)


r = run(pid.main, [])
assert r["fell"], "각도만 보는 PID 는 카트가 밀려 레일 끝에서 넘어져야 한다"
r = run(pid.main, ["--kx", "0.05", "--kv", "0.1", "--target", "1"])
assert not r["fell"] and abs(r["x"] - 1) < 0.1, r
r = run(lqr.main, [])
assert not r["fell"] and abs(r["x"] - 1) < 0.02 and abs(r["angle_deg"]) < 0.5, r
print("ok")
