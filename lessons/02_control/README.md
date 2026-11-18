# 02. 제어 기법 실습 — PID 와 LQR (카트-폴)

## 목표

같은 카트-폴(레일 위 카트 + 경첩 막대)을 PID 와 LQR 로 세워 보고 차이를 숫자와 화면으로 본다.

- PID: 오차 하나(막대 각도)를 보고 힘을 정한다. 이득 3개를 손으로 맞춘다.
- LQR: 상태 전체(위치 · 각도 · 속도 · 각속도)를 보고 힘을 정한다. 이득을 비용 Q, R 에서 계산한다.

ROS 2 없이 MuJoCo Python 만 쓴다. 01 의 `mujoco_basics.py` 루프에 제어 함수 하나를 끼운 구조다.

## 파일

| 파일 | 내용 |
|---|---|
| `cartpole.xml` | 카트(slide, ±2.9 m) + 막대(hinge, 0.6 m). 액추에이터는 카트 힘 `motor` (±20 N). 0 rad = 똑바로 위 |
| `sim.py` | 공통 루프: `control(data, dt)` 호출 → `mj_step`, 0.5 s 마다 출력, t=4 s 에 막대를 3 N 으로 0.1 s 밈(외란) |
| `pid.py` | `PID` 클래스(약 10줄). 각도 PID, 선택으로 카트 위치 바깥 루프(캐스케이드) |
| `lqr.py` | `mjd_transitionFD` 로 선형화 → `solve_discrete_are` 로 K → `u = -K(x - x_goal)` |
| `test_control.py` | 아래 1 · 3 · 4 의 결과가 기대대로인지 확인 |

## 실행

```bash
source scripts/env.sh                                       # .venv (mujoco, numpy, scipy)

# 1. 각도만 보는 PID
python lessons/02_control/pid.py
# 2. 이득 바꿔 보기
python lessons/02_control/pid.py --kd 0
python lessons/02_control/pid.py --kp 15
# 3. 바깥 루프로 카트 위치까지 (캐스케이드 PID)
python lessons/02_control/pid.py --kx 0.05 --kv 0.1 --target 1 --seconds 20
# 4. LQR
python lessons/02_control/lqr.py
python lessons/02_control/lqr.py --q 1 1 1 1 --r 10 --seconds 20

# 창으로 보려면 아무 명령에나 --viewer
python lessons/02_control/lqr.py --viewer

python lessons/02_control/test_control.py                   # ok
```

공통 옵션: `--seconds` 길이, `--angle0` 처음 기울기 [rad], `--push` 외란 힘 [N] (0 이면 없음), `--push-at` 외란 시각.

## 화면에서 볼 것

1. **각도 PID** — 막대는 1 s 안에 서지만 `x` 가 0.3 m/s 로 계속 늘어난다. 외란 뒤 더 빨리 밀려 레일 끝(2.9 m)에 닿고 넘어진다(약 5 s). 각도 오차만 0 으로 만들 뿐, 카트 위치 · 속도는 PID 의 관심 밖이다.
2. **이득** (`--push 0` 으로 외란을 빼고 비교하면 잘 보인다)
   - `--kd 0`: 감쇠가 없어 ±5° 진동이 계속된다.
   - `--kp 15`: 기울기를 못 따라가 6° 쯤 기운 채로 카트가 가속, 2.3 s 만에 레일 끝.
   - `--ki 50`: 남은 각도 오차가 적분으로 지워져 밀려남이 줄어든다(3 s 에 0.39 m, 기본 0.89 m).
   - 셋 다 기본 외란(3 N)을 넣으면 결국 레일 끝에서 넘어진다. 카트 위치를 안 보는 구조의 한계다.
3. **캐스케이드 PID** — 바깥 루프가 "카트를 1 m 로 보내려면 막대를 얼마나 기울일지"를 정하고 안쪽 루프가 그 각도를 따른다. 20 s 에 x ≈ 1.000 m. 목표 각도 = −(Kx(x−target) + Kv·ẋ) 로 **부호가 직관과 반대**다. 카트를 +x 로 보내려면 먼저 −x 로 살짝 빼서 막대를 +x 로 기울여야 하기 때문이다(비최소 위상). 이득 5개를 손으로 맞춰야 한다.
4. **LQR** — 출력의 `K` 4개가 위치 · 각도 · 속도 · 각속도 이득이다. 기본값(Q = diag(10, 100, 1, 1), R = 0.1)으로 10 s 에 x = 1.000 m, 각도 ≈ 0. 외란도 받아낸다. 캐스케이드의 반직관적 부호가 `K` 안에 이미 들어 있다(위치 이득과 각도 이득의 부호가 같다).
5. **Q, R 조절** — `--r 10` 이면 힘을 아껴 느리게 간다(20 s 에 x ≈ 2.0 m 은 아직 수렴 전, 더 길게 돌려 본다). Q 의 각도 항을 키우면 막대를 더 꼿꼿이 세우고, 위치 항을 키우면 카트를 더 서둘러 보낸다.

## 핵심 코드

PID (`pid.py`):

```python
error = measured - self.setpoint
self.integral += error * dt
derivative = (error - self.prev_error) / dt
u = kp * error + ki * self.integral + kd * derivative
```

LQR (`lqr.py`):

```python
mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)   # x[k+1] ≈ A x[k] + B u[k]
P = solve_discrete_are(A, B, Q, R)
K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
u = -K @ (np.concatenate([data.qpos, data.qvel]) - goal)
```

운동 방정식을 손으로 유도하지 않는다. MuJoCo 가 평형점 근처에서 한 스텝을 수치 미분해 A, B 를 준다. 그래서 모델(XML)을 바꿔도 코드는 그대로다.

## 해 볼 것

- `cartpole.xml` 의 막대 `mass` 를 0.2 → 1.0 으로. PID 이득은 다시 맞춰야 하지만 LQR 은 다시 선형화하므로 그대로 돈다.
- `--angle0 0.5` 처럼 크게 기울여 시작. LQR 은 0 rad 근처 선형 모델이라 멀어질수록 성능이 떨어지고, ±20 N 힘 제한에도 걸린다.
- `lqr.py` 출력의 폐루프 고윳값 `|λ|` 를 Q, R 을 바꿔 가며 본다. 1 에 가까울수록 느리다.
- `sim.py` 에서 제어를 매 스텝(500 Hz)이 아니라 10 스텝마다(50 Hz) 계산하도록 바꿔 본다. 실제 로봇의 제어 주기 문제다.

## 문제 해결

- `ModuleNotFoundError: scipy` → `pip install -r requirements.txt` (requirements 에 `scipy>=1.11` 있음).
- 한글이 깨지면(Windows 콘솔) `PYTHONIOENCODING=utf-8`.
