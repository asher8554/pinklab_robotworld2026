# PinkLAB @ 로보월드 2026 — MuJoCo 부스와 안내 로봇

> **이 저장소는 [PinkWink/pinklab_robotworld2026](https://github.com/PinkWink/pinklab_robotworld2026) 의 개인 실습용 fork 입니다.**
> 원본 내용은 그대로 두고, 제어 기법(PID · LQR) 실습 [`lessons/02_control/`](lessons/02_control/) 을 더했습니다. 아래 [제어 기법 실습](#제어-기법-실습-pid--lqr) 절을 보세요.

로보월드 2026(KINTEX 3홀 A03)의 핑크랩 부스를 MuJoCo 로 옮기고, 그 안에서 이동형 양팔 로봇(PL-2M)이
손님에게 컵을 가져다 드리고 부스를 안내하는 번외편 저장소입니다. 웹 브라우저 FMS 에서 로봇을 보고, 직접 몰고,
Nav2 로 보내고, 한국어 문장(LLM)으로 일을 시킵니다.

강좌 저장소([pinklab_mobile_openarm_ros2_mujoco](https://github.com/PinkWink/pinklab_mobile_openarm_ros2_mujoco))와는 별개입니다.
필요한 ROS 2 패키지는 이 저장소로 **복사해서** 고쳤고, 강좌 코드는 고치지 않습니다. ROS 도메인도 44 로 달라서(강좌 43) 한 PC 에서 같이 켜도 섞이지 않습니다.

## 설치와 실행

Ubuntu 24.04 + ROS 2 Jazzy(`ros-jazzy-desktop`)가 있으면:

```bash
sudo apt install fonts-nanum                 # 벽 그래픽의 한글
./scripts/install.sh                         # apt · rosdep · .venv · pip · colcon build
cp .env.example .env                         # OPENAI_API_KEY 입력 (LLM 지령에만 필요)
./scripts/robotworld doctor                  # 환경 점검
./scripts/robotworld start                   # MuJoCo 부스 + 로봇 + Nav2 + MoveIt + 집기·놓기 + Task Manager
./scripts/robotworld fms                     # 다른 터미널: 웹 FMS → http://localhost:8080
./scripts/robotworld stop                    # 끝낼 때
```

- 다른 PC · 태블릿에서 FMS 를 열려면 `./scripts/robotworld fms --host 0.0.0.0`
- 명령줄로 컵 하나 옮기기: `./scripts/robotworld pick pink drinks consult_2`

## 웹 FMS

| 기능 | 방법 |
|---|---|
| 지도 | 부스 지도 · 로봇(AMCL) · Nav2 경로 · 라이다 · 컵 · 장소. 휠 확대, 더블클릭 원래 크기 |
| Nav2 이동 | 지도를 클릭하면 그 자리로(끌면 그 방향을 보고 멈춤), 또는 장소 버튼 |
| 수동 조작 | 방향 버튼을 누르고 있는 동안 / 키보드 W A S D. 손을 떼면 0.4 s 안에 멈춤. 누르면 Nav2 · 미션이 먼저 취소됨 |
| 카메라 | 평소 꺼짐. 켜면 머리 · 베이스 카메라 영상(MJPEG). 꺼져 있으면 영상 인코딩도 하지 않음 |
| LLM 지령 | 문장 → 계획(또는 되묻기 · 답변) → 단계 목록 확인 → 실행. 진행 단계와 결과 문장이 대화창에, 컵 위치 표는 결과대로 갱신 |
| 정지 | 오른쪽 위 ■ 정지: Nav2 · 미션 · 수동 조작 모두 정지 |

지령 예: "2번 테이블 손님에게 물 한 잔 가져다 드려", "흰 옷 입은 손님한테 파란 컵 갖다 드려", "두 테이블에 한 잔씩 드려",
"약국으로 안내해 줘", "핑크랩은 어떤 회사야?"

## 부스와 시나리오

- 부스 설계: `src/robotworld/booth/booth_data.json` (설계 HTML 의 `window.BOOTH`, 최종 평면도 2026-10-14) — 고치지 않는다
- 로봇 시나리오: `src/robotworld/booth/scenario.yaml` — 상담 테이블 의자 하나씩 + 손님, 음료대(0.75 m), 벽 문구 수정(숫자로 보는 실적)
- 로봇 쪽 월드: `src/mobile_openarm_mujoco/worlds/booth.yaml` — 시작 위치, 컵 3개(지름 5 cm 원기둥), 스테이션 3곳(음료대 · 상담 테이블 1 · 2), 도킹 마커, MoveIt 장애물
- 장소 이름: `src/warehouse_lecture/worlds/locations.yaml` — 대기 위치 · 음료대 · 상담 테이블 1/2 · 인포 · 테이블 A/B · 약국 · 교실1/2 · 입구 · 원기둥
- Nav2 지도: `src/mobile_openarm_navigation/maps/booth.*` — SLAM 을 마쳤다고 보고 모델 형상에서 만든 지도 (`scripts/generate_booth_map.py`)
- 좌표: MuJoCo X = 설계 x / 100, Y = −설계 z / 100. 부스는 X 0 ~ 9, Y −6 ~ 0 (메인 통로 +X, 주출입구 −Y)

## 구성

| 경로 | 내용 |
|---|---|
| `src/robotworld/` | 부스 빌더(설계 데이터 → MJCF), 벽 그래픽(Pillow), 설계 데이터와 그림 |
| `src/mobile_openarm_*`, `src/warehouse_*`, `src/openarm_description`, `src/vicpinky_description` | 강좌 저장소에서 복사한 로봇 · 브리지 · Nav2 · MoveIt · 스킬 · LLM 패키지 (아래처럼 고침) |
| `fms/` | 웹 FMS: `server.py`(Flask + rclpy), `static/`, `frame_tap.py`(카메라 JPEG, 켤 때만) |
| `scripts/robotworld` | 실행 래퍼 (도메인 44, `.venv`) |
| `scripts/view_world.py` · `render_views.py` | 로봇 없이 부스만 보기 · 설계 시점 7개 렌더 |
| `tools/extract_booth_html.py` | 설계 HTML 이 바뀌면 `booth/` 를 다시 뽑기 |

강좌 패키지에서 고친 점: 브리지가 부스를 합쳐 넣음(`mjcf_world: robotworld`), 원기둥 물체, 스테이션을 테이블 가장자리 · 방향 · 높이로
정의(어느 방향 테이블이든 도킹 · 집기 · 놓기), 상자 대신 컵 · 부스 장소 · 부스 안내용 LLM 프롬프트, Task Manager 를 launch 에 포함.
YOLO(torch) 는 쓰지 않고 컵 위치는 시뮬레이터 truth 로 찾습니다.

## 제어 기법 실습 (PID · LQR)

이 fork 에서 더한 실습입니다. ROS 2 없이 MuJoCo Python 만 쓰며, 카트-폴(레일 위 카트 + 경첩 막대) 하나로 세 가지 제어기를 비교합니다.

| 제어기 | 보는 것 | 결과 (기본값, t=4 s 에 3 N 외란) |
|---|---|---|
| 각도 PID | 막대 각도 1개 | 막대는 서지만 카트가 0.3 m/s 로 밀려 5.0 s 에 레일 끝에서 넘어짐 |
| 캐스케이드 PID | 바깥 루프(카트 위치) → 안쪽 루프(각도) | 20 s 에 x = 1.000 m. 이득 5개를 손으로 맞춤 |
| LQR | 위치 · 각도 · 속도 · 각속도 4개 | 10 s 에 x = 1.000 m. 이득 K 를 비용 Q, R 에서 계산 |

```bash
source scripts/env.sh                                                        # .venv (mujoco, numpy, scipy)
python lessons/02_control/pid.py                                            # 각도 PID
python lessons/02_control/pid.py --kx 0.05 --kv 0.1 --target 1 --seconds 20 # 캐스케이드 PID
python lessons/02_control/lqr.py                                            # LQR
python lessons/02_control/lqr.py --viewer                                   # 아무 명령에나 --viewer 를 붙이면 MuJoCo 창
python lessons/02_control/test_control.py                                   # 자체 점검 → ok
```

- LQR 은 운동 방정식을 손으로 유도하지 않습니다. `mujoco.mjd_transitionFD` 로 선형화한 A, B 에 `scipy.linalg.solve_discrete_are` 를 써서 K 를 구합니다.
- 설치 없이 체험: [`lessons/02_control/cartpole_demo.html`](lessons/02_control/cartpole_demo.html) 을 내려받아 브라우저로 엽니다. 슬라이더로 이득을 바꾸고 버튼으로 막대를 밀어 봅니다. 물리는 `cartpole.xml` 수치를 옮긴 근사이며, 넘어지는 시각과 K 가 MuJoCo 결과와 맞습니다.
- 자세한 설명 · 관찰 포인트 · 해 볼 것: [`lessons/02_control/README.md`](lessons/02_control/README.md)

## 알려진 한계

- 수동 조작으로 테이블 · 사람에 너무 바짝 붙이면 Nav2 가 제자리 회전을 충돌로 보고 빠져나오지 못한다 → 수동으로 0.5 m 쯤 빼 준 뒤 다시 보낸다.
- 컵은 단단한 원기둥이다(물은 시뮬레이션하지 않는다). 손님에게 직접 건네지 않고 손님 앞 테이블에 놓는다.
- 부스 사람들은 아직 움직이지 않는다.
