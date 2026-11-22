# Isaac Sim 경로 (MuJoCo 실습과 나란히)

MuJoCo 실습을 그대로 두고, 같은 PinkLAB 부스와 이동형 양팔 로봇을 **Isaac Sim 6.2 (Windows)** 에서 돌린다.
WSL 의 ROS 2 스택(Nav2 · MoveIt · 집기·놓기 스킬)은 원본 패키지를 그대로 쓴다.
`isaac/` 밖의 파일은 하나도 바꾸지 않았다. MuJoCo 쪽은 지금처럼 `./scripts/robotworld ...` 로 쓴다.

| | MuJoCo 실습 | Isaac 경로 |
| --- | --- | --- |
| 시뮬레이터 | `mobile_openarm_mujoco` 브리지 (WSL) | `isaac/booth_sim.py` (Windows, Isaac Sim 6.2) |
| ROS 2 스택 | `./scripts/robotworld start` | `isaac/wsl/robotworld-isaac start` |
| ROS_DOMAIN_ID | 44 | 45 (`ROBOTWORLD_ISAAC_DOMAIN_ID`) |
| 카메라 처리 | 브리지 안 핸들러 | `isaac/wsl/vision_bridge.py` 가 `/isaac/<카메라>/image_raw` 를 받아 원본 `ArucoDetector` 로 처리 |

## 구성

| 파일 | 역할 |
| --- | --- |
| `export_assets.py` (WSL) | 원본 빌드 코드로 로봇 URDF · 부스 MJCF · `sim.json`(설정 · 장소 · 카메라) 을 `artifacts/isaac/` 에 쓴다 |
| `booth_sim.py` (Windows) | URDF · MJCF 를 USD 로 바꿔 장면을 만들고, MuJoCo 브리지와 같은 토픽 · 액션을 낸다 |
| `isaac_physics.py` | Isaac 관절 상태를 MuJoCo `Physics` 처럼 보이게 하는 어댑터. 원본 `actions.Controllers` 를 수정 없이 쓴다 |
| `test_isaac_physics.py` | 위 어댑터 자체 시험 (Isaac 없이 실행) |
| `wsl/robotworld-isaac` | WSL 실행 래퍼. 도메인 45 · Fast DDS 로 맞추고 원본 명령을 부른다 |
| `wsl/booth_isaac.launch.py` | 원본 `warehouse.launch.py` 의 함수를 불러 Nav2 · MoveIt · 스킬 · RViz · vision bridge 를 띄운다 |

## 준비 (처음 한 번)

1. Windows 에 Isaac Sim 6.2 를 `C:\isaacsim-6.2` 에 둔다.
2. WSL Ubuntu 24.04 에 ROS 2 Jazzy 와 이 실습 워크스페이스를 빌드해 둔다 (`./scripts/install.sh`).
   래퍼는 이 저장소에 `install/` · `.venv` 가 있으면 그것을, 없으면 `ROBOTWORLD_WS` (기본 `~/pinklab_robotworld2026`) 를 쓴다.
3. WSL 에서 자산을 내보낸다. 로봇 설명이나 부스 설정을 바꾸면 다시 한다.

```bash
isaac/wsl/robotworld-isaac export        # → artifacts/isaac/ (robot.urdf, world/, sim.json)
isaac/wsl/robotworld-isaac env           # 워크스페이스 · 내보내기 폴더 · 도메인 확인
```

## 실행

```powershell
# Windows (저장소 폴더). 첫 실행은 USD 변환으로 조금 더 걸린다. 다시 내보냈으면 --reimport
C:\isaacsim-6.2\python.bat isaac\booth_sim.py                 # 창 모드
C:\isaacsim-6.2\python.bat isaac\booth_sim.py --headless      # 창 없이
```

`[ISAAC-BOOTH] ready` 가 찍히면 WSL 에서 스택을 띄운다.

```bash
isaac/wsl/robotworld-isaac start                      # Nav2 + MoveIt + 집기·놓기 + RViz
isaac/wsl/robotworld-isaac pick pink drinks consult_2 # 다른 터미널: 컵 하나 옮기기
isaac/wsl/robotworld-isaac stop                       # 끝낼 때 (Windows 쪽은 창을 닫는다)
```

- 시작 위치는 양쪽을 같게 준다. `booth_sim.py --spawn drinks` 와 `robotworld-isaac start spawn:=drinks`.
- 그 밖의 명령은 MuJoCo 래퍼와 같다. `drive` (Nav2 · MoveIt 없이), `slam`, `moveit`, `goal`, `arm`, `teleop`, `fms`.
- 실습 클라이언트(`lessons/*`)는 `robotworld-isaac exec python3 lessons/...` 로 같은 도메인에서 돌린다 (`lessons/08_lmm/07_run_plan.py` 확인).
- `booth_sim.py --help` 에 카메라 · 렌더 옵션의 기본값을 고른 이유를 적었다.

## 점검

```powershell
C:\isaacsim-6.2\python.bat isaac\booth_sim.py --headless --seconds 60   # 자체 점검 후 SELF-CHECK PASS|FAIL
```

```bash
ROBOTWORLD_ISAAC_DOMAIN_ID=77 isaac/wsl/robotworld-isaac exec python3 isaac/test_isaac_physics.py   # ok
```

2026-11-18~22 에 확인한 결과.

| 항목 | 결과 |
| --- | --- |
| 자체 점검 | PASS, 실시간 0.63배 (카메라 2대 640x480 · 10 FPS) |
| 주행 오도메트리 | 직진 실제 0.839 m / odom 0.840 m, 회전 실제 121.7° / odom 121.8° |
| 정지 거리 (0.1 m/s 에서 정지) | 14.7 mm (MuJoCo 13 mm) |
| 운반 `pick pink drinks consult_2` (홈 출발) | 5회 중 5회 성공, 슬롯 오차 0.1~4.0 cm |
| Task Manager `07_run_plan.py navigate:drinks pick:cup_pink@drinks navigate:consult_2 place:@consult_2` | 성공 4/4 단계 |
| 웹 FMS | 지도 · 카메라 MJPEG · 장소 목표 이동 확인 (LLM 채팅 제외) |
| `slam` + `teleop` (headless) | teleop 키 입력으로 odom (0.92, 1.08) m 이동, `/map` 645x618 칸 · 0.05 m |
| 창 모드 Isaac + RViz | 부스 · 로봇 렌더, RViz 에 로봇 모델 · 라이다 · MoveIt 패널 표시 (캡처로 확인) |

## MuJoCo 와 다른 점 · 제약

- **느리다.** 실시간 약 0.6배다. MoveIt 의 실행 시간 감시는 벽시계로 재므로 launch 에서 허용 배율을 4로 올렸다.
- **카메라는 경로 추적 + DLSS 다.** 영상마다 물리를 멈추고 같은 자세로 6번 더 그린다 (`--camera-subframes`). 끄면 회전 직후 마커 방향이 약 7° 늦어 도킹이 빗나간다.
- **PhysX 맞춤.** 물리는 CPU 로 돈다 (GPU 에서는 운반 중 쥔 컵이 튀어 나갔다). 바퀴 접촉은 구, 손가락 joint2 는 mimic 대신 따로 구동, 팔 링크 중력은 끈다 (MuJoCo 중력 보상과 같은 효과). 관절 armature 는 MuJoCo 빌드와 같게 준다 (없으면 바퀴가 제동되지 않는다). 오도메트리 속도는 0.1 s 로 평활한다.
- **부스 그림.** MJCF 변환기가 PNG 텍스처와 조명을 옮기지 않는다. ArUco 마커만 원본 PNG 로 다시 붙였고 나머지는 단색이다.
- **없는 것.** 사람은 움직이지 않고 `/warehouse/actor_states` 가 없다. `/vision/detections` (정답 검출기) 가 없어 `detect` 단계는 쓸 수 없다 (부스 계획은 쓰지 않는다). LLM 을 부르는 `report` · `answer` · 자연어 입력은 확인하지 못했다 (2026-11-22 `08_text_mission.py` 실행 시 API 키 만료 401, 새 키는 크레딧 없음 429).
