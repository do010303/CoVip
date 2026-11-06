# Phần robot: tay `assarm` + ESP32 + bộ điều khiển IK / SAC

Thư mục này là mã của nhóm RL (repo gốc `uvnb/rl_arm5`, cùng `uvnb/determine_depth`), chép vào đây để chạy cùng perception trong một repo. **Không sửa logic điều khiển**; perception (`../perception`) chỉ thay module camera lọc màu cũ (`determine_depth/main.py`) và gửi vị trí vật qua cùng một cổng UDP.

| Thư mục / file | Nội dung |
|---|---|
| `sodonguyenly_calibservo/esp32_dualcore_controller/` | firmware ESP32: PWM 5 servo + đọc ADC phản hồi |
| `sodonguyenly_calibservo/baseline_ik_tracker.py` | **Phương pháp 1:** bám camera + IK số + quỹ đạo S-curve, rồi gắp |
| `sodonguyenly_calibservo/rl_deploy_tracker.py` | **Phương pháp 2:** chính sách SAC (Stage B+) bám camera, rồi gắp |
| `sodonguyenly_calibservo/*_calib.ino` | firmware calib từng servo |
| `determine_depth/core/ipc_bridge.py` | giao thức UDP camera → tay (`VisionIPCPublisher` / `VisionIPCSubscriber`) |
| `determine_depth/core/web_streamer.py`, `arm_dynamics.py`, `tracker_filter.py` | dashboard web :8080, CoM/động lực học cho telemetry, lọc alpha-beta |
| `determine_depth/` (còn lại) | perception lọc màu cũ của nhóm RL (`main.py`), calib camera, K ở `configs/camera_calibration.yaml` |
| `ros2_ws/src/assarm_common/` | FK, cấu hình miền train (`config.py`), bộ dựng quan sát 28 chiều cho SAC, script gắp |
| `ros2_ws/src/assarm_description/`, `assarm_gazebo/` | URDF/mesh, mô phỏng Gazebo |
| `ros2_ws/src/assarm_rl_env/`, `assarm_rl_train/` | môi trường RL và script train SAC (Stage A, B, B+) |
| `models/stage_b_plus_final.zip` | model SAC đã train (SB3 2.8, PyTorch 2.11, Python 3.10). Replay buffer không có trong repo |
| `redme.md`, `rl_recommendation_v5_0.md` | ghi chú gốc của nhóm RL (đường dẫn trong đó là máy của nhóm) |

## 1. Phần cứng và firmware

ESP32-WROOM-32, chạy 2 core:
- **Core 1:** PWM 50 Hz cho 5 servo, nội suy mượt mỗi 20 ms.
- **Core 0:** đọc ADC phản hồi và gửi lên ở 20 Hz.

| Khớp | Servo | PWM | Phản hồi ADC |
|---|---|---|---|
| J1 (xoay đế) | TD8120MG | GPIO27 | GPIO35 |
| J2 (vai) | RDS3120MG | GPIO26 | GPIO34 |
| J3 (khuỷu) | MG995 | GPIO25 | GPIO39 |
| J4 (cổ tay) | MG995 | GPIO33 | GPIO36 |
| Ngàm | MG996R | GPIO32 | không có |

Nạp firmware (máy có `arduino-cli` và core ESP32, cài thư viện `ESP32Servo`):
```bash
cd rl_arm5/sodonguyenly_calibservo
arduino-cli compile --fqbn esp32:esp32:esp32 -u -p /dev/ttyUSB0 esp32_dualcore_controller
```

**Giao thức UART 115200 baud:**
- Lệnh gửi xuống: `SET <s1> <s2> <s3> <s4> <grip>\n`, đơn vị độ servo. Góc khớp (rad) = (độ − 90) · π/180. Home là `SET 90 90 90 90 0`.
- ESP32 gửi lên liên tục: `FB <s1> <s2> <s3> <s4> <grip> | ADC <mv1> <mv2> <mv3> <mv4>`.
- Nếu ADC của J1–J3 < 500 mV (chưa nối dây phản hồi), tracker tự chuyển sang **vòng hở**: coi lệnh là vị trí thật. Giàn hiện tại đang chạy vòng hở.

## 2. Hệ toạ độ và miền làm việc

- **Hệ đế:** X sang phải, Y về phía trước, Z lên (m). Trục J1 nằm ở (0; −0,0108; 0,5808).
- Tracker nhận **tâm vật** trong hệ này.
- Tư thế chờ của tracker: servo [90, 70, 58, 90]. Tâm ngàm thấp nhất với tới được: z ≈ 0,13 m.
- **Miền train của SAC** (`assarm_common/config.py`, `SPAWN_LEVELS[3]`): r 0,25–0,33 m tính từ trục J1, góc ±45°, z 0,36–0,44 m. Ngoài miền này chính sách không được đảm bảo.
- Điểm thử chuẩn P0 = (0; 0,29; 0,40): trước J1 30 cm, thấp hơn J1 18 cm.

## 3. Giao thức camera → tay (UDP 127.0.0.1:5555)

Mỗi gói là `struct.pack("!Bffffff?d", 0xAA, x, y, z, vx, vy, vz, is_valid, timestamp)`, đơn vị m và m/s, hệ đế.
- `VisionIPCSubscriber.get_latest()` trả `(pos, vel, valid, age)`.
- Tracker bỏ qua gói cũ hơn 0,6 s.
- Telemetry của tay gửi ra UDP 5556 (JSON), dashboard web đọc từ đó.

Bên gửi có thể là:
- `../perception/scripts/detect_objects.py --udp` (detector 4 vật);
- `../perception/scripts/udp_fake_target.py --xyz X Y Z` (đích giả, để thử tay không cần camera);
- `determine_depth/main.py` (perception lọc màu cũ của nhóm RL).

## 4. Chạy

Đặt sẵn: `PY=~/venvs/arm/bin/python` (venv do `../perception/scripts/setup_pi5.sh` tạo).

### Phương pháp 1: IK (đã gắp thật trên Pi 5)

Nên dùng script bọc của perception. Script này khoá đích 5 s, giữ vật sau khi gắp, Ctrl+C thì mở ngàm và về home, và in rõ từng pha:
```bash
cd ~/CoVip/perception
$PY scripts/ik_grasp_loop.py --port /dev/ttyUSB0 --handoff-dist 0.015        # --mock: không cần tay
```
Chạy tracker gốc (gắp 1 lần rồi thoát, không về home):
```bash
cd ~/CoVip/rl_arm5/sodonguyenly_calibservo
$PY baseline_ik_tracker.py --port /dev/ttyUSB0 --handoff-dist 0.015 [--z-offset 0.05] [--mock]
```
- Trình tự: `WAIT_TARGET` (thấy vật 0,4 s) → `TRACKING` (IK, tối đa 8°/bước, 10 Hz) → `CLOSE` → `LIFT` → `HOLD` → `COMPLETED`. Bám quá 25 s thì `FAILED`.
- **Lưu ý:** `--handoff-dist` mặc định là 0,075 m. Tay chạy vòng hở nên khoảng cách này tính từ lệnh, và ngàm đóng khi thực tế còn cách vật 6–11 cm. Dùng 0,015.
- `--handoff-dist 0`: chỉ bám, không bao giờ gắp. Dùng để đo sai số.

### Phương pháp 2: SAC (mới chạy giả lập)

```bash
$PY -m pip install torch --index-url https://download.pytorch.org/whl/cpu
$PY -m pip install stable-baselines3 gymnasium
cd ~/CoVip/rl_arm5/sodonguyenly_calibservo
$PY rl_deploy_tracker.py --model ../models/stage_b_plus_final.zip --mock
```
- Quan sát 28 chiều (`assarm_common.obs_builder`), hành động là delta 4 khớp, làm mượt EMA 65/35.
- **Điều kiện đóng ngàm được viết cứng:** P_err ≤ 7,5 cm, hoặc khe ngàm ≤ 4/5/6 cm. Gặp đúng lỗi đóng sớm như IK khi chạy vòng hở, nên chưa nên chạy với tay thật.
- Model lưu từ Python 3.10. Trên Python 3.13 (Pi 5) cần kiểm `SAC.load` chạy được trước.

### Train lại SAC (máy có GPU, ROS 2 + Gazebo)

Xem `ros2_ws/src/assarm_rl_train/` (`train_stage_b_plus.py`) và `redme.md` của nhóm RL.

## 5. Benchmark

Hai tracker ghi CSV theo từng bước (`benchmark_baseline_ik.csv`, `benchmark_deep_rl.csv`) trong thư mục chạy:
- vị trí ngàm, đích, sai số, độ nghiêng;
- độ giật (jerk);
- lệnh và phản hồi của 4 servo.

`benchmark_evaluator.py` so sánh hai phương pháp.
