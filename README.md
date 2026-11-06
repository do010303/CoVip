# CoVip: camera nhận diện vật → tay robot gắp (Raspberry Pi 5)

Camera thấy vật, detector tính toạ độ vật trong hệ đế tay, bộ điều khiển kéo tay tới và gắp. Toàn bộ chạy trên **Raspberry Pi 5**, không cần ROS.

```
 camera USB ──► perception/scripts/detect_objects.py ──UDP 127.0.0.1:5555──► perception/scripts/ik_grasp_loop.py ──USB serial──► ESP32 ──► 5 servo
                 core 1,2,3                                                    core 0 (IK của rl_arm5)
                 YOLOv8n-seg ONNX + khớp trụ 3D → hệ đế tay
                 web :8080 (xem ảnh)
```

| Thư mục | Nội dung |
|---|---|
| `perception/` | detector 4 vật (YOLOv8n-seg, onnxruntime), đổi toạ độ camera sang hệ đế tay, script gắp `ik_grasp_loop.py`, gán nhãn và train, cài Pi 5 |
| `rl_arm5/` | phần robot của nhóm RL: tay `assarm` (4 khớp + ngàm), firmware ESP32, IK tracker, SAC tracker, cầu nối UDP, mô hình động học, model SAC. Xem [rl_arm5/README_ROBOT.md](rl_arm5/README_ROBOT.md) |

Trạng thái (2026-11-04): đã gắp thật trụ xanh ⌀50×100 mm trên Pi 5 bằng IK (2/2 lần). SAC mới chạy giả lập. Các vấn đề còn mở nằm ở cuối file.

---

## 1. Detector

- 4 lớp: `hop_nau` (hộp giấy nâu), `tru_xanh` (trụ nhựa xanh ⌀50×100 mm), `hop_pin` (hộp pin), `lo_nap_do` (lọ nắp đỏ ⌀41×81 mm).
- Dataset gồm 13 clip quay ở 2 môi trường. Nhãn được gán tự động bằng YOLOE theo kiểu gợi ý bằng ảnh (`autolabel_objects.py`), sau đó lọc theo ràng buộc, được train 3.614 / val 567 ảnh, chia theo clip.
- YOLOv8n-seg, val **mask mAP50 0,992 / mAP50-95 0,924**. Tập val quay cùng phòng với tập train nên số này lạc quan. Điểm yếu: vật ở xa hay bị sót, đôi khi nhận nhầm ngàm kẹp (cùng màu xanh) thành trụ.
- Model: `perception/weights/objects4_v1_640.onnx` (384×640) và `objects4_v1_320.onnx` (192×320).

| Model | Tốc độ trên Pi 5 (3 luồng) |
|---|---|
| 320 | ~30 fps; còn 7–18 fps khi đang khớp trụ 3D |
| 640 | ~8 fps |

Train lại: dataset không nằm trong repo. Có dataset thì đặt vào `perception/datasets/objects4/`, rồi chạy `python perception/scripts/train_objects.py [--device cpu]`.

## 2. Phần cứng

- Raspberry Pi 5, nguồn 5 V 5 A, có quạt.
- Camera USB, 640×480 hoặc 640×360.
- Tay `assarm` với ESP32 nạp firmware `rl_arm5/sodonguyenly_calibservo/esp32_dualcore_controller`. Servo cấp nguồn riêng.
- Camera gắn **cứng**, đặt trước trục J1 và chúc xuống. Dời camera thì phải đo lại `--cam-pose` (mục 5).

## 3. Cài trên Pi 5 (Raspberry Pi OS / Debian 13, Python 3.13)

```bash
git clone https://github.com/do010303/CoVip.git ~/CoVip
bash ~/CoVip/perception/scripts/setup_pi5.sh
```
`setup_pi5.sh` làm các việc sau:
- cài gói apt (hỏi mật khẩu sudo một lần);
- tạo venv `~/venvs/arm` với numpy, opencv-python, onnxruntime, pyyaml, pyserial;
- thêm user vào nhóm `dialout` (nếu script báo vừa thêm thì đăng xuất, đăng nhập lại);
- liệt kê camera và cổng serial, kiểm đủ file, đo tốc độ model.

Chạy lại bao nhiêu lần cũng được.

Kiểm thiết bị:
```bash
v4l2-ctl --list-devices   # camera USB là mục có "(usb-...)"; pispbe / rpi-hevc-dec là phần cứng bên trong Pi 5, không phải camera
ls /dev/ttyUSB*           # ESP32
```

## 4. Chạy

Mở 2 terminal trên Pi, đặt sẵn:
```bash
PY=~/venvs/arm/bin/python; cd ~/CoVip/perception
```

**Terminal 1: detector.** Ví dụ dưới dùng camera cũ của nhóm RL ở 640×480 với K của nhóm:
```bash
$PY scripts/detect_objects.py --no-ros --device /dev/video0 --width 640 --height 480 \
    --calib ../rl_arm5/determine_depth/configs/camera_calibration.yaml \
    --cam-pose 0 0.019 0.561 30 --j1-trim 3 \
    --target tru_xanh --udp --web 8080 --cpu 1,2,3 --threads 3 \
    --model weights/objects4_v1_320.onnx
```
- Xem ảnh tại `http://<ip-pi>:8080`.
- Với C930e: dùng `--width 640 --height 360` và bỏ `--calib`. Khi đó mặc định là `calib/c930e_720p.npz`, tự đổi K theo độ phân giải.

**Terminal 2: gắp.** Thêm `--mock` để chạy giả lập, tay không động đậy:
```bash
$PY scripts/ik_grasp_loop.py --port /dev/ttyUSB0 --handoff-dist 0.015
```
Trình tự, mỗi pha đều in ra `[n/5]`:
1. Về HOME (90 90 90 90).
2. Ra tư thế chờ.
3. **Khoá đích:** trụ phải đứng yên (lệch ≤ 3 cm) đủ 5 s. Nếu trụ bị dời hoặc mất khỏi ảnh thì đếm lại, có in lý do.
4. Bám tới trụ.
5. Đóng ngàm khi cách đích ≤ 1,5 cm, rồi nhấc.

Sau đó tay **giữ nguyên tư thế, ngàm vẫn kẹp**. **Ctrl+C:** mở ngàm, về HOME rồi thoát; Ctrl+C lần 2 thì dừng ngay. Mỗi lần gắp ghi 1 dòng vào `perception/logs/grasp_loop_*.csv`.

### Tuỳ chọn hay dùng

`detect_objects.py`:

| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--target tru_xanh` | chỉ nhận lớp này |
| `--cam-pose X Y Z PITCH [YAW]` | vị trí camera trong hệ đế tay (mục 5) |
| `--j1-trim DEG` | xoay đích quanh J1 (+ = sang phải), bù việc tay chĩa lệch |
| `--target-offset DX DY DZ` | dời đích gửi cho tay (m). Ví dụ `0 0 0.10` để ngàm lơ lửng trên trụ, dùng khi đo sai số |
| `--udp` / `--web 8080` / `--show` | gửi đích cho tay / xem ảnh qua web / cửa sổ ảnh (khi có màn hình) |
| `--hand-eye calib/T_cam_assarm.npz` | thay `--cam-pose`, dùng hiệu chỉnh bằng tờ ArUco (`calib_cam_assarm.py`) |
| `--bench` | đo tốc độ model |

`ik_grasp_loop.py`:

| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--handoff-dist 0.015` | đóng ngàm khi cách đích ≤ (m). Ngưỡng 7,5 cm mặc định của tracker gốc đóng ngàm quá sớm vì tay chạy vòng hở |
| `--lock-time 5 --lock-tol 0.03` | điều kiện khoá đích |
| `--z-offset 0.05` | ngàm thấp hơn giữa thân trụ 5 cm (servo bị trọng lực kéo võng) thì nâng đích lên |
| `--home S1 S2 S3 S4` | đổi tư thế home (độ servo) |
| `--after place --cycles 10` | chạy liên tục: gắp, đặt lại chỗ cũ, gắp tiếp, đủ 10 lần thì dừng, in tỉ lệ thành công |
| `--mock` | không cần ESP32 |

Thử tay mà không cần camera: thay Terminal 1 bằng `$PY scripts/udp_fake_target.py --xyz 0 0.29 0.40`.

## 5. Đo vị trí camera (`--cam-pose`)

Hệ đế tay `assarm`:
- +X sang phải (đứng sau tay, nhìn về phía trước);
- +Y về phía trước;
- +Z lên.

Trục J1 nằm ở (0; −1,08; 58,08) cm. Đo tâm ống kính bằng thước, rồi quy đổi:

| Số | Cách tính | Ví dụ: trước J1 3 cm, thấp hơn J1 2 cm, ngay giữa, chúc 30° |
|---|---|---|
| X | lệch phải so với trục J1 (m), lệch trái thì âm | `0` |
| Y | (cm trước trục J1 − 1,08) / 100 | `0.019` |
| Z | 0,5808 − (cm thấp hơn J1) / 100 | `0.561` |
| PITCH | độ chúc xuống so với phương ngang | `30` |
| YAW (tuỳ chọn) | camera quay phải (+) hoặc trái (−), độ | |

Kiểm tra: đặt trụ đứng trên bàn, trước trục J1 30 cm, ngay giữa. Detector phải in `ĐÍCH tru_xanh đế` khoảng **(0; 29; 40) cm**.

Đo sai số khi gắp trượt:
1. Detector thêm `--target-offset 0 0 0.10`.
2. Chạy `../rl_arm5/sodonguyenly_calibservo/baseline_ik_tracker.py --port /dev/ttyUSB0 --handoff-dist 0`: tay chỉ bám, không gắp.
3. Nhìn từ trên xuống, đo tâm ngàm lệch trục trụ bao nhiêu:
   - lệch đều → chỉnh X hoặc `--j1-trim`;
   - lệch tăng dần sang hai bên → thêm YAW.

## 6. Lỗi hay gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `REMOTE HOST IDENTIFICATION HAS CHANGED` khi ssh | `ssh-keygen -f ~/.ssh/known_hosts -R <ip-pi>` |
| `setlocale: LC_ALL: cannot change locale` | vô hại; `sudo sed -i 's/^# *\(vi_VN.UTF-8\)/\1/' /etc/locale.gen && sudo locale-gen` |
| SSH giật | `sudo iw wlan0 set power_save off` |
| Dán nhiều dòng, phần sau `ssh ...` không chạy | vào Pi trước rồi mới dán lệnh |
| `invalid float value: 'CX'` | chữ in hoa trong lệnh mẫu là chỗ điền số đo |
| Script gắp không in gì | lệnh chưa chạy: dán đúng 1 dòng, không có khoảng trắng ở đầu |
| `⚠️ không nhận gói UDP từ detector` | Terminal 1 chưa chạy hoặc thiếu `--udp` |
| `[LỖI UART] ... MOCK mode` | ESP32 chưa cắm, sai cổng, hoặc user chưa ở nhóm `dialout` |
| Ngàm đóng khi còn xa | dùng `--handoff-dist 0.015` |

## 7. Còn mở

- Tay thấp hơn đích khoảng 5 cm, ngàm hơi lệch trái, một đầu ngàm lệch bánh răng.
- Vị trí camera chưa phải bản cuối. Sau khi chốt cần hiệu chỉnh "chạm điểm" bằng chính tay, mục tiêu ≤ 1 cm.
- FPS tụt khi đang khớp trụ 3D: suy luận, khớp trụ và vẽ ảnh vẫn chạy nối tiếp.
- Train thêm ảnh có cả tay và vật trong khung. Đo CPU, RAM và độ trễ đầu–cuối trên Pi. Thử 3 vật còn lại. Chạy SAC trên tay thật.
