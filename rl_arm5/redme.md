## Kiểm tra camera
```bash
v4l2-ctl --list-devices
```

Chọn camera để dùng:
```bash
cd /home/quan/Desktop/test5/determine_depth && python3 main.py --camera 2

```
Web stream: `http://localhost:8080` (hoặc `http://<IP_MAY>:8080`)

-------------------------------------------

## Nạp Firmware ESP32
Để nạp lại chương trình điều khiển 2 nhân (Dual-core PWM + ADC Feedback) xuống ESP32:
```bash
cd /home/quan/Desktop/test5/sodonguyenly_calibservo
arduino-cli compile --fqbn esp32:esp32:esp32 -u -p /dev/ttyUSB0 esp32_dualcore_controller
```

-------------------------------------------

## Phương Pháp 1: Baseline Camera Tracking + S-Curve IK

- **Bước 1**: Bật nhận diện Camera trên Terminal 1:
```bash
cd /home/quan/Desktop/test5/determine_depth
python3 main.py --camera 2
```

- **Bước 2**: Chạy điều khiển bám vết IK trên Terminal 2:
```bash
cd /home/quan/Desktop/test5
python3 sodonguyenly_calibservo/baseline_ik_tracker.py
```
*Dữ liệu đo đạc thực nghiệm tự động ghi vào `benchmark_baseline_ik.csv`.*

-------------------------------------------

## Phương Pháp 2: Deep RL SAC (Deploy thực tế)

- Vẫn giữ nguyên Terminal 1 (Camera đang chạy).
- Trên Terminal 2, chạy:
```bash
cd /home/quan/Desktop/test5
python3 sodonguyenly_calibservo/rl_deploy_tracker.py
```
*Dữ liệu đo đạc thực nghiệm tự động ghi vào `benchmark_deep_rl.csv`.*

-------------------------------------------

## Đánh giá & Xuất Biểu Đồ Đối Chứng Thực Nghiệm

Trên Terminal 3, chạy:
```bash
cd /home/quan/Desktop/test5
python3 sodonguyenly_calibservo/benchmark_evaluator.py
```
*Xem bảng số liệu so sánh chi tiết và đồ thị tại `sodonguyenly_calibservo/benchmark_comparison.png`.*

-------------------------------------------

## Xem Trước Quỹ Đạo Động Hình Dấu (+) Trong Gazebo (1.2 - 2.0 cm/s)

Để trực quan hóa quỹ đạo dấu `+` 4 nhánh mượt mà (đảm bảo luôn nằm trong tầm với của cánh tay và trường nhìn camera):

```bash
# Terminal 1: Khởi động mô phỏng cánh tay trong Gazebo
source /opt/ros/humble/setup.bash
source /home/quan/Desktop/test5/ros2_ws/install/setup.bash
ros2 launch assarm_description gazebo.launch.py

# Terminal 2: Chạy script điều khiển vật thể di chuyển hình dấu '+'
cd /home/quan/Desktop/test5
python3 ros2_ws/src/assarm_deploy/assarm_deploy/preview_dynamic_target_gazebo.py
```

-------------------------------------------

## Huấn luyện Nâng Cấp: Stage B+ (Curriculum 2 Giai Đoạn)

Quy trình huấn luyện tuân thủ nghiêm ngặt tính chất Markov (Agent chỉ nhận tọa độ hiện tại $[x, y, z]$ và vận tốc tức thời $[v_x, v_y, v_z]$, tuyệt đối **không** biết trước quỹ đạo tương lai):

- **Giai đoạn 1 (0 -> 30,000 steps)**: Quỹ đạo dấu `+` cố định danh định tại tâm $(0, 0.28, 0.40)$ để mô hình ổn định hàm thưởng (Reward Shaping) và hội tụ kỹ năng bám vết cơ bản (>90%).
- **Giai đoạn 2 (30,000 -> 80,000 steps)**: Tự động ngẫu nhiên hóa vị trí ban đầu (Level 3 toàn dải) + ngẫu nhiên nhánh xuất phát + bổ sung nhiễu gió / rung lắc bất định (Ornstein-Uhlenbeck drift $\pm 5\text{ mm}$) để chống quá khớp và sẵn sàng chuyển giao Sim-to-Real.

Chạy huấn luyện tự động chuyển giai đoạn (Auto Curriculum):
```bash
cd /home/quan/Desktop/test5/ros2_ws
python3 src/assarm_rl_train/assarm_rl_train/train_stage_b_plus.py \
  --warm_start ./models/best_model.zip \
  --timesteps 80000 \
  --stage1_steps 30000 \
  --lr 1e-4
```

Theo dõi tiến trình huấn luyện qua TensorBoard:
```bash
tensorboard --logdir=/home/quan/Desktop/test5/ros2_ws/tb_logs/stage_b_plus/ --port=6006
```
*(Mô hình tốt nhất tự động lưu tại `ros2_ws/models/stage_b_plus/best_model.zip`)*

-------------------------------------------

## Đồng Bộ & Triển Khai Thực Nghiệm Lên Raspberry Pi 4B

### 1. Đẩy toàn bộ dự án từ PC sang Pi:
Từ thư mục dự án trên máy tính, chạy script đồng bộ:
```bash
cd /home/quan/Desktop/test5
./sync_to_piros2.sh
```
*(Script sẽ tự động kết nối `piros2@192.168.50.1` và đồng bộ vào thư mục `~/quan` trên Pi).*

### 2. Chạy trên Raspberry Pi:
Mở 2 Terminal trên Pi (hoặc qua SSH `ssh piros2@192.168.50.1`):

- **Terminal 1: Luồng Thị Giác 3D & Web Dashboard (Core 1)**
  ```bash
  cd ~/quan/determine_depth
  python3 main.py --camera 2 --color red
  ```

- **Terminal 2: Bộ Điều Khiển Robot (Core 0)**
  ```bash
  cd ~/quan/sodonguyenly_calibservo

  # Chạy Phương pháp 1 (IK):
  python3 baseline_ik_tracker.py --port /dev/ttyUSB0

  # Hoặc chạy Phương pháp 2 (Deep RL SAC):
  python3 rl_deploy_tracker.py --model stage_b_final.zip --port /dev/ttyUSB0
  ```

### 3. Xem Bảng Điều Khiển Telemetry 3D thời gian thực:
Trên trình duyệt (điện thoại hoặc laptop):
```
http://192.168.50.1:8080
```
- Hiển thị đầy đủ video trực tiếp, mô hình 3D tương tác xoay 360°, điểm đỏ phát sáng tâm trọng lượng CoM, lực quán tính, mô-men lật, độ giật từng servo và trạng thái thành công.


