# -*- coding: utf-8 -*-
"""
assarm_common — Cấu hình hằng số dùng chung giữa sim và robot thật.

Mọi con số ở đây là giá trị khởi đầu, sẽ hiệu chỉnh bằng đo đạc thật (Phase R).
Xem rl_recommendation_v5_0.md mục 3.x cho nguồn gốc từng giá trị.
"""

import numpy as np

# ===========================================================================
# Giới hạn khớp (từ URDF, rad)
# ===========================================================================
Q_LO_ARM = np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2])
Q_HI_ARM = np.array([+np.pi / 2, +np.pi / 2, +np.pi / 2, +np.pi / 2])

JOINT_NAMES_ARM = [
    "Revolute_Joint_1",
    "Revolute_Joint_2",
    "Revolute_Joint_3",
    "Revolute_Joint_4",
]
JOINT_NAME_GRIPPER = "Revolute_Joint_Active"

# ===========================================================================
# Gripper
# ===========================================================================
Q_STAR = 0.896       # rad — góc đóng ôm khít khối 5 cm (từ FK trên URDF)
EPS_GRIP = 0.1       # rad — dư để có lực kẹp
Q_OPEN = 0.0         # rad — gripper mở hoàn toàn
Q_GRIP_MAX = np.pi / 2  # rad — giới hạn URDF

# Offset tâm lòng ngàm kẹp khi ngón mở tiếp cận khối (hệ tọa độ Plate_1, m)
# Trục xoay ngón tại z=-0.022m, miếng cao su kẹp tại z=-0.065m, y=+0.0105m
GRIP_CLOSE_OFFSET = np.array([0.0, 0.0105, -0.065])

# ===========================================================================
# Cửa sổ VERIFY (rad)
# ===========================================================================
VERIFY_WINDOW_LO = 0.82  # rad
VERIFY_WINDOW_HI = 0.97  # rad
MAX_VERIFY_RETRIES = 2

# ===========================================================================
# Ngưỡng bàn giao (handoff) cho Khối trụ tròn gắp mặt cong
# ===========================================================================
HANDOFF_EX = 0.012          # m — |e_x| dung sai theo trục đóng ngàm (khe hở 15mm)
HANDOFF_EY = 0.018          # m — |e_y| dung sai dọc theo thân ngón
HANDOFF_EZ = 0.025          # m — |e_z| dung sai theo chiều cao thân trụ 10cm (±25mm quanh tâm)
HANDOFF_DPSI = np.pi        # rad — Khối trụ đối xứng tròn xoay 360°, không phụ thuộc Yaw
TARGET_TILT = 0.0           # rad — Tiếp cận tự nhiên ôm mặt cong
HANDOFF_TILT = np.radians(60) # rad — Cho phép góc tiếp cận tự nhiên rộng rãi
HANDOFF_REL_SPEED = 0.05    # m/s — tốc độ tương đối an toàn
HANDOFF_CONSEC_STEPS = 2    # số step liên tiếp đạt chuẩn bàn giao (0.2s)

# ===========================================================================
# Điều khiển
# ===========================================================================
DT_CTRL = 0.1          # s — chu kỳ điều khiển 10 Hz
DT_PHYS = 0.002        # s — max_step_size Gazebo
SIM_SUBSTEPS = int(DT_CTRL / DT_PHYS)  # 50
CMD_SUBSTEPS = 5        # 5 lệnh con 20 ms (khớp firmware ESP32)

MAX_DELTA = 0.08        # rad/step — bước tối đa mỗi action
EPS_ARM = 0.25          # rad — giới hạn |target − q_meas|

# ===========================================================================
# Chuẩn hóa observation (thủ công, range cố định)
# ===========================================================================
NORM_JOINT = np.pi / 2  # chia giới hạn khớp
NORM_VEL = 2.0          # rad/s — clip ±1
NORM_POS = 0.35         # m
NORM_OBJ_VEL = 0.035    # m/s — chuẩn hóa vận tốc khối nhạy bén trong dải 1.2 - 2.0 cm/s
NORM_ERR = 0.1          # m
NORM_TILT = np.pi / 2   # rad

# ===========================================================================
# Reward hyperparameters — Gắp mặt cong hình trụ tròn (Đơn giản hóa triệt để)
# ===========================================================================
W_NEAR = 0.3            # hệ số shaping gần
W_YAW = 0.0             # Triệt tiêu phạt Yaw vì mặt trụ tròn đối xứng xoay 360°
W_TILT = 0.0            # Triệt tiêu phạt Tilt ép góc để tay vươn tự nhiên nhất
W_SPEED = 0.15          # hệ số phạt tốc độ tương đối khi tiếp cận
W_CONTACT = 0.3         # hệ số phạt contact ngón–khối
W_ACTION = 0.005        # hệ số phạt action magnitude
W_RATE = 0.005          # hệ số phạt action rate
REWARD_HANDOFF_GOOD = 15.0   # thưởng bàn giao tốt thành công
REWARD_HANDOFF_BAD = -2.0    # phạt bàn giao nhầm

# ===========================================================================
# Episode
# ===========================================================================
MAX_STEPS_STAGE_A = 60
MAX_STEPS_STAGE_B = 80

# ===========================================================================
# Domain Randomization — khoảng ban đầu (Khối trụ tròn)
# ===========================================================================
DR_CUBE_MASS = (0.07, 0.13)          # kg (0.1 ± 30%)
DR_CUBE_SIZE = (0.047, 0.053)        # m (Đường kính trụ tròn 0.05 ± 3mm)
DR_FRICTION_GRIPPER = (0.5, 2.0)     # μ
DR_EFFORT_SCALE = (0.8, 1.2)         # ± 20%
DR_SERVO_GAIN_SCALE = (0.7, 1.3)     # ± 30%
DR_PENDULUM_LENGTH_SCALE = (0.9, 1.1)  # ± 10%
DR_PENDULUM_DAMPING_RANGE = (0.001, 0.01)  # log-uniform quanh số đo

# ===========================================================================
# Spawn levels cho Khối trụ tròn (Cylinder) — Gắp MẶT TRÒN cong
# Tầm vươn xa thêm +10% (r = 0.26m - 0.33m), cao độ Z = 0.36m - 0.44m
# Khối luôn đứng yên (swing = 0.0) và đối xứng tròn xoay (yaw = 0.0)
# ===========================================================================
SPAWN_LEVELS = {
    0: dict(r=(0.28, 0.31), theta=np.radians(15), z=(0.38, 0.42),
            yaw=0.0, swing=0.0),
    1: dict(r=(0.27, 0.32), theta=np.radians(25), z=(0.37, 0.43),
            yaw=0.0, swing=0.0),
    2: dict(r=(0.26, 0.32), theta=np.radians(35), z=(0.37, 0.43),
            yaw=0.0, swing=0.0),
    3: dict(r=(0.25, 0.33), theta=np.radians(45), z=(0.36, 0.44),
            yaw=0.0, swing=0.0),
}

# ===========================================================================
# Thông số hiệu chỉnh phần cứng Phase R (Đã đo đạc từ flash/code Arduino)
# ===========================================================================
# Joint 1: TD8120MG
SERVO_J1_PULSE_MIN = 490   # us (0 deg)
SERVO_J1_PULSE_MAX = 2520  # us (180 deg)
SERVO_J1_ADC0 = 2533       # mV (0 deg)
SERVO_J1_ADCMID = 1616     # mV (90 deg)
SERVO_J1_ADC180 = 705      # mV (180 deg)
SERVO_J1_RATIO = 2.0       # Cau chia ap 47k+47k tren GPIO35 (Pin 7) theo sodonguyenly_calibservo (truoc day la 1.0 noi thang)

# Joint 2: RDS3120MG
SERVO_J2_PULSE_MIN = 510   # us (0 deg)
SERVO_J2_PULSE_MAX = 2440  # us (180 deg)
SERVO_J2_ADC0 = 756        # mV wiper (0 deg)
SERVO_J2_ADCMID = 3220     # mV wiper (90 deg)
SERVO_J2_ADC180 = 5790     # mV wiper (180 deg)
SERVO_J2_RATIO = 2.0       # Cau chia ap 47k+47k tren GPIO34 (Pin 6)

# Joint 3, 4, Gripper: MG995 / MG996R
SERVO_MG995_PULSE_MIN = 500  # us (0 deg)
SERVO_MG995_PULSE_MAX = 2500 # us (180 deg)
SERVO_MG995_ADC0 = 494       # mV wiper (0 deg)
SERVO_MG995_ADCMID = 3332    # mV wiper (90 deg)
SERVO_MG995_ADC180 = 6186    # mV wiper (180 deg)
SERVO_MG995_RATIO = 2.0      # Cau chia ap 47k+47k (J3 tren GPIO39/Pin 5, J4 tren GPIO36/Pin 4)

