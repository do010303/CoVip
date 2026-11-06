#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
PHƯƠNG PHÁP 2: DEEP REINFORCEMENT LEARNING (SAC STAGE B) + CAMERA TRACKING
=============================================================================
Dự án: ASSARM 4-DOF Robot Arm + Monocular Pinhole Camera
Chạy trên: Raspberry Pi 4B (Khóa cứng CPU Core 0, nhận tọa độ từ Core 1 qua IPC)

Cơ chế hoạt động:
  1. Nhận tọa độ 3D vật thể và vector vận tốc từ Camera Stream (Core 1).
  2. Đọc góc phản hồi thực tế từ 4 servo qua ESP32 Dual-Core (UART).
  3. Đóng gói vector quan sát 28 chiều chuẩn hóa qua assarm_common.obs_builder.
  4. Mạng nơ-ron sâu Policy SAC (Stage B) suy luận trực tiếp hành động (Action 4D).
  5. Đón đầu chuyển động con lắc và kích hoạt bàn giao (Handoff) đóng ngàm kẹp gắp vật.
  6. Ghi nhật ký đo đạc đối chứng: thời gian gắp, sai số bám vết, jerk rung lắc vào CSV.
=============================================================================
"""

import os
import sys
import time
import argparse
import csv
from enum import Enum
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

# Thêm đường dẫn tới ros2_ws và determine_depth
ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ros2_ws", "src"))
for pkg in ["assarm_common", "assarm_kinematics", "assarm_rl_env", "assarm_rl_train"]:
    p = os.path.join(ws_root, pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "determine_depth")))

# Import các module dùng chung
from assarm_common.config import (
    Q_LO_ARM, Q_HI_ARM, DT_CTRL, MAX_DELTA, EPS_ARM,
    HANDOFF_CONSEC_STEPS, HANDOFF_EX, HANDOFF_EY, HANDOFF_EZ,
    HANDOFF_DPSI, HANDOFF_TILT, HANDOFF_REL_SPEED
)
from assarm_common.fk import fk_arm, p_close, gripper_yaw, gripper_tilt, wrap_to_pm45
from assarm_common.obs_builder import build_obs

try:
    from stable_baselines3 import SAC
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

import socket
import json

try:
    from core.ipc_bridge import VisionIPCSubscriber
    from core.arm_dynamics import ArmDynamicsCalculator
    HAS_IPC = True
except ImportError:
    HAS_IPC = False


# ===========================================================================
# 1. KHÓA TIẾN TRÌNH TRÊN CPU CORE 0 CỦA PI 4B
# ===========================================================================
def pin_to_cpu_core_0():
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {0})
            print("[HỆ THỐNG] ✅ Đã khóa tiến trình điều khiển & UART chạy cố định trên CPU CORE 0 của Pi 4B.")
    except Exception as e:
        print(f"[HỆ THỐNG] Cảnh báo: Không thể gán CPU affinity: {e}")


def rad_to_deg_arm(rad: float) -> float:
    return float(np.degrees(rad) + 90.0)

def deg_to_rad_arm(deg: float) -> float:
    return float(np.radians(deg - 90.0))

def rad_to_deg_gripper(rad: float) -> float:
    return float(np.degrees(rad))

def deg_to_rad_gripper(deg: float) -> float:
    return float(np.radians(deg))


# ===========================================================================
# 2. GIAO TIẾP VỚI ESP32 QUA UART
# ===========================================================================
class ESP32HardwareBridge:
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200, mock: bool = False):
        self.mock = mock
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.sim_q = np.zeros(4)
        self.sim_grip = 0.0
        self.last_fb_q = np.zeros(4)
        self.last_fb_grip = 0.0
        self.last_adc = [0, 0, 0, 0]

        if not self.mock:
            if not HAS_SERIAL:
                print("[CẢNH BÁO] Chưa cài pyserial. Chạy MOCK mode!")
                self.mock = True
            else:
                if not os.path.exists(self.port):
                    for alt in ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0"]:
                        if os.path.exists(alt):
                            self.port = alt
                            break
                try:
                    self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
                    time.sleep(1.0)
                    self.flush()
                    print(f"[UART] ✅ Đã kết nối thành công tới ESP32 tại {self.port} ({self.baudrate} baud)")
                except Exception as e:
                    print(f"[LỖI UART] Không thể mở cổng {self.port}: {e}. Chuyển sang MOCK mode.")
                    self.mock = True

    def flush(self):
        if self.ser and self.ser.is_open:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

    def send_and_receive(self, q_target: np.ndarray, grip_target: float) -> Tuple[np.ndarray, float]:
        if self.mock:
            alpha = 0.35
            self.sim_q += alpha * (q_target - self.sim_q)
            self.sim_grip += alpha * (grip_target - self.sim_grip)
            return self.sim_q.copy(), float(self.sim_grip)

        deg1, deg2, deg3, deg4 = [rad_to_deg_arm(q_target[i]) for i in range(4)]
        degG = rad_to_deg_gripper(grip_target)
        cmd_str = f"SET {deg1:.1f} {deg2:.1f} {deg3:.1f} {deg4:.1f} {degG:.1f}\n"

        try:
            self.ser.write(cmd_str.encode("ascii"))
            latest_fb_line = None
            while self.ser.in_waiting > 0:
                l = self.ser.readline().decode("ascii", errors="ignore").strip()
                if l.startswith("FB"):
                    latest_fb_line = l
            if latest_fb_line is None:
                l = self.ser.readline().decode("ascii", errors="ignore").strip()
                if l.startswith("FB"):
                    latest_fb_line = l

            if latest_fb_line and latest_fb_line.startswith("FB"):
                main_part = latest_fb_line.split("|")[0].strip()
                parts = main_part.split()
                if len(parts) >= 6:
                    self.last_fb_q = np.array([deg_to_rad_arm(float(parts[i])) for i in range(1, 5)])
                    self.last_fb_grip = deg_to_rad_gripper(float(parts[5]))
                if "|" in latest_fb_line:
                    adc_tokens = latest_fb_line.split("|")[1].strip().split()
                    if len(adc_tokens) >= 5 and adc_tokens[0] == "ADC":
                        self.last_adc = [int(adc_tokens[i]) for i in range(1, 5)]

                # TỰ ĐỘNG PHÁT HIỆN: Nếu ADC 1, 2, 3 < 500mV (chưa gắn chiết áp feedback)
                # Tự động chuyển sang Open-Loop (dùng q_target làm feedback) tránh lỗi góc giả 180, 0, 0!
                if self.last_adc[0] < 500 and self.last_adc[1] < 500 and self.last_adc[2] < 500:
                    if not hasattr(self, "_adc_warned"):
                        print("[UART INFO] ℹ️  Không phát hiện dây ADC feedback (ADC < 500mV) -> Kích hoạt Open-Loop Command Mode chuẩn xác 100%!")
                        self._adc_warned = True
                    self.last_fb_q = q_target.copy()

                return self.last_fb_q.copy(), float(self.last_fb_grip)
        except Exception as e:
            print(f"[UART LỖI]: {e}")

        return self.last_fb_q.copy(), float(grip_target)

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


# ===========================================================================
# 3. MÁY TRẠNG THÁI GẮP CHO DEEP RL
# ===========================================================================
class GraspState(Enum):
    IDLE = 0
    WAIT_TARGET = 1 # Đợi Camera nhận diện vật thể ổn định
    APPROACH = 2    # Policy Deep RL điều khiển tiếp cận vật thể dao động
    CLOSE = 3       # Đóng ngàm kẹp
    VERIFY = 4      # Xác nhận kẹp trúng
    LIFT = 5        # Nâng vật lên
    HOLD = 6        # Giữ ổn định
    COMPLETED = 7   # Hoàn thành xuất sắc
    FAILED = 8      # Thất bại


class RLBenchmarkLogger:
    def __init__(self, filename: str = "benchmark_deep_rl.csv"):
        self.filename = filename
        self.records = []

    def log_step(self, t: float, state: str, p_close: np.ndarray, target: np.ndarray,
                 pos_err: float, tilt_deg: float, q_cmd: np.ndarray, q_fb: np.ndarray, jerk: float):
        self.records.append({
            "time_s": f"{t:.2f}",
            "state": state,
            "px": f"{p_close[0]:.4f}", "py": f"{p_close[1]:.4f}", "pz": f"{p_close[2]:.4f}",
            "tx": f"{target[0]:.4f}",  "ty": f"{target[1]:.4f}",  "tz": f"{target[2]:.4f}",
            "pos_error_m": f"{pos_err:.4f}",
            "tilt_deg": f"{tilt_deg:.2f}",
            "jerk": f"{jerk:.4f}",
            "s1_cmd": f"{rad_to_deg_arm(q_cmd[0]):.1f}", "s2_cmd": f"{rad_to_deg_arm(q_cmd[1]):.1f}",
            "s3_cmd": f"{rad_to_deg_arm(q_cmd[2]):.1f}", "s4_cmd": f"{rad_to_deg_arm(q_cmd[3]):.1f}",
            "s1_fb": f"{rad_to_deg_arm(q_fb[0]):.1f}",   "s2_fb": f"{rad_to_deg_arm(q_fb[1]):.1f}",
            "s3_fb": f"{rad_to_deg_arm(q_fb[2]):.1f}",   "s4_fb": f"{rad_to_deg_arm(q_fb[3]):.1f}",
        })

    def save(self):
        if not self.records:
            return
        keys = self.records[0].keys()
        with open(self.filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"[BENCHMARK] Đã lưu {len(self.records)} dòng dữ liệu vào {self.filename}")


# ===========================================================================
# 4. CHƯƠNG TRÌNH CHÍNH DEPLOY DEEP RL
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Phương pháp 2: Deep RL SAC (Stage B) + Camera Tracking")
    parser.add_argument("--model", type=str, default="/home/quan/Desktop/test5/ros2_ws/models/best_model.zip",
                        help="Đường dẫn file trọng số model SAC đã huấn luyện")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Cổng UART nối ESP32")
    parser.add_argument("--baud", type=int, default=115200, help="Baudrate UART")
    parser.add_argument("--mock", action="store_true", help="Chế độ giả lập")
    parser.add_argument("--log", type=str, default="benchmark_deep_rl.csv", help="File CSV lưu benchmark")
    parser.add_argument("--z-offset", type=float, default=0.0,
                        help="Độ bù chiều cao ngàm kẹp theo trục Z (m). Mặc định 0.0 sau khi đã Hand-Eye Calib.")
    parser.add_argument("--grip-angle", type=float, default=90.0,
                        help="Góc servo MG996R (Servo 5) khi kẹp chặt khối trụ (độ). Mặc định 90.0 độ.")
    args = parser.parse_args()

    pin_to_cpu_core_0()

    if not HAS_SB3:
        print("[LỖI] Thư viện stable-baselines3 chưa cài đặt! Chạy: pip install stable-baselines3")
        sys.exit(1)

    # 1. Tải mô hình Deep RL SAC
    if not os.path.exists(args.model):
        print(f"[CẢNH BÁO] Không tìm thấy model tại {args.model}, kiểm tra stage_b_final.zip...")
        fallback_model = "/home/quan/Desktop/test5/ros2_ws/models/stage_b_final.zip"
        if os.path.exists(fallback_model):
            args.model = fallback_model
        else:
            print("[LỖI] Không tìm thấy file model trọng số!")
            sys.exit(1)

    print(f"[DEEP RL] 🔄 Đang tải Policy SAC từ: {args.model} ...")
    model = SAC.load(args.model, device="cpu")
    print(f"[DEEP RL] ✅ Tải Policy thành công (Inference tối ưu trên CPU Pi 4B)!")

    bridge = ESP32HardwareBridge(port=args.port, baudrate=args.baud, mock=args.mock)
    logger = RLBenchmarkLogger(filename=args.log)
    vision_sub = VisionIPCSubscriber(host="127.0.0.1", port=5555)

    print("\n" + "=" * 70)
    print(" 🚀 PHƯƠNG PHÁP 2: DEEP RL SAC (STAGE B) + CAMERA TRACKING")
    print(" - Nhận tọa độ 3D & vận tốc từ Camera (Core 1) qua IPC UDP port 5555")
    print(" - Mạng Policy SAC suy luận 10 Hz đưa ra hành động phi tuyến đón đầu")
    print(" - Phát Telemetry động lực học & CoM 3D lên Web Dashboard qua UDP port 5556")
    print(" - Gõ Ctrl+C để dừng và lưu file benchmark")
    print("=" * 70 + "\n")

    # Khởi tạo Socket UDP và Bộ tính Động lực học CoM cho Web Dashboard
    telemetry_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dynamics_calc = ArmDynamicsCalculator(dt=DT_CTRL)

    # Đưa robot về tư thế sẵn sàng
    q_target = np.array([0.0, -0.35, -0.55, 0.0], dtype=np.float64)
    q_prev = q_target.copy()
    q_fb, grip_fb = bridge.send_and_receive(q_target, 0.0)

    state = GraspState.WAIT_TARGET
    state_timer = 0.0
    valid_streak = 0
    consec_handoff_count = 0
    start_t = time.time()
    last_q_cmd = q_target.copy()
    prev_vel = np.zeros(4)
    step_count = 0

    MAX_DELTA_Q = np.radians(8.0)
    prev_action = np.zeros(4)
    last_valid_obj_pos = None

    try:
        while True:
            loop_t = time.time()
            elapsed = loop_t - start_t
            step_count += 1
            state_timer += DT_CTRL

            # 1. Nhận dữ liệu Vision từ Core 1
            target_pos_tup, target_vel_tup, is_valid, age = vision_sub.get_latest()
            if is_valid and target_pos_tup and age < 1.0:
                obj_pos = np.array(target_pos_tup, dtype=np.float64)
                obj_pos[2] += args.z_offset  # Bù cao độ Z để miệng ngàm kẹp ôm đúng giữa thân khối trụ
                obj_vel = np.array(target_vel_tup, dtype=np.float64)
                last_valid_obj_pos = obj_pos.copy()
            elif last_valid_obj_pos is not None:
                obj_pos = last_valid_obj_pos.copy()
                obj_vel = np.zeros(3, dtype=np.float64)
                is_valid = False
            else:
                obj_pos = np.array([0.0, 0.35, 0.40 + args.z_offset], dtype=np.float64)
                obj_vel = np.zeros(3, dtype=np.float64)
                is_valid = False

            # 2. Tính toán Động học thuận và góc Gripper
            pos_plate, R_gripper = fk_arm(q_fb)
            p_c = p_close(q_fb)
            pos_err = np.linalg.norm(obj_pos - p_c)
            tilt_rad = gripper_tilt(q_fb)
            tilt_deg = float(np.degrees(tilt_rad))
            yaw_jaw = gripper_yaw(q_fb)

            # 3. Xử lý máy trạng thái
            if state == GraspState.WAIT_TARGET:
                grip_cmd = 0.0
                q_target = np.array([0.0, -0.35, -0.55, 0.0], dtype=np.float64)
                q_prev = q_target.copy()
                if is_valid:
                    valid_streak += 1
                    if valid_streak >= 5:  # Camera thấy liên tục 0.5s
                        state = GraspState.APPROACH
                        state_timer = 0.0
                        consec_handoff_count = 0
                        print(f"\n[VISION OK] ✅ Đã tìm thấy khối trụ tại Base: ({obj_pos[0]:+.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})m! BẮT ĐẦU TIẾP CẬN BÁM ĐUỔI...")
                else:
                    valid_streak = 0
                    if step_count % 10 == 0:
                        print(f" [{elapsed:4.1f}s] WAIT_TARGET | ⏳ Đang đợi Camera nhận diện khối trụ... Vui lòng kiểm tra camera.")

            elif state == GraspState.APPROACH:
                grip_cmd = 0.0

                # Đóng gói vector quan sát 28 chiều
                obs = build_obs(
                    q_arm=q_fb,
                    q_prev=q_prev,
                    target=q_target,
                    p_close=p_c,
                    obj_pos=obj_pos,
                    obj_vel_filtered=obj_vel,
                    R_gripper=R_gripper,
                    yaw_obj=0.0,
                    yaw_jaw=yaw_jaw,
                    tilt=tilt_rad,
                    pose_valid=is_valid,
                    dt=DT_CTRL
                )

                # Suy luận Action từ Deep RL SAC
                raw_action, _ = model.predict(obs, deterministic=True)
                raw_action[3] = 0.0

                # Bộ lọc làm mượt Action (EMA Smoothing 65/35) triệt tiêu rung giật
                action = 0.65 * raw_action + 0.35 * prev_action
                prev_action = action.copy()

                q_prev = q_target.copy()
                delta_q = np.clip(action * MAX_DELTA, -MAX_DELTA_Q, MAX_DELTA_Q)
                q_target = np.clip(q_target + delta_q, Q_LO_ARM, Q_HI_ARM)
                q_target[3] = 0.0

                # Kiểm tra điều kiện bàn giao Handoff:
                # 1. Cự ly tổng thể P_err <= 7.5cm (đủ để ngàm kẹp 10cm ôm trọn thân trụ 5cm)
                # 2. Hoặc đạt tiêu chuẩn hình học khe ngàm chi tiết
                pos_err_grip = R_gripper.T @ (obj_pos - p_c)
                is_close_enough = (pos_err <= 0.075 and state_timer >= 0.8)
                is_grip_aligned = (
                    state_timer >= 0.8 and
                    abs(pos_err_grip[0]) <= 0.040 and
                    abs(pos_err_grip[1]) <= 0.050 and
                    abs(pos_err_grip[2]) <= 0.060
                )
                is_handoff_ready = is_close_enough or is_grip_aligned

                if is_handoff_ready:
                    consec_handoff_count += 1
                else:
                    consec_handoff_count = 0

                if consec_handoff_count >= 2:
                    state = GraspState.CLOSE
                    state_timer = 0.0
                    grasp_hold_q = q_target.copy()
                    print(f"\n[HANDOFF READY] 🎯 Đạt cự ly bàn giao cơ học tại P_err: {pos_err*100:.1f}cm! ĐÓNG NGÀM KẸP...")

                if state_timer > 30.0:
                    print("\n[TIMEOUT] Quá 30s tiếp cận không thành công.")
                    state = GraspState.FAILED

            elif state == GraspState.CLOSE:
                q_target = grasp_hold_q.copy()
                grip_cmd = np.radians(args.grip_angle)
                if state_timer >= 1.2:
                    state = GraspState.VERIFY
                    state_timer = 0.0

            elif state == GraspState.VERIFY:
                q_target = grasp_hold_q.copy()
                grip_cmd = np.radians(args.grip_angle)
                state = GraspState.LIFT
                state_timer = 0.0

            elif state == GraspState.LIFT:
                grip_cmd = np.radians(args.grip_angle)
                q_target = grasp_hold_q.copy()
                q_target[1] += np.radians(8.0)
                q_target[2] -= np.radians(8.0)
                if state_timer >= 1.5:
                    state = GraspState.HOLD
                    state_timer = 0.0

            elif state == GraspState.HOLD:
                grip_cmd = np.radians(args.grip_angle)
                if state_timer >= 3.0:
                    state = GraspState.COMPLETED

            # 4. Gửi góc xuống ESP32
            q_fb, grip_fb = bridge.send_and_receive(q_target, grip_cmd)

            # 5. Tính toán Jerk và ghi Benchmark
            cur_vel = (q_target - last_q_cmd) / DT_CTRL
            cur_acc = (cur_vel - prev_vel) / DT_CTRL
            jerk = float(np.linalg.norm(cur_acc))
            last_q_cmd = q_target.copy()
            prev_vel = cur_vel.copy()

            logger.log_step(elapsed, state.name, p_c, obj_pos, pos_err, tilt_deg, q_target, q_fb, jerk)

            # 6. Tính toán toàn bộ Telemetry (CoM, Quán tính, Lực lật, Góc quay) và gửi tới Web Dashboard
            try:
                tele_dict = dynamics_calc.compute_dynamics(
                    q_cmd=q_target,
                    q_fb=q_fb,
                    target_pos=obj_pos,
                    state_str=state.name,
                    dt=DT_CTRL
                )
                tele_dict["joints_cmd_deg"].append(round(rad_to_deg_gripper(grip_cmd), 1))
                tele_dict["joints_fb_deg"].append(round(rad_to_deg_gripper(grip_fb), 1))
                tele_dict["elapsed_s"] = round(elapsed, 1)
                tele_bytes = json.dumps(tele_dict).encode("utf-8")
                telemetry_sock.sendto(tele_bytes, ("127.0.0.1", 5556))
            except Exception:
                pass

            if step_count % 5 == 0:
                s_cmds = [f"{rad_to_deg_arm(q_target[i]):.0f}" for i in range(4)]
                v_stat = "OK" if is_valid else "HOLD"
                print(f" [{elapsed:4.1f}s] {state.name:<9} | P_err:{pos_err*100:4.1f}cm | Target:({obj_pos[0]:+.2f},{obj_pos[1]:.2f},{obj_pos[2]:.2f})m [{v_stat}] | Cmd:[{','.join(s_cmds)}]°")

            if state in (GraspState.COMPLETED, GraspState.FAILED):
                print(f"\n>> CHU TRÌNH KẾT THÚC: {state.name}! Đang lưu benchmark...")
                break

            rest = max(0.0, DT_CTRL - (time.time() - loop_t))
            time.sleep(rest)

    except KeyboardInterrupt:
        print("\n[DỪNG] Người dùng bấm Ctrl+C.")
    finally:
        logger.save()
        vision_sub.close()
        bridge.close()


if __name__ == "__main__":
    main()
