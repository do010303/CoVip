#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
CÔNG CỤ HIỆU CHUẨN HAND-EYE CALIBRATION (INTERACTIVE ROBOT JOGGING + CAMERA)
=============================================================================
Dự án: ASSARM 4-DOF Robot Arm + Monocular Camera
Mục đích:
  Xác định chính xác tuyệt đối ma trận chuyển đổi Hand-Eye (Camera <-> Robot Base).

Do các khớp Servo trên tay robot chạy Open-Loop (chưa có feedback chiết áp ADC),
công cụ này cho phép bạn điều khiển (JOG) tay robot bằng phím bấm trực tiếp
cho đến khi ngàm kẹp ôm trọn tâm khối trụ, sau đó bấm 'C' để tính toán chuẩn xác 100%!

BÀN PHÍM ĐIỀU KHIỂN ROBOT TRÊN CỬA SỔ CAMERA:
  • 'A' / 'D' : Xoay đế J1 (Trái / Phải)
  • 'W' / 'S' : Vai J2 (Vươn tới / Thu về)
  • 'R' / 'F' : Khuỷu J3 (Nâng lên / Hạ xuống)
  • 'T' / 'G' : Cổ tay J4 (Xoay ngửa / Xoay úp)
  • 'O' / 'P' : Ngàm kẹp (Mở ra / Kẹp chặt)
  • 'SPACE'   : Reset về tư thế ngắm chuẩn [90, 60, 20, 90]°
  • 'C'       : Tính toán Hand-Eye Calibration
  • 'L'       : LƯU thông số vào configs/config.yaml
  • 'Q'       : Thoát
=============================================================================
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sodonguyenly_calibservo")))
try:
    from baseline_ik_tracker import compute_p_close, rad_to_deg_arm, deg_to_rad_arm, ESP32HardwareBridge
    HAS_ROBOT = True
except ImportError:
    HAS_ROBOT = False

from core.cylinder_detector import CylinderDetector
from core.pinhole_3d import Pinhole3DEstimator
from core.camera_stream import CameraStream


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_hand_eye_config(config_path: str, z_cam_world: float, forward_m: float, lateral_m: float):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "hand_eye" not in cfg:
        cfg["hand_eye"] = {}
    cfg["hand_eye"]["z_cam_world"] = float(round(z_cam_world, 4))
    cfg["hand_eye"]["forward_offset_m"] = float(round(forward_m, 4))
    cfg["hand_eye"]["lateral_offset_m"] = float(round(lateral_m, 4))
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"\n[LƯU THÀNH CÔNG] ✅ Đã cập nhật vào file: {config_path}")
    print(f"  - z_cam_world:      {z_cam_world*100:.2f} cm ({z_cam_world:.4f} m)")
    print(f"  - forward_offset_m: {forward_m*100:.2f} cm ({forward_m:.4f} m)")
    print(f"  - lateral_offset_m: {lateral_m*100:.2f} cm ({lateral_m:.4f} m)")


def main():
    parser = argparse.ArgumentParser(description="Hand-Eye Calibration Tool with Interactive Jogging")
    parser.add_argument("--camera", type=int, default=2, help="Chỉ số thiết bị camera (mặc định 2)")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB1", help="Cổng UART nối ESP32 (/dev/ttyUSB0 hoặc /dev/ttyUSB1)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="File config.yaml")
    args = parser.parse_args()

    # Kiểm tra cổng serial
    if not os.path.exists(args.port):
        if os.path.exists("/dev/ttyUSB0"):
            args.port = "/dev/ttyUSB0"
        elif os.path.exists("/dev/ttyUSB1"):
            args.port = "/dev/ttyUSB1"

    cfg = load_config(args.config)
    hand_eye_cfg = cfg.get("hand_eye", {})
    cur_z_cam = hand_eye_cfg.get("z_cam_world", 0.500)
    cur_forward = hand_eye_cfg.get("forward_offset_m", 0.070)
    cur_lateral = hand_eye_cfg.get("lateral_offset_m", 0.000)

    # Khởi tạo Camera
    stream = CameraStream(src=args.camera, width=640, height=480, fps=30)
    stream.start()
    time.sleep(0.5)

    seg_cfg = cfg.get("segmentation", {})
    detector = CylinderDetector(
        color_mode=seg_cfg.get("color_mode", "blue"),
        blue_lower=tuple(seg_cfg.get("blue_lower", [90, 45, 40])),
        blue_upper=tuple(seg_cfg.get("blue_upper", [135, 255, 255])),
        min_area=seg_cfg.get("min_contour_area", 1000)
    )

    estimator_3d = Pinhole3DEstimator(
        fx=830.5, fy=830.5, cx=320.0, cy=240.0,
        d_real_m=cfg.get("target_object", {}).get("d_real_m", 0.05)
    )

    bridge = ESP32HardwareBridge(port=args.port, baudrate=115200, mock=False) if HAS_ROBOT else None

    # Góc lệnh ban đầu: [90, 60, 20, 90] (tư thế vươn tới sẵn sàng)
    cmd_deg = [90.0, 60.0, 20.0, 90.0]
    grip_deg = 0.0

    if bridge:
        q_send = np.array([deg_to_rad_arm(x) for x in cmd_deg])
        bridge.send_and_receive(q_send, grip_deg)

    print("\n" + "=" * 75)
    print(" 🎯 CÔNG CỤ HIỆU CHUẨN HAND-EYE (INTERACTIVE ROBOT JOGGING)")
    print("=" * 75)
    print(f" Cổng ESP32: {args.port}")
    print(f" Cấu hình hiện tại trong config.yaml:")
    print(f"  • z_cam_world:      {cur_z_cam*100:.1f} cm ({cur_z_cam:.3f} m)")
    print(f"  • forward_offset_m: {cur_forward*100:.1f} cm ({cur_forward:.3f} m)")
    print("-" * 75)
    print(" HƯỚNG DẪN ĐIỀU KHIỂN (Bấm phím trên cửa sổ hình ảnh Camera):")
    print("  • A / D : Quay đế J1      |  W / S : Vươn vai J2")
    print("  • R / F : Nâng khuỷu J3   |  T / G : Khớp cổ tay J4")
    print("  • O / P : Mở/Đóng kẹp     |  SPACE : Reset [90, 60, 20, 90]°")
    print("  • C     : TÍNH TOÁN Hand-Eye Calibration")
    print("  • L     : LƯU vào configs/config.yaml")
    print("  • Q     : Thoát")
    print("=" * 75 + "\n")

    latest_calib = None
    cv2.namedWindow("Hand-Eye Calibration", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame, _ = stream.read(wait_new=True)
            if not ret or frame is None:
                continue

            detection = detector.detect(frame)
            disp_frame = detection["processed_frame"].copy()

            cam_X, cam_Y, cam_Z = 0.0, 0.0, 0.0
            found = detection["found"]
            if found:
                cam_X, cam_Y, cam_Z = estimator_3d.estimate(detection["u"], detection["v"], detection["w_px"])

            # Động học thuận tính từ góc lệnh thực tế gửi xuống ESP32
            cur_q = np.array([deg_to_rad_arm(x) for x in cmd_deg])
            p_arm, _ = compute_p_close(cur_q)

            # Vẽ HUD thông số
            cv2.putText(disp_frame, f"CAM (Pinhole): X={cam_X:+.3f}m Y={cam_Y:+.3f}m Z={cam_Z:.3f}m",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(disp_frame, f"ARM (P_close): X={p_arm[0]:+.3f}m Y={p_arm[1]:.3f}m Z={p_arm[2]:.3f}m",
                        (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(disp_frame, f"Cmd: J1={cmd_deg[0]:.0f} J2={cmd_deg[1]:.0f} J3={cmd_deg[2]:.0f} J4={cmd_deg[3]:.0f} Grip={grip_deg:.0f}",
                        (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

            if latest_calib:
                cv2.putText(disp_frame, f"KQU CALIB: Z_cam={latest_calib['z_cam']*100:.1f}cm | Fwd={latest_calib['fwd']*100:.1f}cm",
                            (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.putText(disp_frame, "Nhan 'L' de LUU vao configs/config.yaml",
                            (15, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
            else:
                cv2.putText(disp_frame, "Dung WASD/RF chinh tay kep om khoi tru, roi bam 'C'",
                            (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

            cv2.imshow("Hand-Eye Calibration", disp_frame)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):  # Reset
                cmd_deg = [90.0, 60.0, 20.0, 90.0]
                if bridge:
                    bridge.send_and_receive(np.array([deg_to_rad_arm(x) for x in cmd_deg]), grip_deg)
            # Jog J1
            elif key in (ord('a'), ord('A')):
                cmd_deg[0] = min(180.0, cmd_deg[0] + 2.0)
                if bridge: bridge.send_and_receive(np.array([deg_to_rad_arm(x) for x in cmd_deg]), grip_deg)
            elif key in (ord('d'), ord('D')):
                cmd_deg[0] = max(0.0, cmd_deg[0] - 2.0)
                if bridge: bridge.send_and_receive(np.array([deg_to_rad_arm(x) for x in cmd_deg]), grip_deg)
            # Jog J2
            elif key in (ord('w'), ord('W')):
                cmd_deg[1] = max(0.0, cmd_deg[1] - 2.0) # Vươn tới
                if bridge: bridge.send_and_receive(np.array([deg_to_rad_arm(x) for x in cmd_deg]), grip_deg)
            elif key in (ord('s'), ord('S')):
                cmd_deg[1] = min(180.0, cmd_deg[1] + 2.0) # Thu về
                if bridge: bridge.send_and_receive(np.array([deg_to_rad_arm(x) for x in cmd_deg]), grip_deg)
            # Jog J3
            elif key in (ord('r'), ord('R')):
                cmd_deg[2] = min(180.0, cmd_deg[2] + 2.0) # Nâng lên
                if bridge: bridge.send_and_receive(np.array([deg_to_rad_arm(x) for x in cmd_deg]), grip_deg)
            elif key in (ord('f'), ord('F')):
                cmd_deg[2] = max(0.0, cmd_deg[2] - 2.0) # Hạ xuống
                if bridge: bridge.send_and_receive(np.array([deg_to_rad_arm(x) for x in cmd_deg]), grip_deg)
            # Jog J4
            elif key in (ord('t'), ord('T')):
                cmd_deg[3] = min(180.0, cmd_deg[3] + 2.0)
                if bridge: bridge.send_and_receive(np.array([deg_to_rad_arm(x) for x in cmd_deg]), grip_deg)
            elif key in (ord('g'), ord('G')):
                cmd_deg[3] = max(0.0, cmd_deg[3] - 2.0)
                if bridge: bridge.send_and_receive(np.array([deg_to_rad_arm(x) for x in cmd_deg]), grip_deg)
            # Gripper
            elif key in (ord('o'), ord('O')):
                grip_deg = 0.0
                if bridge: bridge.send_and_receive(np.array([deg_to_rad_arm(x) for x in cmd_deg]), grip_deg)
            elif key in (ord('p'), ord('P')):
                grip_deg = 90.0
                if bridge: bridge.send_and_receive(np.array([deg_to_rad_arm(x) for x in cmd_deg]), grip_deg)

            # Tính toán Calib
            elif key in (ord('c'), ord('C')):
                if not found:
                    print("\n[CẢNH BÁO] ⚠️ Camera chưa nhận diện được khối trụ!")
                else:
                    calc_forward = p_arm[1] - cam_Z
                    calc_z_cam   = p_arm[2] + cam_Y
                    calc_lateral = p_arm[0] - cam_X
                    latest_calib = {
                        "z_cam": calc_z_cam,
                        "fwd": calc_forward,
                        "lat": calc_lateral
                    }
                    print("\n" + "*" * 60)
                    print(" 🎯 KẾT QUẢ TÍNH TOÁN HIỆU CHUẨN HAND-EYE:")
                    print(f"  • Góc khớp lúc kẹp:         {cmd_deg}°")
                    print(f"  • Độ cao quang tâm Z_cam:   {calc_z_cam*100:.2f} cm ({calc_z_cam:.4f} m)")
                    print(f"  • Khoảng nhô ra trước Fwd:   {calc_forward*100:.2f} cm ({calc_forward:.4f} m)")
                    print(f"  • Độ lệch ngang Lateral:     {calc_lateral*100:.2f} cm ({calc_lateral:.4f} m)")
                    print("  -> Nhấn phím 'L' trên cửa sổ để lưu vào configs/config.yaml!")
                    print("*" * 60 + "\n")

            # Lưu vào config
            elif key in (ord('l'), ord('L')):
                if latest_calib:
                    save_hand_eye_config(args.config, latest_calib["z_cam"], latest_calib["fwd"], latest_calib["lat"])
                    print(">> Đã lưu thành công vào config.yaml! Bạn có thể bấm 'Q' để hoàn tất.")
                else:
                    print("[THÔNG BÁO] Vui lòng bấm 'C' để tính toán trước khi bấm 'L'.")

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        if bridge:
            bridge.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
