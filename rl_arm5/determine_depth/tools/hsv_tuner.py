"""Công cụ GUI trực quan cân chỉnh dải màu HSV thời gian thực.

Hỗ trợ các chế độ nhanh:
  1. BLUE MODE (Mặc định - Khối trụ XANH DƯƠNG): Tối ưu và đơn giản nhất trên nền bảng trắng.
  2. RED MODE (Khối trụ ĐỎ): Lọc 2 nhánh dải đỏ (0-10 & 165-180).
  3. CUSTOM MODE: 6 thanh trượt H/S/V tự do tùy biến.

Phím tắt:
  - Phím [b]: Chọn chế độ XANH DƯƠNG (Blue Mode).
  - Phím [r]: Chọn chế độ ĐỎ (Red Mode).
  - Phím [c]: Chọn chế độ TÙY BIẾN (Custom Mode).
  - Click chuột trái vào khối trụ: Xem giá trị HSV tức thời tại điểm click.
  - Phím [s]: Tự động LƯU dải màu vào `configs/config.yaml`.
  - Phím [q]: Thoát.
"""

import os
import sys
import cv2
import numpy as np
import yaml

clicked_hsv = None

def mouse_callback(event, x, y, flags, param):
    global clicked_hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_hsv = param
        if frame_hsv is not None and 0 <= y < frame_hsv.shape[0] and 0 <= x < frame_hsv.shape[1]:
            clicked_hsv = frame_hsv[y, x]
            h, s, v = clicked_hsv
            print(f"[CLICK] Tọa độ ({x}, {y}) -> HSV = [{h}, {s}, {v}]")

def nothing(x):
    pass

def main():
    global clicked_hsv
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    cam_cfg = cfg.get("camera", {})
    seg_cfg = cfg.get("segmentation", {})

    device_index = cam_cfg.get("device_index", 0)
    width = cam_cfg.get("frame_width", 640)
    height = cam_cfg.get("frame_height", 480)

    color_mode = seg_cfg.get("color_mode", "blue").lower()

    # Dải màu mặc định
    blue_lower = seg_cfg.get("blue_lower", [95, 90, 50])
    blue_upper = seg_cfg.get("blue_upper", [130, 255, 255])
    red_lower1 = seg_cfg.get("red_lower1", [0, 70, 50])
    red_upper1 = seg_cfg.get("red_upper1", [10, 255, 255])
    red_lower2 = seg_cfg.get("red_lower2", [165, 70, 50])
    red_upper2 = seg_cfg.get("red_upper2", [180, 255, 255])

    # Khởi tạo giá trị thanh trượt ban đầu theo chế độ xanh dương
    init_h_min = blue_lower[0] if color_mode == "blue" else 0
    init_h_max = blue_upper[0] if color_mode == "blue" else 179
    init_s_min = blue_lower[1] if color_mode == "blue" else 80
    init_v_min = blue_lower[2] if color_mode == "blue" else 50

    # Mở camera với backend V4L2
    cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device_index)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"[LỖI] Không thể mở camera thiết bị {device_index}!")
        sys.exit(1)

    window_name = "HSV Color Tuner"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Tạo thanh trượt Trackbars
    cv2.createTrackbar("H Min", window_name, int(init_h_min), 179, nothing)
    cv2.createTrackbar("H Max", window_name, int(init_h_max), 179, nothing)
    cv2.createTrackbar("S Min", window_name, int(init_s_min), 255, nothing)
    cv2.createTrackbar("S Max", window_name, 255, 255, nothing)
    cv2.createTrackbar("V Min", window_name, int(init_v_min), 255, nothing)
    cv2.createTrackbar("V Max", window_name, 255, 255, nothing)

    print("=" * 65)
    print(" BẮT ĐẦU CÂN CHỈNH MÀU SẮC (HSV TUNER)")
    print("=" * 65)
    print(f"- Chế độ đang chọn: {color_mode.upper()} MODE")
    print("- [b]: Chuyển sang BLUE MODE (Khối trụ Xanh dương - Khuyên dùng)")
    print("- [r]: Chuyển sang RED MODE (Khối trụ Đỏ)")
    print("- [c]: Chuyển sang CUSTOM MODE (Tùy biến dải màu)")
    print("- Click chuột trái vào vật thể để xem giá trị HSV thực tế")
    print("- [s]: LƯU cấu hình vào config.yaml | [q]: Thoát\n")

    current_hsv_frame = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        current_hsv_frame = hsv
        cv2.setMouseCallback(window_name, mouse_callback, current_hsv_frame)

        # Lấy giá trị thanh trượt
        h_min = cv2.getTrackbarPos("H Min", window_name)
        h_max = cv2.getTrackbarPos("H Max", window_name)
        s_min = cv2.getTrackbarPos("S Min", window_name)
        s_max = cv2.getTrackbarPos("S Max", window_name)
        v_min = cv2.getTrackbarPos("V Min", window_name)
        v_max = cv2.getTrackbarPos("V Max", window_name)

        if color_mode == "blue":
            # Dải màu xanh dương từ trackbar hoặc preset chuẩn
            lower_b = np.array([h_min, s_min, v_min], dtype=np.uint8)
            upper_b = np.array([h_max, s_max, v_max], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_b, upper_b)
        elif color_mode == "red":
            # Dải màu đỏ kép
            lower_r1 = np.array([0, s_min, v_min], dtype=np.uint8)
            upper_r1 = np.array([12, s_max, v_max], dtype=np.uint8)
            lower_r2 = np.array([165, s_min, v_min], dtype=np.uint8)
            upper_r2 = np.array([180, s_max, v_max], dtype=np.uint8)
            m1 = cv2.inRange(hsv, lower_r1, upper_r1)
            m2 = cv2.inRange(hsv, lower_r2, upper_r2)
            mask = cv2.bitwise_or(m1, m2)
        else:
            lower_bound = np.array([h_min, s_min, v_min], dtype=np.uint8)
            upper_bound = np.array([h_max, s_max, v_max], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Lọc nhiễu hình thái học
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

        result = cv2.bitwise_and(frame, frame, mask=mask_cleaned)

        display_frame = frame.copy()
        if color_mode == "blue":
            mode_text = "MODE: BLUE (XANH DUONG)"
            mode_color = (255, 100, 0)
        elif color_mode == "red":
            mode_text = "MODE: RED (DO)"
            mode_color = (0, 0, 255)
        else:
            mode_text = "MODE: CUSTOM"
            mode_color = (0, 255, 255)

        cv2.putText(display_frame, mode_text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        if clicked_hsv is not None:
            ch, cs, cv_val = clicked_hsv
            cv2.putText(display_frame, f"Sampled HSV: [{ch}, {cs}, {cv_val}]", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(display_frame, "[b]: Blue | [r]: Red | [s]: Luu | [q]: Thoat", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)

        # Ghép 3 màn hình
        mask_bgr = cv2.cvtColor(mask_cleaned, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([
            cv2.resize(display_frame, (320, 240)),
            cv2.resize(mask_bgr, (320, 240)),
            cv2.resize(result, (320, 240))
        ])

        cv2.imshow(window_name, combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('b'):
            color_mode = "blue"
            cv2.setTrackbarPos("H Min", window_name, 95)
            cv2.setTrackbarPos("H Max", window_name, 130)
            cv2.setTrackbarPos("S Min", window_name, 90)
            cv2.setTrackbarPos("V Min", window_name, 50)
            print("[CHUYỂN CHẾ ĐỘ] -> BLUE MODE (Xanh dương)")
        elif key == ord('r'):
            color_mode = "red"
            cv2.setTrackbarPos("H Min", window_name, 0)
            cv2.setTrackbarPos("H Max", window_name, 12)
            cv2.setTrackbarPos("S Min", window_name, 70)
            cv2.setTrackbarPos("V Min", window_name, 50)
            print("[CHUYỂN CHẾ ĐỘ] -> RED MODE (Màu đỏ)")
        elif key == ord('c'):
            color_mode = "custom"
            print("[CHUYỂN CHẾ ĐỘ] -> CUSTOM MODE")
        elif key == ord('s'):
            if "segmentation" not in cfg:
                cfg["segmentation"] = {}

            cfg["segmentation"]["color_mode"] = color_mode
            if color_mode == "blue":
                cfg["segmentation"]["blue_lower"] = [int(h_min), int(s_min), int(v_min)]
                cfg["segmentation"]["blue_upper"] = [int(h_max), int(s_max), int(v_max)]
            elif color_mode == "red":
                cfg["segmentation"]["red_lower1"] = [0, int(s_min), int(v_min)]
                cfg["segmentation"]["red_upper1"] = [12, int(s_max), int(v_max)]
                cfg["segmentation"]["red_lower2"] = [165, int(s_min), int(v_min)]
                cfg["segmentation"]["red_upper2"] = [180, int(s_max), int(v_max)]
            else:
                cfg["segmentation"]["hsv_lower"] = [int(h_min), int(s_min), int(v_min)]
                cfg["segmentation"]["hsv_upper"] = [int(h_max), int(s_max), int(v_max)]

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=None)

            print(f"\n[THÀNH CÔNG] Đã lưu dải màu {color_mode.upper()} vào {config_path}!\n")
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
