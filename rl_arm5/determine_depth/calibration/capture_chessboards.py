"""Script chụp ảnh bàn cờ caro (Chessboard) phục vụ hiệu chuẩn Camera.

Hướng dẫn sử dụng:
  - Chuẩn bị tấm bàn cờ in phẳng (mặc định 9x6 góc trong, ô vuông 25mm).
  - Đưa bàn cờ ra trước camera ở nhiều khoảng cách, góc nghiêng và vị trí khác nhau.
  - Khi script phát hiện được các góc bàn cờ (hiển thị màu xanh lá cây):
    Nhấn phím [SPACE] để lưu ảnh.
  - Chụp khoảng 15-25 bức ảnh chất lượng tốt.
  - Nhấn phím [q] để kết thúc.
"""

import os
import sys
import cv2
import yaml

def main():
    # Đọc cấu hình
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    cam_cfg = cfg.get("camera", {})
    calib_cfg = cfg.get("calibration", {})

    device_index = cam_cfg.get("device_index", 0)
    width = cam_cfg.get("frame_width", 640)
    height = cam_cfg.get("frame_height", 480)

    cb_size = tuple(calib_cfg.get("chessboard_size", [9, 6]))
    img_dir = os.path.join(os.path.dirname(__file__), "..", calib_cfg.get("image_dir", "calibration/images"))
    os.makedirs(img_dir, exist_ok=True)

    print(f"=== BẮT ĐẦU CHỤP ẢNH HIỆU CHUẨN CAMERA ===")
    print(f"- Camera Device: {device_index} (Độ phân giải: {width}x{height})")
    print(f"- Kích thước bàn cờ (góc trong): {cb_size[0]} x {cb_size[1]}")
    print(f"- Thư mục lưu ảnh: {os.path.abspath(img_dir)}")
    print(f"- Hướng dẫn: Nhấn [SPACE] để chụp ảnh, nhấn [q] để thoát.\n")

    cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device_index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"[LỖI] Không thể kết nối với Camera thiết bị index {device_index}!")
        sys.exit(1)

    count = len([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])
    print(f"Hiện đã có {count} ảnh trong thư mục.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CẢNH BÁO] Không nhận được frame từ camera.")
            break

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Tìm góc bàn cờ (hỗ trợ cả khi cầm bàn cờ dọc hay ngang)
        active_cb_size = cb_size
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, cb_size, flags)
        if not found and cb_size[0] != cb_size[1]:
            rev_size = (cb_size[1], cb_size[0])
            found, corners = cv2.findChessboardCorners(gray, rev_size, flags)
            if found:
                active_cb_size = rev_size

        # Hiển thị thông báo trạng thái
        if found:
            cv2.drawChessboardCorners(display_frame, active_cb_size, corners, found)
            status_text = f"READY: Phat hien ban co {active_cb_size[0]}x{active_cb_size[1]}! Nhan [SPACE] de luu"
            color = (0, 255, 0)
        else:
            status_text = f"Chua phat hien du {cb_size[0]}x{cb_size[1]} goc... Hay can chinh goc do"
            color = (0, 0, 255)

        cv2.putText(display_frame, status_text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display_frame, f"Da luu: {count} anh | [SPACE]: Luu | [q]: Thoat",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Capture Chessboard Images", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # SPACE
            count += 1
            filename = os.path.join(img_dir, f"chessboard_{count:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[THÀNH CÔNG] Đã lưu ảnh {count}: {filename}")
        elif key == ord('q') or key == 27:  # q hoặc ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nHoàn thành chụp ảnh. Tổng số ảnh hiện có: {count}")
    print(f"Bước tiếp theo: Chạy 'python3 calibration/calibrate_camera.py' để tính ma trận camera.")

if __name__ == "__main__":
    main()
