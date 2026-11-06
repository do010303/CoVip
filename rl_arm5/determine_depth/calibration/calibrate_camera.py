"""Script tính toán ma trận thông số nội tại (K) và hệ số méo (dist) từ tập ảnh bàn cờ.

Kết quả sẽ được tự động lưu vào file cấu hình `configs/camera_calibration.yaml`.
"""

import os
import glob
import cv2
import numpy as np
import yaml

def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    calib_cfg = cfg.get("calibration", {})
    cb_size = tuple(calib_cfg.get("chessboard_size", [9, 6]))
    square_size = float(calib_cfg.get("square_size_mm", 25.0))
    img_dir = os.path.join(os.path.dirname(__file__), "..", calib_cfg.get("image_dir", "calibration/images"))
    output_yaml = os.path.join(os.path.dirname(__file__), "..", calib_cfg.get("calibration_file", "configs/camera_calibration.yaml"))

    print(f"=== BẮT ĐẦU TÍNH TOÁN HIỆU CHUẨN CAMERA (CALIBRATION) ===")
    print(f"- Kích thước bàn cờ: {cb_size[0]} x {cb_size[1]} góc trong")
    print(f"- Kích thước ô vuông: {square_size} mm")
    print(f"- Thư mục ảnh mẫu: {os.path.abspath(img_dir)}")

    image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
    if not image_files:
        print(f"\n[LỖI] Không tìm thấy ảnh nào trong thư mục: {img_dir}")
        print("Vui lòng chạy 'python3 calibration/capture_chessboards.py' để chụp ảnh bàn cờ trước!")
        return

    print(f"- Tìm thấy tổng cộng {len(image_files)} ảnh. Đang trích xuất góc bàn cờ...")

    # Chuẩn bị tọa độ điểm 3D trong không gian thực của bàn cờ
    # Hỗ trợ cả 2 hướng xoay ngang và xoay dọc
    objp_normal = np.zeros((cb_size[0] * cb_size[1], 3), np.float32)
    objp_normal[:, :2] = np.mgrid[0:cb_size[0], 0:cb_size[1]].T.reshape(-1, 2) * square_size

    objp_rev = np.zeros((cb_size[0] * cb_size[1], 3), np.float32)
    objp_rev[:, :2] = np.mgrid[0:cb_size[1], 0:cb_size[0]].T.reshape(-1, 2) * square_size

    objpoints = []  # Điểm 3D trong không gian thực
    imgpoints = []  # Điểm 2D tương ứng trên mặt phẳng ảnh pixel

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    valid_images = 0
    img_shape = None

    for idx, fname in enumerate(image_files, start=1):
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        # Tìm góc bàn cờ (thử cả 2 chiều nếu không đối xứng)
        ret, corners = cv2.findChessboardCorners(gray, cb_size, None)
        active_objp = objp_normal

        if not ret and cb_size[0] != cb_size[1]:
            rev_size = (cb_size[1], cb_size[0])
            ret, corners = cv2.findChessboardCorners(gray, rev_size, None)
            active_objp = objp_rev

        if ret:
            valid_images += 1
            # Tính kích thước trung bình của ô vuông trong ảnh này để chọn cửa sổ sub-pixel an toàn
            c_reshaped = corners.reshape(-1, 2)
            dists = np.linalg.norm(c_reshaped[1:] - c_reshaped[:-1], axis=1)
            mean_sq = float(np.median(dists)) if len(dists) > 0 else 20.0

            # Cửa sổ sub-pixel an toàn (tối đa 1/5 cạnh ô vuông, tránh lấn sang ô bên cạnh gây sai tiêu cự)
            w_sub = max(2, min(4, int(mean_sq / 5.0)))
            corners_sub = cv2.cornerSubPix(gray, corners, (w_sub, w_sub), (-1, -1), criteria)
            objpoints.append(active_objp)
            imgpoints.append(corners_sub)
            print(f"  [{idx}/{len(image_files)}] OK (ô ~ {mean_sq:.1f}px, win={w_sub}x{w_sub}): {os.path.basename(fname)}")
        else:
            print(f"  [{idx}/{len(image_files)}] BỎ QUA (không phát hiện đủ {cb_size[0]}x{cb_size[1]} góc): {os.path.basename(fname)}")

    if valid_images < 5:
        print(f"\n[LỖI] Chỉ có {valid_images} ảnh hợp lệ. Cần ít nhất 10-15 ảnh rõ nét để hiệu chuẩn chính xác!")
        return

    print(f"\nĐang tính toán ma trận camera từ {valid_images} ảnh...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    # Tính sai số reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    print("\n" + "="*50)
    print(" KẾT QUẢ HIỆU CHUẨN THÀNH CÔNG!")
    print("="*50)
    print(f"- Kích thước ảnh: {img_shape[0]}x{img_shape[1]} (Width x Height)")
    print(f"- Tiêu cự fx: {fx:.2f} px")
    print(f"- Tiêu cự fy: {fy:.2f} px")
    print(f"- Tâm quang học cx: {cx:.2f} px")
    print(f"- Tâm quang học cy: {cy:.2f} px")
    print(f"- Hệ số méo (distCoeffs): {dist_coeffs.ravel().tolist()}")
    print(f"- Sai số trung bình (Mean Reprojection Error): {mean_error:.4f} pixels")
    if mean_error < 0.5:
        print("  -> ĐÁNH GIÁ: Rất tốt (< 0.5 px)!")
    elif mean_error < 1.0:
        print("  -> ĐÁNH GIÁ: Đạt tiêu chuẩn (< 1.0 px).")
    else:
        print("  -> ĐÁNH GIÁ: Sai số hơi cao (> 1.0 px). Nên chụp lại ảnh bàn cờ phẳng và đủ sáng hơn.")

    # Lưu vào file camera_calibration.yaml
    output_data = {
        "image_width": int(img_shape[0]),
        "image_height": int(img_shape[1]),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.tolist(),
        "reprojection_error": float(mean_error),
        "is_calibrated": True
    }

    os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, default_flow_style=None)

    print(f"\nĐã lưu thông số vào: {os.path.abspath(output_yaml)}")

if __name__ == "__main__":
    main()
