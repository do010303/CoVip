#!/usr/bin/env python3
"""
Hiệu chuẩn nội tại camera (Phase 2 trong PLAN.md) — dùng ảnh bàn cờ
(chessboard) chụp bằng ĐÚNG camera C920 sẽ gắn lên Pi, ở ĐÚNG độ phân giải
720p MJPEG sẽ deploy, với focus đã khoá cố định trước khi chụp.

Có thể viết/chạy được ngay trên laptop, không cần Pi — chỉ cần camera vật
lý C920 và bàn cờ in ra giấy.

Chạy:
    # 1) Chụp ảnh bàn cờ (giữ nguyên focus khoá, đưa bàn cờ qua nhiều góc/khoảng cách)
    #    Mặc định TỰ TÌM camera rời theo tên thiết bị (giống record_dataset.py),
    #    không dùng --device thì script tự chọn đúng camera USB rời, tránh bị
    #    nhầm sang webcam tích hợp của laptop (thường là /dev/video0).
    python3 scripts/calibrate_camera.py capture --camera-name C930e \\
        --focus 20 --out calib/chessboard_raw

    # Biết chắc device node rồi thì chỉ định thẳng, bỏ qua tự tìm:
    python3 scripts/calibrate_camera.py capture --device /dev/video2 --focus 20

    # 2) Tính K/dist từ các ảnh đã chụp
    python3 scripts/calibrate_camera.py compute --images calib/chessboard_raw \\
        --cols 9 --rows 6 --square-mm 25 --out calib/c930e_720p.npz

Bàn cờ chuẩn OpenCV: --cols/--rows là SỐ GÓC TRONG (inner corners), không
phải số ô — bàn cờ 10x7 ô vuông thì cols=9, rows=6.

Không chắc camera nào là camera rời? Liệt kê tên tất cả thiết bị video:
    for d in /sys/class/video4linux/video*; do echo "$d: $(cat $d/name)"; done
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def find_camera(name):
    """Tìm /dev/videoN theo tên thiết bị (giống record_dataset.py) — tránh
    nhầm sang webcam tích hợp của laptop khi có nhiều camera cắm cùng lúc."""
    for d in sorted(Path("/sys/class/video4linux").glob("video*")):
        name_file, index_file = d / "name", d / "index"
        if not name_file.exists() or not index_file.exists():
            continue
        if name in name_file.read_text() and index_file.read_text().strip() == "0":
            return int(d.name.replace("video", ""))
    return None


def list_cameras():
    found = []
    for d in sorted(Path("/sys/class/video4linux").glob("video*")):
        name_file = d / "name"
        if name_file.exists():
            found.append(f"  {d.name}: {name_file.read_text().strip()}")
    return "\n".join(found) if found else "  (không thấy thiết bị video nào)"


def resolve_device(args):
    if args.device is not None:
        return int(args.device) if str(args.device).isdigit() else args.device
    idx = find_camera(args.camera_name)
    if idx is None:
        print(f"Không tự tìm được camera tên chứa '{args.camera_name}'. "
              f"Danh sách thiết bị video hiện có:\n{list_cameras()}\n"
              f"Dùng --device /dev/videoN để chỉ định thẳng, hoặc --camera-name đúng tên.",
              file=sys.stderr)
        sys.exit(1)
    print(f"Tự tìm thấy camera '{args.camera_name}' -> /dev/video{idx}")
    return idx


def cmd_capture(args):
    idx = resolve_device(args)
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.focus is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, args.focus)
    if not cap.isOpened():
        print(f"Không mở được camera /dev/video{idx}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Nhấn SPACE để lưu ảnh khi thấy bàn cờ hiện rõ, 'q' để thoát.")
    print(f"KHÔNG đổi focus/zoom giữa các lần chụp — focus đang khoá ở {args.focus}.")
    n = 0
    pattern = (args.cols, args.rows)
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        disp = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            cv2.drawChessboardCorners(disp, pattern, corners, found)
        cv2.putText(disp, f"da luu: {n}  [{'THAY BAN CO' if found else 'khong thay ban co'}]",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("calibrate_camera - capture", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and found:
            path = out_dir / f"chess_{n:03d}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"  luu {path}")
            n += 1
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Xong: {n} ảnh -> {out_dir}. Khuyến nghị >= 15-20 ảnh, nhiều góc/khoảng cách khác nhau.")
    return 0


def find_corners(gray, pattern):
    """Dò góc bàn cờ + tinh chỉnh dưới pixel. Trả (corners, lệch so với mặt
    phẳng lý tưởng [px]) hoặc None.

    Cửa sổ tinh chỉnh phải NHỎ HƠN nửa khoảng cách giữa 2 góc kề nhau. Bản cũ
    dùng cố định 11 px (ô quét 23x23): khi bàn cờ ở xa, các góc chỉ cách nhau
    ~18 px nên ô quét trùm sang góc bên cạnh và kéo lệch vị trí góc — calib ra
    tiêu cự sai ~18% (911 thay vì ~770 ở 720p) mà không báo lỗi gì."""
    found, corners = cv2.findChessboardCorners(
        gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    grid = corners.reshape(pattern[1], pattern[0], 2)
    step = min(np.linalg.norm(np.diff(grid, axis=1), axis=2).min(),
               np.linalg.norm(np.diff(grid, axis=0), axis=2).min())
    w = int(np.clip(step * 0.35, 2, 11))
    corners = cv2.cornerSubPix(
        gray, corners, (w, w), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
    flat = (np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)).astype(np.float32)
    H, _ = cv2.findHomography(flat, corners.reshape(-1, 2))
    proj = cv2.perspectiveTransform(flat.reshape(-1, 1, 2), H).reshape(-1, 2)
    plane_err = float(np.sqrt(np.mean(np.sum((proj - corners.reshape(-1, 2)) ** 2, axis=1))))
    return corners, plane_err


def cmd_compute(args):
    pattern = (args.cols, args.rows)
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= args.square_mm / 1000.0  # ra đơn vị mét

    files = []
    for d in args.images:
        files += sorted(Path(d).glob("*.jpg")) + sorted(Path(d).glob("*.png"))
    if not files:
        print(f"Không thấy ảnh nào trong {args.images}", file=sys.stderr)
        return 1

    img_points, names, img_shape = [], [], None
    for f in files:
        gray = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY)
        if img_shape is not None and gray.shape[::-1] != img_shape:
            print(f"  bỏ qua {f}: khác độ phân giải {img_shape}")
            continue
        img_shape = gray.shape[::-1]
        res = find_corners(gray, pattern)
        if res is None:
            print(f"  bỏ qua {f.name}: không thấy bàn cờ")
            continue
        if res[1] > 1.0:
            print(f"  bỏ qua {f.name}: góc dò được không nằm trên một mặt phẳng (lệch {res[1]:.1f}px — nhoè/dò nhầm)")
            continue
        img_points.append(res[0])
        names.append(f.name)

    if len(img_points) < 10:
        print(f"Chỉ dùng được {len(img_points)} ảnh — cần ít nhất 10, khuyến nghị 20+.", file=sys.stderr)
        return 1

    # Mặc định chỉ dùng 2 hệ số méo xuyên tâm: ống kính webcam méo ít, thêm hệ
    # số (k3, tiếp tuyến) chỉ làm kết quả dao động theo bộ ảnh.
    flags = 0 if args.full_dist else (cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
    while True:
        rms, K, dist, rvecs, tvecs, _, _, per_view = cv2.calibrateCameraExtended(
            [objp] * len(img_points), img_points, img_shape, None, None, flags=flags)
        per_view = per_view.ravel()
        worst = int(np.argmax(per_view))
        if per_view[worst] < max(1.0, 3.0 * np.median(per_view)) or len(img_points) <= 10:
            break
        print(f"  bỏ {names[worst]}: lệch {per_view[worst]:.2f}px, gấp nhiều lần các ảnh khác")
        img_points.pop(worst); names.pop(worst)

    # Độ tin cậy của tiêu cự: calib lại 30 lần trên bộ ảnh lấy mẫu lại
    rng = np.random.default_rng(0)
    fs = []
    for _ in range(30):
        ii = rng.integers(0, len(img_points), len(img_points))
        _, Kb, _, _, _ = cv2.calibrateCamera([objp] * len(ii), [img_points[i] for i in ii],
                                             img_shape, None, None, flags=flags)
        fs.append(Kb[0, 0])
    f_sd = float(np.std(fs))

    tilts = [np.degrees(np.arccos(abs(cv2.Rodrigues(r)[0][2, 2]))) for r in rvecs]
    pts = np.vstack([c.reshape(-1, 2) for c in img_points])
    cover = (pts.min(0) / img_shape, pts.max(0) / img_shape)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, K=K, dist=dist, image_size=np.array(img_shape),
             reprojection_error=rms)

    hfov = 2 * np.degrees(np.arctan(img_shape[0] / 2 / K[0, 0]))
    print(f"Dùng {len(img_points)}/{len(files)} ảnh, {img_shape[0]}x{img_shape[1]}.")
    print(f"K =\n{K}")
    print(f"dist = {dist.ravel()}")
    print(f"Tiêu cự fx = {K[0, 0]:.1f} ± {f_sd:.1f} px (góc nhìn ngang {hfov:.1f}°)")
    print(f"Sai số chiếu lại (RMS): {rms:.3f}px "
          f"({'ĐẠT' if rms < 0.5 else 'CHƯA ĐẠT'} mục tiêu <0.5px)")
    print(f"Bàn cờ nghiêng {min(tilts):.0f}–{max(tilts):.0f}°, phủ {cover[0][0]:.2f}–{cover[1][0]:.2f} bề ngang, "
          f"{cover[0][1]:.2f}–{cover[1][1]:.2f} bề dọc ảnh.")
    if f_sd > 0.03 * K[0, 0]:
        print("⚠️  Tiêu cự chưa ổn định (dao động trên 3%). Chụp thêm ảnh bàn cờ GẦN hơn (chiếm 1/3–1/2 "
              "khung) và NGHIÊNG 30–45°: bàn cờ nhỏ, chính diện gần như không cho biết tiêu cự.")
    if max(tilts) < 25:
        print("⚠️  Không ảnh nào nghiêng quá 25° — tiêu cự xác định kém. Chụp thêm ảnh nghiêng 30–45°.")
    print(f"Đã lưu -> {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="Chụp ảnh bàn cờ bằng camera")
    c.add_argument("--device", default=None,
                   help="Chỉ định thẳng /dev/videoN hoặc số index. Bỏ trống = tự tìm theo --camera-name.")
    c.add_argument("--camera-name", default="C930e",
                   help="Tự tìm camera rời theo tên thiết bị khớp chuỗi này (khi không dùng --device). "
                        "Đổi thành 'C920' khi chuyển sang camera deploy thật.")
    c.add_argument("--width", type=int, default=1280)
    c.add_argument("--height", type=int, default=720)
    c.add_argument("--focus", type=int, default=None, help="Giá trị focus cố định (khoá autofocus)")
    c.add_argument("--cols", type=int, default=9, help="Số góc trong theo chiều ngang")
    c.add_argument("--rows", type=int, default=6, help="Số góc trong theo chiều dọc")
    c.add_argument("--out", default=str(ROOT / "calib" / "chessboard_raw"))

    p = sub.add_parser("compute", help="Tính K/dist từ ảnh bàn cờ đã chụp")
    p.add_argument("--images", required=True, nargs="+",
                   help="Một hoặc nhiều thư mục ảnh bàn cờ (cùng độ phân giải)")
    p.add_argument("--full-dist", action="store_true",
                   help="Dùng đủ 5 hệ số méo (mặc định chỉ k1, k2)")
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--square-mm", type=float, default=25.0, help="Kích thước 1 ô vuông bàn cờ (mm)")
    p.add_argument("--out", default=str(ROOT / "calib" / "c930e_720p.npz"))

    args = ap.parse_args()
    if args.cmd == "capture":
        return cmd_capture(args)
    return cmd_compute(args)


if __name__ == "__main__":
    sys.exit(main())
