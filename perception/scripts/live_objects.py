#!/usr/bin/env python3
"""
Chạy detector 4 vật (runs/objects4/<run>/weights/best.pt|onnx) trực tiếp với C930e trên LAPTOP,
hiện mặt nạ, tên, độ tin, fps và khoảng cách ƯỚC LƯỢNG THÔ từ kích thước đã biết.

    ~/venvs/new_rl_ros2/bin/python scripts/live_objects.py                  # .pt trên GPU, 640x360
    ~/venvs/new_rl_ros2/bin/python scripts/live_objects.py --onnx --cpu     # giống Pi hơn (đo tốc độ CPU)
    ~/venvs/new_rl_ros2/bin/python scripts/live_objects.py --video datasets/raw_videos/xa_*.avi

Phím: q/ESC thoát, s lưu ảnh, space dừng hình.

Khoảng cách (chỉ để xem, CHƯA phải bước 2 của lộ trình — chưa dùng trọng lực/IMU):
  - trụ, lọ (đứng thẳng): cạnh ngắn của hình chữ nhật bao nhỏ nhất quanh mặt nạ ≈ đường kính
    -> Z = fx·D / cạnh_ngắn. Đặt nằm hoặc nhìn từ trên xuống thì sai.
  - hộp: cạnh ngắn nằm giữa cạnh nhỏ nhất và cạnh giữa của hộp -> in khoảng Z [min, max].
X, Y theo hệ camera (X phải, Y xuống, Z ra trước), mm.
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from record_dataset import find_camera, open_camera  # noqa: E402

# tên lớp model -> tên trong config/test_objects.yaml
CFG_NAME = {"hop_nau": "hop_giay", "tru_xanh": "tru_xanh", "hop_pin": "hop_pin", "lo_nap_do": "lo_nap_do"}
COLORS = {"hop_nau": (40, 90, 160), "tru_xanh": (255, 120, 0), "hop_pin": (200, 200, 200), "lo_nap_do": (0, 200, 255)}


def load_sizes():
    objs = yaml.safe_load((ROOT / "config" / "test_objects.yaml").read_text())["objects"]
    return {k: (objs[v]["shape"], sorted(objs[v]["size_mm"])) for k, v in CFG_NAME.items()}


def load_K(w, h):
    d = np.load(ROOT / "calib" / "c930e_720p.npz")
    s = w / float(d["image_size"][0])
    K = d["K"].copy()
    K[:2] *= s
    return K, d["dist"]


def estimate(poly, shape, dims, K, dist):
    """-> (X, Y, Z_min, Z_max) mm hoặc None."""
    if len(poly) < 5:
        return None
    (_, _), (rw, rh), _ = cv2.minAreaRect(np.float32(poly))
    short = min(rw, rh)
    if short < 4:
        return None
    fx = K[0, 0]
    if shape == "cylinder":
        z0 = z1 = fx * dims[0] / short  # dims sorted: [D, h] với D < h
    else:
        z0, z1 = fx * dims[0] / short, fx * dims[1] / short
    m = cv2.moments(np.float32(poly))
    if m["m00"] == 0:
        return None
    c = np.array([[[m["m10"] / m["m00"], m["m01"] / m["m00"]]]], np.float32)
    xn, yn = cv2.undistortPoints(c, K, dist)[0, 0]
    z = 0.5 * (z0 + z1)
    return xn * z, yn * z, z0, z1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="v1")
    ap.add_argument("--onnx", action="store_true", help="dùng best.onnx (384x640) thay best.pt")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--focus", type=int, default=0)
    ap.add_argument("--exposure", type=int, default=None)
    ap.add_argument("--cam", type=int, default=None)
    ap.add_argument("--video", default=None, help="chạy trên file video thay vì camera")
    a = ap.parse_args()

    from ultralytics import YOLO
    wdir = ROOT / "runs" / "objects4" / a.run / "weights"
    model = YOLO(str(wdir / ("best.onnx" if a.onnx else "best.pt")), task="segment")
    device = "cpu" if a.cpu else 0
    imgsz = (384, 640) if a.onnx else 640
    sizes = load_sizes()

    if a.video:
        cap = cv2.VideoCapture(sorted(glob.glob(a.video))[0])
    else:
        idx = a.cam if a.cam is not None else find_camera()
        if idx is None:
            raise SystemExit("Không tìm thấy C930e. Dùng --cam <index>.")
        cap = open_camera(idx, a.width, a.height, 30, a.focus, a.exposure)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Không đọc được khung đầu.")
    if a.video:
        frame = cv2.resize(frame, (a.width, a.height))
    h, w = frame.shape[:2]
    K, dist = load_K(w, h)
    print(f"Ảnh {w}x{h}, fx={K[0, 0]:.1f}, model {'ONNX' if a.onnx else 'PT'} trên {'CPU' if a.cpu else 'GPU'}")

    t_hist, paused, n = [], False, 0
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            if a.video:
                frame = cv2.resize(frame, (a.width, a.height))
            t0 = time.time()
            r = model.predict(frame, imgsz=imgsz, conf=a.conf, device=device, verbose=False)[0]
            t_hist = (t_hist + [time.time() - t0])[-30:]
            view = frame.copy()
            if r.masks is not None:
                overlay = view.copy()
                for c, s, poly in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.masks.xy):
                    name = r.names[int(c)]
                    col = COLORS.get(name, (0, 255, 0))
                    pts = np.int32(poly)
                    if len(pts) < 3:
                        continue
                    cv2.fillPoly(overlay, [pts], col)
                    cv2.polylines(view, [pts], True, col, 2)
                    shape, dims = sizes[name]
                    e = estimate(poly, shape, dims, K, dist)
                    x0, y0 = pts.min(0)
                    txt = f"{name} {s:.2f}"
                    if e:
                        X, Y, z0, z1 = e
                        zt = f"Z{z0 / 10:.0f}cm" if abs(z1 - z0) < 5 else f"Z{z0 / 10:.0f}-{z1 / 10:.0f}cm"
                        txt2 = f"{zt} X{X / 10:+.0f} Y{Y / 10:+.0f}"
                    else:
                        txt2 = ""
                    cv2.putText(view, txt, (int(x0), max(14, int(y0) - 18)), 0, 0.5, col, 2)
                    cv2.putText(view, txt2, (int(x0), max(28, int(y0) - 4)), 0, 0.45, col, 1)
                view = cv2.addWeighted(overlay, 0.3, view, 0.7, 0)
            ms = 1000 * np.mean(t_hist)
            cv2.putText(view, f"{ms:.0f} ms/khung ({1000 / ms:.0f} fps) | q thoat, s luu, space dung",
                        (8, h - 10), 0, 0.5, (0, 255, 0), 1)
        cv2.imshow("live_objects", view)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord(" "):
            paused = not paused
        if k == ord("s"):
            p = ROOT / "runs" / "objects4" / f"live_{int(time.time())}_{n}.jpg"
            cv2.imwrite(str(p), view)
            print(f"Lưu {p}")
            n += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
