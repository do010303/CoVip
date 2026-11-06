#!/usr/bin/env python3
"""
Detector 4 vật gắp (YOLOv8n-seg, train bởi train_objects.py) chạy bằng onnxruntime THUẦN —
không cần ultralytics/torch, nên chạy được trên Pi 4 (và laptop).

Ra cho mỗi vật: tên, độ tin, mặt nạ, tâm ảnh, và khoảng cách ƯỚC LƯỢNG THÔ từ kích thước đã biết
(chưa dùng IMU/trọng lực — xem live_objects.py). Mỗi vật chỉ có MỘT cái: giữ phát hiện tốt nhất mỗi lớp.

Trên Pi (trong ~/aeroscript):
    # 1) đo tốc độ, không cần camera
    python3 scripts/detect_objects.py --bench --model weights/objects4_v1_640.onnx
    python3 scripts/detect_objects.py --bench --model weights/objects4_v1_320.onnx
    # 2) chạy với camera + ROS, xem ảnh trên trình duyệt:
    python3 scripts/detect_objects.py --model weights/objects4_v1_320.onnx
    #    http://192.168.50.1:8080/stream?topic=/detector/image&type=ros_compressed
    # không ROS, chỉ in ra terminal:
    python3 scripts/detect_objects.py --no-ros

Trên laptop có màn hình:  python3 scripts/detect_objects.py --no-ros --show --device <chỉ số C930e>

Nối tay + RL (INTEGRATION_RL.md, thay determine_depth/main.py; tracker IK/SAC của RL giữ nguyên):
    python3 scripts/detect_objects.py --no-ros --hand-eye calib/T_cam_assarm.npz --udp --target auto \
        --web 8080 --cpu 1,2,3 --threads 3 --model weights/objects4_v1_320.onnx
  -> vị trí tâm gắp trong hệ đế assarm (m) gửi UDP 127.0.0.1:5555 (gói của determine_depth), dashboard RL :8080.
  --target auto | hop_nau | tru_xanh | hop_pin | lo_nap_do (auto: vật trong miền SAC, gần trục J1 nhất).
Cách ĐƠN GIẢN (1 trụ, không bục/tờ ArUco): thay --hand-eye bằng vị trí camera đo thước + góc chúc:
    --cam-pose X Y Z PITCH   (m, m, m, độ; tâm ống kính trong hệ đế, chúc xuống PITCH độ)
  -> trụ/lọ đứng thẳng khớp 3D theo kích thước đã biết; hộp chỉ ước lượng thô.

Topic: /detector/objects (std_msgs/String, JSON mỗi khung), /detector/image/compressed (JPEG, chỉ khi có
người xem). Log CSV: logs/detect_<giờ>.csv. Kiểm logic giải mã: --self-test (cần ultralytics, chỉ laptop).
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
NAMES = ["hop_nau", "tru_xanh", "hop_pin", "lo_nap_do"]
# Kích thước mm, khớp config/test_objects.yaml (hop_nau = hop_giay). cylinder: [D, h]; box: cạnh tăng dần
SIZES = {"hop_nau": ("box", [60, 62, 100]), "tru_xanh": ("cylinder", [50, 100]),
         "hop_pin": ("box", [50, 61, 160]), "lo_nap_do": ("cylinder", [41, 81])}
COLORS = {"hop_nau": (40, 90, 160), "tru_xanh": (255, 120, 0), "hop_pin": (200, 200, 200), "lo_nap_do": (0, 200, 255)}


class Detector:
    def __init__(self, model, threads=4, conf=0.5, iou=0.5):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.H, self.W = self.sess.get_inputs()[0].shape[2:]
        self.conf, self.iou = conf, iou
        self.t_infer = 0.0

    def letterbox(self, img):
        h, w = img.shape[:2]
        r = min(self.H / h, self.W / w)
        nw, nh = round(w * r), round(h * r)
        top, left = (self.H - nh) // 2, (self.W - nw) // 2
        out = np.full((self.H, self.W, 3), 114, np.uint8)
        out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return out, r, top, left, nh, nw

    def __call__(self, img):
        """-> list dict(name, conf, box [x1 y1 x2 y2], mask bool HxW, poly Nx2) theo toạ độ ảnh gốc."""
        h, w = img.shape[:2]
        lb, r, top, left, nh, nw = self.letterbox(img)
        blob = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None], np.float32) / 255.0
        t0 = time.time()
        out0, proto = self.sess.run(None, {self.inp: blob})
        self.t_infer = time.time() - t0
        p = out0[0].T  # (N, 4 + nc + 32)
        nc = len(NAMES)
        scores = p[:, 4:4 + nc]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(p)), cls]
        keep = conf >= self.conf
        if not keep.any():
            return []
        p, cls, conf = p[keep], cls[keep], conf[keep]
        xywh = p[:, :4].copy()
        xywh[:, :2] -= xywh[:, 2:] / 2
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), self.conf, self.iou)
        idx = np.array(idx).reshape(-1)
        # mỗi vật chỉ có một cái: giữ cái tốt nhất mỗi lớp
        best = {}
        for i in idx:
            c = int(cls[i])
            if c not in best or conf[i] > conf[best[c]]:
                best[c] = i
        _, mh, mw = proto[0].shape
        P = proto[0].reshape(proto.shape[1], -1)
        dets = []
        for c, i in sorted(best.items()):
            x1, y1, bw, bh = xywh[i]
            m = 1.0 / (1.0 + np.exp(-(p[i, 4 + nc:] @ P)))
            m = cv2.resize(m.reshape(mh, mw).astype(np.float32), (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            crop = np.zeros_like(m)
            xa, ya = int(max(0, x1)), int(max(0, y1))
            xb, yb = int(min(self.W, x1 + bw + 1)), int(min(self.H, y1 + bh + 1))
            crop[ya:yb, xa:xb] = m[ya:yb, xa:xb]
            crop = cv2.resize(crop[top:top + nh, left:left + nw], (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
            cs, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            poly = max(cs, key=cv2.contourArea).reshape(-1, 2)
            box = [(x1 - left) / r, (y1 - top) / r, (x1 + bw - left) / r, (y1 + bh - top) / r]
            dets.append(dict(name=NAMES[c], conf=float(conf[i]), box=box, mask=crop, poly=poly))
        return dets


def load_K(path, w, rotate=0):
    """K, dist cho ảnh rộng w. Nhận .npz của CoVip (K, dist, image_size) hoặc .yaml của determine_depth
    (camera_matrix, distortion_coefficients, image_width). rotate=180: ảnh xoay 180° -> tâm ảnh lật."""
    if str(path).endswith((".yaml", ".yml")):
        import yaml
        y = yaml.safe_load(open(path, encoding="utf-8"))
        K = np.asarray(y["camera_matrix"], float).reshape(3, 3)
        dist = np.asarray(y["distortion_coefficients"], float).ravel()
        w0, h0 = float(y["image_width"]), float(y["image_height"])
    else:
        d = np.load(path)
        K, dist = d["K"].astype(float).copy(), d["dist"]
        w0, h0 = (float(v) for v in d["image_size"])
    K[:2] *= w / w0
    if rotate == 180:  # xoay ảnh 180°: u' = W-1-u, v' = H-1-v; méo xuyên tâm giữ nguyên, méo tiếp tuyến đổi dấu
        h = h0 * w / w0
        K[0, 2], K[1, 2] = w - 1 - K[0, 2], h - 1 - K[1, 2]
        dist = np.asarray(dist, float).ravel().copy()
        if len(dist) >= 4:
            dist[2:4] *= -1
    return K, dist


def estimate(det, K, dist):
    """Khoảng cách thô: trụ đứng thẳng -> cạnh ngắn hình bao ≈ D; hộp -> khoảng [cạnh nhỏ, cạnh giữa].
    -> dict(x, y, z, z_min, z_max) mm hệ camera (X phải, Y xuống, Z trước) hoặc None."""
    shape, dims = SIZES[det["name"]]
    if len(det["poly"]) < 5:
        return None
    (_, _), (rw, rh), _ = cv2.minAreaRect(np.float32(det["poly"]))
    short = min(rw, rh)
    if short < 4:
        return None
    fx = K[0, 0]
    z0, z1 = (fx * dims[0] / short,) * 2 if shape == "cylinder" else (fx * dims[0] / short, fx * dims[1] / short)
    ys, xs = np.nonzero(det["mask"])
    if len(xs) == 0:
        return None
    c = np.array([[[xs.mean(), ys.mean()]]], np.float32)
    xn, yn = cv2.undistortPoints(c, K, dist)[0, 0]
    z = 0.5 * (z0 + z1)
    return dict(x=float(xn * z), y=float(yn * z), z=float(z), z_min=float(z0), z_max=float(z1))


def _text(view, s, org, scale, col):
    """Chữ to, viền đen cho dễ đọc trên mọi nền."""
    x = int(np.clip(org[0], 2, view.shape[1] - 10))
    y = int(np.clip(org[1], 18, view.shape[0] - 4))
    cv2.putText(view, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(view, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)


def draw(img, dets, ms_infer, fps):
    view = img.copy()
    over = img.copy()
    texts = []  # vẽ chữ SAU khi trộn mặt nạ để chữ không bị mờ
    for d in dets:
        col = COLORS[d["name"]]
        over[d["mask"]] = col
        cv2.polylines(view, [np.int32(d["poly"])], True, col, 2)
        x0, y0 = int(d["box"][0]), int(d["box"][1])
        texts.append((f"{d['name']} {d['conf']:.2f}", (x0, y0 - 30), 0.7, col))
        e = d.get("est")
        if e:
            zt = (f"Z {e['z'] / 10:.0f} cm" if e["z_max"] - e["z_min"] < 5
                  else f"Z {e['z_min'] / 10:.0f}-{e['z_max'] / 10:.0f} cm")
            texts.append((f"{zt}  X{e['x'] / 10:+.0f} Y{e['y'] / 10:+.0f}", (x0, y0 - 8), 0.65, col))
        b = d.get("base")
        if b is not None:
            tag = "TARGET " if d.get("is_target") else ""
            dom = "" if d.get("in_domain") else " (ngoai mien SAC)"
            texts.append((f"{tag}de: {b[0] * 100:+.1f} {b[1] * 100:.1f} {b[2] * 100:.1f} cm{dom}",
                          (x0, int(d["box"][3]) + 24), 0.6, (0, 0, 255) if d.get("is_target") else col))
        if d.get("is_target"):
            cv2.polylines(view, [np.int32(d["poly"])], True, (0, 0, 255), 3)
    view = cv2.addWeighted(over, 0.3, view, 0.7, 0)
    for t in texts:
        _text(view, *t)
    _text(view, f"{fps:.1f} fps | infer {ms_infer:.0f} ms", (8, view.shape[0] - 12), 0.8, (0, 255, 0))
    return view


INTEGRATED_CAM = ("User Facing", "Integrated", "HD Webcam", "IR Camera")


def find_device(device):
    """'auto' -> camera USB RỜI đầu tiên (bỏ webcam tích hợp của laptop); chuỗi khác số/đường dẫn ->
    tìm theo tên (vd 'C930e', 'USB CAMERA'). Trả số/đường dẫn cho cv2.VideoCapture."""
    if str(device).isdigit() or str(device).startswith("/dev/"):
        return int(device) if str(device).isdigit() else device
    cands = []
    for d in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:])):
        try:
            name, idx = (d / "name").read_text().strip(), (d / "index").read_text().strip()
        except OSError:
            continue
        if idx != "0":
            continue
        if device == "auto" and "usb" not in str((d / "device").resolve()).lower():
            continue  # Pi 5: bỏ nút nội bộ pispbe / rpi-hevc-dec, chỉ nhận camera cắm USB
        if device == "auto" and not any(k in name for k in INTEGRATED_CAM):
            cands.append((d.name, name))
        elif device != "auto" and device.lower() in name.lower():
            cands.append((d.name, name))
    if not cands:
        raise SystemExit(f"Không thấy camera '{device}'. Xem: v4l2-ctl --list-devices; hoặc --device /dev/videoN")
    print(f"Camera: /dev/{cands[0][0]} ({cands[0][1]})")
    return f"/dev/{cands[0][0]}"


class Camera(threading.Thread):
    """Luôn giữ khung mới nhất (như perception_node.Camera) + khoá lấy nét."""

    def __init__(self, device, width, height, focus=0, exposure=None, rotate=0):
        super().__init__(daemon=True)
        self.rotate = rotate
        idx = find_device(device)
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if focus is not None and focus >= 0:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_FOCUS, focus)
        if exposure is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được camera {device}")
        self._lock = threading.Lock()
        self._frame, self._id, self._stop = None, 0, False
        self.t_frame = 0.0  # lúc nhận khung mới nhất (time.time())

    def run(self):
        while not self._stop:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                if self.rotate == 180:
                    f = cv2.rotate(f, cv2.ROTATE_180)
                self._frame, self._id, self.t_frame = f, self._id + 1, time.time()

    def latest(self):
        with self._lock:
            return self._frame, self._id

    def stop(self):
        self._stop = True
        self.cap.release()


class VideoSource(threading.Thread):
    """Như Camera nhưng đọc file video đúng nhịp thật (thử cả luồng trên laptop không cần camera)."""

    def __init__(self, path, width, height):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được {path}")
        self.size, self.dt = (width, height), 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self._lock = threading.Lock()
        self._frame, self._id, self._stop, self.t_frame = None, 0, False, 0.0

    def run(self):
        while not self._stop:
            ok, f = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            with self._lock:
                self._frame, self._id, self.t_frame = cv2.resize(f, self.size), self._id + 1, time.time()
            time.sleep(self.dt)

    def latest(self):
        with self._lock:
            return self._frame, self._id

    def stop(self):
        self._stop = True
        self.cap.release()


def bench(det, image, n):
    img = cv2.imread(str(image)) if image else None
    if img is None:
        img = np.random.default_rng(0).integers(0, 255, (360, 640, 3), np.uint8)
    img = cv2.resize(img, (640, 360))
    for _ in range(3):
        det(img)
    ti, tt = [], []
    for _ in range(n):
        t0 = time.time()
        dets = det(img)
        tt.append(time.time() - t0)
        ti.append(det.t_infer)
    print(f"Model {det.H}x{det.W}: suy luận trung vị {1000 * np.median(ti):.0f} ms "
          f"(p90 {1000 * np.percentile(ti, 90):.0f}), cả giải mã mặt nạ {1000 * np.median(tt):.0f} ms "
          f"-> ~{1 / np.median(tt):.1f} fps; thấy {[(d['name'], round(d['conf'], 2)) for d in dets]}")


def self_test():
    """So kết quả giải mã onnxruntime với ultralytics trên ảnh val (laptop)."""
    from ultralytics import YOLO
    ok_all = True
    imgs = sorted((ROOT / "datasets" / "objects4" / "images" / "val").glob("*.jpg"))[::60][:8]
    for tag in ("640", "320"):
        onnx = ROOT / "weights" / f"objects4_v1_{tag}.onnx"
        det = Detector(onnx, conf=0.5)
        ref = YOLO(str(onnx), task="segment")
        worst = 0.0
        for f in imgs:
            img = cv2.resize(cv2.imread(str(f)), (640, 360))
            mine = {d["name"]: d for d in det(img)}
            r = ref.predict(img, imgsz=(det.H, det.W), conf=0.5, device="cpu", verbose=False)[0]
            for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                n = NAMES[int(c)]
                if n not in mine:
                    worst = max(worst, 99)
                    continue
                worst = max(worst, float(np.abs(np.array(mine[n]["box"]) - b).max()))
        ok = worst < 3.0
        ok_all &= ok
        print(f"[self-test] {tag}: lệch hộp lớn nhất so với ultralytics {worst:.1f} px -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    return ok_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "weights" / "objects4_v1_640.onnx"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--device", default="auto",
                    help="auto (camera USB rời, bỏ webcam laptop), /dev/videoN, số, hoặc một phần tên (vd C930e)")
    ap.add_argument("--video", default=None, help="file video thay camera (lặp lại), để thử trên laptop")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--focus", type=int, default=0, help="-1 = tự lấy nét")
    ap.add_argument("--rotate", type=int, default=0, choices=(0, 180), help="180: camera gắn ngược")
    ap.add_argument("--exposure", type=float, default=None)
    ap.add_argument("--calib", default=str(ROOT / "calib" / "c930e_720p.npz"))
    ap.add_argument("--no-ros", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--image-every", type=int, default=1, help="phát ảnh vẽ mỗi N khung")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--bench", action="store_true", help="đo tốc độ, không cần camera")
    ap.add_argument("--bench-image", default=None)
    ap.add_argument("--bench-n", type=int, default=30)
    ap.add_argument("--hand-eye", default=None, help="calib/T_cam_assarm.npz (calib_cam_assarm.py) -> toạ độ hệ đế assarm")
    ap.add_argument("--cam-pose", type=float, nargs="+", default=None, metavar="V",
                    help="X Y Z PITCH [YAW]: camera đo bằng thước (m) + góc chúc xuống (độ) — thay --hand-eye")
    ap.add_argument("--udp", action="store_true", help="gửi vật đích sang tracker IK/SAC qua UDP (cần --hand-eye)")
    ap.add_argument("--udp-port", type=int, default=5555)
    ap.add_argument("--target-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("DX", "DY", "DZ"),
                    help="dời đích GỬI cho tay (m, hệ đế), vd 0 0 0.10 = ngàm lơ lửng trên đỉnh trụ 10 cm để đo sai số")
    ap.add_argument("--j1-trim", type=float, default=0.0, metavar="DEG",
                    help="xoay đích GỬI quanh trục J1 (độ, + = sang phải): bù servo J1 lệch tâm, "
                         "vd tay chĩa lệch trái 3° khi lệnh thẳng -> --j1-trim 3")
    ap.add_argument("--target", default="auto", help="auto hoặc tên lớp (khi chọn 1 lớp: bỏ qua mọi lớp khác)")
    ap.add_argument("--classes", default=None, help="chỉ nhận các lớp này, vd tru_xanh,lo_nap_do (mặc định: "
                    "--target nếu là tên lớp, không thì cả 4)")
    ap.add_argument("--rl-root", default=None, help="thư mục rl_arm5 (Pi: ~/quan)")
    ap.add_argument("--web", type=int, default=0, help="cổng dashboard WebStreamer của RL (8080); 0 = tắt")
    ap.add_argument("--cpu", default=None, help="ghim tiến trình vào các core, vd 1,2,3 (core 0 cho tracker)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return 0 if self_test() else 1

    det = Detector(a.model, a.threads, a.conf)
    if a.bench:
        bench(det, a.bench_image, a.bench_n)
        return 0

    if a.cpu:
        try:
            os.sched_setaffinity(0, {int(c) for c in a.cpu.split(",")})
            print(f"Ghim CPU core {a.cpu}")
        except (AttributeError, OSError) as e:
            print(f"⚠️  Không ghim được CPU: {e}")
    K, dist = load_K(a.calib, a.width, a.rotate)
    he = sender = web = None
    if a.hand_eye or a.cam_pose or a.udp or a.web:
        from assarm_link import HandEye, TargetSender, add_rl_paths, find_rl_root, in_rl_domain, pick_target
        rl = find_rl_root(a.rl_root)
        if rl is None and (a.udp or a.web):
            raise SystemExit("Không thấy rl_arm5 (determine_depth/core/ipc_bridge.py). Dùng --rl-root ~/quan")
        if rl is not None:
            add_rl_paths(rl)
        if a.hand_eye:
            he = HandEye.load(a.hand_eye, K, dist)
            print(f"Hand-eye: camera ở {np.round(he.cam_pos * 100, 1)} cm hệ đế, chúc {he.pitch_deg:.1f}°, "
                  f"bục z = {he.plat_z:.3f} m")
        elif a.cam_pose:
            if len(a.cam_pose) not in (4, 5):
                raise SystemExit("--cam-pose cần X Y Z PITCH [YAW]")
            he = HandEye.from_ruler(*a.cam_pose[:4], K, dist, yaw_deg=a.cam_pose[4] if len(a.cam_pose) == 5 else 0.0)
            print(f"Camera (đo thước): {np.round(he.cam_pos * 100, 1)} cm hệ đế, chúc {he.pitch_deg:.1f}° "
                  f"— không bục: trụ/lọ khớp theo kích thước, hộp thô")
        elif a.udp:
            raise SystemExit("--udp cần --hand-eye calib/T_cam_assarm.npz hoặc --cam-pose X Y Z PITCH")
        if a.udp:
            sender = TargetSender(port=a.udp_port)
            print(f"Gửi vật đích ({a.target}) -> UDP 127.0.0.1:{a.udp_port}")
        if a.web:
            from core.web_streamer import WebStreamer
            web = WebStreamer(port=a.web)
            web.start()
    cam = (VideoSource(a.video, a.width, a.height) if a.video
           else Camera(a.device, a.width, a.height, a.focus, a.exposure, a.rotate))
    cam.start()

    ros = None
    if not a.no_ros:
        import rclpy
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
        rclpy.init()
        node = rclpy.create_node("detect_objects")
        pub_obj = node.create_publisher(String, "/detector/objects", 5)
        pub_img = node.create_publisher(CompressedImage, "/detector/image/compressed", 1)
        ros = (rclpy, node, pub_obj, pub_img, String, CompressedImage)

    log_w = None
    if not a.no_log:
        (ROOT / "logs").mkdir(exist_ok=True)
        log_path = ROOT / "logs" / f"detect_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        log_f = open(log_path, "w", newline="")
        log_w = csv.writer(log_f)
        log_w.writerow(["t", "frame", "infer_ms", "name", "conf", "u", "v", "x_mm", "y_mm", "z_mm", "z_min", "z_max",
                        "bx_mm", "by_mm", "bz_mm", "chk_mm", "in_domain", "target"])
        print(f"Log: {log_path}")
    print(f"Model {Path(a.model).name} ({det.H}x{det.W}), {a.threads} luồng. Ctrl+C để dừng.")

    last_id, n, t_win, n_win, fps, t_print = -1, 0, time.time(), 0, 0.0, 0.0
    prev_target = None
    offset = np.asarray(a.target_offset, float)
    if np.any(offset):
        print(f"Đích gửi cho tay = vật + ({offset[0] * 100:+.1f}, {offset[1] * 100:+.1f}, {offset[2] * 100:+.1f}) cm")
    j1_xy = np.array([0.0, -0.0108])          # trục J1 trong hệ đế RL (X phải, Y trước)
    tc, ts = np.cos(np.radians(a.j1_trim)), np.sin(np.radians(a.j1_trim))

    def send_point(p):
        q = np.asarray(p, float) + offset
        if a.j1_trim:                          # quay theo chiều kim đồng hồ nhìn từ trên (sang phải)
            dx, dy = q[:2] - j1_xy
            q[:2] = j1_xy + (tc * dx + ts * dy, -ts * dx + tc * dy)
        return q

    if a.j1_trim:
        print(f"Bù J1: xoay đích gửi {a.j1_trim:+.1f}° quanh trục J1 (+ = sang phải)")
    allowed = (set(a.classes.split(",")) if a.classes
               else ({a.target} if a.target != "auto" else None))
    if allowed:
        bad = allowed - set(NAMES)
        if bad:
            raise SystemExit(f"Lớp không có: {bad}. Có: {NAMES}")
        print(f"Chỉ nhận lớp: {sorted(allowed)}")
    try:
        while True:
            frame, fid = cam.latest()
            if frame is None or fid == last_id:
                time.sleep(0.002)
                continue
            last_id = fid
            t_frame = cam.t_frame
            dets = [d for d in det(frame) if allowed is None or d["name"] in allowed]
            for d in dets:
                d["est"] = estimate(d, K, dist)
                if he is not None:
                    if he.table is not None:
                        loc = he.locate(d["name"], d["poly"])
                    elif d["est"]:
                        from assarm_link import CYL
                        p0 = he.to_base(np.array([d["est"][k] for k in ("x", "y", "z")]) / 1000)
                        loc = he.locate_cylinder(d["name"], d["poly"], p0) if d["name"] in CYL else dict(p=p0)
                    else:
                        loc = None
                    d["base"] = None if loc is None else loc["p"]
                    if d["base"] is not None:
                        d["in_domain"], d["why"] = in_rl_domain(d["base"])
                        e = d["est"]
                        # đối chứng: điểm theo kích thước đã biết (mm, hệ camera) đổi sang hệ đế
                        d["chk_mm"] = (float(np.linalg.norm(he.to_base(np.array([e["x"], e["y"], e["z"]]) / 1000)
                                                            - d["base"]) * 1000) if e else float("nan"))
            tgt = None
            if he is not None:
                tgt = pick_target(dets, a.target, prev_target)
                if tgt is not None:
                    tgt["is_target"] = True
                prev_target = tgt["name"] if tgt is not None else None
                if sender is not None:
                    ok_send = tgt is not None and (tgt.get("in_domain") or a.target != "auto")
                    sender.send(send_point(tgt["base"]) if ok_send else None, t_frame)
            n += 1
            n_win += 1
            now = time.time()
            if now - t_win >= 2.0:
                fps, t_win, n_win = n_win / (now - t_win), now, 0
            out = []
            for d in dets:
                e = d["est"] or {}
                u, v = d["poly"].mean(0)
                out.append(dict(name=d["name"], conf=round(d["conf"], 3), u=round(float(u), 1), v=round(float(v), 1),
                                box=[round(x, 1) for x in d["box"]], **{k: round(x, 1) for k, x in e.items()}))
                b = d.get("base")
                if b is not None:
                    out[-1].update(base_m=[round(float(v_), 4) for v_ in b], in_domain=bool(d.get("in_domain")),
                                   target=bool(d.get("is_target")))
                if log_w:
                    bb = [f"{1000 * v_:.1f}" for v_ in b] if b is not None else ["nan"] * 3
                    log_w.writerow([f"{now:.3f}", n, f"{1000 * det.t_infer:.1f}", d["name"], f"{d['conf']:.3f}",
                                    f"{u:.1f}", f"{v:.1f}"] + [f"{e.get(k, float('nan')):.1f}"
                                                               for k in ("x", "y", "z", "z_min", "z_max")]
                                   + bb + [f"{d.get('chk_mm', float('nan')):.1f}", int(bool(d.get("in_domain"))),
                                           int(bool(d.get("is_target")))])
            if now - t_print >= 1.0:
                t_print = now
                s = ", ".join(f"{o['name']} {o['conf']:.2f} Z~{o.get('z', 0) / 10:.0f}cm" for o in out) or "—"
                if he is not None:
                    if tgt is not None:
                        b = tgt["base"]
                        s += (f" | ĐÍCH {tgt['name']} đế ({b[0] * 100:+.1f}, {b[1] * 100:.1f}, {b[2] * 100:.1f}) cm"
                              f", lệch đối chứng {tgt.get('chk_mm', float('nan')):.0f} mm"
                              + ("" if tgt.get("in_domain") else f" ⚠️ {tgt.get('why')}"))
                    else:
                        s += " | không có vật đích" + ("" if a.target == "auto" else f" '{a.target}'")
                print(f"[{fps:4.1f} fps, suy luận {1000 * det.t_infer:4.0f} ms] {s}", flush=True)
            web_on = web is not None and web._active_clients > 0
            need_img = a.show or web_on or (ros and ros[3].get_subscription_count() > 0 and n % a.image_every == 0)
            view = draw(frame, dets, 1000 * det.t_infer, fps) if need_img else None
            if web_on:
                e = (tgt or {}).get("est") or {}
                web.update_frame(view, metrics=dict(
                    fps=fps, coords_3d=[e.get(k, 0.0) / 1000 for k in ("x", "y", "z")],
                    coords_base=list(tgt["base"]) if tgt is not None else [0.0, 0.0, 0.0],
                    target_found=tgt is not None))
            if ros:
                rclpy_, node, pub_obj, pub_img, String, CompressedImage = ros
                pub_obj.publish(String(data=json.dumps(dict(t=now, objects=out))))
                if view is not None and pub_img.get_subscription_count() > 0:
                    okj, jpg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if okj:
                        m = CompressedImage()
                        m.header.stamp = node.get_clock().now().to_msg()
                        m.format = "jpeg"
                        m.data = jpg.tobytes()
                        pub_img.publish(m)
                rclpy_.spin_once(node, timeout_sec=0)
            if a.show:
                cv2.imshow("detect_objects", view)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        if sender is not None:
            sender.send(None, time.time())
            sender.close()
        if web is not None:
            web.stop()
        if log_w:
            log_f.close()
        if ros:
            ros[1].destroy_node()
            if ros[0].ok():
                ros[0].shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
