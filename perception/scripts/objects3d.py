#!/usr/bin/env python3
"""
Đo vật trên mặt bàn bằng 1 camera cố định (PERCEPTION.md, V1–V2): ảnh → mặt
bàn → vị trí, kích thước, hình dạng (hộp / trụ). Chỉ cv2 + numpy, không ROS.

Hệ BÀN (table, mét): gốc = tâm tờ calib mặt bàn lúc calib, x sang phải, y lên
(nhìn tờ giấy xuôi chữ), z vuông góc mặt bàn hướng LÊN (về phía camera) —
giống quy ước bảng trong board_detect.py. T_table2cam: p_cam = R·p_table + t.

Mặt bàn KHÔNG lấy từ bảng vẽ (bảng đứng thẳng, Bước 4 PLAN.md) mà calib 1
lần bằng tờ marker nằm phẳng trên bàn (`make_table_sheet.py`, ID 40–48).

Đo vật không cần biết trước kích thước: coi vật là lăng trụ đứng (hộp: đáy chữ
nhật w×d xoay yaw; trụ: đáy tròn đường kính D), cao h. Tìm tham số sao cho bóng
chiếu của lăng trụ lên ảnh trùng mask của vật nhất (IoU lớn nhất). Ước lượng
ban đầu tính thẳng từ hình học: cạnh đứng của vật chiếu xuống mặt bàn thành các
tia đi qua chân camera, nên góc mở của mask nhìn từ chân camera cho bề ngang,
điểm gần nhất cho mép đáy, điểm xa nhất cho đỉnh.

    python3 scripts/objects3d.py --self-test
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from board_detect import _square, _to_T, make_detect_fn  # noqa: E402

# ── Tờ calib mặt bàn: lưới 3×3 marker DICT_4X4_1000, ID 40–48 ────────────────
# Không dùng 0–3 (bảng vẽ), 10 (đầu gắp), 20 (vật ở G2): các từ điển 4x4 của
# OpenCV dùng chung 50 mã đầu, nên ID trùng là nhầm thật.
SHEET_DICT = "DICT_4X4_1000"
SHEET_IDS = tuple(range(40, 49))
SHEET_MARKER_M = 0.030
SHEET_PITCH_M = 0.060


def sheet_marker_centre(mid):
    """Tâm marker `mid` trong hệ tờ giấy (m). 40 trên-trái ... 48 dưới-phải."""
    k = mid - SHEET_IDS[0]
    r, c = divmod(k, 3)
    return np.array([(c - 1) * SHEET_PITCH_M, (1 - r) * SHEET_PITCH_M, 0.0])


SHEET_OBJ = {m: _square(*sheet_marker_centre(m)[:2], SHEET_MARKER_M / 2) for m in SHEET_IDS}


def load_camera_calib(path, width, height):
    """K/dist từ file .npz của calibrate_camera.py, hoặc .yaml của
    lidar_camera_calib (K, D, image_width/height), quy đổi về độ phân giải đang
    chạy (giống load_camera_calib trong run_pi4_ros2.py — chép lại để khỏi kéo
    theo rclpy/onnxruntime)."""
    if str(path).endswith((".yaml", ".yml")):
        import yaml
        with open(path, encoding="utf-8") as f:
            y = yaml.safe_load(f)
        K = np.asarray(y["K"], np.float64).reshape(3, 3)
        dist = np.asarray(y["D"], np.float64).ravel()
        w0, h0 = int(y["image_width"]), int(y["image_height"])
    else:
        d = np.load(path)
        K = d["K"].astype(np.float64)
        dist = d["dist"].astype(np.float64).ravel()
        w0, h0 = (int(v) for v in d["image_size"])
    sx, sy = width / w0, height / h0
    if abs(sx - sy) > 0.01:
        print(f"⚠️  Calib làm ở {w0}x{h0}, đang chạy {width}x{height}: khác tỉ lệ khung, K quy đổi có thể sai.")
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K, dist


# ── Tờ calib mặt bàn ──────────────────────────────────────────────────────────
class SheetDetector:
    """Dò tờ calib mặt bàn -> T_sheet2cam. Dùng mọi góc của mọi marker thấy được
    (≥ 3 marker) nên chính xác hơn 4 marker của bảng vẽ."""

    def __init__(self, K, dist, min_markers=3):
        self.K, self.dist = np.asarray(K, np.float64), np.asarray(dist, np.float64)
        self.min_markers = min_markers
        self._detect = make_detect_fn(SHEET_DICT)

    def detect(self, frame):
        """-> (T_sheet2cam, {id: 4 góc px}, sai số chiếu lại px) hoặc None."""
        corners, ids, _ = self._detect(frame)
        if ids is None:
            return None
        seen = {int(i): c.reshape(4, 2).astype(np.float64)
                for c, i in zip(corners, ids.flatten()) if int(i) in SHEET_OBJ}
        if len(seen) < self.min_markers:
            return None
        obj = np.vstack([SHEET_OBJ[i] for i in seen]).astype(np.float64)
        img = np.vstack([seen[i] for i in seen])
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok or tvec[2, 0] <= 0:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, self.K, self.dist, rvec, tvec)
        proj = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)[0].reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))
        return _to_T(rvec, tvec), seen, err


def average_poses(Ts):
    """Trung bình nhiều pose gần nhau (tịnh tiến: trung vị; xoay: trung bình rồi
    trực giao hoá bằng SVD)."""
    Ts = np.asarray(Ts)
    U, _, Vt = np.linalg.svd(Ts[:, :3, :3].mean(0))
    out = np.eye(4)
    out[:3, :3] = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    out[:3, 3] = np.median(Ts[:, :3, 3], axis=0)
    return out


def save_table(path, T_table2cam, K, dist, image_size, err_px, n_frames):
    np.savez(path, T_table2cam=T_table2cam, K=K, dist=dist, image_size=np.array(image_size),
             reprojection_error=err_px, n_frames=n_frames, stamp=time.time())


def load_table(path, width, height):
    """T_table2cam đã lưu. Đổi độ phân giải không làm đổi T (chỉ K đổi)."""
    d = np.load(path)
    w0, h0 = (int(v) for v in d["image_size"])
    if (w0, h0) != (width, height):
        print(f"ℹ️  Mặt bàn calib ở {w0}x{h0}, đang chạy {width}x{height} (T không đổi, K tự quy đổi).")
    return d["T_table2cam"].astype(np.float64)


# ── Hình học mặt bàn ──────────────────────────────────────────────────────────
class Table:
    """Mặt bàn z=0 trong hệ bàn, nhìn từ camera (K, dist)."""

    def __init__(self, T_table2cam, K, dist):
        self.T = np.asarray(T_table2cam, np.float64)
        self.K, self.dist = np.asarray(K, np.float64), np.asarray(dist, np.float64).ravel()
        self.R, self.t = self.T[:3, :3], self.T[:3, 3]
        self.rvec = cv2.Rodrigues(self.R)[0]
        self.cam = -self.R.T @ self.t                  # tâm camera trong hệ bàn
        if self.cam[2] <= 0:
            raise ValueError("Camera nằm DƯỚI mặt bàn theo T_table2cam — tờ calib bị lật?")

    @property
    def height(self):
        return float(self.cam[2])

    @property
    def foot(self):
        """Chân camera: hình chiếu tâm camera xuống mặt bàn (x, y)."""
        return self.cam[:2].copy()

    def pix_to_table(self, uv):
        """Pixel (N×2) -> điểm trên mặt bàn (N×3, z=0). Tia không cắt bàn: NaN."""
        uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        n = cv2.undistortPoints(uv, self.K, self.dist).reshape(-1, 2)
        d = (self.R.T @ np.column_stack([n, np.ones(len(n))]).T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -self.cam[2] / d[:, 2]
        p = self.cam + s[:, None] * d
        p[~(s > 0)] = np.nan
        return p

    def project(self, P):
        """Điểm hệ bàn (N×3) -> pixel (N×2)."""
        P = np.ascontiguousarray(np.asarray(P, np.float64).reshape(-1, 3))
        return cv2.projectPoints(P, self.rvec, self.t, self.K, self.dist)[0].reshape(-1, 2)

    def to_cam(self, P):
        return (self.R @ np.asarray(P, np.float64).reshape(-1, 3).T).T + self.t


# ── Mô hình lăng trụ ─────────────────────────────────────────────────────────
N_CIRCLE = 32


def prism_points(shape, p):
    """Các điểm 3D (hệ bàn) đủ để vẽ bóng chiếu của lăng trụ.
    box: p = (cx, cy, yaw, w, d, h); cylinder: p = (cx, cy, D, h)."""
    if shape == "box":
        cx, cy, yaw, w, d, h = p
        s = np.linspace(-0.5, 0.5, 3)                       # 3 điểm mỗi cạnh: bù méo ống kính
        loc = np.vstack([np.column_stack([s * w, np.full(3, -d / 2)]),
                         np.column_stack([np.full(3, w / 2), s * d]),
                         np.column_stack([-s * w, np.full(3, d / 2)]),
                         np.column_stack([np.full(3, -w / 2), -s * d])])
        c, sn = np.cos(yaw), np.sin(yaw)
        xy = loc @ np.array([[c, sn], [-sn, c]]) + [cx, cy]
    else:
        cx, cy, D, h = p
        a = np.linspace(0, 2 * np.pi, N_CIRCLE, endpoint=False)
        xy = np.column_stack([cx + D / 2 * np.cos(a), cy + D / 2 * np.sin(a)])
    n = len(xy)
    return np.vstack([np.column_stack([xy, np.zeros(n)]), np.column_stack([xy, np.full(n, h)])])


def silhouette(table, shape, p):
    """Đa giác bóng chiếu (pixel, lồi) của lăng trụ."""
    uv = table.project(prism_points(shape, p))
    if not np.all(np.isfinite(uv)):
        return None
    return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)


def initial_guess(table, contour_uv):
    """Ước lượng thẳng từ hình học (xem đầu file) -> dict hoặc None.
    Trả tâm (x, y), bề ngang D, cao h, hướng nhìn u (từ chân camera ra vật)."""
    P = table.pix_to_table(contour_uv)
    P = P[np.all(np.isfinite(P), axis=1)]
    if len(P) < 5:
        return None
    C = table.foot
    rel = P[:, :2] - C
    cen = rel.mean(0)
    if np.linalg.norm(cen) < 1e-4:                         # camera nhìn thẳng xuống ngay trên vật
        cen = np.array([0.0, 1e-4])
    u = cen / np.linalg.norm(cen)
    v = np.array([-u[1], u[0]])
    r, l = rel @ u, rel @ v
    phi = np.arctan2(l, r)
    half = 0.5 * float(phi.max() - phi.min())
    r_near, r_far = float(r.min()), float(r.max())
    if r_near <= 0:                                        # chân camera nằm trong vết vật
        D = float(l.max() - l.min())
        rho = r_near + D / 2
    else:
        sn = min(np.sin(half), 0.95)
        rho = r_near / (1 - sn)                            # tiếp tuyến từ chân camera tới vòng đáy
        D = 2 * rho * sn
    phi_c = 0.5 * float(phi.max() + phi.min())
    centre = C + rho * (np.cos(phi_c) * u + np.sin(phi_c) * v)
    H = table.height
    far_base = rho + D / 2
    h = H * (1 - far_base / r_far) if r_far > far_base else 0.0
    h = float(np.clip(h, 0.005, 0.8 * H))
    return dict(xy=centre, D=float(D), h=h, u=u, v=v, foot=C, r_near=r_near, r_far=r_far,
                phi_c=phi_c, half=half, H=H)


def box_starts(g, yaws=np.radians([0, 22.5, 45, 67.5]), aspects=(1.0, 1.6, 2.5, 1 / 1.6, 1 / 2.5)):
    """Điểm xuất phát cho hộp: với mỗi góc xoay + tỉ lệ cạnh, chọn cỡ hộp sao cho
    góc nhìn ngang của đáy (từ chân camera) bằng góc ngang của mask (cạnh đứng
    chiếu theo tia từ chân camera nên mặt trên không làm rộng góc này), đặt mép
    gần đúng mép gần của mask, rồi tính h từ mép xa (đỉnh). Không dùng bề ngang
    D của trụ: hộp quay mặt vào camera thì D trụ to hơn thật 15–20 % -> kẹt nghiệm sai."""
    u0 = np.cos(g["phi_c"]) * g["u"] + np.sin(g["phi_c"]) * g["v"]      # hướng nhìn tới tâm vật
    ang_u = float(np.arctan2(u0[1], u0[0]))
    span = 2.0 * g["half"]
    r_near = max(g["r_near"], 1e-3)
    out = []
    for dy in yaws:
        yaw = ang_u + dy
        c, sn = np.cos(dy), np.sin(dy)
        for ar in aspects:
            # đáy cỡ (ar × 1): góc theo hướng nhìn (a) và ngang (l), mép gần đặt ở r_near
            bx = np.array([-ar, ar, ar, -ar]) / 2
            by = np.array([-1, -1, 1, 1]) / 2
            a, lat = bx * c - by * sn, bx * sn + by * c
            a = a - a.min()

            def ang(sc):
                phi = np.arctan2(sc * lat, r_near + sc * a)
                return float(phi.max() - phi.min())
            lo, hi = 1e-3, 1.0                         # cỡ (m): góc tăng dần theo cỡ -> chia đôi
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if ang(mid) < span else (lo, mid)
            d = 0.5 * (lo + hi)
            w = ar * d
            ext = d * float(a.max())                  # bề sâu theo hướng nhìn
            xy = g["foot"] + u0 * (r_near + ext / 2)
            far_base = r_near + ext
            h = g["H"] * (1 - far_base / g["r_far"]) if g["r_far"] > far_base else 0.01
            out.append(np.array([xy[0], xy[1], yaw, w, d, float(np.clip(h, 0.005, 0.8 * g["H"]))]))
    return out


class _Raster:
    """So bóng chiếu với mask trong 1 ô cắt nhỏ (≤ `size` px) để mỗi lần thử rẻ.
    Có `gray` (ảnh xám cùng khung) thì thêm bản đồ khoảng cách tới cạnh ảnh bên
    trong mask, để chấm điểm các mép NẰM TRONG bóng chiếu (mép trên phía trước
    của hộp, cung trước của nắp trụ) — chỉ bóng chiếu thì hộp dẹt bị mơ hồ giữa
    sâu và cao."""

    EDGE_CAP_PX = 4.0

    def __init__(self, mask, bbox, size=96, gray=None):
        x0, y0, x1, y1 = bbox
        self.s = min(1.0, size / max(x1 - x0, y1 - y0))
        self.off = np.array([x0, y0], np.float64)
        crop = mask[y0:y1, x0:x1]
        w = max(1, int(round((x1 - x0) * self.s)))
        h = max(1, int(round((y1 - y0) * self.s)))
        self.m = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA) > 127
        self.n = int(self.m.sum())
        self.buf = np.zeros_like(self.m, np.uint8)
        self.edt = None            # khoảng cách tới cạnh ảnh gần nhất (px ô cắt)
        self.epx = None            # các pixel cạnh ảnh bên trong vật (≤ 100 điểm)
        if gray is not None:
            g = cv2.resize(gray[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_AREA)
            g = cv2.GaussianBlur(g, (3, 3), 0)
            gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
            mag = np.hypot(gx, gy)
            inner = cv2.erode(self.m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            v = mag[inner]
            if v.size > 20 and v.max() > 40:
                e = inner & (mag > max(40.0, 0.35 * float(np.percentile(v, 98))))
                if e.sum() > 8:
                    self.edt = cv2.distanceTransform((~e).astype(np.uint8), cv2.DIST_L2, 3)
                    ys, xs = np.nonzero(e)
                    k = np.linspace(0, len(xs) - 1, min(100, len(xs))).astype(int)
                    self.epx = np.column_stack([xs[k], ys[k]]).astype(np.float64)

    def iou(self, poly):
        if poly is None:
            return 0.0
        self.buf[:] = 0
        q = np.round((poly - self.off) * self.s * 16).astype(np.int32)   # 4 bit lẻ: khớp dưới 1 px
        cv2.fillConvexPoly(self.buf, q, 1, cv2.LINE_8, 4)
        b = self.buf.astype(bool)
        inter = int(np.count_nonzero(b & self.m))
        return inter / max(1, self.n + int(np.count_nonzero(b)) - inter)

    def edge_cost(self, segs):
        """Chamfer 2 chiều giữa mép trong dự đoán (`segs`: S×2×2 pixel) và cạnh
        ảnh bên trong vật. 0 = trùng khít, 1 = lệch ≥ EDGE_CAP_PX. Dự đoán không
        có mép trong nào trong khi ảnh có cạnh -> 1 (hộp dẹt giả)."""
        if self.edt is None:
            return 0.0
        if segs is None or len(segs) == 0:
            return 1.0
        cap = self.EDGE_CAP_PX
        S = (np.asarray(segs, np.float64) - self.off) * self.s
        a, b = S[:, 0], S[:, 1]
        # chiều dự đoán -> ảnh: điểm giữa và 2 đầu mỗi đoạn tra bản đồ khoảng cách
        q = np.vstack([a, b, 0.5 * (a + b)])
        H, W = self.edt.shape
        x = np.clip(np.round(q[:, 0]).astype(int), 0, W - 1)
        y = np.clip(np.round(q[:, 1]).astype(int), 0, H - 1)
        fwd = np.mean(np.minimum(self.edt[y, x], cap)) / cap
        # chiều ảnh -> dự đoán: mỗi pixel cạnh tới đoạn gần nhất
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-9)
        P = self.epx[:, None, :]
        t = np.clip(((P - a) * ab).sum(2) / L2, 0, 1)
        d = np.sqrt((((a + t[..., None] * ab) - P) ** 2).sum(2)).min(1)
        rev = np.mean(np.minimum(d, cap)) / cap
        return float(0.5 * (fwd + rev))


def inner_edge_points(table, shape, p, hull, margin_px=2.5, n_per_edge=8):
    """Các đoạn (pixel, S×2×2) dọc mép MẶT TRÊN nằm bên trong bóng chiếu — mép
    mà camera thấy được nhưng không thuộc đường viền."""
    if hull is None:
        return None
    if shape == "box":
        cx, cy, yaw, w, d, h = p
        loc = np.array([[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]])
        c, s = np.cos(yaw), np.sin(yaw)
        top = np.column_stack([loc @ np.array([[c, s], [-s, c]]) + [cx, cy], np.full(4, h)])
        t = np.linspace(0, 1, n_per_edge)[:, None]
        chains = [top[i] + t * (top[(i + 1) % 4] - top[i]) for i in range(4)]
    else:
        cx, cy, D, h = p
        a = np.linspace(0, 2 * np.pi, 49)
        chains = [np.column_stack([cx + D / 2 * np.cos(a), cy + D / 2 * np.sin(a), np.full(49, h)])]
    # bóng chiếu lồi: điểm ở trong (cách viền > margin) khi nằm phía trong mọi cạnh
    hv = hull.astype(np.float64)
    e = np.roll(hv, -1, axis=0) - hv
    area2 = float(np.sum(hv[:, 0] * np.roll(hv[:, 1], -1) - np.roll(hv[:, 0], -1) * hv[:, 1]))
    nrm = np.column_stack([-e[:, 1], e[:, 0]]) / np.maximum(np.linalg.norm(e, axis=1), 1e-9)[:, None]
    if area2 < 0:                       # diện tích có dấu > 0: phía trong nằm bên trái mỗi cạnh
        nrm = -nrm
    segs = []
    for P in chains:
        uv = table.project(P)
        depth = ((uv[:, None, :] - hv[None, :, :]) * nrm[None]).sum(2).min(1)
        ok = depth > margin_px
        both = ok[:-1] & ok[1:]
        if both.any():
            segs.append(np.stack([uv[:-1][both], uv[1:][both]], axis=1))
    return np.concatenate(segs) if segs else None


def nelder_mead(f, x0, step, iters=120, tol=1e-6):
    """Cực tiểu f (không cần scipy trên Pi)."""
    n = len(x0)
    pts = [np.asarray(x0, float)]
    for i in range(n):
        x = pts[0].copy()
        x[i] += step[i]
        pts.append(x)
    vals = [f(x) for x in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0] + 0.5 * (x - pts[0]) for x in pts]
                vals = [vals[0]] + [f(x) for x in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def _valid(shape, p):
    if shape == "box":
        return p[3] > 0.004 and p[4] > 0.004 and p[5] > 0.002
    return p[2] > 0.004 and p[3] > 0.002


EDGE_WEIGHT = 0.06       # 1 px lệch mép trong ≈ 1,5% IoU
H_SCAN = (0.6, 0.8, 1.25, 1.6)   # hệ số chiều cao thử lại sau khi khớp hộp


def fit_prism(table, mask, bbox, contour_uv, shape, guess=None, iters=90, gray=None):
    """Khớp 1 loại lăng trụ với mask (+ cạnh ảnh nếu có `gray`).
    -> (tham số, IoU, điểm cạnh 0..1 (0 = mép trong trùng cạnh ảnh))."""
    g = guess or initial_guess(table, contour_uv)
    if g is None:
        return None, 0.0, 1.0
    ras = _Raster(mask, bbox, gray=gray)
    w_e = EDGE_WEIGHT if ras.edt is not None else 0.0

    def parts(p):
        if not _valid(shape, p):
            return 0.0, 1.0
        hull = silhouette(table, shape, p)
        iou = ras.iou(hull)
        e = ras.edge_cost(inner_edge_points(table, shape, p, hull)) if w_e else 0.0
        return iou, e

    def cost(p):
        iou, e = parts(p)
        return 1.0 - iou + w_e * e

    D, h = g["D"], g["h"]
    if shape == "cylinder":
        x0 = np.array([g["xy"][0], g["xy"][1], D, h])
        step = np.array([0.15 * D, 0.15 * D, 0.2 * D, 0.25 * h + 0.005])
        p, _ = nelder_mead(cost, x0, step, iters)
        p, _ = nelder_mead(cost, p, step * 0.2, iters // 2)
        return (p,) + parts(p)
    # hộp: chấm 20 điểm xuất phát (box_starts), tối ưu ngắn 3 cái tốt nhất, rồi
    # tối ưu kỹ cái tốt nhất (hộp dẹt dễ kẹt cực tiểu cục bộ)
    step = np.array([0.1 * D, 0.1 * D, np.radians(8), 0.15 * D, 0.15 * D, 0.2 * h + 0.005])
    starts = sorted(((x0, cost(x0)) for x0 in box_starts(g)), key=lambda r: r[1])
    cand = sorted((nelder_mead(cost, x0, step, 50) for x0, _ in starts[:3]), key=lambda r: r[1])
    p, v = nelder_mead(cost, cand[0][0], step * 0.5, iters)
    p, v = nelder_mead(cost, p, step * 0.2, iters // 2)              # khởi động lại simplex
    # hộp dài/dẹt: (sâu hơn + thấp hơn) và (nông hơn + cao hơn) cho bóng gần như nhau ->
    # dò dọc chiều cao: ép h khác đi rồi tối ưu lại ngắn, giữ cái tốt nhất
    for m in H_SCAN:
        q0 = _box_same_far(table, p, p[5] * m)
        if q0 is None:
            continue
        q, vq = nelder_mead(cost, q0, step * 0.3, iters // 3)
        if vq < v:
            p, v = q, vq
    p = normalize_box(p)
    return (p,) + parts(p)


def _box_same_far(table, p, h_new):
    """Hộp cùng mép gần + cùng mép xa trên ảnh (đỉnh xa chiếu xuống bàn) nhưng cao h_new:
    đổi bề sâu theo hướng nhìn (cạnh gần hướng nhìn hơn) và dời tâm theo hướng nhìn."""
    H = table.height
    if not 0.002 < h_new < 0.8 * H:
        return None
    cx, cy, yaw, w, d, h = p
    rel = np.array([cx, cy]) - table.foot
    r_c = float(np.linalg.norm(rel))
    if r_c < 1e-4:
        return None
    u0 = rel / r_c
    a = yaw - np.arctan2(u0[1], u0[0])
    c, s = abs(np.cos(a)), abs(np.sin(a))
    ext = w * c + d * s
    r_n = r_c - ext / 2
    far = (r_n + ext) * H / (H - h)                    # mép xa của đỉnh chiếu xuống bàn
    ext_new = far * (H - h_new) / H - r_n
    q = np.array(p, float)
    if c >= s:
        q[3] = (ext_new - d * s) / c
    else:
        q[4] = (ext_new - w * c) / s
    if q[3] <= 0.004 or q[4] <= 0.004:
        return None
    q[:2] = np.array([cx, cy]) + u0 * ((r_n + ext_new / 2) - r_c)
    q[5] = h_new
    return q


def normalize_box(p):
    """yaw về [-45°, 45°), đổi w/d cho khớp (hộp xoay 90° là chính nó)."""
    p = np.array(p, float)
    k = int(np.floor((p[2] + np.pi / 4) / (np.pi / 2)))
    p[2] -= k * np.pi / 2
    if k % 2:
        p[3], p[4] = p[4], p[3]
    return p


def measure_object(table, mask, bbox, contour_uv, gray=None, box_margin=0.012, min_iou=0.75):
    """Đo 1 vật: khớp cả hộp lẫn trụ, chọn cái khớp hơn. `gray`: ảnh xám cả
    khung (để dùng cạnh bên trong vật), None = chỉ dùng mask.
    Hộp có nhiều tham số hơn nên chỉ chọn hộp khi IoU hơn trụ ≥ box_margin.
    -> dict(shape, xy, size(w,d,h), yaw, iou, params) hoặc None."""
    g = initial_guess(table, contour_uv)
    if g is None:
        return None
    pc, ic, ec = fit_prism(table, mask, bbox, contour_uv, "cylinder", g, gray=gray)
    pb, ib, eb = fit_prism(table, mask, bbox, contour_uv, "box", g, gray=gray)
    if pb is not None and ib - EDGE_WEIGHT * eb > ic - EDGE_WEIGHT * ec + box_margin:
        shape, p, iou = "box", pb, ib
        size, yaw = (p[3], p[4], p[5]), p[2]
    else:
        shape, p, iou = "cylinder", pc, ic
        size, yaw = (p[2], p[2], p[3]), 0.0
    if iou < min_iou:
        shape = "object"            # không giống lăng trụ: vẫn báo vị trí, kích thước gần đúng
    return dict(shape=shape, xy=np.array(p[:2]), size=np.array(size, float), yaw=float(yaw),
                iou=float(iou), params=np.array(p), iou_box=float(ib), iou_cyl=float(ic))


def quick_measure(table, contour_uv):
    """Ước lượng thẳng (không khớp) — dùng khi vật đang bị dời, cần nhanh."""
    g = initial_guess(table, contour_uv)
    if g is None:
        return None
    return dict(shape="?", xy=g["xy"], size=np.array([g["D"], g["D"], g["h"]]), yaw=0.0, iou=0.0,
                params=None)


# ── Self-test ────────────────────────────────────────────────────────────────
def synth_camera():
    """Camera giả giống chỗ đặt thật: 640×360, cao ~45 cm trên bàn, nhìn chếch."""
    K = np.array([[385.0, 0, 320], [0, 385.0, 180], [0, 0, 1]])
    dist = np.array([0.02, -0.06, 0.0, 0.0, 0.0])
    # camera ở (0, -0.45, 0.42) trong hệ bàn, nhìn về điểm (0, 0.02, 0)
    cam = np.array([0.0, -0.45, 0.42])
    z = np.array([0.0, 0.02, 0.0]) - cam
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])                       # hàng = trục camera trong hệ bàn
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ cam
    return K, dist, T


def render_scene(table, shape, p, size=(360, 640), supersample=4, bg=None, rng=None):
    """Ảnh giả vẽ độc lập với bộ khớp: mặt trên sáng, mặt bên tối dần theo hướng,
    chỉ vẽ mặt quay về camera (lăng trụ lồi thì các mặt đó không che nhau).
    -> (mask uint8, ảnh xám uint8)."""
    H, W = size
    ss = supersample
    img = np.full((H * ss, W * ss), 90 if bg is None else 0, np.uint8)
    mask = np.zeros_like(img)
    if shape == "box":
        cx, cy, yaw, w, d, h = p
        loc = np.array([[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]])
        c, s = np.cos(yaw), np.sin(yaw)
        xy = loc @ np.array([[c, s], [-s, c]]) + [cx, cy]
    else:
        cx, cy, D, h = p
        a = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        xy = np.column_stack([cx + D / 2 * np.cos(a), cy + D / 2 * np.sin(a)])
    n = len(xy)
    light = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    ctr = np.array([cx, cy])
    for i in range(n):
        j = (i + 1) % n
        mid = np.array([*(0.5 * (xy[i] + xy[j])), h / 2])
        nrm = np.array([*(0.5 * (xy[i] + xy[j]) - ctr), 0.0])
        nrm /= np.linalg.norm(nrm)
        if nrm @ (table.cam - mid) <= 0:
            continue
        quad = np.array([[*xy[i], 0], [*xy[j], 0], [*xy[j], h], [*xy[i], h]])
        uv = np.round(table.project(quad) * ss).astype(np.int32)
        cv2.fillConvexPoly(img, uv, int(110 + 60 * max(0.0, nrm @ light)))
        cv2.fillConvexPoly(mask, uv, 255)
    uv = np.round(table.project(np.column_stack([xy, np.full(n, h)])) * ss).astype(np.int32)
    cv2.fillPoly(img, [uv], 205)
    cv2.fillPoly(mask, [uv], 255)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA)
    if rng is not None:
        img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
    return mask, img


def render_mask(table, shape, p, size=(360, 640), supersample=4):
    """Mask 'thật' vẽ độc lập: phủ từng mặt của lăng trụ ở độ phân giải ×4."""
    H, W = size
    big = np.zeros((H * supersample, W * supersample), np.uint8)
    if shape == "box":
        cx, cy, yaw, w, d, h = p
        loc = np.array([[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]])
        c, s = np.cos(yaw), np.sin(yaw)
        xy = loc @ np.array([[c, s], [-s, c]]) + [cx, cy]
        n = 4
    else:
        cx, cy, D, h = p
        a = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        xy = np.column_stack([cx + D / 2 * np.cos(a), cy + D / 2 * np.sin(a)])
        n = len(xy)
    for i in range(n):                              # các mặt bên (tứ giác)
        j = (i + 1) % n
        quad = np.array([[*xy[i], 0], [*xy[j], 0], [*xy[j], h], [*xy[i], h]])
        uv = table.project(quad) * supersample
        cv2.fillConvexPoly(big, np.round(uv).astype(np.int32), 255)
    for z in (0.0, h):
        uv = table.project(np.column_stack([xy, np.full(n, z)])) * supersample
        cv2.fillPoly(big, [np.round(uv).astype(np.int32)], 255)
    return cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)


def synth_sheet_image(table_nodist, size=(360, 640), background=None, T_sheet2table=None):
    """Ảnh giả tờ calib mặt bàn nằm trên bàn (ảnh không méo ống kính).
    `background`: ảnh BGR/xám để dán tờ giấy lên; `T_sheet2table`: tờ đặt lệch."""
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, SHEET_DICT))
    ppm = 4000                                       # px/m của ảnh tờ giấy
    page = np.full((int(0.2 * ppm), int(0.2 * ppm)), 255, np.uint8)
    for m in SHEET_IDS:
        c = sheet_marker_centre(m)
        s = int(SHEET_MARKER_M * ppm)
        x0 = int((c[0] + 0.1) * ppm - s / 2)
        y0 = int((0.1 - c[1]) * ppm - s / 2)
        page[y0:y0 + s, x0:x0 + s] = cv2.aruco.generateImageMarker(adict, m, s)
    # mép trang tính theo tâm pixel (cv2.warpPerspective: pixel i phủ [i-0.5, i+0.5])
    n = page.shape[0]
    src = np.array([[-0.5, -0.5], [n - 0.5, -0.5], [n - 0.5, n - 0.5], [-0.5, n - 0.5]], np.float32)
    corners = np.array([[-0.1, 0.1, 0, 1], [0.1, 0.1, 0, 1], [0.1, -0.1, 0, 1], [-0.1, -0.1, 0, 1]])
    if T_sheet2table is not None:
        corners = (T_sheet2table @ corners.T).T
    Hm = cv2.getPerspectiveTransform(src, table_nodist.project(corners[:, :3]).astype(np.float32))
    H, W = size
    warped = cv2.warpPerspective(page, Hm, (W, H), borderValue=0, flags=cv2.INTER_LINEAR)
    cover = cv2.warpPerspective(np.full_like(page, 255), Hm, (W, H), borderValue=0, flags=cv2.INTER_LINEAR)
    if background is None:
        background = np.full((H, W), 120, np.uint8)
    bgf = background.astype(np.float32)
    a = cover.astype(np.float32) / 255.0
    if bgf.ndim == 3:
        a, warped = a[..., None], warped[..., None]
    return np.clip(bgf * (1 - a) + warped.astype(np.float32) * a, 0, 255).astype(np.uint8)


def mask_contour(mask):
    m = (mask > 127).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(c)
    pad = 4
    bbox = (max(0, x - pad), max(0, y - pad), min(mask.shape[1], x + w + pad), min(mask.shape[0], y + h + pad))
    return c.astype(np.float64), bbox


def _self_test():
    ok_all = True
    K, dist, T = synth_camera()
    table = Table(T, K, dist)

    # 1) pixel -> bàn -> pixel phải về đúng chỗ
    P = np.array([[0.05, 0.03, 0], [-0.1, 0.12, 0], [0.12, -0.08, 0]])
    back = table.pix_to_table(table.project(P))
    e = np.abs(back - P).max() * 1000
    ok = e < 0.05
    ok_all &= ok
    print(f"[self-test] pixel <-> mặt bàn: lệch {e:.4f} mm -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")

    # 2) tờ calib: chiếu ảnh tờ giấy giả, dò lại pose
    tbl0 = Table(T, K, np.zeros(5))          # ảnh giả không có méo ống kính -> dò bằng dist=0
    img = synth_sheet_image(tbl0)
    res = SheetDetector(K, np.zeros(5)).detect(img)
    if res is None:
        print("[self-test] tờ calib: KHÔNG dò được")
        ok_all = False
    else:
        Ts, seen, err = res
        # cái cần là điểm trên MẶT BÀN đúng, không phải gốc toạ độ (gốc lệch dọc tia
        # nhìn gần như không làm lệch điểm trên bàn): so lưới điểm ±15 cm
        g = np.array([[x, y, 0] for x in np.linspace(-0.15, 0.15, 7) for y in np.linspace(-0.12, 0.15, 7)])
        uv = tbl0.project(g)
        P = Table(Ts, K, np.zeros(5)).pix_to_table(uv)
        e = np.nanmax(np.linalg.norm(P - g, axis=1)) * 1000
        dang = np.degrees(np.arccos(np.clip(Ts[:3, 2] @ T[:3, 2], -1, 1)))
        ok = e < 2.0 and len(seen) == 9
        ok_all &= ok
        print(f"[self-test] tờ calib: {len(seen)}/9 marker, chiếu lại {err:.2f}px, pháp tuyến lệch {dang:.2f}°; "
              f"điểm trên bàn (vùng 30×27 cm) lệch lớn nhất {e:.2f} mm -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")

    # 3) khớp hộp / trụ: vị trí, kích thước, phân loại
    cases = [("box", (0.04, 0.02, np.radians(20), 0.080, 0.060, 0.040)),
             ("box", (-0.08, 0.10, np.radians(-35), 0.050, 0.050, 0.050)),
             ("box", (0.10, -0.05, np.radians(5), 0.120, 0.040, 0.030)),
             ("cylinder", (0.00, 0.08, 0.070, 0.090)),
             ("cylinder", (-0.10, -0.04, 0.040, 0.110)),
             ("cylinder", (0.09, 0.12, 0.055, 0.025))]
    rng = np.random.default_rng(1)
    for shape, p in cases:
        mask, gray = render_scene(table, shape, p, rng=rng)
        cont, bbox = mask_contour(mask)
        true_size = (p[2], p[2], p[3])
        if shape == "box":
            tp = normalize_box(p)
            true_size = (tp[3], tp[4], tp[5])
        r0 = measure_object(table, mask, bbox, cont)                 # chỉ mask
        t0 = time.perf_counter()
        r = measure_object(table, mask, bbox, cont, gray=gray)       # mask + cạnh ảnh
        ms = (time.perf_counter() - t0) * 1000
        dxy = np.linalg.norm(r["xy"] - p[:2]) * 1000
        dxy0 = np.linalg.norm(r0["xy"] - p[:2]) * 1000
        dsz = np.abs(r["size"] - true_size).max() * 1000
        ok = r["shape"] == shape and dxy < 5 and dsz < 8
        ok_all &= ok
        print(f"[self-test] {shape:8s} thật {np.round(np.array(true_size) * 1000).astype(int)} mm -> "
              f"nhận {r['shape']:8s} {np.round(r['size'] * 1000).astype(int)} mm, tâm lệch {dxy:.1f} mm "
              f"(chỉ mask: {dxy0:.1f}), kích thước lệch ≤ {dsz:.1f} mm, IoU hộp {r['iou_box']:.3f} / trụ "
              f"{r['iou_cyl']:.3f}, {ms:.0f} ms -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    return ok_all


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if _self_test() else 1)
    print(__doc__)
