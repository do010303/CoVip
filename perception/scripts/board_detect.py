"""
Dò bảng workspace 4 marker ArUco (và 1 marker đơn gắn trên tay) trên ảnh
THÔ của camera, trả pose trong hệ camera OpenCV (X phải, Y xuống, Z ra trước).

Chỉ dùng cv2 + numpy để run_pi4_ros2.py trên Pi nạp được mà không cần ROS.

Hệ trục bảng (giống vs_lib/vision/vision_aruco_detector.py): gốc ở tâm bảng
(dấu +), x sang phải, y lên trên, z vuông góc mặt bảng hướng về người nhìn.
ID 0 trên-trái, 1 trên-phải, 2 dưới-phải, 3 dưới-trái.

    python3 scripts/board_detect.py --self-test
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def make_detect_fn(dict_name):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    try:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return lambda img: detector.detectMarkers(img)
    except AttributeError:      # OpenCV < 4.7
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return lambda img: cv2.aruco.detectMarkers(img, aruco_dict, parameters=params)


def _square(cx, cy, half):
    return np.array([[cx - half, cy + half, 0], [cx + half, cy + half, 0],
                     [cx + half, cy - half, 0], [cx - half, cy - half, 0]], np.float32)


def _to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = np.asarray(tvec).ravel()
    return T


class BoardDetector:
    """Bảng workspace: marker cạnh `marker_size_m`, tâm marker cách tâm bảng
    ±`marker_offset_m`. Mặc định = bảng in thật của tay mới (30mm, ±75mm)."""

    def __init__(self, K, dist, marker_offset_m=0.075, marker_size_m=0.030,
                 dict_name="DICT_4X4_1000", min_markers=2):
        self.K, self.dist = np.asarray(K, np.float64), np.asarray(dist, np.float64)
        o, h = marker_offset_m, marker_size_m / 2
        self.obj = {0: _square(-o, o, h), 1: _square(o, o, h),
                    2: _square(o, -o, h), 3: _square(-o, -o, h)}
        self.min_markers = min_markers
        self._detect = make_detect_fn(dict_name)
        self.half_m = marker_offset_m + marker_size_m      # nửa cạnh khung bao 4 marker
        self.last_seen = {}        # id -> 4 góc (px) của LẦN DÒ GẦN NHẤT (rỗng = không thấy)
        self.last_roi = None       # (x0, y0, x1, y1) vùng vừa dò, None = cả khung
        self._roi_ok = False       # lần trước thấy đủ marker -> lần này chỉ dò quanh bảng
        self._runs = 0

    def _roi(self, T_hint, shape, margin=0.35, min_px=24):
        """Khung bao bảng trên ảnh (theo pose đang nhớ) nới thêm `margin` mỗi
        phía. Dò ArUco chỉ trong vùng này rẻ hơn nhiều so với cả khung hình —
        chi phí dò tỉ lệ với số điểm ảnh."""
        b = self.half_m
        pts = np.array([[-b, b, 0], [b, b, 0], [b, -b, 0], [-b, -b, 0]], np.float64)
        uv = cv2.projectPoints(pts, cv2.Rodrigues(T_hint[:3, :3])[0], T_hint[:3, 3],
                               self.K, self.dist)[0].reshape(-1, 2)
        if not np.all(np.isfinite(uv)):
            return None
        H, W = shape[:2]
        lo, hi = uv.min(0), uv.max(0)
        pad = np.maximum((hi - lo) * margin, min_px)
        x0, y0 = np.floor(np.maximum(lo - pad, 0)).astype(int)
        x1, y1 = np.ceil(np.minimum(hi + pad, [W, H])).astype(int)
        if x1 - x0 < 16 or y1 - y0 < 16 or (x1 - x0) * (y1 - y0) > 0.8 * W * H:
            return None            # vùng quá nhỏ / gần bằng cả khung: dò cả khung
        return int(x0), int(y0), int(x1), int(y1)

    def detect(self, frame, T_hint=None):
        """-> (T_board2cam 4x4 [m], số marker thấy, sai số chiếu lại px) hoặc None.

        `T_hint` = pose bảng đang nhớ. Có nó, và lần dò trước thấy đủ 4 marker,
        thì chỉ dò trong vùng quanh bảng. Thiếu marker (bị che, hoặc bảng vừa
        bị dời ra khỏi vùng) thì lần kế dò lại cả khung; cứ 10 lần cũng dò cả
        khung 1 lần."""
        self._runs += 1
        roi = None
        if T_hint is not None and self._roi_ok and self._runs % 10:
            roi = self._roi(T_hint, frame.shape)
        self.last_roi = roi
        x0, y0 = (roi[0], roi[1]) if roi else (0, 0)
        corners, ids, _ = self._detect(frame[roi[1]:roi[3], roi[0]:roi[2]] if roi else frame)
        if ids is None:
            self.last_seen, self._roi_ok = {}, False
            return None
        obj, img, seen = [], [], {}
        for c, i in zip(corners, ids.flatten().tolist()):
            if i in self.obj:
                c = c.reshape(4, 2) + [x0, y0]
                obj.append(self.obj[i])
                img.append(c)
                seen[i] = c.copy()
        self.last_seen = seen
        self._roi_ok = len(seen) == len(self.obj)
        if len(obj) < self.min_markers:
            return None
        obj = np.vstack(obj).astype(np.float64)
        img = np.vstack(img).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok or tvec[2][0] <= 0:
            return None
        proj = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)[0].reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))
        return _to_T(rvec, tvec), len(obj) // 4, err


def cam_to_board(T_board2cam, p_cam):
    """Điểm trong hệ camera -> hệ BẢNG (cùng đơn vị với T, tức mét).

    Hệ bảng: gốc ở dấu + giữa bảng, x sang phải, y lên trên (nhìn vào bảng),
    z vuông góc mặt bảng hướng RA phía người nhìn/camera. Vậy z = khoảng cách
    từ điểm tới mặt bảng (z > 0: ở phía trước bảng; z = 0: chạm bảng)."""
    R, t = T_board2cam[:3, :3], T_board2cam[:3, 3]
    return R.T @ (np.asarray(p_cam, np.float64).ravel() - t)


class BoardTracker:
    """Giữ pose bảng ổn định giữa các lần dò.

    Bảng và camera đứng yên, còn mỗi lần dò rung vài phần mười mm → lấy trung
    bình `window` lần gần nhất (tịnh tiến: trung bình; xoay: trung bình ma
    trận rồi trực giao hoá bằng SVD — đúng khi các mẫu gần nhau). Nếu lần dò
    mới lệch khỏi trung bình quá `jump_m` / `jump_deg` thì coi như bảng hoặc
    camera vừa bị dời: bỏ lịch sử, bắt đầu lại. Lần dò chỉ thấy 2-3 marker
    kém chính xác hơn hẳn, nên khi đã có mẫu đủ 4 marker thì bỏ qua chúng.

    Bảng/camera bị dời: 2 lần dò LIÊN TIẾP cùng lệch khỏi trung bình quá
    `move_m` / `move_deg` (lớn hơn rung của một lần dò) và khớp nhau thì bỏ
    lịch sử, chỉ giữ 2 mẫu đó -> trễ tối đa 2 lần dò (trung bình cả cửa sổ 8
    mẫu ở 3-5 lần dò/giây sẽ trễ 2-3 giây). Một mẫu lệch lẻ loi không đổi gì
    — một lần dò hỏng không được phép kéo pose đi."""

    def __init__(self, window=8, jump_m=0.02, jump_deg=8.0, move_m=0.004, move_deg=1.5, full_markers=4):
        self.window, self.jump_m, self.jump_deg = window, jump_m, jump_deg
        # Từ bao nhiêu marker trở lên thì coi lần dò là "đủ tin". Mặc định 4;
        # đặt 3 khi cần bám bảng trong lúc đầu công cụ che mất 1 marker.
        self.full = full_markers
        self.move_m, self.move_deg = move_m, move_deg
        self._pend = []            # các mẫu lệch LIÊN TIẾP đang chờ xác nhận (T, n_marker)
        self._hist = []            # list (T, n_marker)
        self.T = None              # pose đã làm mượt (board -> camera, mét)
        self.n_markers = 0
        self.err_px = None
        self.t_last = None         # time.time() của lần dò thành công gần nhất

    @staticmethod
    def _diff(Ta, Tb):
        """(lệch tịnh tiến [m], lệch góc [độ]) giữa 2 pose."""
        c = np.clip((np.trace(Ta[:3, :3].T @ Tb[:3, :3]) - 1) / 2, -1, 1)
        return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), float(np.degrees(np.arccos(c)))

    def _set(self, n_markers, err_px, now):
        Ts = np.array([h[0] for h in self._hist])
        U, _, Vt = np.linalg.svd(Ts[:, :3, :3].mean(0))
        R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = R, Ts[:, :3, 3].mean(0)
        self.T, self.n_markers, self.err_px, self.t_last = out, n_markers, err_px, now
        return out

    def update(self, T, n_markers, err_px, now):
        best = max([n for _, n in self._hist], default=0)
        if self.T is None:
            self._hist = [(T, n_markers)]
            return self._set(n_markers, err_px, now)
        if n_markers >= self.full and best < self.full:
            # có mẫu đủ 4 marker rồi thì bỏ các mẫu yếu trước đó
            self._hist, self._pend = [(T, n_markers)], []
            return self._set(n_markers, err_px, now)

        # Mẫu yếu (2-3 marker) trong khi đang có pose từ mẫu 4 marker: chỉ tin
        # khi nó lệch LỚN và 3 lần liên tiếp cùng nói một chỗ (bảng dời thật
        # trong lúc bị che); còn lại bỏ qua.
        weak = n_markers < self.full and best >= self.full
        d, a = self._diff(self.T, T)
        if weak:
            off, need = d > self.jump_m or a > self.jump_deg, 3
        else:
            off, need = d > self.move_m or a > self.move_deg, 2
        if not off:
            self._pend = []
            self.t_last = now
            if weak:
                return self.T
            self._hist = (self._hist + [(T, n_markers)])[-self.window:]
            return self._set(n_markers, err_px, now)

        # Mẫu lệch: một mẫu lẻ loi không đổi gì (chống nhiễu); đủ `need` mẫu
        # liên tiếp khớp NHAU thì coi là bảng/camera vừa bị dời -> bám theo.
        if self._pend:
            dp, ap = self._diff(self._pend[-1][0], T)
            if dp > self.jump_m or ap > self.jump_deg:
                self._pend = []
        self._pend.append((T, n_markers))
        self.t_last = now
        if len(self._pend) < need:
            return self.T
        self._hist, self._pend = self._pend, []
        return self._set(n_markers, err_px, now)

    def age(self, now):
        return None if self.t_last is None else now - self.t_last


def draw_workspace(frame, K, dist, T_board2cam, draw_half_m=0.05, tip_board_m=None):
    """Vẽ lên ảnh, nét mảnh để không che bút: khung vùng vẽ (vuông
    ±draw_half_m quanh dấu +), dấu + ở tâm, trục x (đỏ) / y (lục) của bảng dài
    15 mm. Có `tip_board_m` (đầu bút trong hệ bảng) thì vẽ thêm hình chiếu
    vuông góc của đầu bút xuống mặt bảng (chấm tím) và đoạn nối."""
    rvec = cv2.Rodrigues(T_board2cam[:3, :3])[0]
    tvec = T_board2cam[:3, 3]
    h = draw_half_m
    pts = [[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0],            # 0-3 khung
           [0, 0, 0], [0.015, 0, 0], [0, 0.015, 0]]                    # 4 tâm, 5 x, 6 y
    if tip_board_m is not None:
        x, y, z = (float(v) for v in tip_board_m)
        pts += [[x, y, 0], [x, y, z]]                                  # 7 chân, 8 đầu bút
    uv = cv2.projectPoints(np.array(pts, np.float64), rvec, tvec, K, dist)[0].reshape(-1, 2)
    if not np.all(np.isfinite(uv)) or np.abs(uv).max() > 1e5:
        return
    p = [tuple(int(round(v)) for v in q) for q in uv]
    inside = tip_board_m is not None and abs(tip_board_m[0]) <= h and abs(tip_board_m[1]) <= h
    cv2.polylines(frame, [np.array(p[:4], np.int32)], True,
                  (0, 255, 0) if inside else (0, 200, 255), 1, cv2.LINE_AA)
    cv2.line(frame, p[4], p[5], (0, 0, 255), 1, cv2.LINE_AA)
    cv2.line(frame, p[4], p[6], (0, 255, 0), 1, cv2.LINE_AA)
    if tip_board_m is not None:
        cv2.line(frame, p[7], p[8], (255, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, p[7], 2, (255, 0, 255), -1, cv2.LINE_AA)


def draw_board_tracking(frame, K, dist, T_board2cam, marker_obj, seen, seen_scale=1.0):
    """Vẽ phần BÁM BẢNG, nét mảnh: 4 marker chiếu từ pose bảng đang nhớ —
    LỤC nếu lần dò gần nhất thấy marker đó, ĐỎ nếu đang bị che / không thấy
    (vị trí suy từ pose nhớ) — kèm số ID nhỏ; chấm vàng = góc marker dò được
    thật trên ảnh. Ô lệch khỏi marker thật hay chấm vàng tức là pose bảng đang
    nhớ đã cũ (bảng hoặc camera vừa bị dời). `seen_scale`: nhân toạ độ góc dò
    được khi ảnh vẽ nhỏ hơn ảnh dò (K truyền vào phải là K của ảnh vẽ)."""
    rvec = cv2.Rodrigues(T_board2cam[:3, :3])[0]
    tvec = T_board2cam[:3, 3]
    ids = sorted(marker_obj)
    pts = np.vstack([marker_obj[i] for i in ids]).astype(np.float64)
    uv = cv2.projectPoints(pts, rvec, tvec, K, dist)[0].reshape(-1, 2)
    if not np.all(np.isfinite(uv)) or np.abs(uv).max() > 1e5:
        return
    uv = np.round(uv).astype(np.int32)
    fs = 0.3 * frame.shape[1] / 640
    for k, i in enumerate(ids):
        q = uv[4 * k: 4 * k + 4]
        col = (0, 255, 0) if i in seen else (0, 0, 255)
        cv2.polylines(frame, [q], True, col, 1, cv2.LINE_AA)
        org = (int(q[:, 0].min()), int(q[:, 1].min()) - 3)      # nhãn nằm trên marker
        cv2.putText(frame, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
    for c in seen.values():
        for x, y in np.asarray(c) * seen_scale:
            cv2.circle(frame, (int(round(x)), int(round(y))), 1, (0, 255, 255), -1, cv2.LINE_AA)


class SingleMarkerDetector:
    """1 marker ArUco đơn dán trên phần chuyển động của tay (vd hộp bút) —
    dùng cho hand-eye kiểu marker."""

    def __init__(self, K, dist, marker_id=0, marker_size_m=0.020, dict_name="DICT_4X4_50"):
        self.K, self.dist = np.asarray(K, np.float64), np.asarray(dist, np.float64)
        self.marker_id = marker_id
        self.obj = _square(0.0, 0.0, marker_size_m / 2)
        self._detect = make_detect_fn(dict_name)

    def detect(self, frame):
        """-> T_marker2cam 4x4 [m] hoặc None."""
        corners, ids, _ = self._detect(frame)
        if ids is None:
            return None
        ids = ids.flatten().tolist()
        if self.marker_id not in ids:
            return None
        img = corners[ids.index(self.marker_id)].reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(self.obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        return _to_T(rvec, tvec) if ok else None


def T_to_pose(T):
    """4x4 -> (xyz, quaternion xyzw) để nhét vào PoseStamped."""
    R, t = T[:3, :3], T[:3, 3]
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    return [float(v) for v in t], [float(v) for v in q]


def pose_to_T(position, orientation):
    """PoseStamped.pose -> 4x4."""
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [position.x, position.y, position.z]
    return T


def _self_test():
    """Chiếu ảnh bảng in (600dpi) vào 1 camera giả ở tư thế biết trước, kể cả
    khi 2 marker bị che, rồi kiểm pose dò được."""
    root = next(p for p in Path(__file__).resolve().parents if (p / "ros2_ws").is_dir())
    png = (root / "ros2_ws/src/visual_servoing/aruco_markers"
           / "workspace_board_newarm_A4_600dpi.png")
    page = cv2.imread(str(png))
    if page is None:
        print(f"không đọc được {png}")
        return False
    ppm = 600 / 0.0254                      # px trên mét của ảnh in
    H, W = page.shape[:2]
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    dist = np.zeros(5)
    det = BoardDetector(K, dist)
    # tâm bảng trong ảnh trang in: tìm bằng chính 4 marker
    c, ids, _ = det._detect(page)
    centre_px = np.mean([c[i].reshape(4, 2).mean(0) for i in range(len(ids))], axis=0)
    ok_all = True
    for name, rvec, tvec, hide in [("thẳng 45cm", [0.0, 0, 0], [0.0, 0, 0.45], False),
                                   ("xiên 55cm", [0.35, -0.5, 0.1], [0.04, -0.03, 0.55], False),
                                   ("xiên, che 2 marker trên", [0.35, -0.5, 0.1], [0.04, -0.03, 0.55], True)]:
        # điểm trang (px) -> bảng (m): x phải, y lên
        pts_px = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64)
        pts_b = np.column_stack([(pts_px[:, 0] - centre_px[0]) / ppm,
                                 -(pts_px[:, 1] - centre_px[1]) / ppm, np.zeros(4)])
        # bảng nhìn từ camera: trục y bảng ngược trục y ảnh -> lật quanh x
        R_flip = np.diag([1.0, -1.0, -1.0])
        R = cv2.Rodrigues(np.array(rvec))[0] @ R_flip
        T_true = np.eye(4); T_true[:3, :3] = R; T_true[:3, 3] = tvec
        uv = cv2.projectPoints(pts_b, cv2.Rodrigues(R)[0], np.array(tvec), K, dist)[0].reshape(-1, 2)
        Hm = cv2.getPerspectiveTransform(pts_px.astype(np.float32), uv.astype(np.float32))
        img = cv2.warpPerspective(page, Hm, (1280, 720), borderValue=(90, 90, 90), flags=cv2.INTER_AREA)
        if not hide:
            img_full = img
        if hide:
            top = cv2.projectPoints(np.array([[-0.1, 0.05, 0], [0.1, 0.05, 0], [0.1, 0.1, 0], [-0.1, 0.1, 0]]),
                                    cv2.Rodrigues(R)[0], np.array(tvec), K, dist)[0].reshape(-1, 2)
            cv2.fillPoly(img, [top.astype(np.int32)], (60, 60, 60))
        res = det.detect(img)
        if res is None:
            print(f"[self-test] {name}: KHÔNG dò được bảng")
            ok_all = False
            continue
        T, n, err = res
        dpos = np.linalg.norm(T[:3, 3] - T_true[:3, 3]) * 1000
        dang = np.degrees(np.arccos(np.clip(T[:3, 2] @ T_true[:3, 2], -1, 1)))
        # 2 marker: hình học yếu hơn hẳn -> ngưỡng lỏng hơn
        ok = (dpos < 2.0 and dang < 2.5) if n >= 4 else (dpos < 10.0 and dang < 3.0)
        ok_all &= ok
        print(f"[self-test] {name}: {n} marker, tâm lệch {dpos:.2f} mm, pháp tuyến lệch {dang:.2f}°, "
              f"chiếu lại {err:.2f}px -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
        # Đầu bút so với bảng: đặt 1 điểm biết trước trong hệ bảng, đổi sang hệ
        # camera bằng pose THẬT (như node bút đo được), rồi đổi ngược về hệ bảng
        # bằng pose DÒ được — sai lệch = sai số mà người dùng sẽ thấy.
        worst = 0.0
        for pb in ([0.0, 0.0, 0.02], [0.03, -0.04, 0.05], [-0.05, 0.05, 0.0], [0.02, 0.01, 0.12]):
            p_cam = T_true[:3, :3] @ np.array(pb) + T_true[:3, 3]
            worst = max(worst, np.linalg.norm(cam_to_board(T, p_cam) - pb) * 1000)
        ok = worst < (2.0 if n >= 4 else 10.0)
        ok_all &= ok
        print(f"            đầu bút so với bảng: lệch lớn nhất {worst:.2f} mm -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")

    # BoardTracker: 30 lần dò có nhiễu quanh 1 pose -> trung bình phải gần pose
    # thật hơn từng lần lẻ; bảng bị dời 5cm -> phải bám theo ngay, không kéo lê.
    rng = np.random.default_rng(0)
    T0 = np.eye(4); T0[:3, :3] = cv2.Rodrigues(np.array([0.3, -0.4, 0.1]))[0] @ np.diag([1.0, -1, -1])
    T0[:3, 3] = [0.02, -0.01, 0.40]
    def noisy(Tb):
        Tn = Tb.copy()
        Tn[:3, :3] = cv2.Rodrigues(rng.normal(0, 0.004, 3))[0] @ Tb[:3, :3]
        Tn[:3, 3] += rng.normal(0, 0.0008, 3)
        return Tn
    trk, single = BoardTracker(), []
    for k in range(30):
        Tn = noisy(T0); single.append(np.linalg.norm(Tn[:3, 3] - T0[:3, 3]) * 1000)
        Ts = trk.update(Tn, 4, 0.3, float(k))
    e_avg = np.linalg.norm(Ts[:3, 3] - T0[:3, 3]) * 1000
    orth = np.abs(Ts[:3, :3] @ Ts[:3, :3].T - np.eye(3)).max()
    T1 = T0.copy(); T1[:3, 3] += [0.05, 0, 0]
    trk.update(T1, 4, 0.3, 31.0)
    Ts = trk.update(T1, 4, 0.3, 31.2)                   # 2 lần dò liên tiếp cùng chỗ mới -> bám
    e_jump = np.linalg.norm(Ts[:3, 3] - T1[:3, 3]) * 1000
    weak = trk.update(noisy(T1), 2, 0.3, 32.0)          # mẫu 2 marker cùng chỗ: phải bị bỏ qua
    kept = np.allclose(weak, Ts)
    ok = e_avg < np.mean(single) and orth < 1e-9 and e_jump < 0.01 and kept
    ok_all &= ok
    print(f"[self-test] BoardTracker: lệch từng lần TB {np.mean(single):.2f} mm -> sau làm mượt {e_avg:.2f} mm; "
          f"bảng dời 50mm -> bám sau 2 lần dò (lệch {e_jump:.3f} mm); bỏ mẫu 2 marker: {kept} -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    # Dời CHẬM 6 mm mỗi lần dò (dưới ngưỡng nhảy 20 mm) rồi dừng: pose phải
    # đuổi kịp trong 2 lần dò, không phải chờ hết cửa sổ trung bình.
    trk, Tm, lag = BoardTracker(), T0.copy(), []
    for k in range(12):
        trk.update(noisy(T0), 4, 0.3, float(k))
    for k in range(8):
        Tm = Tm.copy(); Tm[:3, 3] += [0.006, 0, 0]
        Ts = trk.update(noisy(Tm), 4, 0.3, 12.0 + k)
        lag.append(np.linalg.norm(Ts[:3, 3] - Tm[:3, 3]) * 1000)
    for k in range(2):
        Ts = trk.update(noisy(Tm), 4, 0.3, 20.0 + k)
    e_stop = np.linalg.norm(Ts[:3, 3] - Tm[:3, 3]) * 1000
    ok = max(lag[2:]) < 10.0 and e_stop < 2.5
    ok_all &= ok
    print(f"[self-test] BoardTracker dời chậm 6 mm/lần dò: trễ lớn nhất {max(lag[2:]):.1f} mm lúc đang dời, "
          f"{e_stop:.1f} mm sau khi dừng 2 lần dò -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")

    # Dò trong vùng quanh bảng phải cho cùng pose với dò cả khung
    det_f, det_r = BoardDetector(K, dist), BoardDetector(K, dist)
    r_full = det_f.detect(img_full)
    det_r.detect(img_full)                                   # lần 1: cả khung, bật ROI cho lần 2
    r_roi = det_r.detect(img_full, r_full[0])
    same = (r_roi is not None and det_r.last_roi is not None
            and np.linalg.norm(r_roi[0][:3, 3] - r_full[0][:3, 3]) * 1000 < 0.05)
    ok_all &= bool(same)
    print(f"[self-test] dò theo vùng {det_r.last_roi}: cùng pose với dò cả khung -> "
          f"{'ĐẠT' if same else 'KHÔNG ĐẠT'}")
    # Một lần dò hỏng lẻ loi (lệch 40 mm / 12 độ) không được kéo pose đi; mẫu
    # 3 marker lệch 6 độ lặp lại nhiều lần cũng không (dưới ngưỡng nhảy).
    trk = BoardTracker()
    for k in range(6):
        Ts0 = trk.update(noisy(T0), 4, 0.3, float(k))
    bad = T0.copy(); bad[:3, 3] += [0.04, 0, 0]
    bad[:3, :3] = cv2.Rodrigues(np.array([0.0, np.radians(12), 0.0]))[0] @ T0[:3, :3]
    one = np.allclose(trk.update(bad, 4, 0.3, 6.0), Ts0)
    Ts1 = trk.update(noisy(T0), 4, 0.3, 7.0)
    tilt6 = T0.copy(); tilt6[:3, :3] = cv2.Rodrigues(np.array([0.0, np.radians(6), 0.0]))[0] @ T0[:3, :3]
    tilt6[:3, 3] += [0.006, 0, -0.010]
    for k in range(10):
        Tw = trk.update(tilt6, 3, 0.3, 8.0 + k)
    weak_kept = np.allclose(Tw, Ts1)
    ok = one and weak_kept
    ok_all &= ok
    print(f"[self-test] BoardTracker chống nhiễu: 1 lần dò hỏng bị bỏ qua: {one}; mẫu 3 marker lệch 6°/12mm "
          f"lặp 10 lần bị bỏ qua: {weak_kept} -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    return ok_all


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if _self_test() else 1)
    print(__doc__)
