#!/usr/bin/env python3
"""
Nối perception CoVip (detect_objects.py) với bộ điều khiển IK/SAC của nhóm RL (rl_arm5) — INTEGRATION_RL.md.

- Hệ ĐẾ `assarm` (m): X phải, Y trước (hướng tay vươn tới), Z lên; gốc ở mặt bàn ngay dưới đế
  (tâm base_link cao 0,56 m). Giống `determine_depth/core/hand_eye_trans.py` và sim (`r·sinθ, r·cosθ, z`).
- `calib/T_cam_assarm.npz` (do calib_cam_assarm.py ghi): T_cam2base (p_base = R·p_cam + t, có quay vì
  camera chúc xuống) + độ cao mặt bục `plat_z`.
- Vị trí vật: tia camera cắt MẶT BỤC (objects3d.initial_guess: tiếp tuyến tới vòng đáy) -> tâm đáy (x, y),
  tâm gắp z = plat_z + chiều cao/2 (chiều cao đã biết của lớp, chọn theo chiều cao đo được).
  Đối chứng: khoảng cách theo kích thước đã biết (detect_objects.estimate) đổi sang hệ đế.
- Gửi: UDP 5555 đúng gói `VisionIPCPublisher` của determine_depth; vận tốc lọc bằng chính
  `assarm_common.perception_filter.AlphaBetaFilter3D` (như lúc train).

Đường dẫn rl_arm5: --rl-root, biến môi trường RL_ROOT, hoặc tự tìm (2026.1/rl_arm5 trên laptop, ~/quan trên Pi).

    python3 scripts/assarm_link.py --self-test
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

# Miền vị trí vật mà SAC stage_b_plus đã train (SPAWN_LEVELS[3]); thay bằng số trong assarm_common nếu nạp được
RL_DOMAIN = dict(r=(0.25, 0.33), theta_deg=45.0, z=(0.36, 0.44))
# Chiều cao có thể của mỗi lớp (m) theo các tư thế đặt; z tâm gắp = mặt bục + chiều cao/2
HEIGHTS = {"hop_nau": (0.060, 0.062, 0.100), "tru_xanh": (0.100,), "hop_pin": (0.050, 0.061, 0.160),
           "lo_nap_do": (0.081,)}
# Trụ đứng thẳng: (đường kính, cao) m — dùng cho cách không có mặt bục
CYL = {"tru_xanh": (0.050, 0.100), "lo_nap_do": (0.041, 0.081)}
# Đáy hộp khi đặt NẰM (2 cạnh lớn, m)
BOX_FOOT = {"hop_nau": (0.100, 0.062), "hop_pin": (0.160, 0.061)}


def find_rl_root(rl_root=None):
    cands = [rl_root, os.environ.get("RL_ROOT"), ROOT.parent / "rl_arm5", Path.home() / "quan",
             Path.home() / "rl_arm5"]
    for c in cands:
        if c and (Path(c) / "determine_depth" / "core" / "ipc_bridge.py").exists():
            return Path(c).resolve()
    return None


def add_rl_paths(rl_root):
    """Cho import `core.ipc_bridge`, `core.web_streamer` (determine_depth) và `assarm_common`."""
    for p in (rl_root / "determine_depth", rl_root / "ros2_ws" / "src" / "assarm_common"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from assarm_common.config import SPAWN_LEVELS
        lv = SPAWN_LEVELS[max(SPAWN_LEVELS)]
        RL_DOMAIN.update(r=tuple(lv["r"]), theta_deg=float(np.degrees(lv["theta"])), z=tuple(lv["z"]))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  Không nạp được assarm_common.config ({e}); dùng miền mặc định {RL_DOMAIN}")


def in_rl_domain(p, margin=0.0):
    """-> (True/False, lý do). p = (x, y, z) hệ đế."""
    x, y, z = p
    r = float(np.hypot(x, y))
    th = float(np.degrees(np.arctan2(x, y)))
    (r0, r1), (z0, z1), tmax = RL_DOMAIN["r"], RL_DOMAIN["z"], RL_DOMAIN["theta_deg"]
    if not (r0 - margin <= r <= r1 + margin):
        return False, f"r {r * 100:.0f} cm ngoài [{r0 * 100:.0f}, {r1 * 100:.0f}]"
    if abs(th) > tmax + np.degrees(margin / max(r, 1e-3)):
        return False, f"góc {th:+.0f}° ngoài ±{tmax:.0f}°"
    if not (z0 - margin <= z <= z1 + margin):
        return False, f"z {z * 100:.0f} cm ngoài [{z0 * 100:.0f}, {z1 * 100:.0f}]"
    return True, ""


def trans(x, y, z):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


def _poly_area(poly):
    """Diện tích đa giác (công thức shoelace). Không dùng np.cross 2D: numpy ≥ 2.x đã bỏ (Pi 5)."""
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


class HandEye:
    """Camera -> hệ đế assarm + mặt bục (objects3d.Table với hệ 'bàn' = hệ đế dời lên plat_z)."""

    def __init__(self, T_cam2base, plat_z, K, dist):
        self.K, self.dist = np.asarray(K, np.float64), np.asarray(dist, np.float64).ravel()
        """plat_z = None: không có mặt bục (cách đơn giản, đo camera bằng thước) -> vị trí theo kích thước đã biết."""
        from objects3d import Table
        self.T = np.asarray(T_cam2base, np.float64)
        self.R, self.t = self.T[:3, :3], self.T[:3, 3]
        self.plat_z = None if plat_z is None else float(plat_z)
        self.table = None
        if self.plat_z is not None:
            self.table = Table(np.linalg.inv(self.T) @ trans(0, 0, self.plat_z), K, dist)
        self._cache = {}  # tên -> (tâm mặt nạ px, diện tích, kết quả): khớp hộp tốn ~40 ms, vật đứng yên thì dùng lại

    @classmethod
    def from_ruler(cls, x, y, z, pitch_deg, K, dist, yaw_deg=0.0):
        """Camera đo bằng thước: tâm ống kính (x, y, z) m trong hệ đế, nhìn về phía trước (+Y),
        chúc xuống pitch_deg (app thước thuỷ đặt trên lưng camera), lệch trái/phải yaw_deg (+ = quay sang phải)."""
        p, yw = np.radians(pitch_deg), np.radians(yaw_deg)
        fwd = np.array([np.sin(yw) * np.cos(p), np.cos(yw) * np.cos(p), -np.sin(p)])
        right = np.array([np.cos(yw), -np.sin(yw), 0.0])
        down = np.cross(fwd, right)
        T = np.eye(4)
        T[:3, :3] = np.column_stack([right, down, fwd])  # cột = trục X, Y, Z camera trong hệ đế
        T[:3, 3] = (x, y, z)
        return cls(T, None, K, dist)

    @classmethod
    def load(cls, path, K, dist):
        d = np.load(path)
        return cls(d["T_cam2base"], float(d["plat_z"]), K, dist)

    def to_base(self, p_cam):
        return self.R @ np.asarray(p_cam, np.float64) + self.t

    def vec_to_base(self, v_cam):
        return self.R @ np.asarray(v_cam, np.float64)

    @property
    def cam_pos(self):
        return self.t.copy()

    @property
    def pitch_deg(self):
        """Góc chúc xuống của trục quang (dương = nhìn xuống)."""
        z_axis = self.R[:, 2]
        return float(np.degrees(np.arcsin(-z_axis[2])))

    def locate(self, name, poly):
        """Mặt nạ (đa giác pixel) -> dict(p=(x, y, z) tâm gắp hệ đế, h_est, h_used) hoặc None.
        Không có mặt bục -> None (dùng vị trí theo kích thước, xem detect_objects)."""
        if self.table is None:
            return None
        from objects3d import initial_guess
        poly = np.asarray(poly, np.float64).reshape(-1, 2)
        cen, area = poly.mean(0), _poly_area(poly)
        c = self._cache.get(name)
        if c is not None and np.linalg.norm(c[0] - cen) < 1.5 and abs(c[1] - area) < 0.03 * max(area, 1.0):
            return c[2]
        g = initial_guess(self.table, poly)
        if g is None:
            return None
        hs = HEIGHTS.get(name, (g["h"],))
        h = min(hs, key=lambda v: abs(v - g["h"]))
        if name in BOX_FOOT and g["r_near"] > 0:
            # hộp: công thức trụ (tiếp tuyến vòng đáy) lệch tới 10 cm. Mép trước của hộp = điểm đáy gần chân
            # camera nhất (r_near); bề ngang nhìn thấy cho biết cạnh nào quay về camera; tâm = mép trước +
            # nửa cạnh còn lại theo hướng nhìn. Giả định hộp quay mặt về camera (yaw ≈ hướng nhìn).
            w_seen = 2 * g["r_near"] * np.tan(g["half"])
            a, b = BOX_FOOT[name]
            front, depth = (a, b) if abs(a - w_seen) < abs(b - w_seen) else (b, a)
            phi = g["phi_c"]
            dirn = np.cos(phi) * g["u"] + np.sin(phi) * g["v"]
            x, y = g["foot"] + (g["r_near"] + depth / 2) * dirn
            x, y = self._fit_box(name, poly, (x, y), h)
        else:
            x, y = g["xy"]
        out = dict(p=np.array([x, y, self.plat_z + h / 2]), h_est=float(g["h"]), h_used=float(h),
                   D_est=float(g["D"]))
        self._cache[name] = (cen, area, out)
        return out


    def locate_cylinder(self, name, poly, p0):
        """Không có mặt bục: trụ ĐỨNG THẲNG (trục // Z đế), kích thước đã biết (CYL) -> khớp tâm 3D (x, y, z)
        sao cho bóng chiếu trùng mặt nạ nhất (IoU). p0: điểm khởi đầu hệ đế (theo đường kính)."""
        import cv2
        from objects3d import nelder_mead
        poly = np.asarray(poly, np.float64).reshape(-1, 2)
        cen, area = poly.mean(0), _poly_area(poly)
        c = self._cache.get(name)
        if c is not None and np.linalg.norm(c[0] - cen) < 1.5 and abs(c[1] - area) < 0.03 * max(area, 1.0):
            return c[2]
        D, H = CYL[name]
        a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring = np.column_stack([D / 2 * np.cos(a), D / 2 * np.sin(a)])
        Rinv, rvec0, tvec0 = self.R.T, np.zeros(3), np.zeros(3)
        x0, y0, bw, bh = cv2.boundingRect(np.float32(poly))
        pad = max(bw, bh)
        off = np.array([x0 - pad, y0 - pad], np.float32)
        size = (bh + 2 * pad, bw + 2 * pad)
        mask = np.zeros(size, np.uint8)
        cv2.fillPoly(mask, [np.int32(poly - off)], 1)
        area_m = float(mask.sum())

        def sil(p):
            P = np.vstack([np.column_stack([ring + p[:2], np.full(len(ring), z)]) for z in (p[2] - H / 2, p[2] + H / 2)])
            Pc = (Rinv @ (P - self.t).T).T
            if np.any(Pc[:, 2] <= 0.02):
                return None
            uv = cv2.projectPoints(Pc, rvec0, tvec0, self.K, self.dist)[0].reshape(-1, 2)
            return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)

        def cost(p):
            s_ = sil(p)
            if s_ is None:
                return 1.0
            m = np.zeros(size, np.uint8)
            cv2.fillPoly(m, [np.int32(np.round(s_ - off))], 1)
            inter = float((m & mask).sum())
            return 1.0 - inter / (area_m + float(m.sum()) - inter + 1e-9)

        p, val = nelder_mead(cost, np.asarray(p0, float), np.array([0.01, 0.01, 0.01]), iters=150, tol=1e-6)
        out = dict(p=p, iou=1.0 - float(val))
        self._cache[name] = (cen, area, out)
        return out

    def _fit_box(self, name, poly, xy0, h):
        """Khớp tâm (x, y) của hộp kích thước đã biết, nằm hướng về trục J1 (cạnh dài dọc hoặc ngang hướng
        vươn tới), sao cho bóng chiếu trùng mặt nạ nhất (IoU). 2 tham số, vài ms."""
        import cv2
        from objects3d import nelder_mead, silhouette
        poly = np.asarray(poly, np.float32).reshape(-1, 2)
        x0, y0, bw, bh = cv2.boundingRect(poly)
        pad = 40
        off = np.array([x0 - pad, y0 - pad], np.float32)
        size = (bh + 2 * pad, bw + 2 * pad)
        mask = np.zeros(size, np.uint8)
        cv2.fillPoly(mask, [np.int32(poly - off)], 1)
        area_m = float(mask.sum())
        a, b = BOX_FOOT[name]

        def cost(p, yaw_off):
            cx, cy = p
            yaw = -np.arctan2(cx, cy) + yaw_off
            s = silhouette(self.table, "box", (cx, cy, yaw, a, b, h))
            if s is None:
                return 1.0
            m = np.zeros(size, np.uint8)
            cv2.fillPoly(m, [np.int32(s - off)], 1)
            inter = float((m & mask).sum())
            return 1.0 - inter / (area_m + float(m.sum()) - inter + 1e-9)

        best = None
        for yaw_off in (0.0, np.pi / 2):
            p, c = nelder_mead(lambda q: cost(q, yaw_off), np.array(xy0, float), np.array([0.01, 0.01]),
                               iters=60, tol=1e-5)
            if best is None or c < best[1]:
                best = (p, c)
        return best[0]


def pick_target(dets, target, prev_name=None):
    """Chọn 1 vật để gửi cho tay. dets: list dict có name, conf, base (hoặc None), in_domain.
    target = tên lớp, hoặc 'auto': vật trong miền SAC, ưu tiên giữ vật đang chọn, rồi gần trục J1 nhất."""
    ok = [d for d in dets if d.get("base") is not None]
    if target != "auto":
        ok = [d for d in ok if d["name"] == target]
        return ok[0] if ok else None
    ok = [d for d in ok if d.get("in_domain")]
    if not ok:
        return None
    for d in ok:
        if d["name"] == prev_name:
            return d
    return min(ok, key=lambda d: float(np.hypot(d["base"][0], d["base"][1])))


class TargetSender:
    """Gửi 1 vật đích qua UDP 5555 (gói của determine_depth) + vận tốc lọc alpha-beta như lúc train."""

    def __init__(self, host="127.0.0.1", port=5555, lost_reset_s=0.5):
        from core.ipc_bridge import VisionIPCPublisher
        from assarm_common.perception_filter import AlphaBetaFilter3D
        self.pub = VisionIPCPublisher(host=host, port=port)
        self.filt = AlphaBetaFilter3D()
        self.lost_reset_s = lost_reset_s
        self.t_last = None
        self.pos = self.vel = None

    def send(self, p_base, t_frame):
        """p_base: (3,) hoặc None (mất vật). t_frame: lúc chụp khung (time.time())."""
        if p_base is None:
            if self.t_last is not None and t_frame - self.t_last > self.lost_reset_s:
                self.filt._pos = None  # lần thấy lại: khởi động lại bộ lọc, vận tốc = 0
            self.pub.publish(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, is_valid=False)
            return None
        if self.t_last is not None:
            self.filt.dt = float(np.clip(t_frame - self.t_last, 0.01, 0.5))
        self.t_last = t_frame
        self.pos, self.vel = self.filt.update(np.asarray(p_base, np.float64))
        self.pub.publish(*self.pos, *self.vel, is_valid=True)
        return self.pos, self.vel

    def close(self):
        self.pub.close()


def synth_setup(cam_xyz=(0.0, 0.06, 0.50), look_at=(0.0, 0.29, 0.36), plat_z=0.35):
    """Camera giả đặt trước J1, chúc xuống nhìn vùng gắp; K của C930e ở 640x360 (không méo)."""
    K = np.array([[385.8, 0, 320], [0, 384.1, 181.7], [0, 0, 1]])
    dist = np.zeros(5)
    cam = np.asarray(cam_xyz, float)
    z = np.asarray(look_at, float) - cam
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])  # cột = trục camera trong hệ đế
    T[:3, 3] = cam
    return HandEye(T, plat_z, K, dist), K, dist


def self_test():
    from objects3d import render_mask, mask_contour
    ok_all = True
    rl = find_rl_root()
    if rl is None:
        print("[self-test] ⚠️ không thấy rl_arm5 — bỏ phần UDP/bộ lọc")
    else:
        add_rl_paths(rl)
    he, K, dist = synth_setup()
    print(f"[self-test] camera giả ở {np.round(he.cam_pos, 3)} m, chúc {he.pitch_deg:.0f}°")
    ok = abs(he.pitch_deg - np.degrees(np.arctan2(0.14, 0.23))) < 0.5
    ok_all &= ok
    print(f"[self-test] góc chúc tính lại đúng -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    # vật thật trên bục ở 5 điểm, sai số vị trí tâm gắp
    cases = [("tru_xanh", "cylinder", (0.050, 0.100)), ("lo_nap_do", "cylinder", (0.041, 0.081)),
             ("hop_nau", "box", (0.100, 0.062, 0.060)), ("hop_pin", "box", (0.160, 0.061, 0.050))]
    pts = [(0.0, 0.29), (0.15, 0.25), (-0.15, 0.25), (0.0, 0.26), (0.0, 0.32)]
    for name, shape, dims in cases:
        errs = []
        t_fit = 0.0
        for k, (bx, by) in enumerate(pts):
            tb = he.table  # hệ bục: x, y như hệ đế
            if shape == "cylinder":
                p = (bx, by, dims[0], dims[1])
                h = dims[1]
            else:
                # điểm chẵn: cạnh dài nằm ngang trước tay; điểm lẻ: cạnh dài dọc hướng vươn tới
                yaw = -np.arctan2(bx, by) + (np.pi / 2 if k % 2 else 0.0)
                p = (bx, by, yaw, dims[0], dims[1], dims[2])
                h = dims[2]
            m = render_mask(tb, shape, p)
            if m.max() == 0:
                errs.append(np.inf)
                continue
            c, _ = mask_contour(m)
            t0 = time.time()
            r = he.locate(name, c)
            t_fit = max(t_fit, time.time() - t0)
            truth = np.array([bx, by, he.plat_z + h / 2])
            errs.append(np.inf if r is None else float(np.linalg.norm(r["p"] - truth)) * 1000)
        lim = 6.0 if shape == "cylinder" else 15.0
        ok = max(errs) < lim
        ok_all &= ok
        print(f"[self-test] {name:9s} 5 điểm trên bục: lệch tâm gắp {np.round(errs, 1)} mm (≤ {lim}) "
              f"{1000 * t_fit:.0f} ms -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    D = [dict(name="hop_nau", conf=0.9, base=np.array([0.1, 0.30, 0.40]), in_domain=True),
         dict(name="tru_xanh", conf=0.8, base=np.array([0.0, 0.27, 0.40]), in_domain=True),
         dict(name="lo_nap_do", conf=0.99, base=np.array([0.0, 0.10, 0.40]), in_domain=False)]
    ok = (pick_target(D, "auto")["name"] == "tru_xanh" and pick_target(D, "auto", "hop_nau")["name"] == "hop_nau"
          and pick_target(D, "lo_nap_do")["name"] == "lo_nap_do" and pick_target(D, "hop_pin") is None)
    ok_all &= ok
    print(f"[self-test] chọn vật đích (auto / giữ vật cũ / theo tên) -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    # cách đơn giản: camera đo bằng thước + góc chúc, trụ xanh, vị trí theo đường kính đã biết
    import detect_objects as dobj
    he_r = HandEye.from_ruler(0.0, 0.06, 0.50, he.pitch_deg, K, dist)
    ok = np.allclose(he_r.T, he.T, atol=1e-6)
    ok_all &= ok
    print(f"[self-test] dựng camera từ thước + góc chúc khớp camera giả -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    errs = []
    for bx, by in pts:
        m = render_mask(he.table, "cylinder", (bx, by, 0.05, 0.10))
        c, _ = mask_contour(m)
        d = dict(name="tru_xanh", poly=c, mask=m > 127)
        e = dobj.estimate(d, K, dist)
        p0 = he_r.to_base(np.array([e["x"], e["y"], e["z"]]) / 1000)
        he_r._cache.clear()
        t0 = time.time()
        pb = he_r.locate_cylinder("tru_xanh", c, p0)["p"]
        t_fit = time.time() - t0
        errs.append(float(np.linalg.norm(pb - [bx, by, he.plat_z + 0.05])) * 1000)
    ok = max(errs) < 6.0
    ok_all &= ok
    print(f"[self-test] trụ xanh không bục (khớp 3D theo kích thước): lệch {np.round(errs, 1)} mm (≤ 6), "
          f"{1000 * t_fit:.0f} ms "
          f"-> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    ok = in_rl_domain((0.0, 0.29, 0.40))[0] and not in_rl_domain((0.0, 0.29, 0.10))[0] \
        and not in_rl_domain((0.3, 0.05, 0.40))[0]
    ok_all &= ok
    print(f"[self-test] kiểm miền SAC (trong / thấp / lệch góc) -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    if rl is not None:
        from core.ipc_bridge import VisionIPCSubscriber
        port = 55871
        sub = VisionIPCSubscriber(port=port)
        tx = TargetSender(port=port)
        t0 = time.time()
        for k in range(20):  # vật chạy đều 2 cm/s theo X
            tx.send((0.02 * k * 0.1, 0.29, 0.40), t0 + 0.1 * k)
        time.sleep(0.05)
        pos, vel, valid, age = sub.get_latest()
        ok = valid and pos is not None and abs(pos[1] - 0.29) < 1e-3 and abs(vel[0] - 0.02) < 0.004 and age < 0.5
        ok_all &= ok
        print(f"[self-test] UDP 5555 qua VisionIPCSubscriber thật: pos {np.round(pos, 3)}, "
              f"v {np.round(vel, 3)} m/s -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
        tx.send(None, t0 + 2.1)
        time.sleep(0.05)
        _, _, valid, _ = sub.get_latest()
        ok = not valid
        ok_all &= ok
        print(f"[self-test] mất vật -> is_valid=False -> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
        tx.close()
        sub.close()
    return ok_all


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    print(__doc__)
