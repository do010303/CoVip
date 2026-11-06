#!/usr/bin/env python3
"""
Hiệu chỉnh camera -> hệ đế tay `assarm` (camera đặt trước J1, chúc xuống) bằng tờ A4 ArUco ID 40–48
(`calib/table_sheet_A4.pdf`, in 100 %, marker 30 mm, lưới 60 mm) đặt PHẲNG trên mặt bục. Ghi
`calib/T_cam_assarm.npz` (T_cam2base + độ cao mặt bục) cho detect_objects.py --hand-eye.

Đặt tờ (đứng sau tay, nhìn về phía trước như camera):
  - chữ trên tờ đọc xuôi: mép TRÊN của tờ hướng ra xa tay (+Y đế), mép PHẢI sang phải (+X đế);
    xoay khác thì khai --sheet-yaw (độ, quanh trục Z đế, ngược chiều kim đồng hồ nhìn từ trên).
  - đo bằng thước TÂM marker giữa (ID 44) trong hệ đế: X = lệch ngang so với trục J1 (phải +),
    Y = khoảng cách ra phía trước tính từ trục J1, rồi khai --sheet-xy X Y (m).
  - --plat-z: độ cao mặt bục so với mặt bàn đặt chân tay (m); tâm base_link cao 0,56 m.

    python3 scripts/calib_cam_assarm.py --sheet-xy 0 0.29 --plat-z 0.35            # camera C930e
    python3 scripts/calib_cam_assarm.py --sheet-xy 0 0.29 --plat-z 0.35 --image anh.jpg
    python3 scripts/calib_cam_assarm.py --check     # xem lại: vẽ trục đế + vùng SAC lên ảnh, bỏ tờ ra rồi đặt vật
    python3 scripts/calib_cam_assarm.py --self-test

Không cần màn hình: thêm --no-show (in kết quả ra terminal, lưu ảnh kiểm vào logs/).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from assarm_link import RL_DOMAIN, HandEye, rot_z, trans  # noqa: E402
from objects3d import SheetDetector, average_poses, load_camera_calib  # noqa: E402

OUT = ROOT / "calib" / "T_cam_assarm.npz"


def solve(T_sheet2cam, sheet_xy, plat_z, sheet_yaw_deg):
    T_sheet2base = trans(sheet_xy[0], sheet_xy[1], plat_z) @ rot_z(sheet_yaw_deg)
    return T_sheet2base @ np.linalg.inv(T_sheet2cam)


def collect(src, det, n, timeout_s=15.0):
    """Lấy n pose tờ từ nguồn ảnh -> (T trung bình, sai số chiếu lại trung vị px, số khung, khung cuối)."""
    Ts, errs, last, t0 = [], [], None, time.time()
    while len(Ts) < n and time.time() - t0 < timeout_s:
        frame = src()
        if frame is None:
            break
        last = frame
        r = det.detect(frame)
        if r is None:
            continue
        T, seen, err = r
        if len(seen) >= 6:
            Ts.append(T)
            errs.append(err)
    if not Ts:
        return None
    return average_poses(Ts), float(np.median(errs)), len(Ts), last


def draw_check(frame, he):
    """Vẽ trục đế (X đỏ, Y xanh lá) trên mặt bục và vòng miền SAC để kiểm bằng mắt."""
    view = frame.copy()
    tb = he.table
    (r0, r1), tmax = RL_DOMAIN["r"], np.radians(RL_DOMAIN["theta_deg"])
    for r in (r0, r1):
        a = np.linspace(-tmax, tmax, 40)
        P = np.column_stack([r * np.sin(a), r * np.cos(a), np.zeros_like(a)])
        uv = tb.project(P)
        cv2.polylines(view, [np.int32(uv)], False, (0, 200, 255), 1)
    for a in (-tmax, tmax):
        P = np.array([[r0 * np.sin(a), r0 * np.cos(a), 0], [r1 * np.sin(a), r1 * np.cos(a), 0]])
        cv2.polylines(view, [np.int32(tb.project(P))], False, (0, 200, 255), 1)
    o = np.array([[0, 0.29, 0], [0.05, 0.29, 0], [0, 0.34, 0]])
    uv = np.int32(tb.project(o))
    cv2.arrowedLine(view, tuple(uv[0]), tuple(uv[1]), (0, 0, 255), 2)
    cv2.arrowedLine(view, tuple(uv[0]), tuple(uv[2]), (0, 255, 0), 2)
    for x in (-0.1, 0.0, 0.1):
        for y in (0.25, 0.29, 0.33):
            u, v = np.int32(tb.project(np.array([[x, y, 0]]))[0])
            cv2.drawMarker(view, (int(u), int(v)), (255, 255, 255), cv2.MARKER_CROSS, 8, 1)
    cv2.putText(view, "buc: X do, Y xanh (5 cm) tai (0, 29 cm); vang = mien SAC; + luoi 10x4 cm",
                (8, 18), 0, 0.45, (0, 255, 255), 1)
    return view


def self_test():
    from assarm_link import synth_setup
    from objects3d import synth_sheet_image
    ok_all = True
    for yaw in (0.0, 90.0):
        he, K, dist = synth_setup()
        T_sheet2plat = trans(0.02, 0.30, 0) @ rot_z(yaw)
        img = synth_sheet_image(he.table, T_sheet2table=T_sheet2plat)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        r = collect(lambda: img, SheetDetector(K, dist), 3, timeout_s=2)
        if r is None:
            print(f"[self-test] yaw {yaw:.0f}°: không thấy tờ -> KHÔNG ĐẠT")
            ok_all = False
            continue
        T_sc, err, _, _ = r
        T = solve(T_sc, (0.02, 0.30), he.plat_z, yaw)
        dt = np.linalg.norm(T[:3, 3] - he.T[:3, 3]) * 1000
        dR = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ he.T[:3, :3]) - 1) / 2, -1, 1)))
        ok = dt < 3.0 and dR < 0.5
        ok_all &= ok
        print(f"[self-test] tờ xoay {yaw:.0f}°: camera lệch {dt:.1f} mm, {dR:.2f}° (chiếu lại {err:.2f} px) "
              f"-> {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    return ok_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet-xy", type=float, nargs=2, metavar=("X", "Y"), help="tâm ID 44 trong hệ đế (m)")
    ap.add_argument("--plat-z", type=float, help="độ cao mặt bục so với mặt bàn (m)")
    ap.add_argument("--sheet-yaw", type=float, default=0.0)
    ap.add_argument("--image", default=None, help="ảnh tĩnh thay cho camera")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--focus", type=int, default=0)
    ap.add_argument("--calib", default=str(ROOT / "calib" / "c930e_720p.npz"))
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true", help="chỉ vẽ kiểm bằng file --out đã có")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return 0 if self_test() else 1

    K, dist = load_camera_calib(a.calib, a.width, a.height)
    cam = None
    if a.image:
        img = cv2.resize(cv2.imread(a.image), (a.width, a.height))
        src = lambda: img  # noqa: E731
    else:
        from detect_objects import Camera
        cam = Camera(a.device, a.width, a.height, a.focus)
        cam.start()
        time.sleep(1.0)

        def src():
            f, _ = cam.latest()
            time.sleep(0.03)
            return f
    try:
        if a.check:
            he = HandEye.load(a.out, K, dist)
            print(f"Camera ở {np.round(he.cam_pos * 100, 1)} cm (hệ đế), chúc {he.pitch_deg:.1f}°, bục z = {he.plat_z:.3f} m")
            while True:
                view = draw_check(src(), he)
                if a.no_show:
                    p = ROOT / "logs" / f"calib_check_{time.strftime('%H%M%S')}.jpg"
                    p.parent.mkdir(exist_ok=True)
                    cv2.imwrite(str(p), view)
                    print(f"Ảnh kiểm: {p}")
                    break
                cv2.imshow("calib_cam_assarm --check", view)
                if (cv2.waitKey(30) & 0xFF) in (27, ord("q")):
                    break
            return 0
        if a.sheet_xy is None or a.plat_z is None:
            raise SystemExit("Cần --sheet-xy X Y và --plat-z (m). Xem --help.")
        r = collect(src, SheetDetector(K, dist), a.frames)
        if r is None:
            raise SystemExit("Không thấy tờ ArUco (cần ≥ 6 marker trong khung). Kiểm ánh sáng, focus, tờ trong tầm nhìn.")
        T_sc, err, n, last = r
        T = solve(T_sc, a.sheet_xy, a.plat_z, a.sheet_yaw)
        he = HandEye(T, a.plat_z, K, dist)
        np.savez(a.out, T_cam2base=T, plat_z=a.plat_z, sheet_xy=np.array(a.sheet_xy), sheet_yaw=a.sheet_yaw,
                 T_sheet2cam=T_sc, K=K, dist=dist, image_size=np.array([a.width, a.height]),
                 reprojection_error=err, n_frames=n, stamp=time.time())
        c = he.cam_pos
        print(f"Đã lưu {a.out}  ({n} khung, chiếu lại {err:.2f} px)")
        print(f"Camera trong hệ đế: X {c[0] * 100:+.1f} cm, Y {c[1] * 100:+.1f} cm, Z {c[2] * 100:.1f} cm; "
              f"chúc xuống {he.pitch_deg:.1f}°")
        warn = []
        if err > 1.0:
            warn.append("sai số chiếu lại > 1 px: tờ cong/nhoè?")
        if not (0.3 < c[2] < 0.7):
            warn.append("độ cao camera lạ — kiểm --plat-z")
        if c[1] > a.sheet_xy[1]:
            warn.append("camera nằm TRƯỚC tờ — kiểm hướng tờ (--sheet-yaw) hoặc dấu của Y")
        for w in warn:
            print(f"⚠️  {w}")
        print("So bằng thước vị trí camera ở trên. Rồi kiểm: python3 scripts/calib_cam_assarm.py --check")
        view = draw_check(last, he)
        p = ROOT / "logs" / f"calib_assarm_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        p.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(p), view)
        print(f"Ảnh kiểm: {p}")
        if not a.no_show:
            cv2.imshow("calib_cam_assarm", view)
            cv2.waitKey(0)
    finally:
        if cam:
            cam.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
