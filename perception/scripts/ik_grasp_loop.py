#!/usr/bin/env python3
"""ik_grasp_loop.py — gắp trụ bằng tracker IK của nhóm RL (baseline_ik_tracker.py), có khoá đích và về home.

Không sửa mã RL: nạp BaselineVisualTracker + ESP32HardwareBridge từ rl_arm5/sodonguyenly_calibservo
(Pi: ~/quan/sodonguyenly_calibservo). Vị trí vật nhận qua UDP 5555 từ detect_objects.py --udp.

Mặc định (--after hold):
    HOME → tư thế chờ → KHOÁ ĐÍCH: thấy trụ liên tục, đứng yên (lệch ≤ --lock-tol) đủ --lock-time giây
         → TRACKING → CLOSE → LIFT → GIỮ NGUYÊN tư thế, ngàm vẫn kẹp (không tự mở)
    Ctrl+C (bất cứ lúc nào) → MỞ ngàm → về HOME mượt → thoát. Ctrl+C lần 2: dừng ngay.
    FAILED (bám > 25 s không tới) → mở ngàm → HOME → khoá đích lại từ đầu.

--after place: chạy liên tục để đo tỉ lệ: sau khi nhấc thì đặt vật về chỗ cũ, mở ngàm, nhấc ngàm,
    về HOME, nghỉ --rest giây, rồi gắp lần nữa (--cycles N để dừng sau N lần).

Chạy (Pi):
    ~/venvs/arm/bin/python ~/aeroscript/scripts/ik_grasp_loop.py --port /dev/ttyUSB0 --handoff-dist 0.015
    ... --lock-time 5 --lock-tol 0.03     # mặc định: trụ đứng yên 5 s (lệch ≤ 3 cm) mới gắp
    ... --mock                            # không cần tay
    ... --after place --cycles 10         # 10 lần gắp-đặt-lại liên tục
    ... --home 90 70 58 90                # home = tư thế chờ thay vì 90 90 90 90 (firmware)
"""
import argparse
import csv
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def find_tracker_dir(arg):
    cands = [Path(arg)] if arg else []
    cands += [HERE.parent.parent / "rl_arm5" / "sodonguyenly_calibservo",    # máy dev: 2026.1/CoVip/scripts
              Path.home() / "quan" / "sodonguyenly_calibservo"]              # Pi
    for c in cands:
        if (c / "baseline_ik_tracker.py").is_file():
            return c
    raise SystemExit(f"Không thấy baseline_ik_tracker.py trong {[str(c) for c in cands]}; dùng --tracker-dir")


class TargetLock:
    """Đích phải hợp lệ liên tục và đứng yên (lệch ≤ tol so với trung bình) đủ hold_s giây."""

    def __init__(self, hold_s, tol, gap_s=0.5):
        self.hold_s, self.tol, self.gap_s = hold_s, tol, gap_s
        self.why = None
        self.reset()

    def reset(self, why=None):
        if why and self.t0 is not None:
            self.why = why                     # lý do đếm lại, vòng in sẽ báo
        self.t0 = self.t_last = None
        self.pts = []

    def update(self, p, now):
        """p: điểm hợp lệ hoặc None. Trả (đã_khoá, số giây đã giữ)."""
        if p is None:
            if self.t_last is not None and now - self.t_last > self.gap_s:
                self.reset(f"mất trụ > {self.gap_s:.1f} s")
            return False, 0.0 if self.t0 is None else now - self.t0
        if self.pts:
            d = float(np.linalg.norm(p - np.mean(self.pts, 0)))
            if d > self.tol:
                self.reset(f"trụ dời {d * 100:.1f} cm > {self.tol * 100:.0f} cm")   # vật bị dời -> đếm lại
        if self.t0 is None:
            self.t0 = now
        self.t_last = now
        self.pts.append(np.asarray(p, float))
        held = now - self.t0
        return held >= self.hold_s, held

    @property
    def mean(self):
        return np.mean(self.pts, 0) if self.pts else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--mock", action="store_true", help="giả lập tay (không cần ESP32)")
    ap.add_argument("--z-offset", type=float, default=0.0, help="bù z đích (m), như tracker gốc")
    ap.add_argument("--grip-angle", type=float, default=90.0, help="góc đóng ngàm (độ)")
    ap.add_argument("--handoff-dist", type=float, default=0.015, help="đóng ngàm khi P_err ≤ (m)")
    ap.add_argument("--lock-time", type=float, default=5.0, help="trụ phải đứng yên bao lâu (s) mới bắt đầu gắp")
    ap.add_argument("--lock-tol", type=float, default=0.03, help="'đứng yên' = lệch ≤ (m) so với trung bình")
    ap.add_argument("--home", type=float, nargs=4, default=(90, 90, 90, 90), metavar=("S1", "S2", "S3", "S4"),
                    help="tư thế home (độ servo); mặc định 90 90 90 90 = home firmware")
    ap.add_argument("--home-time", type=float, default=2.5, help="thời gian đi về/ra khỏi home (s)")
    ap.add_argument("--after", choices=("hold", "place"), default="hold",
                    help="hold = giữ vật, giữ tư thế tới khi Ctrl+C; place = đặt lại, về home, gắp tiếp")
    ap.add_argument("--rest", type=float, default=3.0, help="(--after place) nghỉ ở home giữa 2 lần (s)")
    ap.add_argument("--cycles", type=int, default=0, help="(--after place) số lần gắp (0 = tới Ctrl+C)")
    ap.add_argument("--tracker-dir", default=None, help="thư mục chứa baseline_ik_tracker.py")
    ap.add_argument("--log", default=None, help="CSV tóm tắt từng lần gắp (mặc định logs/grasp_loop_<giờ>.csv)")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)   # in ngay từng dòng (kể cả qua ssh/pipe)
    except AttributeError:
        pass
    print("ik_grasp_loop: khởi động, nạp mã RL...")

    tdir = find_tracker_dir(a.tracker_dir)
    sys.path.insert(0, str(tdir))
    import baseline_ik_tracker as bt  # noqa: E402  (tự thêm determine_depth vào sys.path)
    if not bt.HAS_IPC:
        raise SystemExit(f"Không import được core.ipc_bridge (determine_depth cạnh {tdir})")

    bt.pin_to_cpu_core_0()
    S = bt.GraspState
    dt = bt.DT_CTRL
    q_home = np.array([bt.deg_to_rad_arm(v) for v in a.home])
    g_open, g_close = bt.Q_GRIPPER_OPEN, np.radians(a.grip_angle)
    deg = lambda q: [int(round(bt.rad_to_deg_arm(v))) for v in q]  # noqa: E731

    bridge = bt.ESP32HardwareBridge(port=a.port, baudrate=a.baud, mock=a.mock)
    vision = bt.VisionIPCSubscriber(host="127.0.0.1", port=5555)
    ctl = bt.BaselineVisualTracker(dt=dt, grip_angle=a.grip_angle, handoff_dist=a.handoff_dist)
    lock = TargetLock(a.lock_time, a.lock_tol)
    tele = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    log_path = Path(a.log) if a.log else HERE.parent / "logs" / f"grasp_loop_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(logf)
    log.writerow(["cycle", "result", "t_lock_to_end_s", "lock_x", "lock_y", "lock_z", "tx", "ty", "tz",
                  "p_err_handoff_cm", "q_grasp_deg"])

    st = dict(q=q_home.copy(), g=g_open)   # lệnh đang gửi (vòng hở: chính là vị trí tay)

    def send(q, g):
        st["q"], st["g"] = np.asarray(q, float).copy(), float(g)
        return bridge.send_and_receive(st["q"], st["g"])

    def tick(t0):
        time.sleep(max(0.0, dt - (time.time() - t0)))

    def move(q_goal, g, duration, label):
        """Đi mượt (S-curve) từ lệnh hiện tại tới q_goal, giữ ngàm g."""
        print(f"   -> {label} ({duration:.1f} s): {deg(q_goal)}°")
        for q in bt.generate_scurve_trajectory(st["q"], np.asarray(q_goal, float), duration=duration, dt=dt):
            t0 = time.time()
            send(q, g)
            tick(t0)

    def wait(sec, g=None):
        t_end = time.time() + sec
        while time.time() < t_end:
            t0 = time.time()
            send(st["q"], st["g"] if g is None else g)
            tick(t0)

    def release_and_home():
        if st["g"] != g_open:
            print("   -> mở ngàm")
            wait(1.0, g_open)
        move(q_home, g_open, a.home_time, "HOME")

    print("\n" + "=" * 70)
    print(" IK GẮP TRỤ (bộ điều khiển baseline_ik_tracker của RL)")
    print(f" - Khoá đích: đứng yên {a.lock_time:.1f} s (lệch ≤ {a.lock_tol * 100:.0f} cm) | đóng ngàm khi "
          f"P_err ≤ {a.handoff_dist * 100:.1f} cm | bù z {a.z_offset * 100:+.1f} cm | ngàm {a.grip_angle:.0f}°")
    print(f" - Sau khi nhấc: {'GIỮ NGUYÊN tới Ctrl+C' if a.after == 'hold' else 'đặt lại, về home, gắp tiếp'}"
          f" | home {list(a.home)}° | log {log_path}")
    print(" - Ctrl+C: mở ngàm, về home rồi thoát")
    print("=" * 70)

    phase = lambda n, txt: print(f"\n[{n}/5] {txt}")  # noqa: E731

    phase(1, "VỀ HOME (tư thế lúc bật không biết trước, firmware tự ramp)")
    send(q_home, g_open)        # tư thế lúc bật không biết trước: gửi thẳng (firmware tự ramp), đợi tới nơi
    wait(a.home_time)

    n_ok = n_fail = 0
    cycle = 0
    try:
        while a.cycles == 0 or cycle < a.cycles:
            cycle += 1
            print(f"\n=== Lần gắp {cycle} ===")
            if cycle > 1:
                wait(a.rest, g_open)
            phase(2, "RA TƯ THẾ CHỜ (ngàm mở)")
            move(ctl.q_ready, g_open, a.home_time, "tư thế chờ")
            phase(3, f"KHOÁ ĐÍCH: cần thấy trụ đứng yên {a.lock_time:.0f} s (lệch ≤ {a.lock_tol * 100:.0f} cm)")
            ctl.start(initial_q=st["q"])
            ctl.q_cmd = st["q"].copy()
            lock.reset()
            vision.get_latest()              # bỏ gói cũ

            locked, t_lock, lock_p = False, None, None
            handoff_err = float("nan")
            k, t_print = 0, 0.0
            while True:
                t0 = time.time()
                pos, _vel, valid, age = vision.get_latest()
                ok = bool(valid and pos and age < 0.6)
                tgt = None
                if ok:
                    tgt = np.array(pos, float)
                    tgt[2] += a.z_offset

                if not locked:                       # chỉ đứng ở tư thế chờ, đếm thời gian khoá
                    locked, held = lock.update(tgt, t0)
                    send(ctl.q_ready, g_open)
                    if lock.why:
                        print(f"   ↺ đếm lại: {lock.why}")
                        lock.why = None
                    if t0 - t_print >= 0.5:
                        t_print = t0
                        m = lock.mean
                        if age > 2.0:
                            where = f"⚠️ không nhận gói UDP từ detector ({'chưa có gói nào' if age >= 999 else f'mất {age:.0f} s'}) — Terminal 1 có chạy --udp không?"
                        elif m is None:
                            where = "detector đang chạy nhưng chưa thấy trụ"
                        else:
                            bar = "#" * int(10 * held / a.lock_time)
                            where = f"[{bar:<10}] trụ ({m[0] * 100:+.1f}, {m[1] * 100:.1f}, {m[2] * 100:.1f}) cm"
                        print(f"   KHOÁ ĐÍCH {held:3.1f}/{a.lock_time:.0f} s {where}")
                    if locked:
                        lock_p, t_lock = lock.mean, time.time()
                        print(f"   🔒 Đã khoá đích ({lock_p[0] * 100:+.1f}, {lock_p[1] * 100:.1f}, {lock_p[2] * 100:.1f}) cm")
                        phase(4, f"BÁM TỚI TRỤ (tối đa 8°/bước, đóng ngàm khi P_err ≤ {a.handoff_dist * 100:.1f} cm "
                                 f"2 bước liền; quá 25 s = thất bại)")
                    tick(t0)
                    continue

                prev = ctl.state
                p_before, _ = bt.compute_p_close(st["q"])       # đúng cái tracker dùng để quyết đóng ngàm
                q_cmd, g_cmd, state = ctl.step(tgt, ok, st["q"], st["g"])
                q_fb, _ = send(q_cmd, g_cmd)
                p_c, _ = bt.compute_p_close(q_fb)
                err = float(np.linalg.norm(ctl.target_pos - p_c))
                if prev == S.TRACKING and state == S.CLOSE:
                    handoff_err = float(np.linalg.norm(ctl.target_pos - p_before))
                k += 1
                if state != prev:
                    msg = {S.TRACKING: "bắt đầu bám", S.CLOSE: f"tới nơi (P_err {handoff_err * 100:.1f} cm) -> ĐÓNG NGÀM 1,2 s",
                           S.VERIFY: "kiểm tra kẹp", S.LIFT: "NHẤC lên (+8° vai, −8° khuỷu, 1,5 s)",
                           S.HOLD: "giữ 3 s", S.COMPLETED: "xong", S.FAILED: "quá 25 s chưa tới"}.get(state, "")
                    if state == S.CLOSE:
                        phase(5, "GẮP")
                    print(f"   ▶ {prev.name} -> {state.name}: {msg}")
                if k % 5 == 0 or state != prev:
                    tp = ctl.target_pos
                    print(f" [{time.time() - t_lock:5.1f}s] {state.name:<11} | P_err {err * 100:4.1f} cm | "
                          f"đích ({tp[0]:+.2f}, {tp[1]:.2f}, {tp[2]:.2f}) m {'[OK]' if ok else '[--]'} | "
                          f"Cmd {deg(q_cmd)}° ngàm {int(round(bt.rad_to_deg_gripper(g_cmd)))}°")
                try:
                    tele.sendto(f'{{"state":"{state.name}","cycle":{cycle}}}'.encode(), ("127.0.0.1", 5556))
                except OSError:
                    pass
                if state in (S.COMPLETED, S.FAILED):
                    break
                tick(t0)

            q_grasp = ctl.initial_grasp_q.copy()
            tp = ctl.target_pos
            log.writerow([cycle, state.name, f"{time.time() - t_lock:.1f}",
                          *(f"{v:.4f}" for v in lock_p), *(f"{v:.4f}" for v in tp),
                          f"{handoff_err * 100:.1f}", " ".join(str(v) for v in deg(q_grasp))])
            logf.flush()

            if state == S.FAILED:
                n_fail += 1
                print(f">> Lần {cycle}: THẤT BẠI (bám {ctl.state_timer:.0f} s chưa tới đích) -> mở ngàm, về home, thử lại")
                release_and_home()
                continue

            n_ok += 1
            print(f">> Lần {cycle}: GẮP XONG (đóng ngàm ở P_err {handoff_err * 100:.1f} cm)")
            if a.after == "hold":
                print("   GIỮ NGUYÊN tư thế + ngàm kẹp. Ctrl+C để MỞ ngàm và về home.")
                t_note = time.time()
                while True:                              # chỉ thoát bằng Ctrl+C
                    t0 = time.time()
                    send(st["q"], g_close)
                    if t0 - t_note >= 10.0:
                        t_note = t0
                        print(f"   ... đang giữ vật ({deg(st['q'])}°), Ctrl+C để thả + về home")
                    tick(t0)
            # --after place
            move(q_grasp, g_close, 1.5, "hạ vật về chỗ cũ")
            wait(0.8, g_open)
            lift = q_grasp.copy()
            lift[1] += np.radians(8.0)
            lift[2] -= np.radians(8.0)
            move(lift, g_open, 1.2, "nhấc ngàm khỏi vật")
            move(q_home, g_open, a.home_time, "HOME")
            print(f"   Tổng: {n_ok} thành công / {n_ok + n_fail} lần")
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Mở ngàm, về home...  (Ctrl+C lần nữa = dừng ngay)")
        try:
            release_and_home()
        except KeyboardInterrupt:
            print("[Ctrl+C lần 2] Dừng ngay, không về home.")
    finally:
        print(f"\nKết quả: {n_ok}/{n_ok + n_fail} lần gắp thành công. Log: {log_path}")
        logf.close()
        vision.close()
        bridge.close()


if __name__ == "__main__":
    main()
