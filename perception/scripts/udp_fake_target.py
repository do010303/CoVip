#!/usr/bin/env python3
"""
Gửi 1 vật đích GIẢ qua UDP 5555 (đúng gói của determine_depth) để thử tay + bộ điều khiển IK/SAC của nhóm RL
KHÔNG cần camera (bước S3, INTEGRATION_RL.md): xác nhận chiều hệ đế, firmware ESP32, model trên Pi.

    python3 scripts/udp_fake_target.py --xyz 0 0.29 0.40             # điểm cố định, 30 Hz
    python3 scripts/udp_fake_target.py --xyz 0.10 0.27 0.40 --seconds 20
    python3 scripts/udp_fake_target.py --xyz 0 0.29 0.40 --blink 3   # mất vật 1 s mỗi 3 s (thử is_valid=False)

Rồi ở terminal khác (trong ~/quan/sodonguyenly_calibservo):
    python3 rl_deploy_tracker.py --model ~/quan/models/stage_b_plus_final.zip --mock     # chưa nối tay
    python3 rl_deploy_tracker.py --model ~/quan/models/stage_b_plus_final.zip --port /dev/ttyUSB0
Hệ đế (m): X phải, Y trước (hướng tay vươn tới), Z lên từ mặt bàn; miền SAC đã train: r 0,25–0,33, ±45°, z 0,36–0,44.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assarm_link import add_rl_paths, find_rl_root, in_rl_domain  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xyz", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = tới khi Ctrl+C")
    ap.add_argument("--blink", type=float, default=0.0, help="mỗi N giây mất vật 1 s")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--rl-root", default=None)
    a = ap.parse_args()
    rl = find_rl_root(a.rl_root)
    if rl is None:
        raise SystemExit("Không thấy rl_arm5 (determine_depth/core/ipc_bridge.py). Dùng --rl-root ~/quan")
    add_rl_paths(rl)
    from core.ipc_bridge import VisionIPCPublisher
    ok, why = in_rl_domain(a.xyz)
    if not ok:
        print(f"⚠️  Điểm ngoài miền SAC đã train ({why}) — policy có thể làm bậy; IK thì vẫn thử được.")
    pub = VisionIPCPublisher(port=a.port)
    print(f"Gửi đích ({a.xyz[0]:+.3f}, {a.xyz[1]:.3f}, {a.xyz[2]:.3f}) m tới 127.0.0.1:{a.port}, {a.hz:.0f} Hz. Ctrl+C dừng.")
    t0 = time.time()
    try:
        while not a.seconds or time.time() - t0 < a.seconds:
            el = time.time() - t0
            valid = not (a.blink and (el % a.blink) > a.blink - 1.0)
            pub.publish(*a.xyz, 0.0, 0.0, 0.0, is_valid=valid)
            time.sleep(1.0 / a.hz)
    except KeyboardInterrupt:
        pass
    finally:
        pub.publish(0, 0, 0, 0, 0, 0, is_valid=False)
        pub.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
