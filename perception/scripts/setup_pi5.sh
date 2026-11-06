#!/usr/bin/env bash
# setup_pi5.sh — cài môi trường chạy flow detect + IK tay trên Raspberry Pi 5 (chạy 1 lần, TRÊN Pi).
#
#   bash ~/aeroscript/scripts/setup_pi5.sh
#
# Làm: kiểm OS/Python, cài gói apt (cần sudo), tạo venv ~/venvs/arm, pip numpy/opencv/onnxruntime/
# pyyaml/pyserial, thêm user vào nhóm dialout (cổng ESP32), liệt kê camera + cổng serial, đo tốc độ model.
# Chạy lại an toàn (bước nào đã xong thì bỏ qua).
set -uo pipefail

VENV="${VENV:-$HOME/venvs/arm}"
# Thư mục perception = cha của scripts/ (~/aeroscript khi đẩy bằng deploy_to_pi.sh, hoặc <repo>/perception khi clone).
AERO="${AERO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Mã robot/RL: <repo>/rl_arm5 nếu clone repo CoVip, không thì ~/quan.
if [ -z "${QUAN:-}" ]; then
    if [ -d "$AERO/../rl_arm5/determine_depth" ]; then QUAN="$(cd "$AERO/../rl_arm5" && pwd)"; else QUAN="$HOME/quan"; fi
fi
echo "perception: $AERO | robot/RL: $QUAN"

echo "== 1. Hệ thống =="
uname -m
grep -E '^(PRETTY_NAME)=' /etc/os-release || true
grep -m1 -i "model" /proc/cpuinfo || cat /proc/device-tree/model 2>/dev/null; echo
python3 --version
command -v vcgencmd >/dev/null && vcgencmd measure_temp

echo; echo "== 2. Gói apt (hỏi mật khẩu sudo) =="
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-pip python3-dev v4l-utils libgl1 libglib2.0-0

echo; echo "== 3. Venv $VENV =="
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q numpy opencv-python onnxruntime pyyaml pyserial
"$VENV/bin/python" - <<'EOF'
import numpy, cv2, onnxruntime, yaml, serial
print("numpy", numpy.__version__, "| opencv", cv2.__version__, "| onnxruntime", onnxruntime.__version__,
      "| aruco", hasattr(cv2, "aruco"), "| pyserial", serial.__version__)
EOF

echo; echo "== 4. Quyền cổng serial =="
if id -nG "$USER" | grep -qw dialout; then echo "đã ở nhóm dialout"
else sudo usermod -aG dialout "$USER"; echo "đã thêm vào dialout -> ĐĂNG XUẤT/ĐĂNG NHẬP LẠI (hoặc reboot) rồi mới mở được /dev/ttyUSB0"; fi

echo; echo "== 5. Thiết bị =="
v4l2-ctl --list-devices 2>/dev/null || ls /dev/video*
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "chưa thấy ESP32 (/dev/ttyUSB*): cắm cáp USB ESP32 vào Pi"

echo; echo "== 6. File đã đẩy lên =="
for f in "$AERO/scripts/detect_objects.py" "$AERO/scripts/assarm_link.py" "$AERO/weights/objects4_v1_640.onnx" \
         "$AERO/weights/objects4_v1_320.onnx" "$AERO/calib/c930e_720p.npz" \
         "$QUAN/sodonguyenly_calibservo/baseline_ik_tracker.py" "$QUAN/determine_depth/core/ipc_bridge.py" \
         "$QUAN/ros2_ws/src/assarm_common/assarm_common/perception_filter.py"; do
    [ -f "$f" ] && echo "  có    $f" || echo "  THIẾU $f"
done

echo; echo "== 7. Tốc độ detector (ảnh ngẫu nhiên, 3 luồng) =="
for m in 320 640; do
    "$VENV/bin/python" "$AERO/scripts/detect_objects.py" --bench --bench-n 30 --threads 3 \
        --model "$AERO/weights/objects4_v1_${m}.onnx" 2>&1 | tail -1
done
command -v vcgencmd >/dev/null && vcgencmd measure_temp && vcgencmd get_throttled

echo; echo "XONG. Gửi toàn bộ output này cho Claude."
