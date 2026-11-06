#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preview_dynamic_target_gazebo.py
Xem trước chuyển động của vật thể trong Gazebo (Stage B+: 1.2 - 2.0 cm/s).

- Tự động phát hiện world đang mở (pendulum_grasp_world hoặc empty).
- Tự động spawn khối trụ tròn màu xanh (d=5cm, h=10cm) nếu chưa có trong Gazebo!
- Di chuyển vật thể ở vận tốc chuẩn xác 1.2 - 2.0 cm/s trong vùng nhìn của Camera.
"""

import subprocess
import time
import argparse
import numpy as np

def detect_running_world() -> str:
    """Tự động kiểm tra xem world nào đang mở trong Gazebo."""
    try:
        res = subprocess.run(
            ['ign', 'service', '-l'],
            capture_output=True, text=True, timeout=2.0
        )
        lines = res.stdout.splitlines()
        for line in lines:
            if '/world/pendulum_grasp_world/' in line:
                return 'pendulum_grasp_world'
            elif '/world/empty/' in line:
                return 'empty'
    except Exception:
        pass
    return 'empty'

def spawn_cylinder_if_needed(world_name: str, model_name: str, x: float, y: float, z: float):
    """Tự động tạo một khối trụ tròn màu xanh dương nổi bật trong Gazebo nếu chưa có."""
    sdf_str = f"""
    <sdf version="1.6">
      <model name="{model_name}">
        <static>true</static>
        <pose>{x} {y} {z} 0 0 0</pose>
        <link name="link">
          <visual name="visual_cylinder">
            <geometry>
              <cylinder>
                <radius>0.025</radius>
                <length>0.10</length>
              </cylinder>
            </geometry>
            <material>
              <ambient>0.0 0.8 1.0 1.0</ambient>
              <diffuse>0.0 0.8 1.0 1.0</diffuse>
              <specular>0.5 0.5 0.5 1.0</specular>
            </material>
          </visual>
        </link>
      </model>
    </sdf>
    """
    escaped_sdf = sdf_str.replace('\n', ' ').replace('"', '\\"')
    req = f'sdf: "{escaped_sdf}"'
    subprocess.run(
        ['ign', 'service', '-s', f'/world/{world_name}/create',
         '--reqtype', 'ignition.msgs.EntityFactory',
         '--reptype', 'ignition.msgs.Boolean',
         '--req', req, '--timeout', '1000'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

def main():
    parser = argparse.ArgumentParser(description="Xem trước chuyển động ngẫu nhiên động của vật trong Gazebo")
    parser.add_argument("--speed", type=float, default=0.018, help="Vận tốc đỉnh mong muốn (m/s). Mặc định 0.018 m/s (1.8 cm/s)")
    parser.add_argument("--world", type=str, default=None, help="Tên world trong Gazebo (tự động phát hiện nếu để trống)")
    parser.add_argument("--hz", type=float, default=20.0, help="Tần số cập nhật vị trí (Hz)")
    parser.add_argument("--arm_len", type=float, default=0.035, help="Độ dài mỗi nhánh dấu cộng (m, mặc định 0.035m = 3.5cm)")
    args = parser.parse_args()

    world_name = args.world if args.world else detect_running_world()
    model_name = "target_cylinder" if world_name == "empty" else "cube_pendulum"
    z_offset = 0.0 if world_name == "empty" else 0.175

    dt = 1.0 / args.hz
    v_max = float(np.clip(args.speed, 0.012, 0.020))
    L = float(args.arm_len)  # 3.5 cm mỗi nhánh dấu cộng
    
    # Tính tần số góc để vận tốc cực đại đạt đúng v_max: v_max = L * omega / 2 => omega = 2 * v_max / L
    omega = 2.0 * v_max / L
    T_stroke = 2.0 * np.pi / omega
    T_cycle = 4.0 * T_stroke

    # Vị trí gốc tâm của dấu cộng (+) ở phía trước cánh tay: (X=0.0, Y=0.28, Z=0.40)
    base_x = 0.00
    base_y = 0.28  # 28 cm phía trước
    base_z = 0.40  # 40 cm cao độ

    print("=" * 75)
    print(" 🎬 XEM TRƯỚC CHUYỂN ĐỘNG VẬT THỂ TRONG GAZEBO: HÌNH DẤU CỘNG (+)")
    print(f" - Đang kết nối tới Gazebo World: [{world_name}] | Model: [{model_name}]")
    print(f" - Dạng quỹ đạo: Hình Dấu Cộng (+) 4 nhánh (Tâm: X=0cm, Y=28cm)")
    print(f" - Chiều dài mỗi nhánh: ±{L * 100:.1f} cm (Phạm vi: X∈[{-L*100:.1f}, {L*100:.1f}]cm, Y∈[{(base_y-L)*100:.1f}, {(base_y+L)*100:.1f}]cm)")
    print(f" - Vận tốc cực đại: {v_max * 100:.2f} cm/s (Vận tốc trung bình: ~{v_max * 2 / np.pi * 100:.2f} cm/s)")
    print(f" - Chu kỳ hoàn thành 1 dấu cộng (+): {T_cycle:.1f} giây (Mỗi nhánh: {T_stroke:.1f}s)")
    print(" - Trình tự: Tâm ➔ +X (Phải) ➔ Tâm ➔ +Y (Xa) ➔ Tâm ➔ -X (Trái) ➔ Tâm ➔ -Y (Gần) ➔ Tâm")
    print(" - Nhấn Ctrl+C để dừng.")
    print("=" * 75 + "\n")

    # Tự động spawn khối trụ nếu là world empty
    if world_name == "empty":
        print(f"--> Đang tự động nạp khối trụ màu xanh ({model_name}) vào Gazebo...")
        spawn_cylinder_if_needed(world_name, model_name, base_x, base_y, base_z)
        time.sleep(0.5)

    t_start = time.time()
    step = 0

    try:
        while True:
            t = time.time() - t_start
            step += 1

            # Quỹ đạo hình dấu cộng (+) 4 nhánh mượt mà
            t_mod = t % T_cycle
            stroke = int(t_mod // T_stroke)
            t_s = t_mod - stroke * T_stroke

            # Hàm nội suy êm 0 -> 1 -> 0
            s = 0.5 * (1.0 - np.cos(omega * t_s))
            s_dot = 0.5 * omega * np.sin(omega * t_s)

            if stroke == 0:    # Nhánh 1: Tâm -> +X (Phải) -> Tâm
                cur_x = base_x + L * s
                cur_y = base_y
                vx = L * s_dot
                vy = 0.0
                branch_name = "+X (Phải)"
            elif stroke == 1:  # Nhánh 2: Tâm -> +Y (Vươn xa) -> Tâm
                cur_x = base_x
                cur_y = base_y + L * s
                vx = 0.0
                vy = L * s_dot
                branch_name = "+Y (Vươn xa)"
            elif stroke == 2:  # Nhánh 3: Tâm -> -X (Trái) -> Tâm
                cur_x = base_x - L * s
                cur_y = base_y
                vx = -L * s_dot
                vy = 0.0
                branch_name = "-X (Trái)"
            else:              # Nhánh 4: Tâm -> -Y (Lùi gần) -> Tâm
                cur_x = base_x
                cur_y = base_y - L * s
                vx = 0.0
                vy = -L * s_dot
                branch_name = "-Y (Lùi gần)"

            speed_real = np.hypot(vx, vy)
            cur_z = base_z

            # Ràng buộc an toàn trong Camera FOV và tầm với Robot
            r_cur = float(np.hypot(cur_x, cur_y))
            theta_cur = float(np.arctan2(cur_x, cur_y))
            r_clamped = float(np.clip(r_cur, 0.25, 0.33))
            theta_clamped = float(np.clip(theta_cur, -np.radians(35), np.radians(35)))

            final_x = r_clamped * np.sin(theta_clamped)
            final_y = r_clamped * np.cos(theta_clamped)

            # Gửi lệnh cập nhật pose tới Gazebo GUI
            z_model = cur_z + z_offset
            req = f'name: "{model_name}", position: {{x: {final_x:.4f}, y: {final_y:.4f}, z: {z_model:.4f}}}, orientation: {{z: 0.0, w: 1.0}}'
            
            subprocess.run(
                ['ign', 'service', '-s', f'/world/{world_name}/set_pose',
                 '--reqtype', 'ignition.msgs.Pose',
                 '--reptype', 'ignition.msgs.Boolean',
                 '--req', req, '--timeout', '200'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            if step % 10 == 0:
                print(f" [{t:5.1f}s] [{branch_name:<11}] Tọa độ: X={final_x*100:+5.1f}cm, Y={final_y*100:4.1f}cm | Vận tốc: {speed_real*100:4.2f} cm/s")

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n>> Đã dừng xem trước chuyển động.")

if __name__ == "__main__":
    main()
