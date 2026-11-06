#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_dynamic_motion.py
Trực quan hóa hoạt hình 2D/3D chuyển động của vật thể trong không gian làm việc (Stage B+: 1.2 - 2.0 cm/s).

- Hiển thị góc nhìn từ trên xuống (Top-down X-Y plane):
  + Vị trí Robot Base tại (0, 0)
  + Vị trí Camera tại (0, 0.07m) và hình nón trường nhìn (Camera FOV 60°)
  + Vùng làm việc của cánh tay robot (Workspace arc: r = 0.25 - 0.33m, theta = ±35°)
  + Quỹ đạo chuyển động thời gian thực của khối trụ với vector vận tốc v = 1.2 - 2.0 cm/s
- Xuất file ảnh động GIF hoặc hiển thị cửa sổ trực tiếp.
"""

import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge, Polygon

def generate_preview(output_file: str = "dynamic_motion_preview.gif", duration_s: float = 12.0, fps: int = 15):
    v_max = 0.018     # 1.8 cm/s đỉnh (nằm chuẩn trong khoảng 1.2 - 2.0 cm/s)
    L = 0.035         # 3.5 cm mỗi nhánh dấu cộng
    omega = 2.0 * v_max / L
    T_stroke = 2.0 * np.pi / omega  # ~3.05 giây mỗi nhánh
    T_cycle = 4.0 * T_stroke        # ~12.2 giây cả chu trình dấu cộng

    base_x = 0.0
    base_y = 0.28
    base_z = 0.40

    total_frames = int(duration_s * fps)
    t_arr = np.linspace(0, duration_s, total_frames)
    x_arr = []
    y_arr = []
    vx_arr = []
    vy_arr = []

    for t in t_arr:
        t_mod = t % T_cycle
        stroke = int(t_mod // T_stroke)
        t_s = t_mod - stroke * T_stroke

        s = 0.5 * (1.0 - np.cos(omega * t_s))
        s_dot = 0.5 * omega * np.sin(omega * t_s)

        if stroke == 0:    # +X (Phải)
            cur_x = base_x + L * s
            cur_y = base_y
            vx = L * s_dot
            vy = 0.0
        elif stroke == 1:  # +Y (Xa)
            cur_x = base_x
            cur_y = base_y + L * s
            vx = 0.0
            vy = L * s_dot
        elif stroke == 2:  # -X (Trái)
            cur_x = base_x - L * s
            cur_y = base_y
            vx = -L * s_dot
            vy = 0.0
        else:              # -Y (Gần)
            cur_x = base_x
            cur_y = base_y - L * s
            vx = 0.0
            vy = -L * s_dot

        r_cur = float(np.hypot(cur_x, cur_y))
        theta_cur = float(np.arctan2(cur_x, cur_y))
        r_clamped = float(np.clip(r_cur, 0.25, 0.33))
        theta_clamped = float(np.clip(theta_cur, -np.radians(35), np.radians(35)))

        fx = r_clamped * np.sin(theta_clamped)
        fy = r_clamped * np.cos(theta_clamped)

        x_arr.append(fx)
        y_arr.append(fy)
        vx_arr.append(vx)
        vy_arr.append(vy)

    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    vx_arr = np.array(vx_arr)
    vy_arr = np.array(vy_arr)

    # Khởi tạo đồ thị
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#1a1a24')
    ax.set_facecolor('#12121a')

    # Vẽ Robot Base tại (0,0)
    ax.plot(0, 0, marker='s', markersize=14, color='#ff7b00', label='Robot Base (0, 0)')

    # Vẽ Vị trí Camera tại (0, 0.07m)
    ax.plot(0, 0.07, marker='o', markersize=10, color='#00d4ff', label='Camera (0, 0.07m)')

    # Vẽ Camera FOV cone (Góc nhìn ngang ~60 độ từ Y=0.07m tới Y=0.45m)
    fov_y = 0.45
    fov_half_w = (fov_y - 0.07) * np.tan(np.radians(30))
    fov_poly = Polygon([[0, 0.07], [-fov_half_w, fov_y], [fov_half_w, fov_y]],
                       closed=True, facecolor='#00d4ff', alpha=0.10, edgecolor='#00d4ff', linestyle='--', label='Camera FOV (60°)')
    ax.add_patch(fov_poly)

    # Vẽ Vùng làm việc Robot (Workspace Arc r=0.25 - 0.33m, theta=±35°)
    # Lưu ý: Trục Y là hướng duỗi ra trước (theta=0), Trục X là ngang
    theta_arc = np.linspace(-np.radians(35), np.radians(35), 60)
    arc_inner_x = 0.25 * np.sin(theta_arc)
    arc_inner_y = 0.25 * np.cos(theta_arc)
    arc_outer_x = 0.33 * np.sin(theta_arc[::-1])
    arc_outer_y = 0.33 * np.cos(theta_arc[::-1])
    ws_x = np.concatenate([arc_inner_x, arc_outer_x])
    ws_y = np.concatenate([arc_inner_y, arc_outer_y])
    ws_poly = Polygon(np.column_stack([ws_x, ws_y]), closed=True,
                      facecolor='#39ff14', alpha=0.12, edgecolor='#39ff14', linestyle=':', label='Vùng làm việc Robot (r=25-33cm)')
    ax.add_patch(ws_poly)

    # Quỹ đạo vẽ trước mờ
    ax.plot(x_arr, y_arr, color='#ff007f', alpha=0.35, lw=2.5, linestyle='--', label='Quỹ đạo hình dấu cộng (+)')

    # Đối tượng động
    trail_line, = ax.plot([], [], color='#ff007f', lw=3.0, alpha=0.85)
    target_dot, = ax.plot([], [], marker='o', markersize=14, color='#ff0055', markeredgecolor='white', markeredgewidth=1.5, label='Khối trụ di động (v=1.2-2.0 cm/s)')
    vel_quiver = ax.quiver([0], [0], [0], [0], color='#ffff00', scale=0.1, scale_units='xy', width=0.008, label='Vector vận tốc v')
    info_text = ax.text(0.03, 0.95, '', transform=ax.transAxes, color='white', fontsize=11, family='monospace',
                        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#222233', alpha=0.8, edgecolor='#444466'))

    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.02, 0.46)
    ax.set_xlabel('Trục X Robot (m) [Ngang]', color='white', fontsize=12)
    ax.set_ylabel('Trục Y Robot (m) [Chiều sâu vươn tới]', color='white', fontsize=12)
    ax.set_title('Mô Phỏng Quỹ Đạo Hình Dấu Cộng (+) Stage B+ (v = 1.2 - 2.0 cm/s)', color='white', fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(colors='white')
    ax.grid(True, color='#333344', linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', facecolor='#222233', edgecolor='#444466', labelcolor='white', fontsize=9)

    def init():
        trail_line.set_data([], [])
        target_dot.set_data([], [])
        info_text.set_text('')
        return trail_line, target_dot, vel_quiver, info_text

    def update(frame):
        cur_x = x_arr[frame]
        cur_y = y_arr[frame]
        cur_vx = vx_arr[frame]
        cur_vy = vy_arr[frame]
        cur_speed = np.hypot(cur_vx, cur_vy)

        start_trail = max(0, frame - int(2.5 * fps))
        trail_line.set_data(x_arr[start_trail:frame+1], y_arr[start_trail:frame+1])
        target_dot.set_data([cur_x], [cur_y])

        vel_quiver.set_offsets([cur_x, cur_y])
        # scale arrow dài ra 2 lần để mắt dễ thấy
        vel_quiver.set_UVC([cur_vx * 2.5], [cur_vy * 2.5])

        t_cur = t_arr[frame]
        info_text.set_text(
            f"Thời gian: {t_cur:4.1f} s\n"
            f"Tọa độ X : {cur_x*100:+5.1f} cm\n"
            f"Tọa độ Y : {cur_y*100:5.1f} cm\n"
            f"Vận tốc  : {cur_speed*100:4.2f} cm/s\n"
            f"Trạng thái: AN TOÀN TRONG FOV ✅"
        )
        return trail_line, target_dot, vel_quiver, info_text

    ani = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=False, interval=1000/fps)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    print(f"--> Đang xuất file hoạt hình mô phỏng: {output_file}...")
    try:
        ani.save(output_file, writer='pillow', fps=fps)
        print(f"✅ ĐÃ XUẤT THÀNH CÔNG: {output_file}")
    except Exception as e:
        print(f"Lỗi xuất gif: {e}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="/home/quan/Desktop/test5/ros2_ws/models/stage_b_plus/dynamic_motion_preview.gif")
    args = parser.parse_args()
    generate_preview(args.output)
