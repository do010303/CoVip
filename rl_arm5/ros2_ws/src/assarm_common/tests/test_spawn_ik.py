# -*- coding: utf-8 -*-
"""
Validation script cho vùng spawn (r = 0.08 - 0.19 m).

Kiểm tra tầm với IK và độ nghiêng gripper < 15° với 1000 mẫu ngẫu nhiên
trên các cấp spawn (Level 0..3) sử dụng tra cứu grid 2D chính xác từ FK.
"""

from typing import Tuple
import numpy as np
import pytest
from assarm_common.fk import p_close, gripper_tilt

# Precompute 2D grid của (q2, q3) -> (r, z, tilt) với q3 ≈ -q2
_q2_grid = np.linspace(0.1, 1.3, 150)
_dq3_grid = np.linspace(-np.radians(15), np.radians(15), 80)
_grid_q2, _grid_dq3 = np.meshgrid(_q2_grid, _dq3_grid)
_grid_q2 = _grid_q2.flatten()
_grid_q3 = -_grid_q2 + _grid_dq3.flatten()

_r_vals = []
_z_vals = []
for _q2_val, _q3_val in zip(_grid_q2, _grid_q3):
    _q = np.array([0.0, _q2_val, _q3_val, 0.0])
    _p = p_close(_q)
    _r_vals.append(np.sqrt(_p[0]**2 + _p[1]**2))
    _z_vals.append(_p[2])

_r_arr = np.array(_r_vals)
_z_arr = np.array(_z_vals)


def solve_ik_numerical(target_p_close: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Giải IK tìm góc khớp q cho target_p_close sử dụng tra cứu grid 2D chính xác."""
    x, y, z = target_p_close
    y_j1 = -0.010797
    x_j1 = -0.000006
    q1 = np.arctan2(y - y_j1, x - x_j1)

    r_target = np.sqrt((x - x_j1)**2 + (y - y_j1)**2)
    dist_sq = (_r_arr - r_target)**2 + (_z_arr - z)**2
    min_idx = np.argmin(dist_sq)
    min_dist = np.sqrt(dist_sq[min_idx])

    q_sol = np.array([q1, _grid_q2[min_idx], _grid_q3[min_idx], 0.0])
    return q_sol, (min_dist < 0.003)


def test_spawn_region_reachability():
    """Sample 1000 điểm ngẫu nhiên trong vùng spawn r=0.08-0.19m và kiểm tra IK + Tilt."""
    np.random.seed(42)
    n_samples = 1000

    levels = {
        'Level 0': (0.10, 0.14),
        'Level 1': (0.09, 0.16),
        'Level 2': (0.08, 0.18),
        'Level 3': (0.08, 0.19),
    }

    print("\n" + "="*70)
    print(" KẾT QUẢ KIỂM TRA IK VÙNG SPAWN (r = 0.08 - 0.19 m)")
    print("="*70)

    for level_name, (r_min, r_max) in levels.items():
        r = np.random.uniform(r_min, r_max, n_samples)
        theta = np.random.uniform(-np.radians(45), np.radians(45), n_samples)

        z_min = 0.176 + 0.44 * (r - 0.10)
        z_max = 0.268 - 0.05 * (r - 0.10)
        z = z_min + np.random.rand(n_samples) * (z_max - z_min)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        ik_success_count = 0
        tilt_ok_count = 0

        for i in range(n_samples):
            target = np.array([x[i], y[i], z[i]])
            q_sol, ok = solve_ik_numerical(target)
            if ok:
                ik_success_count += 1
                tilt = gripper_tilt(q_sol)
                if np.degrees(tilt) <= 15.0:
                    tilt_ok_count += 1

        ik_rate = ik_success_count / n_samples * 100
        tilt_rate = tilt_ok_count / max(ik_success_count, 1) * 100
        overall_rate = tilt_ok_count / n_samples * 100

        print(f"{level_name:8s} | r=[{r_min:.2f}-{r_max:.2f}]m | IK Tới: {ik_rate:5.1f}% | Nghiêng <15°: {tilt_rate:5.1f}% | Thỏa ĐK: {overall_rate:5.1f}%")
        assert overall_rate >= 80.0, f"{level_name} tỉ lệ thỏa < 80% ({overall_rate:.1f}%)"


if __name__ == '__main__':
    test_spawn_region_reachability()
