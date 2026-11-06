"""Bộ lọc làm mịn dữ liệu thời gian thực (Temporal Smoothing).

Giải quyết triệt để hiện tượng rung giật (Jitter) tọa độ (X, Y, Z) do sai lệch viền pixel.
Cung cấp:
  - MovingAverageFilter3D: Bộ lọc trung bình trượt đơn giản và hiệu quả.
  - KalmanFilter3D: Bộ lọc Kalman 1D trên từng trục cho độ mượt cao.
"""

from collections import deque
from typing import Tuple, Optional
import numpy as np

class MovingAverageFilter3D:
    def __init__(self, window_size: int = 5, max_missing_frames: int = 5):
        self.window_size = max(1, window_size)
        self.max_missing_frames = max_missing_frames
        self.history = deque(maxlen=self.window_size)
        self.missing_count = 0

    def update(self, X: float, Y: float, Z: float) -> Tuple[float, float, float]:
        """Cập nhật tọa độ mới và trả về giá trị trung bình trượt."""
        self.missing_count = 0
        self.history.append((X, Y, Z))

        arr = np.array(self.history, dtype=np.float32)
        mean_vals = np.mean(arr, axis=0)
        return float(mean_vals[0]), float(mean_vals[1]), float(mean_vals[2])

    def notify_lost(self):
        """Thông báo khung hình hiện tại không phát hiện được vật thể."""
        self.missing_count += 1
        if self.missing_count > self.max_missing_frames:
            self.reset()

    def reset(self):
        """Xóa lịch sử khi mất dấu vật thể."""
        self.history.clear()
        self.missing_count = 0


class KalmanFilter1D:
    """Bộ lọc Kalman 1 chiều với mô hình vị trí."""
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.05):
        self.Q = float(process_noise)      # Độ không chắc chắn của mô hình
        self.R = float(measurement_noise)  # Độ nhiễu của phép đo cảm biến
        self.x = None                      # Trạng thái ước lượng
        self.P = 1.0                       # Ma trận hiệp phương sai sai số

    def update(self, measurement: float) -> float:
        if self.x is None:
            self.x = measurement
            return self.x

        # 1. Dự đoán (Predict)
        self.P += self.Q

        # 2. Cập nhật (Update)
        K = self.P / (self.P + self.R)     # Hệ số Kalman Gain
        self.x += K * (measurement - self.x)
        self.P *= (1.0 - K)

        return float(self.x)

    def reset(self):
        self.x = None
        self.P = 1.0


class KalmanFilter3D:
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.05, max_missing_frames: int = 5):
        self.filter_x = KalmanFilter1D(process_noise, measurement_noise)
        self.filter_y = KalmanFilter1D(process_noise, measurement_noise)
        self.filter_z = KalmanFilter1D(process_noise, measurement_noise)
        self.max_missing_frames = max_missing_frames
        self.missing_count = 0

    def update(self, X: float, Y: float, Z: float) -> Tuple[float, float, float]:
        self.missing_count = 0
        smooth_x = self.filter_x.update(X)
        smooth_y = self.filter_y.update(Y)
        smooth_z = self.filter_z.update(Z)
        return smooth_x, smooth_y, smooth_z

    def notify_lost(self):
        self.missing_count += 1
        if self.missing_count > self.max_missing_frames:
            self.reset()

    def reset(self):
        self.filter_x.reset()
        self.filter_y.reset()
        self.filter_z.reset()
        self.missing_count = 0
