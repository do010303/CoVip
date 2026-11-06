"""Unit test và kiểm thử tự động cho toàn bộ hệ thống thị giác máy tính 3D.
"""

import unittest
import cv2
import numpy as np
import yaml

from core.pinhole_3d import Pinhole3DEstimator
from core.tracker_filter import MovingAverageFilter3D, KalmanFilter3D
from core.cylinder_detector import CylinderDetector

class TestPinhole3D(unittest.TestCase):
    def setUp(self):
        # Thiết lập camera chuẩn: fx=600, fy=600, cx=320, cy=240, đường kính 0.1m (10cm)
        self.estimator = Pinhole3DEstimator(fx=600.0, fy=600.0, cx=320.0, cy=240.0, d_real_m=0.1)

    def test_center_object_at_one_meter(self):
        """Vật thể nằm chính giữa ảnh (u=320, v=240), khoảng cách Z = 1.0m."""
        # Theo công thức: w_px = (fx * D_real) / Z = (600 * 0.1) / 1.0 = 60 pixels
        w_px = 60.0
        u, v = 320.0, 240.0

        X, Y, Z = self.estimator.estimate(u, v, w_px)

        self.assertAlmostEqual(Z, 1.0, places=3, msg="Z phải bằng 1.0 mét")
        self.assertAlmostEqual(X, 0.0, places=3, msg="X tại tâm phải bằng 0.0 mét")
        self.assertAlmostEqual(Y, 0.0, places=3, msg="Y tại tâm phải bằng 0.0 mét")

    def test_offset_object(self):
        """Vật thể lệch sang phải 60px (u=380) và xuống dưới 30px (v=270)."""
        w_px = 60.0  # Z = 1.0m
        u, v = 380.0, 270.0

        X, Y, Z = self.estimator.estimate(u, v, w_px)

        # X = (380 - 320) * 1.0 / 600 = 60 / 600 = +0.1 mét (+10 cm)
        # Y = (270 - 240) * 1.0 / 600 = 30 / 600 = +0.05 mét (+5 cm)
        self.assertAlmostEqual(Z, 1.0, places=3)
        self.assertAlmostEqual(X, 0.1, places=3)
        self.assertAlmostEqual(Y, 0.05, places=3)

    def test_full_metrics(self):
        metrics = self.estimator.get_full_metrics(u=320, v=240, w_px=60, h_px=120)
        self.assertAlmostEqual(metrics["Z"], 1.0, places=3)
        self.assertAlmostEqual(metrics["estimated_height_m"], 0.2, places=3)  # 120 * 1.0 / 600 = 0.2m (20cm)

class TestTrackerFilters(unittest.TestCase):
    def test_moving_average_smoothing(self):
        filt = MovingAverageFilter3D(window_size=3)
        # Giá trị đo có nhiễu quanh Z = 1.0m
        filt.update(0.0, 0.0, 1.0)
        filt.update(0.0, 0.0, 1.1)
        sx, sy, sz = filt.update(0.0, 0.0, 0.9)

        # Trung bình của (1.0 + 1.1 + 0.9) / 3 = 1.0
        self.assertAlmostEqual(sz, 1.0, places=4)

    def test_kalman_filter_convergence(self):
        filt = KalmanFilter3D(process_noise=0.01, measurement_noise=0.05)
        for _ in range(20):
            sx, sy, sz = filt.update(0.1, -0.05, 1.5)
        self.assertAlmostEqual(sz, 1.5, delta=0.05)

class TestCylinderDetector(unittest.TestCase):
    def test_synthetic_cylinder_detection(self):
        """Tạo ảnh nhân tạo nền trắng (255) và một khối trụ sẫm màu ở giữa."""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # Bảng trắng

        # Vẽ khối trụ màu xanh đậm (BGR: [120, 50, 20])
        # Kích thước: x=280, y=140, w=80, h=200 -> tâm u=320, v=240
        cv2.rectangle(img, (280, 140), (280 + 80, 140 + 200), (120, 50, 20), -1)

        # Detector với dải HSV bắt màu tối/màu sắc (khác bảng trắng)
        # Bảng trắng trong HSV có S rất thấp (< 30) và V rất cao (> 200)
        # Vật thể sẫm màu/có màu sẽ có S > 40 hoặc V < 200
        detector = CylinderDetector(
            color_mode="single",
            hsv_lower=(0, 30, 20),
            hsv_upper=(180, 255, 230),
            min_area=500
        )

        res = detector.detect(img)
        self.assertTrue(res["found"], "Phải phát hiện được khối trụ trên ảnh nhân tạo")
        self.assertAlmostEqual(res["u"], 320.0, delta=3.0)
        self.assertAlmostEqual(res["v"], 240.0, delta=3.0)
        self.assertAlmostEqual(res["w_px"], 80.0, delta=4.0)
        self.assertAlmostEqual(res["h_px"], 200.0, delta=4.0)

    def test_blue_cylinder_detection(self):
        """Kiểm tra nhận diện khối trụ MÀU XANH DƯƠNG trên nền bảng trắng."""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # Bảng trắng

        # Vẽ khối trụ màu xanh dương đậm (BGR: [220, 50, 20])
        # x=280, y=140, w=80, h=200 -> tâm u=320, v=240
        cv2.rectangle(img, (280, 140), (280 + 80, 140 + 200), (220, 50, 20), -1)

        blue_detector = CylinderDetector(color_mode="blue", min_area=500)
        res = blue_detector.detect(img)

        self.assertTrue(res["found"], "Phải phát hiện được khối trụ màu xanh dương")
        self.assertAlmostEqual(res["u"], 320.0, delta=3.0)
        self.assertAlmostEqual(res["v"], 240.0, delta=3.0)
        self.assertAlmostEqual(res["w_px"], 80.0, delta=4.0)

    def test_red_cylinder_detection(self):
        """Kiểm tra nhận diện khối trụ MÀU ĐỎ trên nền bảng trắng."""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # Bảng trắng

        # Vẽ khối trụ màu đỏ tươi (BGR: [20, 20, 230])
        # x=280, y=140, w=80, h=200 -> tâm u=320, v=240
        cv2.rectangle(img, (280, 140), (280 + 80, 140 + 200), (20, 20, 230), -1)

        red_detector = CylinderDetector(color_mode="red", min_area=500)
        res = red_detector.detect(img)

        self.assertTrue(res["found"], "Phải phát hiện được khối trụ màu đỏ")
        self.assertAlmostEqual(res["u"], 320.0, delta=3.0)
        self.assertAlmostEqual(res["v"], 240.0, delta=3.0)
        self.assertAlmostEqual(res["w_px"], 80.0, delta=4.0)

if __name__ == "__main__":
    unittest.main()
