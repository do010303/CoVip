"""Module nhận diện khối trụ và trích xuất đặc trưng 2D.

Bao gồm:
  - Khử méo ảnh (Undistort) dựa trên thông số Camera Calibration.
  - Lọc ngưỡng màu HSV và tiền xử lý hình thái học (Morphological Filtering).
  - Tìm Contour khối trụ lớn nhất, trích xuất tâm (u, v) và đường kính ảnh w_px.
"""

from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np

class CylinderDetector:
    def __init__(
        self,
        color_mode: str = "blue",
        blue_lower: Tuple[int, int, int] = (95, 90, 50),
        blue_upper: Tuple[int, int, int] = (130, 255, 255),
        hsv_lower: Tuple[int, int, int] = (95, 90, 50),
        hsv_upper: Tuple[int, int, int] = (130, 255, 255),
        red_lower1: Tuple[int, int, int] = (0, 70, 50),
        red_upper1: Tuple[int, int, int] = (10, 255, 255),
        red_lower2: Tuple[int, int, int] = (165, 70, 50),
        red_upper2: Tuple[int, int, int] = (180, 255, 255),
        min_area: float = 500.0,
        blur_kernel: int = 5,
        morph_kernel: int = 5,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None
    ):
        self.color_mode = color_mode.lower()

        # Dải màu xanh dương (tối ưu nhất trên nền bảng trắng)
        self.blue_lower = np.array(blue_lower, dtype=np.uint8)
        self.blue_upper = np.array(blue_upper, dtype=np.uint8)

        # Dải màu tổng quát (custom)
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)

        # Cấu hình 2 dải màu đỏ
        self.red_lower1 = np.array(red_lower1, dtype=np.uint8)
        self.red_upper1 = np.array(red_upper1, dtype=np.uint8)
        self.red_lower2 = np.array(red_lower2, dtype=np.uint8)
        self.red_upper2 = np.array(red_upper2, dtype=np.uint8)

        self.min_area = min_area
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.morph_kernel = morph_kernel

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.undistort_enabled = (camera_matrix is not None and dist_coeffs is not None)

        # Bản đồ tối ưu remap (tính trước 1 lần, nhanh gấp 3 lần cv2.undistort)
        self.map1 = None
        self.map2 = None
        self._map_shape = None

        self._structuring_element = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.morph_kernel, self.morph_kernel)
        )

    def set_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """Cập nhật thông số hiệu chuẩn camera và làm mới bản đồ remap."""
        self.camera_matrix = np.array(camera_matrix, dtype=np.float32)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
        self.undistort_enabled = True
        self.map1 = None
        self.map2 = None
        self._map_shape = None

    def _init_remap_maps(self, shape: Tuple[int, int]):
        """Tính toán trước bản đồ pixel mapping (chỉ chạy 1 lần lúc khởi động)."""
        h, w = shape[:2]
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.camera_matrix,
            (w, h),
            cv2.CV_16SC2
        )
        self._map_shape = (h, w)

    def set_blue_range(self, lower: Tuple[int, int, int], upper: Tuple[int, int, int]):
        """Cập nhật dải màu xanh dương."""
        self.color_mode = "blue"
        self.blue_lower = np.array(lower, dtype=np.uint8)
        self.blue_upper = np.array(upper, dtype=np.uint8)

    def set_hsv_range(self, lower: Tuple[int, int, int], upper: Tuple[int, int, int]):
        """Cập nhật dải màu phân đoạn HSV tùy biến."""
        self.color_mode = "single"
        self.hsv_lower = np.array(lower, dtype=np.uint8)
        self.hsv_upper = np.array(upper, dtype=np.uint8)

    def set_red_ranges(
        self,
        lower1: Tuple[int, int, int],
        upper1: Tuple[int, int, int],
        lower2: Tuple[int, int, int],
        upper2: Tuple[int, int, int]
    ):
        """Cập nhật dải màu kép cho màu đỏ."""
        self.color_mode = "red"
        self.red_lower1 = np.array(lower1, dtype=np.uint8)
        self.red_upper1 = np.array(upper1, dtype=np.uint8)
        self.red_lower2 = np.array(lower2, dtype=np.uint8)
        self.red_upper2 = np.array(upper2, dtype=np.uint8)

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Khử méo thấu kính siêu tốc bằng cv2.remap và làm mịn ảnh."""
        if self.undistort_enabled and self.camera_matrix is not None and self.dist_coeffs is not None:
            h, w = frame.shape[:2]
            if self.map1 is None or self._map_shape != (h, w):
                self._init_remap_maps((h, w))
            # Áp dụng remap tối ưu (nhanh gấp 3-4 lần cv2.undistort lặp lại)
            frame_processed = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        else:
            frame_processed = frame

        blurred = cv2.GaussianBlur(frame_processed, (self.blur_kernel, self.blur_kernel), 0)
        return frame_processed, blurred

    def segment(self, blurred: np.ndarray) -> np.ndarray:
        """Tách ngưỡng màu HSV và khử nhiễu hình thái học."""
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        if self.color_mode == "blue":
            # Phân đoạn màu xanh dương (Hue 95 - 130)
            mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        elif self.color_mode == "red":
            # Phân đoạn màu đỏ bằng cách ghép 2 dải ngưỡng
            mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Lọc nhiễu muối tiêu (Opening) và lấp đầy khoảng trống bề mặt (Closing)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._structuring_element)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._structuring_element)
        return mask

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Phát hiện khối trụ trong khung hình.

        Returns:
            Dict chứa kết quả:
              - 'found' (bool): Có phát hiện được khối trụ không.
              - 'u', 'v' (float): Tọa độ tâm pixel của khối trụ.
              - 'w_px' (float): Đường kính chiếu của khối trụ theo pixel.
              - 'h_px' (float): Chiều cao của khối trụ theo pixel.
              - 'bbox' (Tuple[int, int, int, int]): (x, y, w, h) của bounding box.
              - 'contour' (np.ndarray): Đường viền đối tượng lớn nhất.
              - 'mask' (np.ndarray): Ảnh nhị phân phân đoạn.
              - 'processed_frame' (np.ndarray): Khung hình đã khử méo.
        """
        processed_frame, blurred = self.preprocess(frame)
        mask = self.segment(blurred)

        # Tìm đường viền các đối tượng
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result: Dict[str, Any] = {
            "found": False,
            "u": 0.0,
            "v": 0.0,
            "w_px": 0.0,
            "h_px": 0.0,
            "bbox": (0, 0, 0, 0),
            "contour": None,
            "mask": mask,
            "processed_frame": processed_frame
        }

        if not contours:
            return result

        # Lọc các contour ứng viên cho khối trụ (loại bỏ ngàm kẹp robot có diện tích/bề ngang quá khổ)
        best_contour = None
        best_rect = None
        best_diameter = 0.0
        best_height = 0.0
        best_score = -1.0

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue

            # Ngàm kẹp robot khi áp sát camera có diện tích cực đại (> 80.000 px) và bề ngang choán hết màn hình (> 450 px)
            x_box, y_box, w_box, h_box = cv2.boundingRect(c)
            if w_box > 420 or area > 85000:
                continue # Bỏ qua ngàm kẹp robot

            rect = cv2.minAreaRect(c)
            (center_x, center_y), (rect_w, rect_h), angle = rect
            diameter_px = min(rect_w, rect_h)
            height_px = max(rect_w, rect_h)

            if diameter_px <= 0:
                continue

            # Khối trụ 5cm ở cự ly làm việc (>= 12cm) có đường kính tối đa ~340px
            if diameter_px > 330:
                continue

            ratio = height_px / max(1.0, diameter_px)
            # Khối trụ đứng có chiều cao >= đường kính (ratio ~ 1.2 đến 3.5)
            # Điểm số đánh giá: ưu tiên diện tích lớn và tỷ lệ dài/rộng gần 2.0
            score = area / (1.0 + abs(ratio - 2.0))

            if score > best_score:
                best_score = score
                best_contour = c
                best_rect = rect
                best_diameter = diameter_px
                best_height = height_px

        # Fallback: Nếu không tìm thấy contour lý tưởng, tìm contour lớn nhất có w_box < 450
        if best_contour is None:
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area:
                    continue
                x_b, y_b, w_b, h_b = cv2.boundingRect(c)
                if w_b < 450 and area > best_score:
                    best_score = area
                    best_contour = c
                    best_rect = cv2.minAreaRect(c)
                    best_diameter = min(best_rect[1])
                    best_height = max(best_rect[1])

        if best_contour is None:
            return result

        # Bounding box thẳng đứng để vẽ khung hiển thị
        x, y, w, h = cv2.boundingRect(best_contour)

        # Tính trọng tâm chính xác theo Moments
        moments = cv2.moments(best_contour)
        if moments["m00"] != 0:
            u = moments["m10"] / moments["m00"]
            v = moments["m01"] / moments["m00"]
        else:
            u = best_rect[0][0]
            v = best_rect[0][1]

        result.update({
            "found": True,
            "u": float(u),
            "v": float(v),
            "w_px": float(best_diameter),
            "h_px": float(best_height),
            "bbox": (x, y, w, h),
            "rotated_rect": best_rect,
            "contour": best_contour
        })

        return result
