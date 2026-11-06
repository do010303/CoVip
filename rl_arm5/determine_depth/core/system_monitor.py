"""Module giám sát tài nguyên hệ thống và tối ưu phần cứng cho Raspberry Pi 4B.

Bao gồm:
  - Đo nhiệt độ CPU (SoC Thermal Zone).
  - Kiểm tra trạng thái suy giảm xung nhịp (Thermal Throttling) và sụt áp.
  - Phân bổ lõi CPU (CPU Affinity / Core Pinning) để cô lập tiến trình.
  - Cấu hình số luồng tính toán OpenCV (tránh tranh chấp tài nguyên với AI & Pixhawk).
"""

import os
import sys
import subprocess
from typing import Optional, Set

class SystemMonitor:
    def __init__(self, target_core: Optional[int] = 1, opencv_threads: int = 1):
        """Khởi tạo bộ giám sát hệ thống.

        Args:
            target_core (Optional[int]): Lõi CPU muốn ghim tiến trình (0, 1, 2, 3 trên Pi 4B).
                                         Mặc định Core 1 để chừa Core 0 cho I/O (Pixhawk, ESP32)
                                         và Core 2, 3 cho AI Inference.
            opencv_threads (int): Số luồng CPU cho OpenCV (mặc định 1 để không chiếm dụng đa nhân).
        """
        self.target_core = target_core
        self.opencv_threads = opencv_threads
        self.thermal_path = self._find_thermal_path()

        # Áp dụng cấu hình tối ưu phần cứng ngay khi khởi tạo
        self.apply_cpu_affinity(target_core)
        self.apply_opencv_threads(opencv_threads)

    def _find_thermal_path(self) -> Optional[str]:
        """Tìm đường dẫn cảm biến nhiệt độ phần cứng Linux."""
        candidates = [
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/devices/virtual/thermal/thermal_zone0/temp"
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def get_cpu_temp(self) -> float:
        """Đọc nhiệt độ CPU hiện tại (đơn vị: độ C).

        Returns:
            float: Nhiệt độ CPU (ví dụ 48.5). Trả về -1.0 nếu không đọc được.
        """
        if self.thermal_path:
            try:
                with open(self.thermal_path, "r") as f:
                    return float(f.read().strip()) / 1000.0
            except Exception:
                pass

        # Thử lệnh vcgencmd riêng của Raspberry Pi
        if sys.platform != "win32":
            try:
                output = subprocess.check_output(["vcgencmd", "measure_temp"], timeout=0.1).decode()
                # Định dạng output: temp=48.5'C
                if "temp=" in output:
                    val = output.split("=")[1].split("'")[0]
                    return float(val)
            except Exception:
                pass

        return -1.0

    def get_throttle_status(self) -> str:
        """Kiểm tra tình trạng sụt áp / nghẽn nhiệt độ của Raspberry Pi (vcgencmd get_throttled).

        Returns:
            str: "OK" nếu bình thường, hoặc mã cảnh báo ("THROTTLED", "LOW_VOLTAGE",...).
        """
        if sys.platform == "win32":
            return "N/A"

        try:
            output = subprocess.check_output(["vcgencmd", "get_throttled"], timeout=0.1).decode().strip()
            # Định dạng: throttled=0x0
            if "throttled=" in output:
                hex_str = output.split("=")[1]
                val = int(hex_str, 16)
                if val == 0:
                    return "OK"
                reasons = []
                if val & 0x1:
                    reasons.append("Sụt áp (Low Voltage)")
                if val & 0x2:
                    reasons.append("Giảm xung ARM")
                if val & 0x4:
                    reasons.append("Bị nghẽn nhiệt (Throttled)")
                if val & 0x8:
                    reasons.append("Chạm giới hạn nhiệt")
                if val & 0x10000:
                    reasons.append("Từng bị sụt áp")
                if val & 0x40000:
                    reasons.append("Từng bị nghẽn nhiệt")
                return ", ".join(reasons) if reasons else f"Hex: {hex_str}"
        except Exception:
            pass

        return "N/A"

    def apply_cpu_affinity(self, core_id: Optional[int]):
        """Ghim tiến trình vào 1 lõi CPU duy nhất (Core Pinning) để tránh tranh chấp."""
        if core_id is None or sys.platform == "win32":
            return

        if hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, {core_id})
                print(f"[TỐI ƯU PI4B] Đã ghim tiến trình vào CPU Core {core_id} (giữ nguyên Core 0 cho Pixhawk/ESP32, Core 2-3 cho AI).")
            except Exception as e:
                print(f"[CẢNH BÁO] Không thể ghim lõi CPU: {e}")

    def apply_opencv_threads(self, n_threads: int):
        """Giới hạn số luồng của OpenCV để không chiếm dụng đa nhân."""
        try:
            import cv2
            cv2.setNumThreads(n_threads)
            print(f"[TỐI ƯU PI4B] Đã cấu hình OpenCV chạy tối đa {n_threads} luồng.")
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể cấu hình luồng OpenCV: {e}")
