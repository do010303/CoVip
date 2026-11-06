"""Module truyền dữ liệu tọa độ 3D qua cổng Serial / UART (USB).

Hỗ trợ:
  - Tự động nhận diện hoặc chỉ định cổng: /dev/ttyUSB0, /dev/ttyACM0,...
  - Xử lý mất kết nối an toàn (Graceful reconnection), không làm đơ/crash pipeline xử lý ảnh.
  - Định dạng chuỗi ngắn gọn tối ưu cho Arduino/STM32/ESP32 hoặc định dạng JSON.
"""

import time
import json
import threading
from typing import Tuple, Optional

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class SerialSender:
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        enabled: bool = False,
        data_format: str = "string",
        send_rate_hz: float = 30.0
    ):
        self.port = port
        self.baudrate = baudrate
        self.enabled = enabled and SERIAL_AVAILABLE
        self.data_format = data_format.lower()
        self.min_interval = 1.0 / max(1.0, send_rate_hz)
        self.last_send_time = 0.0

        self.ser: Optional[serial.Serial] = None
        self.is_connected = False
        self.lock = threading.Lock()

        if not SERIAL_AVAILABLE and enabled:
            print("[CẢNH BÁO] Thư viện 'pyserial' chưa được cài đặt. Không thể mở cổng Serial.")

        if self.enabled:
            self.connect()

    def connect(self) -> bool:
        """Thử kết nối tới cổng Serial."""
        if not SERIAL_AVAILABLE:
            return False

        with self.lock:
            try:
                self.ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=0.05,
                    write_timeout=0.05
                )
                self.is_connected = True
                print(f"[SERIAL] Kết nối thành công tới {self.port} (Baudrate: {self.baudrate})")
                return True
            except (serial.SerialException, OSError) as e:
                self.is_connected = False
                self.ser = None
                return False

    def send(self, found: bool, X: float, Y: float, Z: float, distance: float = 0.0) -> bool:
        """Gửi tọa độ (X, Y, Z) tới vi điều khiển nếu đủ chu kỳ gửi."""
        if not self.enabled:
            return False

        now = time.time()
        if now - self.last_send_time < self.min_interval:
            return False

        if not self.is_connected or self.ser is None:
            # Thử kết nối lại mỗi 2 giây nếu chưa cắm dây
            if now - self.last_send_time > 2.0:
                self.connect()
                self.last_send_time = now
            return False

        # Đóng gói dữ liệu
        flag = 1 if found else 0
        if self.data_format == "json":
            payload = {
                "f": flag,
                "x": round(X, 3),
                "y": round(Y, 3),
                "z": round(Z, 3),
                "d": round(distance, 3)
            }
            msg = json.dumps(payload) + "\n"
        else:
            # Định dạng chuỗi ngắn gọn, dễ parse trên STM32/Arduino:
            # Ví dụ: "X:0.052,Y:-0.012,Z:0.450,F:1\n"
            msg = f"X:{X:+.3f},Y:{Y:+.3f},Z:{Z:.3f},F:{flag}\n"

        with self.lock:
            try:
                self.ser.write(msg.encode("ascii"))
                self.ser.flush()
                self.last_send_time = now
                return True
            except (serial.SerialException, OSError) as e:
                self.is_connected = False
                self.ser = None
                return False

    def close(self):
        """Đóng cổng serial an toàn."""
        with self.lock:
            if self.ser is not None and self.ser.is_open:
                try:
                    self.ser.close()
                except Exception:
                    pass
            self.is_connected = False
            self.ser = None

    @staticmethod
    def list_available_ports():
        """Liệt kê danh sách các cổng Serial đang có trên máy."""
        if not SERIAL_AVAILABLE:
            return []
        return [p.device for p in serial.tools.list_ports.comports()]
