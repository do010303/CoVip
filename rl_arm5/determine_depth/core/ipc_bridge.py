# -*- coding: utf-8 -*-
"""
Module truyền thông liên tiến trình tốc độ cao (IPC Bridge) giữa Vision (Core 1) và Controller (Core 0).

Sử dụng UDP Socket trên localhost (127.0.0.1:5555):
- Độ trễ cực thấp (< 0.1ms)
- Hoàn toàn phi khóa (Non-blocking / Zero lock contention)
- Cách ly hoàn toàn lỗi: Camera crash thì Controller không bị crash, và ngược lại.
"""

import socket
import struct
import time
from typing import Optional, Tuple


# Định dạng gói tin nhị phân siêu gọn (29 bytes):
# 1 byte header (0xAA)
# 3 float32: x_base, y_base, z_base (12 bytes)
# 3 float32: vx_base, vy_base, vz_base (12 bytes)
# 1 bool: is_valid (1 byte)
# 1 float64 (double): timestamp (8 bytes, tránh lỗi làm tròn của float32 trên Epoch 2026)
PACKET_FMT = "!Bffffff?d"
HEADER_BYTE = 0xAA
PORT_DEFAULT = 5555


class VisionIPCPublisher:
    """Được gọi từ main.py của determine_depth trên Core 1 để phát tọa độ (30 FPS)."""
    def __init__(self, host: str = "127.0.0.1", port: int = PORT_DEFAULT):
        self.dest = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, x: float, y: float, z: float,
                vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
                is_valid: bool = True):
        now = float(time.time())
        data = struct.pack(PACKET_FMT, HEADER_BYTE,
                           float(x), float(y), float(z),
                           float(vx), float(vy), float(vz),
                           bool(is_valid), now)
        try:
            self.sock.sendto(data, self.dest)
        except Exception:
            pass

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


class VisionIPCSubscriber:
    """Được gọi từ Robot Controller trên Core 0 để nhận tọa độ mới nhất."""
    def __init__(self, host: str = "127.0.0.1", port: int = PORT_DEFAULT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.setblocking(False)

        self.last_pos = None
        self.last_vel = None
        self.last_valid = False
        self.last_time = 0.0

    def get_latest(self) -> Tuple[Optional[Tuple[float, float, float]],
                                  Optional[Tuple[float, float, float]],
                                  bool, float]:
        """
        Đọc xả hết bộ đệm UDP để lấy gói tin mới nhất (không bị trễ FIFO).
        Trả về: ((x, y, z), (vx, vy, vz), is_valid, age_in_seconds)
        """
        packet_received = None
        while True:
            try:
                data, _ = self.sock.recvfrom(64)
                if len(data) in (struct.calcsize("!Bffffff?d"), struct.calcsize("!Bffffff?f")):
                    packet_received = data
            except BlockingIOError:
                break
            except Exception:
                break

        if packet_received is not None:
            fmt = "!Bffffff?d" if len(packet_received) == struct.calcsize("!Bffffff?d") else "!Bffffff?f"
            header, x, y, z, vx, vy, vz, valid, t_stamp = struct.unpack(fmt, packet_received)
            if header == HEADER_BYTE:
                self.last_pos = (x, y, z)
                self.last_vel = (vx, vy, vz)
                self.last_valid = valid
                self.last_time = float(time.time())

        age = time.time() - self.last_time if self.last_time > 0 else 999.0
        return self.last_pos, self.last_vel, self.last_valid, age

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
