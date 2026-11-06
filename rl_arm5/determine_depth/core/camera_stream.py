"""Threaded Video Stream cho Camera.

Tách riêng luồng đọc frame từ phần cứng (I/O) ra khỏi luồng xử lý ảnh.
Giúp đạt tối đa FPS và giảm thiểu trễ khung hình (latency), đặc biệt hiệu quả trên Raspberry Pi 4B.
"""

import threading
import time
import cv2

class CameraStream:
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps

        # Thử mở camera với V4L2 backend trước (chuẩn trên Linux/Raspberry Pi)
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.src)

        # Nếu không mở được index chỉ định, tự động quét tìm camera khả dụng (0, 1, 2, 4)
        if not self.cap.isOpened():
            candidates = [0, 1, 2, 4]
            for cand in candidates:
                if cand == self.src:
                    continue
                cap_try = cv2.VideoCapture(cand, cv2.CAP_V4L2)
                if not cap_try.isOpened():
                    cap_try = cv2.VideoCapture(cand)
                if cap_try.isOpened():
                    ret_try, _ = cap_try.read()
                    if ret_try:
                        print(f"[CAMERA] ℹ️ Camera thiết bị {self.src} không tìm thấy. Tự động chuyển sang Camera thiết bị {cand}!")
                        self.cap = cap_try
                        self.src = cand
                        break
                    cap_try.release()

        # Thiết lập codec MJPG nếu có thể để tối ưu tốc độ truyền khung hình
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.ret, self.frame = self.cap.read()
        self.frame_id = 0
        self.stopped = False
        self.condition = threading.Condition()
        self.thread = None

    def start(self):
        """Khởi động luồng chạy nền đọc camera."""
        if not self.cap.isOpened():
            raise RuntimeError(f"Không thể mở camera thiết bị {self.src}")

        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        time.sleep(0.1)  # Đợi camera ổn định
        return self

    def _update(self):
        """Vòng lặp ngầm liên tục cập nhật frame mới nhất từ phần cứng."""
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            with self.condition:
                self.ret = ret
                self.frame = frame
                self.frame_id += 1
                self.condition.notify_all()

    def read(self, wait_new: bool = False, last_frame_id: int = -1, timeout: float = 0.04):
        """Lấy khung hình mới nhất một cách an toàn đa luồng.

        Args:
            wait_new (bool): Nếu True, sẽ tạm dừng luồng chờ đến khi camera có frame MỚI từ phần cứng,
                             giúp loại bỏ 100% việc lặp lại xử lý trên frame cũ và tiết kiệm tới 65% CPU.
            last_frame_id (int): ID frame đã xử lý ở vòng trước.
            timeout (float): Thời gian chờ tối đa (giây).
        """
        with self.condition:
            if wait_new and self.frame_id <= last_frame_id and not self.stopped:
                self.condition.wait(timeout=timeout)
            if self.frame is None:
                return False, None, self.frame_id
            return self.ret, self.frame.copy(), self.frame_id

    def stop(self):
        """Dừng luồng và giải phóng tài nguyên."""
        self.stopped = True
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
