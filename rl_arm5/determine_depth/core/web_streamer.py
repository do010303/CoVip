"""Luồng phát video trực tiếp và Dashboard hiển thị thông số thời gian thực (Telemetry Dashboard).

Đặc điểm tối ưu cho hệ thống nhúng / Raspberry Pi 4B:
  - Sử dụng thư viện chuẩn của Python (http.server), KHÔNG CẦN Flask hay phụ thuộc bên ngoài.
  - Hoàn toàn Offline: Tích hợp sẵn Canvas 3D Perspective Viewer (Vanilla JS) không cần internet/CDN.
  - Giao diện Mission Control cao cấp:
      + Luồng hình ảnh camera MJPEG chất lượng cao
      + Mô hình 3D tương tác của cánh tay robot với góc xoay Orbit (chuột / cảm ứng điện thoại)
      + Điểm đỏ phát sáng hiển thị chính xác tâm trọng lượng (CoM - Center of Mass)
      + Bảng thông số: Lực quán tính tổng thể, Mô-men lật, Độ giật (Jerk) từng servo
      + Góc xoay các servo S1-S5, hướng ngàm kẹp (Pitch, Yaw, Roll, Tilt)
      + Sai số bám vết (P_err), tốc độ tiếp cận (v_rel), trạng thái thành công
  - Tích hợp UDP Receiver (Port 5556) nhận Telemetry thời gian thực từ Robot Controller.
"""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
import cv2
import numpy as np


class WebStreamer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080,
                 stream_fps: int = 15, jpeg_quality: int = 65,
                 telemetry_udp_port: int = 5556):
        self.host = host
        self.port = port
        self.stream_fps = stream_fps
        self.jpeg_quality = jpeg_quality
        self.telemetry_udp_port = telemetry_udp_port

        self._frame = None
        self._metrics = {}
        self._lock = threading.Lock()
        self._active_clients = 0
        self._server = None
        self._thread = None
        self._udp_thread = None
        self._running = False

        # Dữ liệu Telemetry thời gian thực mặc định
        self._telemetry = {
            "state": "IDLE",
            "success": False,
            "pos_err_cm": 0.0,
            "v_rel_cms": 0.0,
            "com_cm": [-1.08, 6.43, 40.62],
            "f_inertial_N": 0.0,
            "m_pitch_Nm": 0.32,
            "m_roll_Nm": -0.05,
            "jerk_each_degs3": [0.0, 0.0, 0.0, 0.0],
            "jerk_total_rms": 0.0,
            "joints_cmd_deg": [90.0, 69.9, 58.5, 90.0, 0.0],
            "joints_fb_deg": [90.0, 69.9, 58.5, 90.0, 0.0],
            "orientation": {"yaw": 0.0, "pitch": 51.5, "roll": 51.5, "tilt": 51.5},
            "joints_3d": [
                {"name": "Base", "pos": [0.0, 0.0, 56.0]},
                {"name": "J1",   "pos": [0.0, 0.0, 54.6]},
                {"name": "J2",   "pos": [-2.6, 0.0, 50.9]},
                {"name": "J3",   "pos": [-0.6, 8.4, 39.5]},
                {"name": "J4",   "pos": [-0.4, 16.2, 31.8]},
                {"name": "EE",   "pos": [-0.5, 21.1, 22.6]}
            ],
            "target_cm": None,
            "ee_pos_cm": [-0.5, 21.1, 22.6],
            "cam_coords_m": [0.0, 0.0, 0.0],
            "fps_cam": 0.0,
            "cpu_temp": 0.0,
            "uart_ok": False,
            "timestamp": time.time()
        }

    def update_frame(self, frame: np.ndarray, metrics: Optional[Dict[str, Any]] = None):
        """Cập nhật frame hiển thị mới nhất từ luồng chính (không bị block)."""
        if self._active_clients > 0:
            with self._lock:
                self._frame = frame.copy() if frame is not None else None
                if metrics:
                    self._metrics = metrics.copy()
                    # Đồng bộ các thông số camera vào telemetry nếu có
                    if "fps" in metrics:
                        self._telemetry["fps_cam"] = round(float(metrics["fps"]), 1)
                    if "cpu_temp" in metrics:
                        self._telemetry["cpu_temp"] = round(float(metrics["cpu_temp"]), 1)
                    if "coords_3d" in metrics:
                        c = metrics["coords_3d"]
                        self._telemetry["cam_coords_m"] = [round(float(c[0]), 3), round(float(c[1]), 3), round(float(c[2]), 3)]
                    if "coords_base" in metrics and metrics.get("target_found", False):
                        cb = metrics["coords_base"]
                        self._telemetry["target_cm"] = [round(float(cb[0] * 100), 1), round(float(cb[1] * 100), 1), round(float(cb[2] * 100), 1)]
                    if "uart_ok" in metrics:
                        self._telemetry["uart_ok"] = bool(metrics["uart_ok"])

    def update_telemetry(self, tele_dict: Dict[str, Any]):
        """Cập nhật trực tiếp dữ liệu động lực học từ Controller."""
        with self._lock:
            self._telemetry.update(tele_dict)
            self._telemetry["timestamp"] = time.time()

    def _telemetry_udp_listener(self):
        """Luồng ngầm nhận telemetry JSON từ robot controller qua UDP port 5556."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", self.telemetry_udp_port))
            sock.settimeout(0.5)
            while self._running:
                try:
                    data, _ = sock.recvfrom(4096)
                    parsed = json.loads(data.decode("utf-8"))
                    self.update_telemetry(parsed)
                except socket.timeout:
                    continue
                except Exception:
                    continue
        except Exception as e:
            print(f"[WEB STREAMER] Cảnh báo cổng UDP Telemetry {self.telemetry_udp_port}: {e}")
        finally:
            sock.close()

    def start(self):
        """Khởi động Web server và UDP listener trong luồng chạy ngầm daemon."""
        streamer_self = self

        class StreamHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Tắt log in tràn màn hình terminal
                return

            def do_GET(self):
                if self.path == "/" or self.path == "/index.html":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    html = streamer_self._build_dashboard_html()
                    self.wfile.write(html.encode("utf-8"))

                elif self.path == "/api/telemetry":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.end_headers()
                    with streamer_self._lock:
                        tele_copy = streamer_self._telemetry.copy()
                    self.wfile.write(json.dumps(tele_copy).encode("utf-8"))

                elif self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                    self.end_headers()

                    with streamer_self._lock:
                        streamer_self._active_clients += 1

                    interval = 1.0 / max(1, streamer_self.stream_fps)
                    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), streamer_self.jpeg_quality]

                    try:
                        while streamer_self._running:
                            t_start = time.time()
                            frame_to_send = None
                            with streamer_self._lock:
                                if streamer_self._frame is not None:
                                    frame_to_send = streamer_self._frame

                            if frame_to_send is not None:
                                ret, jpeg = cv2.imencode(".jpg", frame_to_send, encode_params)
                                if ret:
                                    self.wfile.write(b"--FRAME\r\n")
                                    self.send_header("Content-Type", "image/jpeg")
                                    self.send_header("Content-Length", str(len(jpeg)))
                                    self.end_headers()
                                    self.wfile.write(jpeg.tobytes())
                                    self.wfile.write(b"\r\n")

                            dt = time.time() - t_start
                            sleep_time = interval - dt
                            if sleep_time > 0:
                                time.sleep(sleep_time)
                            else:
                                time.sleep(0.005)
                    except (ConnectionResetError, BrokenPipeError):
                        pass
                    finally:
                        with streamer_self._lock:
                            streamer_self._active_clients = max(0, streamer_self._active_clients - 1)
                else:
                    self.send_error(404)

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), StreamHandler)
            self._running = True
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()

            self._udp_thread = threading.Thread(target=self._telemetry_udp_listener, daemon=True)
            self._udp_thread.start()

            lan_ip = self._get_lan_ip()
            print(f"[LOCAL STREAM] ✅ Đã kích hoạt Mission Control Web Dashboard thành công!")
            print(f"  👉 Trên máy tính/điện thoại, mở trình duyệt: http://{lan_ip}:{self.port}")
            print(f"  (Hoặc ngay trên màn hình Pi: http://localhost:{self.port})")
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể khởi động Web Server trên cổng {self.port}: {e}")

    def _get_lan_ip(self) -> str:
        """Lấy địa chỉ IP mạng nội bộ thực tế của Raspberry Pi."""
        for target in ["192.168.50.143", "192.168.50.1", "10.255.255.255", "8.8.8.8"]:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect((target, 80))
                ip = s.getsockname()[0]
                s.close()
                if ip and not ip.startswith("127."):
                    return ip
            except Exception:
                pass
        return "192.168.50.1"

    def stop(self):
        """Dừng Web server an toàn."""
        self._running = False
        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass

    def _build_dashboard_html(self) -> str:
        """Tạo giao diện Mission Control Dashboard chuyên nghiệp, hoàn toàn Offline."""
        return """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASSARM Mission Control & CoM Telemetry</title>
    <style>
        :root {
            --bg-main: #090d16;
            --bg-card: #131c2e;
            --bg-card-hover: #1a273f;
            --border: #22324f;
            --accent-cyan: #06b6d4;
            --accent-blue: #3b82f6;
            --accent-red: #ef4444;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --accent-purple: #8b5cf6;
            --text-main: #f1f5f9;
            --text-dim: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: var(--bg-main);
            color: var(--text-main);
            min-height: 100vh;
            padding: 16px;
        }
        /* Top Navigation Bar */
        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 12px 20px;
            border-radius: 12px;
            margin-bottom: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }
        .title-group h1 {
            font-size: 1.35rem;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .title-group p {
            font-size: 0.8rem;
            color: var(--text-dim);
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            background: #1e293b;
            border: 1px solid #334155;
            color: #f8fafc;
        }
        .status-badge.WAIT_TARGET { border-color: var(--accent-yellow); color: var(--accent-yellow); }
        .status-badge.APPROACH { border-color: var(--accent-blue); color: #60a5fa; }
        .status-badge.CLOSE_GRIP { border-color: var(--accent-purple); color: #c084fc; }
        .status-badge.LIFT { border-color: #f97316; color: #fb923c; }
        .status-badge.HOLD { border-color: var(--accent-green); color: #4ade80; }
        .status-badge.COMPLETED { border-color: #2dd4bf; color: #2dd4bf; box-shadow: 0 0 12px rgba(45,212,191,0.4); }

        /* Main Dashboard Grid */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1.15fr 1fr;
            gap: 16px;
        }
        @media (max-width: 1024px) {
            .dashboard-grid { grid-template-columns: 1fr; }
        }

        /* Card Container */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
        }
        .card-header {
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .card-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #38bdf8;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .card-body {
            padding: 16px;
        }

        /* Video Container */
        .video-wrapper {
            position: relative;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            aspect-ratio: 4/3;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .video-wrapper img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }
        .video-hud {
            position: absolute;
            bottom: 10px;
            left: 10px;
            right: 10px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(6px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 8px 12px;
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            color: #cbd5e1;
        }

        /* 3D Canvas Box */
        .canvas-box {
            position: relative;
            width: 100%;
            height: 380px;
            background: radial-gradient(circle at center, #111a2e 0%, #080c16 100%);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #1f2a40;
            cursor: grab;
        }
        .canvas-box:active { cursor: grabbing; }
        .canvas-overlay-hint {
            position: absolute;
            top: 10px;
            right: 10px;
            font-size: 0.72rem;
            background: rgba(0, 0, 0, 0.6);
            padding: 4px 8px;
            border-radius: 4px;
            color: #94a3b8;
            pointer-events: none;
        }
        .com-legend {
            position: absolute;
            bottom: 10px;
            left: 10px;
            font-size: 0.72rem;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 6px;
            padding: 6px 10px;
            display: flex;
            flex-direction: column;
            gap: 4px;
            pointer-events: none;
        }
        .legend-item { display: flex; align-items: center; gap: 6px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; }
        .dot-red { background: #ef4444; box-shadow: 0 0 6px #ef4444; }
        .dot-green { background: #10b981; }
        .dot-blue { background: #38bdf8; }

        /* Metric Grid & Gauges */
        .metrics-subgrid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-top: 16px;
        }
        .metric-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
        }
        .metric-label {
            font-size: 0.75rem;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: #f8fafc;
            display: flex;
            align-items: baseline;
            gap: 4px;
        }
        .metric-unit {
            font-size: 0.75rem;
            font-weight: normal;
            color: var(--text-dim);
        }

        /* Servo Sliders / Angle Bars */
        .servo-bars {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 10px;
        }
        .servo-row {
            display: grid;
            grid-template-columns: 80px 1fr 60px;
            align-items: center;
            gap: 10px;
            font-size: 0.8rem;
        }
        .progress-track {
            height: 8px;
            background: #1e293b;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            border-radius: 4px;
            transition: width 0.1s ease;
        }
        .jerk-fill {
            background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444);
        }

        .footer-bar {
            margin-top: 16px;
            text-align: center;
            font-size: 0.78rem;
            color: #64748b;
        }
    </style>
</head>
<body>

    <!-- Header Topbar -->
    <div class="topbar">
        <div class="title-group">
            <h1>🤖 ASSARM 4-DOF MISSION CONTROL</h1>
            <p>Monocular Pinhole 3D Vision &bull; Dual-Core Real-time Controller &bull; Dynamics & CoM Telemetry</p>
        </div>
        <div id="statusBadge" class="status-badge WAIT_TARGET">
            <span id="statusIndicator">●</span> <span id="statusText">WAIT_TARGET</span>
        </div>
    </div>

    <!-- Main 2-Column Grid -->
    <div class="dashboard-grid">

        <!-- CỘT TRÁI: Luồng Camera & Trạng thái bám vết -->
        <div style="display: flex; flex-direction: column; gap: 16px;">
            <!-- Camera Stream Card -->
            <div class="card">
                <div class="card-header">
                    <span class="card-title">📹 LUỒNG VIDEO CAMERA TRỰC TIẾP</span>
                    <span id="camFpsText" style="font-size: 0.8rem; color: #38bdf8;">FPS: --</span>
                </div>
                <div class="card-body" style="padding: 10px;">
                    <div class="video-wrapper">
                        <img src="/stream.mjpg" alt="Live Camera Video">
                        <div class="video-hud">
                            <span>Camera 3D: <strong id="hudCamCoords">(0.00, 0.00, 0.00) m</strong></span>
                            <span>CPU Temp: <strong id="hudCpuTemp">--°C</strong></span>
                            <span>ESP32: <strong id="hudUartStatus" style="color:#10b981;">OK</strong></span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Tracking & Nhiệm Vụ Card -->
            <div class="card">
                <div class="card-header">
                    <span class="card-title">🎯 TIẾN TRÌNH TIẾP CẬN VÀ SAI SỐ</span>
                    <span id="trackingDistLabel" style="font-size: 0.85rem; font-weight: 600; color: #4ade80;">-- cm</span>
                </div>
                <div class="card-body">
                    <div class="metrics-subgrid" style="margin-top: 0;">
                        <div class="metric-card">
                            <div class="metric-label">Sai số tiếp cận (P_err)</div>
                            <div class="metric-value"><span id="valPosErr">--</span> <span class="metric-unit">cm</span></div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Tốc độ tiếp cận (V_rel)</div>
                            <div class="metric-value"><span id="valVrel">--</span> <span class="metric-unit">cm/s</span></div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Vị trí Ngàm kẹp (EE)</div>
                            <div class="metric-value" style="font-size: 0.95rem;" id="valEePos">[0, 0, 0] cm</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Khối trụ mục tiêu (Target)</div>
                            <div class="metric-value" style="font-size: 0.95rem; color: #facc15;" id="valTargetPos">Đang tìm...</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- CỘT PHẢI: Mô hình 3D tương tác CoM & Động lực học -->
        <div style="display: flex; flex-direction: column; gap: 16px;">
            <!-- 3D Arm Viewer & CoM Card -->
            <div class="card">
                <div class="card-header">
                    <span class="card-title">📐 MÔ HÌNH 3D ROBOT & TÂM TRỌNG LƯỢNG (CoM)</span>
                    <span style="font-size: 0.8rem; color: #94a3b8;">Khối lượng Arm: ~508g</span>
                </div>
                <div class="card-body" style="padding: 10px;">
                    <div class="canvas-box" id="canvasContainer">
                        <canvas id="arm3dCanvas" width="600" height="380"></canvas>
                        <div class="canvas-overlay-hint">🖱️ Giữ chuột trái / Chạm để xoay 3D | Cuộn chuột để Zoom</div>
                        <div class="com-legend">
                            <div class="legend-item"><span class="dot dot-red"></span> <strong>Tâm trọng lượng CoM</strong></div>
                            <div class="legend-item"><span class="dot dot-blue"></span> Khớp xương robot</div>
                            <div class="legend-item"><span class="dot dot-green"></span> Khối trụ mục tiêu</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Động lực học & Độ giật Card -->
            <div class="card">
                <div class="card-header">
                    <span class="card-title">⚡ ĐỘNG LỰC HỌC, QUÁN TÍNH & ĐỘ GIẬT (JERK)</span>
                    <span id="valJerkRms" style="font-size: 0.8rem; color: #38bdf8;">Jerk RMS: -- deg/s³</span>
                </div>
                <div class="card-body">
                    <!-- Hàng chỉ số Quán tính & Mô-men -->
                    <div class="metrics-subgrid" style="margin-top: 0; margin-bottom: 12px;">
                        <div class="metric-card">
                            <div class="metric-label">Tọa độ CoM (X, Y, Z)</div>
                            <div class="metric-value" style="font-size: 0.95rem; color: #f87171;" id="valComCoords">[-1.1, 6.4, 40.6] cm</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Lực quán tính F_inertial</div>
                            <div class="metric-value"><span id="valFinertial">0.00</span> <span class="metric-unit">N</span></div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Mô-men uốn Pitch (M_p)</div>
                            <div class="metric-value"><span id="valMpitch">0.32</span> <span class="metric-unit">N&middot;m</span></div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Tư thế Ngàm (P/Y/Roll)</div>
                            <div class="metric-value" style="font-size: 0.85rem;" id="valOrientation">P: 51° | Y: 0° | T: 51°</div>
                        </div>
                    </div>

                    <!-- Thanh đo góc Servo & Độ giật từng khớp -->
                    <div class="metric-label" style="margin-top: 10px;">Góc quay Servo & Thanh độ giật từng khớp:</div>
                    <div class="servo-bars">
                        <div class="servo-row">
                            <span>S1 (Đế Yaw)</span>
                            <div class="progress-track"><div id="fillS1" class="progress-fill" style="width: 50%;"></div></div>
                            <span id="txtS1" style="text-align: right;">90.0°</span>
                        </div>
                        <div class="servo-row">
                            <span>S2 (Vai)</span>
                            <div class="progress-track"><div id="fillS2" class="progress-fill" style="width: 40%;"></div></div>
                            <span id="txtS2" style="text-align: right;">70.0°</span>
                        </div>
                        <div class="servo-row">
                            <span>S3 (Khuỷu)</span>
                            <div class="progress-track"><div id="fillS3" class="progress-fill" style="width: 35%;"></div></div>
                            <span id="txtS3" style="text-align: right;">60.0°</span>
                        </div>
                        <div class="servo-row">
                            <span>S4 (Cổ tay)</span>
                            <div class="progress-track"><div id="fillS4" class="progress-fill" style="width: 50%;"></div></div>
                            <span id="txtS4" style="text-align: right;">90.0°</span>
                        </div>
                        <div class="servo-row">
                            <span>S5 (Ngàm)</span>
                            <div class="progress-track"><div id="fillS5" class="progress-fill" style="width: 0%; background: #c084fc;"></div></div>
                            <span id="txtS5" style="text-align: right;">MỞ</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

    </div>

    <div class="footer-bar">
        Hệ thống điều khiển ASSARM 4-DOF &bull; Raspberry Pi 4B Dual-Core Isolated Architecture &bull; 100% Offline Canvas 3D
    </div>

    <!-- JAVASCRIPT THUẦN (OFFLINE): 3D PERSPECTIVE RENDERER & TELEMETRY POLLING -->
    <script>
        // --- 1. Bộ kết xuất 3D phối cảnh (Vanilla 3D Perspective Projection Engine) ---
        const canvas = document.getElementById("arm3dCanvas");
        const ctx = canvas.getContext("2d");

        // Camera Orbit Controls
        let rotX = 0.35; // rad (elevation)
        let rotY = -0.55; // rad (azimuth)
        let zoom = 4.2;   // zoom scale factor
        let isDragging = false;
        let lastMouseX = 0, lastMouseY = 0;

        const container = document.getElementById("canvasContainer");

        // Tự động resize canvas theo kích thước container
        function resizeCanvas() {
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
        }
        window.addEventListener("resize", resizeCanvas);
        resizeCanvas();

        // Chuột & Cảm ứng xoay 3D
        container.addEventListener("mousedown", (e) => {
            isDragging = true;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        });
        window.addEventListener("mouseup", () => isDragging = false);
        window.addEventListener("mousemove", (e) => {
            if (!isDragging) return;
            const dx = e.clientX - lastMouseX;
            const dy = e.clientY - lastMouseY;
            rotY += dx * 0.008;
            rotX += dy * 0.008;
            rotX = Math.max(-1.4, Math.min(1.4, rotX));
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        });

        // Hỗ trợ cảm ứng trên điện thoại
        container.addEventListener("touchstart", (e) => {
            if (e.touches.length === 1) {
                isDragging = true;
                lastMouseX = e.touches[0].clientX;
                lastMouseY = e.touches[0].clientY;
            }
        });
        window.addEventListener("touchend", () => isDragging = false);
        window.addEventListener("touchmove", (e) => {
            if (!isDragging || e.touches.length !== 1) return;
            const dx = e.touches[0].clientX - lastMouseX;
            const dy = e.touches[0].clientY - lastMouseY;
            rotY += dx * 0.01;
            rotX += dy * 0.01;
            rotX = Math.max(-1.4, Math.min(1.4, rotX));
            lastMouseX = e.touches[0].clientX;
            lastMouseY = e.touches[0].clientY;
        });

        // Cuộn chuột zoom in/out
        container.addEventListener("wheel", (e) => {
            e.preventDefault();
            zoom *= (e.deltaY > 0 ? 0.92 : 1.08);
            zoom = Math.max(2.0, Math.min(10.0, zoom));
        });

        // Phép chiếu 3D -> 2D
        function project3D(x, y, z) {
            // Hệ tọa độ Robot: X sang phải, Y hướng tới trước, Z hướng lên cao
            // Dịch tâm về giữa vùng làm việc: Y ~ 20cm, Z ~ 35cm
            let px = x;
            let py = y - 18.0;
            let pz = z - 32.0;

            // Xoay quanh trục Z (rotY)
            let cosY = Math.cos(rotY), sinY = Math.sin(rotY);
            let x1 = px * cosY - py * sinY;
            let y1 = px * sinY + py * cosY;
            let z1 = pz;

            // Xoay quanh trục X (rotX)
            let cosX = Math.cos(rotX), sinX = Math.sin(rotX);
            let y2 = y1 * cosX - z1 * sinX;
            let z2 = y1 * sinX + z1 * cosX;
            let x2 = x1;

            // Phối cảnh Perspective
            const distance = 80.0;
            const fov = 350.0;
            const depth = distance + y2;
            const scale = (fov / Math.max(10.0, depth)) * (zoom / 4.0);

            const cx = canvas.width / 2;
            const cy = canvas.height / 2;

            return {
                x: cx + x2 * scale,
                y: cy - z2 * scale,
                depth: depth,
                scale: scale
            };
        }

        // Dữ liệu Arm Joints & CoM hiện tại
        let currentJoints = [
            {name: "Base", pos: [0.0, 0.0, 56.0]},
            {name: "J1",   pos: [0.0, 0.0, 54.6]},
            {name: "J2",   pos: [-2.6, 0.0, 50.9]},
            {name: "J3",   pos: [-0.6, 8.4, 39.5]},
            {name: "J4",   pos: [-0.4, 16.2, 31.8]},
            {name: "EE",   pos: [-0.5, 21.1, 22.6]}
        ];
        let currentCoM = [-1.08, 6.43, 40.62];
        let currentTarget = null;

        // Vòng lặp vẽ 3D Canvas 60 FPS
        function render3D() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // 1. Vẽ lưới mặt phẳng đáy Z = 0
            ctx.strokeStyle = "#1a2538";
            ctx.lineWidth = 1;
            const gridSize = 35;
            const step = 7;
            for (let g = -gridSize; g <= gridSize; g += step) {
                let p1 = project3D(g, 0, 0);
                let p2 = project3D(g, 45, 0);
                ctx.beginPath();
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.stroke();

                let p3 = project3D(-gridSize, g + 20, 0);
                let p4 = project3D(gridSize, g + 20, 0);
                ctx.beginPath();
                ctx.moveTo(p3.x, p3.y);
                ctx.lineTo(p4.x, p4.y);
                ctx.stroke();
            }

            // 2. Vẽ trục tọa độ O (0, 0, 0)
            const o = project3D(0, 0, 0);
            const ox = project3D(8, 0, 0);
            const oy = project3D(0, 8, 0);
            const oz = project3D(0, 0, 8);

            // Trục X (Đỏ)
            ctx.strokeStyle = "#ef4444"; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(ox.x, ox.y); ctx.stroke();
            // Trục Y (Xanh lá)
            ctx.strokeStyle = "#10b981"; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(oy.x, oy.y); ctx.stroke();
            // Trục Z (Xanh dương)
            ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(oz.x, oz.y); ctx.stroke();

            // 3. Vẽ trụ đỡ Base từ Z = 0 lên Base Z = 56
            if (currentJoints.length > 0) {
                const bFloor = project3D(0, 0, 0);
                const bTop = project3D(currentJoints[0].pos[0], currentJoints[0].pos[1], currentJoints[0].pos[2]);
                ctx.strokeStyle = "#334155";
                ctx.lineWidth = 5 * (bTop.scale / 10);
                ctx.beginPath();
                ctx.moveTo(bFloor.x, bFloor.y);
                ctx.lineTo(bTop.x, bTop.y);
                ctx.stroke();
            }

            // 4. Vẽ các thanh xương robot (Bones)
            for (let i = 0; i < currentJoints.length - 1; i++) {
                const ptA = currentJoints[i].pos;
                const ptB = currentJoints[i+1].pos;
                const p2dA = project3D(ptA[0], ptA[1], ptA[2]);
                const p2dB = project3D(ptB[0], ptB[1], ptB[2]);

                // Thanh xương phát sáng Cyber
                ctx.strokeStyle = i === currentJoints.length - 2 ? "#818cf8" : "#38bdf8";
                ctx.lineWidth = Math.max(3, 7 * (p2dA.scale / 10));
                ctx.lineCap = "round";
                ctx.beginPath();
                ctx.moveTo(p2dA.x, p2dA.y);
                ctx.lineTo(p2dB.x, p2dB.y);
                ctx.stroke();
            }

            // 5. Vẽ các khớp cầu (Joint Spheres)
            for (let i = 0; i < currentJoints.length; i++) {
                const pt = currentJoints[i].pos;
                const p2d = project3D(pt[0], pt[1], pt[2]);
                const r = Math.max(3, 6 * (p2d.scale / 10));

                ctx.fillStyle = i === currentJoints.length - 1 ? "#a855f7" : "#0284c7";
                ctx.beginPath();
                ctx.arc(p2d.x, p2d.y, r, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }

            // 6. Vẽ Mục tiêu (Target Cylinder) nếu có
            if (currentTarget) {
                const pT = project3D(currentTarget[0], currentTarget[1], currentTarget[2]);
                const pT_floor = project3D(currentTarget[0], currentTarget[1], 0);

                // Đường dóng rơi xuống sàn
                ctx.strokeStyle = "rgba(16, 185, 129, 0.4)";
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.moveTo(pT.x, pT.y);
                ctx.lineTo(pT_floor.x, pT_floor.y);
                ctx.stroke();
                ctx.setLineDash([]);

                // Khối trụ mục tiêu phát sáng xanh lá
                ctx.fillStyle = "#10b981";
                ctx.beginPath();
                ctx.arc(pT.x, pT.y, 8 * (pT.scale / 10), 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = "#a7f3d0";
                ctx.lineWidth = 2;
                ctx.stroke();

                // Đường nét đứt sai số từ EE -> Target
                if (currentJoints.length > 0) {
                    const ee = currentJoints[currentJoints.length - 1].pos;
                    const pEE = project3D(ee[0], ee[1], ee[2]);
                    ctx.strokeStyle = "#facc15";
                    ctx.lineWidth = 1.5;
                    ctx.setLineDash([3, 3]);
                    ctx.beginPath();
                    ctx.moveTo(pEE.x, pEE.y);
                    ctx.lineTo(pT.x, pT.y);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }
            }

            // 7. VẼ ĐIỂM ĐỎ PHÁT SÁNG TRỌNG TÂM (CENTER OF MASS - CoM)
            if (currentCoM) {
                const pCoM = project3D(currentCoM[0], currentCoM[1], currentCoM[2]);
                const pCoM_floor = project3D(currentCoM[0], currentCoM[1], 0);

                // Đường dóng hình học xuống sàn
                ctx.strokeStyle = "rgba(239, 68, 68, 0.5)";
                ctx.lineWidth = 1.5;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(pCoM.x, pCoM.y);
                ctx.lineTo(pCoM_floor.x, pCoM_floor.y);
                ctx.stroke();
                ctx.setLineDash([]);

                // Vòng tròn bóng đổ sàn
                ctx.fillStyle = "rgba(239, 68, 68, 0.2)";
                ctx.beginPath();
                ctx.arc(pCoM_floor.x, pCoM_floor.y, 6 * (pCoM_floor.scale / 10), 0, Math.PI * 2);
                ctx.fill();

                // Quả cầu đỏ CoM phát sáng rực rỡ
                const radius = Math.max(5, 9 * (pCoM.scale / 10));
                const glow = ctx.createRadialGradient(pCoM.x, pCoM.y, 0, pCoM.x, pCoM.y, radius * 2.2);
                glow.addColorStop(0, "rgba(239, 68, 68, 1)");
                glow.addColorStop(0.5, "rgba(239, 68, 68, 0.8)");
                glow.addColorStop(1, "rgba(239, 68, 68, 0)");

                ctx.fillStyle = glow;
                ctx.beginPath();
                ctx.arc(pCoM.x, pCoM.y, radius * 2.2, 0, Math.PI * 2);
                ctx.fill();

                ctx.fillStyle = "#ef4444";
                ctx.beginPath();
                ctx.arc(pCoM.x, pCoM.y, radius, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = "#fee2e2";
                ctx.lineWidth = 2;
                ctx.stroke();

                // Nhãn chữ CoM
                ctx.fillStyle = "#fca5a5";
                ctx.font = "bold 11px sans-serif";
                ctx.fillText("CoM", pCoM.x + radius + 4, pCoM.y + 4);
            }

            requestAnimationFrame(render3D);
        }
        requestAnimationFrame(render3D);


        // --- 2. Cập nhật Telemetry qua API JSON (10 Hz) ---
        async function fetchTelemetry() {
            try {
                const res = await fetch("/api/telemetry");
                if (res.ok) {
                    const data = await res.json();
                    updateUI(data);
                }
            } catch (err) {
                // Ignore network glitch
            }
        }
        setInterval(fetchTelemetry, 100);

        function updateUI(t) {
            // Badge trạng thái
            const badge = document.getElementById("statusBadge");
            const stText = document.getElementById("statusText");
            if (badge && stText) {
                stText.textContent = t.state || "IDLE";
                badge.className = "status-badge " + (t.state || "WAIT_TARGET");
            }

            // HUD Camera
            const camFps = document.getElementById("camFpsText");
            if (camFps) camFps.textContent = "FPS: " + (t.fps_cam || 0.0);
            const hudCam = document.getElementById("hudCamCoords");
            if (hudCam && t.cam_coords_m) {
                hudCam.textContent = `(${t.cam_coords_m[0]}, ${t.cam_coords_m[1]}, ${t.cam_coords_m[2]}) m`;
            }
            const hudTemp = document.getElementById("hudCpuTemp");
            if (hudTemp) hudTemp.textContent = (t.cpu_temp || 0) + "°C";
            const hudUart = document.getElementById("hudUartStatus");
            if (hudUart) {
                hudUart.textContent = t.uart_ok ? "OK" : "NO";
                hudUart.style.color = t.uart_ok ? "#10b981" : "#ef4444";
            }

            // Sai số & Vận tốc
            const valPosErr = document.getElementById("valPosErr");
            if (valPosErr) valPosErr.textContent = t.pos_err_cm !== undefined ? t.pos_err_cm : "--";
            const distLabel = document.getElementById("trackingDistLabel");
            if (distLabel) distLabel.textContent = (t.pos_err_cm || 0) + " cm";

            const valVrel = document.getElementById("valVrel");
            if (valVrel) valVrel.textContent = t.v_rel_cms !== undefined ? t.v_rel_cms : "--";

            const valEe = document.getElementById("valEePos");
            if (valEe && t.ee_pos_cm) valEe.textContent = `[${t.ee_pos_cm[0]}, ${t.ee_pos_cm[1]}, ${t.ee_pos_cm[2]}] cm`;

            const valTarget = document.getElementById("valTargetPos");
            if (valTarget) {
                if (t.target_cm) {
                    valTarget.textContent = `[${t.target_cm[0]}, ${t.target_cm[1]}, ${t.target_cm[2]}] cm`;
                    valTarget.style.color = "#4ade80";
                } else {
                    valTarget.textContent = "Đang tìm...";
                    valTarget.style.color = "#facc15";
                }
            }

            // Quán tính, Mô-men, CoM
            const valCom = document.getElementById("valComCoords");
            if (valCom && t.com_cm) {
                valCom.textContent = `[${t.com_cm[0]}, ${t.com_cm[1]}, ${t.com_cm[2]}] cm`;
            }
            const valF = document.getElementById("valFinertial");
            if (valF) valF.textContent = (t.f_inertial_N || 0.0).toFixed(3);
            const valM = document.getElementById("valMpitch");
            if (valM) valM.textContent = (t.m_pitch_Nm || 0.0).toFixed(3);

            const valOri = document.getElementById("valOrientation");
            if (valOri && t.orientation) {
                valOri.textContent = `P: ${Math.round(t.orientation.pitch)}° | Y: ${Math.round(t.orientation.yaw)}° | T: ${Math.round(t.orientation.tilt)}°`;
            }

            const valJerk = document.getElementById("valJerkRms");
            if (valJerk) valJerk.textContent = `Jerk RMS: ${t.jerk_total_rms || 0} deg/s³`;

            // Thanh servo
            if (t.joints_fb_deg) {
                const s = t.joints_fb_deg;
                const setBar = (fillId, txtId, deg, minDeg, maxDeg) => {
                    const elFill = document.getElementById(fillId);
                    const elTxt = document.getElementById(txtId);
                    if (elFill) {
                        const pct = Math.max(0, Math.min(100, ((deg - minDeg) / (maxDeg - minDeg)) * 100));
                        elFill.style.width = pct + "%";
                    }
                    if (elTxt) elTxt.textContent = deg.toFixed(1) + "°";
                };
                setBar("fillS1", "txtS1", s[0] || 90, 0, 180);
                setBar("fillS2", "txtS2", s[1] || 90, 0, 180);
                setBar("fillS3", "txtS3", s[2] || 90, 0, 180);
                setBar("fillS4", "txtS4", s[3] || 90, 0, 180);

                const elGrip = document.getElementById("txtS5");
                const fillGrip = document.getElementById("fillS5");
                if (elGrip && fillGrip) {
                    const g = s[4] || 0;
                    fillGrip.style.width = Math.min(100, (g / 90) * 100) + "%";
                    elGrip.textContent = g > 45 ? "ĐÓNG (" + Math.round(g) + "°)" : "MỞ (" + Math.round(g) + "°)";
                }
            }

            // Đồng bộ dữ liệu vào Canvas 3D
            if (t.joints_3d && t.joints_3d.length > 0) currentJoints = t.joints_3d;
            if (t.com_cm) currentCoM = t.com_cm;
            currentTarget = t.target_cm || null;
        }
    </script>
</body>
</html>
"""
