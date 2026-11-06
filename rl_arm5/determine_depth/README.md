# HỆ THỐNG XÁC ĐỊNH TỌA ĐỘ 3D KHỐI TRỤ BẰNG OPENCV & CAMERA PINHOLE

Hệ thống thị giác máy tính nhận diện khối trụ tròn trên nền bảng trắng và xác định tọa độ không gian 3 chiều $(X, Y, Z)$ so với camera đơn (Monocular Camera) sử dụng mô hình Pinhole Camera, tối ưu hóa cho Raspberry Pi 4B và PC Linux.

---

## 1. CẤU TRÚC DỰ ÁN

```
xla/
├── configs/
│   ├── config.yaml               # File cấu hình trung tâm (camera, kích thước vật thể, lọc nhiễu, dải màu HSV)
│   └── camera_calibration.yaml   # Ma trận nội tại K và hệ số méo sau khi calibrate
├── calibration/
│   ├── capture_chessboards.py    # Chụp ảnh bàn cờ caro phục vụ hiệu chuẩn camera
│   └── calibrate_camera.py       # Tính toán ma trận K (fx, fy, cx, cy) và hệ số méo
├── tools/
│   └── hsv_tuner.py              # Công cụ Trackbar cân chỉnh dải màu HSV thời gian thực
├── core/
│   ├── camera_stream.py          # Luồng đọc camera đa luồng (Threaded Capture) tối ưu FPS
│   ├── cylinder_detector.py      # Tiền xử lý, lọc màu HSV, Morphology, trích xuất (u, v) & w_px
│   ├── pinhole_3d.py             # Tính toán tọa độ 3D (X, Y, Z) và Back-projection
│   └── tracker_filter.py         # Bộ lọc làm mịn thời gian thực (Moving Average & Kalman 1D)
├── tests/
│   └── test_pipeline.py          # Bộ kiểm thử tự động (Unit Tests)
├── main.py                       # Pipeline chạy thời gian thực hiển thị HUD thông số
├── requirements.txt              # Danh sách thư viện cần thiết
├── Welcome.md                    # Tài liệu yêu cầu ban đầu
└── README.md                     # Tài liệu hướng dẫn sử dụng chi tiết
```

---

## 2. NGUYÊN LÝ HÌNH HỌC PINHOLE CAMERA

Hệ tọa độ gắn với Camera:
- Trục **Z** (độ sâu): Hướng thẳng từ quang tâm camera ra không gian phía trước.
- Trục **X** (ngang): Hướng sang phải (âm = bên trái tâm quang học, dương = bên phải).
- Trục **Y** (đứng): Hướng xuống dưới (âm = phía trên tâm quang học, dương = phía dưới).

### Công thức tính:
1. **Khoảng cách độ sâu ($Z$):**
   $$Z = \frac{f_x \times D_{real}}{w_{px}}$$
   *(trong đó $D_{real}$ là đường kính thực tế của khối trụ tính theo mét, $w_{px}$ là độ rộng pixel của khối trụ trên ảnh, $f_x$ là tiêu cự camera theo pixel).*

2. **Tọa độ ngang ($X$) và đứng ($Y$):**
   $$X = \frac{(u - c_x) \times Z}{f_x}, \quad Y = \frac{(v - c_y) \times Z}{f_y}$$
   *(trong đó $(u, v)$ là tọa độ tâm pixel của khối trụ, $(c_x, c_y)$ là tâm quang học của camera).*

---

## 3. HƯỚNG DẪN CÀI ĐẶT & CHẠY NHANH

### Bước 1: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy trực tiếp chương trình
Nếu bạn muốn chạy thử nghiệm ngay với thông số camera mặc định:
```bash
python3 main.py
```

Các phím tắt trên cửa sổ hiển thị:
- `q` hoặc `ESC`: Thoát chương trình.
- `u`: Bật/Tắt tính năng khử méo thấu kính (`cv2.undistort`).
- `f`: Bật/Tắt bộ lọc làm mịn tọa độ (`Moving Average` / `Kalman`).
- `m`: Ẩn/Hiện cửa sổ nhỏ Picture-in-Picture hiển thị mặt nạ nhị phân (Mask).
- `s`: Chụp và lưu ảnh khung hình hiện tại (`snapshot_XXX.jpg`).

---

## 4. QUY TRÌNH CÂN CHỈNH ĐỂ ĐẠT ĐỘ CHÍNH XÁC CAO NHẤT

Để hệ thống đo đạc tọa độ $(X, Y, Z)$ chính xác đến từng milimét, hãy thực hiện lần lượt 2 bước sau:

### Bước 4.1: Hiệu chuẩn Camera (Camera Calibration)
1. In mẫu bàn cờ caro $9 \times 6$ góc trong (mỗi ô vuông chuẩn $25\text{ mm}$), dán phẳng lên bề mặt cứng.
2. Chạy script chụp ảnh:
   ```bash
   python3 calibration/capture_chessboards.py
   ```
   - Di chuyển bàn cờ ra trước camera ở nhiều khoảng cách và góc nghiêng khác nhau.
   - Khi hiện dòng chữ màu xanh lá cây `READY: Phat hien ban co!`, nhấn phím `SPACE` để chụp. Chụp từ 15 đến 25 tấm ảnh.
   - Nhấn `q` để kết thúc.
3. Chạy script tính toán ma trận camera:
   ```bash
   python3 calibration/calibrate_camera.py
   ```
   Script sẽ tự động trích xuất $f_x, f_y, c_x, c_y$, vector hệ số méo `distCoeffs` và lưu vào [configs/camera_calibration.yaml](file:///home/hoa-wicom/xla/configs/camera_calibration.yaml).

### Bước 4.2: Cân chỉnh dải màu HSV cho vật thể
Mặc định hệ thống đã tối ưu cho **khối trụ màu XANH DƯƠNG (Blue)** trên nền bảng trắng ($H: 95 \div 130, S: 90 \div 255, V: 50 \div 255$).
Khi môi trường ánh sáng thay đổi hoặc muốn tinh chỉnh lại:
1. Đặt khối trụ lên nền bảng trắng trước camera.
2. Chạy công cụ tuner:
   ```bash
   python3 tools/hsv_tuner.py
   ```
   - Nhấn phím `b` để chọn nhanh chế độ **Blue Mode** (hoặc `r` cho màu đỏ, `c` để tùy biến).
   - Click chuột trái vào khối trụ để xem giá trị màu HSV thực tế.
   - Nhấn phím `s`: Cấu hình tối ưu sẽ được tự động lưu trực tiếp vào [configs/config.yaml](file:///home/hoa-wicom/xla/configs/config.yaml).
   - Nhấn `q` để thoát.

---

## 5. CẤU HÌNH HỆ THỐNG ([configs/config.yaml](file:///home/hoa-wicom/xla/configs/config.yaml))

Bạn có thể chỉnh sửa các thông số thực tế trong file `config.yaml`:
- `color_mode`: `"blue"` (mặc định), `"red"`, hoặc `"custom"`.
- `d_real_m`: Đường kính thực tế của khối trụ (ví dụ: `0.1` mét = $10\text{ cm}$).
- `frame_width`, `frame_height`: Độ phân giải khung hình (mặc định $640 \times 480$ để đảm bảo FPS cao trên Pi 4B).
- `filter_type`: Lựa chọn `"moving_average"` hoặc `"kalman"`.
- `window_size`: Số khung hình trượt để làm mịn (khuyến nghị từ 5 đến 10).

---

## 6. CHẠY KIỂM THỬ ĐƠN VỊ (UNIT TESTS)

Để kiểm tra độ chính xác của các thuật toán hình học 3D, bộ lọc và module nhận diện:
```bash
python3 -m unittest tests/test_pipeline.py
```
Toàn bộ 8 bài test kiểm tra tam giác đồng dạng, back-projection 3D, khử nhiễu lọc mượt và nhận diện ảnh khối trụ xanh dương / đỏ sẽ được chạy tự động.
