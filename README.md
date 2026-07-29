# 🧩 Cube Vision Solver

> **Ứng dụng quét và giải khối Rubik tự động thời gian thực sử dụng Thị giác máy tính.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Algorithm](https://img.shields.io/badge/Solver-Kociemba%20Two--Phase-orange.svg)](#features)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

Hệ thống nhận diện trạng thái các mặt Rubik qua Webcam hoặc Camera IP không dây, phân loại màu sắc chính xác trong nhiều điều kiện ánh sáng và tính toán lời giải tối ưu ($\le 20$ bước) trong thời gian dưới 1 giây.

---

## ✨ Tính năng chính

* 🎯 **Hai chế độ quét linh hoạt (Dual Scan Modes)**
  * **Grid Scan:** Lưới cố định $3 \times 3$ tại trung tâm màn hình, hỗ trợ canh chỉnh thủ công ổn định.
  * **Contour Scan:** Tự động phát hiện viền sticker bằng Canny đa kênh và Morphological Filter—quét tự động ở mọi góc độ/vị trí.
* 🎨 **Nhận diện màu sắc tiên tiến (CIEDE2000)**
  * Tính khoảng cách màu **CIEDE2000** trong không gian **LAB** (khắc phục hoàn toàn nhược điểm của không gian HSV), phân biệt chính xác màu Đỏ và Cam.
  * **Temporal Averaging:** Lấy mẫu màu qua 8 khung hình liên tiếp để loại bỏ nhiễu ánh sáng tức thời.
  * **Median Filter:** Lọc điểm chói lóa và tự động xử lý logo/chữ in trên mặt sticker Trắng.
* ⚙️ **Hiệu chỉnh màu sắc trực quan (On-the-fly Calibration)**
  * Chế độ lấy mẫu màu thực tế linh hoạt, lưu thông số baseline môi trường vào file `calibration.json`.
* ⚡ **Giải thuật Kociemba tối ưu**
  * Tích hợp thuật toán Kociemba Two-Phase cho ra dãy nước đi tiêu chuẩn WCA ($\le 20$ bước) gần như tức thì.
* 📱 **Hỗ trợ đa dạng Camera**
  * Tương thích với Webcam tích hợp, USB Camera ngoài hoặc luồng Camera IP qua WiFi từ điện thoại (MJPEG stream).

---

## 🛠️ Công nghệ & Thư viện sử dụng

| Thành phần | Thư viện / Kỹ thuật | Chức năng |
| :--- | :--- | :--- |
| **Ngôn ngữ** | Python 3.8+ | Cấu trúc và xử lý logic ứng dụng |
| **Thị giác máy tính** | OpenCV, NumPy | Xử lý ảnh, Canny đa kênh, Morphological Closing & Dilate |
| **Xử lý màu sắc** | CIEDE2000 (`colormath`), LAB Space | Đo độ lệch màu theo cảm nhận mắt người, chống chói |
| **Phân cụm** | OpenCV `kmeans` ($k=1$) | Trích xuất màu chủ đạo cho từng vùng sticker |
| **Thuật toán giải** | `kociemba` | Thuật toán Two-Phase tìm lời giải $\le 20$ bước |
| **Giao diện & Overlay** | OpenCV HighGUI | Hiển thị HUD thời gian thực, khung nhận diện và Cube Net 2D |

---

## 🚀 Hướng dẫn cài đặt

### 1. Yêu cầu hệ thống
* **Python 3.8** trở lên
* Webcam máy tính hoặc Điện thoại cài ứng dụng IP Camera

### 2. Cài đặt môi trường

Cài đặt các gói phụ thuộc bằng lệnh:

```bash
# Clone repository về máy
git clone https://github.com/Kyosaki6/Cube-Vision-Solver.git
cd Cube-Vision-Solver

# Tạo và kích hoạt môi trường ảo (Khuyên dùng)
python -m venv venv
source venv/bin/activate      # Trên Linux/macOS
# venv\Scripts\activate       # Trên Windows

# Cài đặt các thư viện yêu cầu
pip install -r requirements.txt
```

### 3. Chạy ứng dụng

**Chạy với Webcam mặc định:**

```bash
python main.py
```

**Chạy với Camera IP / Điện thoại không dây:**

```bash
python main.py --source "http://admin:pass@192.168.1. X:8081/video"
```

---

## 🎮 Bảng điều khiển & Phím tắt

| Phím | Thao tác |
| --- | --- |
| A | Chuyển đổi giữa **Grid Scan** $\leftrightarrow$ **Contour Scan** |
| Space | Chụp/Quét mặt Rubik hiện tại |
| S | Bắt đầu tính toán lời giải (Sau khi đã quét đủ 6 mặt) |
| C | Bật / Tắt chế độ Calibration màu |
| R | Reset các mặt đã quét / Reset cấu hình Calibration |
| U | Hoàn tác (Undo) mặt vừa quét gần nhất |
| [ / ] | Xoay mặt vừa quét ngược chiều / theo chiều kim đồng hồ |
| Q | Thoát ứng dụng |

---

## 🎯 Hướng dẫn Calibration (Hiệu chỉnh màu)

Để hệ thống nhận diện chính xác dưới ánh sáng phòng (đèn vàng, thiếu sáng, chói sáng):

1. Nhấn phím C để vào chế độ Calibration.
2. Đưa mặt Rubik vào đầy khung lưới $3 \times 3$.
3. Nhấn các phím số từ 1 đến 6 để lấy mẫu màu tương ứng:
* 1: Trắng (White) | 2: Đỏ (Red) | 3: Cam (Orange)
* 4: Vàng (Yellow) | 5: Xanh lá (Green) | 6: Xanh dương (Blue)


4. Lặp lại với cả 6 màu.
5. Nhấn phím C một lần nữa để lưu thông số vào file `calibration.json`.

---

## 📂 Cấu trúc dự án

```
Cube-Vision-Solver/
├── main.py                  # Entry point chính điều khiển luồng ứng dụng
├── calibration.json         # File lưu thông số baseline màu sắc thực tế
├── vision/
│   ├── color_detector.py    # Chuyển đổi LAB, tính CIEDE2000 & lưu/tải calibration
│   ├── contour_detector.py  # Canny đa kênh, Morphology, K-means lọc màu sticker
│   ├── state_extractor.py   # Chuyển 6 mặt quét -> Chuỗi 54 ký tự Kociemba
│   └── stream.py            # Đọc luồng video MJPEG từ Camera IP
├── ui/
│   └── overlay.py           # Vẽ lưới Grid, Contour, HUD trạng thái & Cube Net 2D
├── solver/
│   └── cube_solver.py       # Wrapper gọi thuật toán Kociemba Solver
└── requirements.txt         # Danh sách các thư viện cần thiết
```

---

## 👥 Tác giả & Đơn vị thực hiện

* **Sinh viên thực hiện:**
* **Nguyễn Trường Phước** — MSSV: 24100153
* **Huỳnh Quang Thịnh** — MSSV: 24100847
* **Nguyễn Đức Hoàng Nam** — MSSV: 24105787


* **Giảng viên hướng dẫn:** TS. Nguyễn Văn Tới
* **Đơn vị:** **Trường Công nghệ thông tin Phenikaa — Đại học Phenikaa**

---

## Giấy phép

File này giả định dự án sử dụng giấy phép MIT. Nếu bạn muốn một giấy phép khác, hãy chỉnh lại phần này và thêm file `LICENSE` tương ứng.
