import sys
import os
import json
import math
import cv2
import numpy as np
from pathlib import Path

# Cấu hình môi trường hiển thị cho Ubuntu/Linux
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QPushButton, QSplitter, QTextEdit, 
                             QFileDialog, QGroupBox, QRadioButton, QButtonGroup, 
                             QMessageBox, QSizePolicy)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

class ImageCanvas(QWidget):
    """Khu vực hiển thị ảnh 3/4 bên trái, hỗ trợ bắt sự kiện Click chuột"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_win = parent
        self.img_orig = None
        self.img_display = None
        self.points = []
        self.setMouseTracking(True)
        
        # SỬA LỖI: Cho phép Widget Canvas tự do co giãn theo mọi hướng
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Đặt kích thước gợi ý tối thiểu nhỏ để không làm cứng cửa sổ chính
        self.setMinimumSize(400, 300)

    def set_image(self, cv_img):
        self.img_orig = cv_img.copy()
        self.update_display()

    def update_display(self):
        if self.img_orig is None:
            return
        
        self.img_display = self.img_orig.copy()
        h, w = self.img_display.shape[:2]
        cfg = self.main_win.config

        # 1. Vẽ ROI (Màu xanh dương)
        r = cfg["roi"]
        if r != [0, 0, 0, 0] and len(r) == 4:
            cv2.rectangle(self.img_display, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (255, 0, 0), 2)
        
        # 2. Vẽ Jaws Open/Close (Màu xanh lá)
        for state in ["open", "close"]:
            pts = cfg["jaw_info"][state]["points"]
            dists = cfg["jaw_info"][state]["distance"]
            if len(pts) >= 2:
                p1, p2 = tuple(pts[0]), tuple(pts[1])
                cv2.line(self.img_display, p1, p2, (0, 255, 0), 2)
                cv2.putText(self.img_display, f"{dists[0]:.1f}px", p2, 1, 1, (0, 255, 0), 1)
            if len(pts) >= 4:
                p3, p4 = tuple(pts[2]), tuple(pts[3])
                cv2.line(self.img_display, p3, p4, (0, 255, 0), 2)
                cv2.putText(self.img_display, f"{dists[1]:.1f}px", p4, 1, 1, (0, 255, 0), 1)

        # 3. Vẽ Đoạn thẳng Distance (Màu xanh lá)
        d_pts = cfg["distance"]["points"]
        if len(d_pts) >= 2:
            p1, p2 = tuple(d_pts[0]), tuple(d_pts[1])
            cv2.line(self.img_display, p1, p2, (0, 255, 0), 2)
            cv2.putText(self.img_display, f"{cfg['distance']['dist1']:.1f}px", p2, 1, 1, (0, 255, 0), 1)
        if len(d_pts) >= 3:
            p3 = tuple(d_pts[2])
            cv2.line(self.img_display, p1, p3, (0, 255, 0), 2)
            cv2.putText(self.img_display, f"{cfg['distance']['dist2']:.1f}px", p3, 1, 1, (0, 255, 0), 1)

        # 4. Vẽ Mask (Màu vàng)
        m = cfg["mask"]
        if m["center"] != [0, 0]:
            cv2.circle(self.img_display, tuple(m["center"]), 5, (0, 255, 255), -1)
            if m["min_radius"] > 0:
                cv2.circle(self.img_display, tuple(m["center"]), m["min_radius"], (0, 255, 255), 2)
            if m["max_radius"] > 0:
                cv2.circle(self.img_display, tuple(m["center"]), m["max_radius"], (0, 255, 255), 2)

        # 5. Vẽ Detected Circles (Màu cam)
        for circ in cfg["circles_info"]["detected_circles"]:
            cv2.circle(self.img_display, tuple(circ['center']), int(circ['radius']), (0, 165, 255), 2)
            cv2.circle(self.img_display, tuple(circ['center']), 2, (0, 0, 255), -1)

        # 6. Vẽ Reference Points (Màu trắng)
        ref_pts = cfg["ref_points"]
        for p in ref_pts: 
            cv2.circle(self.img_display, p, 4, (255, 255, 255), -1)
        if len(ref_pts) == 2:
            cv2.arrowedLine(self.img_display, ref_pts[0], ref_pts[1], (255, 255, 255), 2, tipLength=0.1)

        # 7. Vẽ Workpiece Check ROI (Màu hồng cánh sen)
        wpc = cfg["workpiece_check"]
        if wpc["center"] != [0, 0] and wpc["center_roi_radius_px"] > 0:
            cv2.circle(self.img_display, tuple(wpc["center"]), wpc["center_roi_radius_px"], (255, 0, 255), 2)

        # 8. Vẽ các điểm đang click dở dang
        for p in self.points:
            cv2.circle(self.img_display, p, 4, (0, 0, 255), -1)

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.img_display is None:
            painter.setPen(QPen(Qt.GlobalColor.gray, 1, Qt.PenStyle.DashLine))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Bấm 'Mở ảnh nguồn' để bắt đầu cấu hình")
            return

        h, w, ch = self.img_display.shape
        bytes_per_line = ch * w
        q_img = QImage(self.img_display.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Tự động co giãn theo kích thước thực tế của vùng Left Widget khi người dùng kéo vạch chia
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)

    def mousePressEvent(self, event):
        if self.img_orig is None or event.button() != Qt.MouseButton.LeftButton:
            return

        h_orig, w_orig = self.img_orig.shape[:2]
        q_img = QImage(self.img_orig.data, w_orig, h_orig, w_orig*3, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        x_offset = (self.width() - scaled_pixmap.width()) // 2
        y_offset = (self.height() - scaled_pixmap.height()) // 2
        
        x_clicked = event.position().x() - x_offset
        y_clicked = event.position().y() - y_offset
        
        if x_clicked < 0 or x_clicked >= scaled_pixmap.width() or y_clicked < 0 or y_clicked >= scaled_pixmap.height():
            return
            
        scale_x = w_orig / scaled_pixmap.width()
        scale_y = h_orig / scaled_pixmap.height()
        x = int(x_clicked * scale_x)
        y = int(y_clicked * scale_y)

        mode = self.main_win.current_mode()
        if not mode: return

        self.main_win.save_history()

        if mode == 'roi':
            if len(self.points) >= 2: self.points = []
            self.points.append((x, y))
            if len(self.points) == 2:
                x1, y1 = self.points[0]
                x2, y2 = self.points[1]
                self.main_win.config["roi"] = [min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)]
                self.points = []

        elif mode in ["open", "close"]:
            if len(self.points) >= 4: self.points = []
            self.points.append((x, y))
            state = mode
            if len(self.points) == 2:
                dists = math.sqrt((self.points[1][0]-self.points[0][0])**2 + (self.points[1][1]-self.points[0][1])**2)
                self.main_win.config["jaw_info"][state]["distance"].append(dists)
                self.main_win.config["jaw_info"][state]["points"].extend([self.points[0], self.points[1]])
            elif len(self.points) == 4:
                distl = math.sqrt((self.points[3][0]-self.points[2][0])**2 + (self.points[3][1]-self.points[2][1])**2)
                self.main_win.config["jaw_info"][state]["distance"].append(distl)
                self.main_win.config["jaw_info"][state]["points"].extend([self.points[2], self.points[3]])
                self.points = []

        elif mode == "distance":
            if len(self.points) >= 3: self.points = []
            self.points.append((x, y))
            if len(self.points) == 2:
                dist1 = math.sqrt((self.points[1][0]-self.points[0][0])**2 + (self.points[1][1]-self.points[0][1])**2)
                self.main_win.config["distance"]["dist1"] = dist1
                self.main_win.config["distance"]["points"].extend([self.points[0], self.points[1]])
            elif len(self.points) == 3:
                dist2 = math.sqrt((self.points[2][0]-self.points[0][0])**2 + (self.points[2][1]-self.points[0][1])**2)
                self.main_win.config["distance"]["dist2"] = dist2
                self.main_win.config["distance"]["points"].append(self.points[2])
                self.points = []

        elif mode == 'mask':
            if len(self.points) >= 3:
                self.points = []
                self.main_win.config["mask"] = {"center": [0, 0], "min_radius": 0, "max_radius": 0}
            self.points.append((x, y))
            if len(self.points) == 1:
                self.main_win.config["mask"]["center"] = [x, y]
            elif len(self.points) == 2:
                dist = math.sqrt((x-self.points[0][0])**2 + (y-self.points[0][1])**2)
                self.main_win.config["mask"]["min_radius"] = int(dist)
            elif len(self.points) == 3:
                dist = math.sqrt((x-self.points[0][0])**2 + (y-self.points[0][1])**2)
                self.main_win.config["mask"]["max_radius"] = int(dist)
                self.points = []

        elif mode == 'circle':
            self.points.append((x, y))
            if len(self.points) == 2:
                r = math.sqrt((x - self.points[0][0])**2 + (y - self.points[0][1])**2)
                self.main_win.config["circles_info"]["detected_circles"].append({'center': self.points[0], 'radius': r})
                self.points = []

        elif mode == 'ref':
            if len(self.main_win.config["ref_points"]) >= 2:
                self.main_win.config["ref_points"] = []
            self.main_win.config["ref_points"].append((x, y))

        elif mode == 'wpc':
            self.points.append((x, y))
            if len(self.points) == 2:
                p1, p2 = self.points[0], self.points[1]
                r = int(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))
                self.main_win.config["workpiece_check"]["center"] = self.points[0]
                self.main_win.config["workpiece_check"]["center_roi_radius_px"] = r
                
                img_gray = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2GRAY)
                mask = np.zeros(img_gray.shape, dtype=np.uint8)
                cv2.circle(mask, p1, r, 255, -1)
                avg_brightness = cv2.mean(img_gray, mask=mask)[0]
                self.main_win.config["workpiece_check"]["brightness_threshold"] = int(avg_brightness)
                self.points = []

        self.update_display()
        self.main_win.update_json_text()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hikrobot Chuck Vision Config Tool Engine [PyQt6]")
        self.resize(1280, 720)
        
        self.default_config = {
            "roi": [0, 0, 0, 0],
            "mask": {"center": [0, 0], "min_radius": 0, "max_radius": 0},
            "circles_info": {"detected_circles": [], "min_radius": 0, "max_radius": 0, "min_dist": 0},
            "ref_points": [],
            "jaw_info": {
                "open": {"points": [], "distance": []},
                "close": {"points": [], "distance": []}
            },
            "distance": {"points": [], "dist1": 0, "dist2": 0},
            "workpiece_check": {
                "center": [], "center_roi_radius_px": 0, "brightness_threshold": 0,
                "hough_wpc_min_radius": 0, "hough_wpc_max_radius": 0, "template_path": ""
            }
        }
        self.config = self.default_config.copy()
        self.history = []
        self.cv_img = None
        self.config_save_path = None

        self.init_ui()

    def init_ui(self):
        # Bộ chia chính màn hình (Splitter)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # SỬA LỖI KÉO DÃN: Cấu hình độ dày và đường vạch chia rõ ràng để dễ kéo trên Ubuntu
        main_splitter.setHandleWidth(7)
        main_splitter.setStyleSheet("QSplitter::handle { background-color: #bdc3c7; }")
        self.setCentralWidget(main_splitter)

        # ---------------- KHU VỰC BÊN TRÁI (3/4) ----------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        top_bar = QHBoxLayout()
        self.btn_open_img = QPushButton("📂 Mở ảnh nguồn")
        self.btn_open_img.clicked.connect(self.open_image)
        self.btn_undo = QPushButton("↩️ Quay lại (Undo)")
        self.btn_undo.clicked.connect(self.undo_action)
        self.btn_clear = QPushButton("🗑️ Xóa hết")
        self.btn_clear.clicked.connect(self.clear_all)
        top_bar.addWidget(self.btn_open_img)
        top_bar.addWidget(self.btn_undo)
        top_bar.addWidget(self.btn_clear)
        top_bar.addStretch()
        left_layout.addLayout(top_bar)

        self.canvas = ImageCanvas(self)
        left_layout.addWidget(self.canvas, stretch=1)

        mode_group = QGroupBox("Chế độ lựa chọn thông số đo")
        mode_layout = QHBoxLayout(mode_group)
        # SỬA LỖI CỨNG CỬA SỔ: Cho phép panel chế độ thu nhỏ linh hoạt
        mode_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        
        self.btn_group = QButtonGroup(self)
        modes = [
            ("roi", "Vùng ROI [R]"),
            ("mask", "Vòng mặt nạ [M]"),
            ("circle", "Lỗ Chấu [O]"),
            ("ref", "Góc tham chiếu [F]"),
            ("open", "Chấu Mở [K]"),
            ("close", "Chấu Đóng [H]"),
            ("distance", "Khoảng cách [D]"),
            ("wpc", "Vùng Phôi [W]")
        ]
        
        for i, (mode_id, mode_name) in enumerate(modes):
            radio = QRadioButton(mode_name)
            if i == 0: radio.setChecked(True)
            mode_layout.addWidget(radio)
            self.btn_group.addButton(radio, i)
            radio.setProperty("mode_id", mode_id)

        left_layout.addWidget(mode_group)
        
        # Đặt chính sách co giãn linh hoạt cho widget bên trái
        left_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_widget.setMinimumSize(500, 400)
        main_splitter.addWidget(left_widget)

        # ---------------- KHU VỰC BÊN PHẢI (1/4) ----------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)

        json_group = QGroupBox("Cấu trúc Config JSON hiện tại")
        json_layout = QVBoxLayout(json_group)
        
        self.json_viewer = QTextEdit()
        self.json_viewer.setReadOnly(True)
        self.json_viewer.setFont(QFont("Courier New", 10))
        
        # SỬA LỖI: Đặt SizePolicy cho khung chữ JSON có khả năng thu hẹp xuống mức rất nhỏ
        self.json_viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.json_viewer.setMinimumWidth(150) 
        json_layout.addWidget(self.json_viewer)
        right_layout.addWidget(json_group, stretch=1)

        bottom_right_layout = QHBoxLayout()
        self.btn_open_config = QPushButton("📁 Tải Config")
        self.btn_open_config.clicked.connect(self.load_config_file)
        self.btn_save_config = QPushButton("💾 Lưu Config")
        self.btn_save_config.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
        self.btn_save_config.clicked.connect(self.save_config_file)
        
        bottom_right_layout.addWidget(self.btn_open_config)
        bottom_right_layout.addWidget(self.btn_save_config)
        right_layout.addLayout(bottom_right_layout)

        # Đặt chính sách co giãn linh hoạt cho widget bên phải
        right_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        right_widget.setMinimumSize(250, 400)
        main_splitter.addWidget(right_widget)

        # THAY ĐỔI QUAN TRỌNG: Thiết lập lại cơ chế phân phối kích thước của QSplitter
        # Tham số đầu tiên (0) và (1) tương ứng với độ ưu tiên co giãn khi kéo cửa sổ tổng
        main_splitter.setStretchFactor(0, 3) # Cực bên trái giữ tỷ lệ 3 phần
        main_splitter.setStretchFactor(1, 1) # Cực bên phải giữ tỷ lệ 1 phần
        
        # Đặt kích thước phân vùng ban đầu (Tổng là 1280)
        main_splitter.setSizes([960, 320])
        self.update_json_text()

    def current_mode(self):
        selected_button = self.btn_group.checkedButton()
        if selected_button:
            return selected_button.property("mode_id")
        return None

    def save_history(self):
        import copy
        self.history.append(copy.deepcopy(self.config))
        if len(self.history) > 20: self.history.pop(0)

    def undo_action(self):
        if self.history:
            self.config = self.history.pop()
            self.canvas.points = []
            self.canvas.update_display()
            self.update_json_text()
        else:
            QMessageBox.information(self, "Thông báo", "Không có hành động nào để hoàn tác.")

    def clear_all(self):
        self.save_history()
        self.config = self.default_config.copy()
        if self.cv_img is not None:
            self.config["roi"] = [0, 0, self.cv_img.shape[1], self.cv_img.shape[0]]
        self.canvas.points = []
        self.canvas.update_display()
        self.update_json_text()

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh mâm cặp", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.cv_img = cv2.imread(file_path)
            if self.cv_img is None:
                QMessageBox.critical(self, "Lỗi", "Không thể đọc tệp ảnh này.")
                return
            self.config["roi"] = [0, 0, self.cv_img.shape[1], self.cv_img.shape[0]]
            self.canvas.set_image(self.cv_img)
            self.update_json_text()

    def update_json_text(self):
        self.json_viewer.setText(json.dumps(self.config, indent=4, ensure_ascii=False))

    def load_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Tải file cấu hình JSON", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    loaded_cfg = json.load(f)
                if "roi" in loaded_cfg and "mask" in loaded_cfg:
                    self.config = loaded_cfg
                    self.config_save_path = file_path
                    self.update_json_text()
                    if self.cv_img is not None:
                        self.canvas.update_display()
                    QMessageBox.information(self, "Thành công", f"Đã tải cấu hình từ:\n{file_path}")
                else:
                    QMessageBox.warning(self, "Cảnh báo", "File JSON không đúng cấu trúc tham số của hệ thống.")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể đọc file JSON:\n{str(e)}")

    def save_config_file(self):
        circles = self.config["circles_info"]["detected_circles"]
        if circles:
            radii = [c['radius'] for c in circles]
            self.config["circles_info"]["min_radius"] = int(min(radii) - 2)
            self.config["circles_info"]["max_radius"] = int(max(radii) + 2)
            if len(circles) >= 2:
                dists = []
                for i in range(len(circles)):
                    for j in range(i + 1, len(circles)):
                        p1, p2 = circles[i]['center'], circles[j]['center']
                        dists.append(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))
                self.config["circles_info"]["min_dist"] = int(min(dists) - 5) if min(dists) > 5 else 5

        wpc = self.config["workpiece_check"]
        if wpc["center_roi_radius_px"] > 0:
            r = wpc["center_roi_radius_px"]
            wpc["hough_wpc_min_radius"] = int(r * 0.7)
            wpc["hough_wpc_max_radius"] = int(r * 1.3)
            wpc["hough_wpc_min_dist"] = int(r * 0.5)

        self.update_json_text()

        default_name = "chuck_config.json" if not self.config_save_path else self.config_save_path
        file_path, _ = QFileDialog.getSaveFileName(self, "Lưu cấu hình JSON", default_name, "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.config, f, indent=4, ensure_ascii=False)
                self.config_save_path = file_path
                QMessageBox.information(self, "Thành công", f"Đã lưu tệp cấu hình thành công tại:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể ghi dữ liệu ra file:\n{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())