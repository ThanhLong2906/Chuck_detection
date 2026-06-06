'''================================
Giao diện đo lường tham số từ ảnh/video cho hệ thống thị giác mâm cặp của Hikrobot
Chế đô: Thủ công, đo khoảng cách từ tâm mâm đến tâm của lỗ chấu
===================================
'''
import sys
import os
import json
import math
import cv2
import numpy as np
from pathlib import Path
import copy
from dotenv import load_dotenv
# Cấu hình môi trường hiển thị cho Ubuntu/Linux
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"
import sys
src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))

from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QPushButton, QSplitter, QTextEdit, 
                             QFileDialog, QGroupBox, QRadioButton, QButtonGroup, 
                             QMessageBox, QSizePolicy)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from RobotVision.MvImport.MvCameraControl_class import *
CURRENT_DIR = Path(__file__).parent
load_dotenv(dotenv_path=CURRENT_DIR.parent.parent / ".env", override=True)
DATA_DIR = os.getenv("DATA_DIR")

class CameraWorker(QThread):
    """Luồng phụ đọc dữ liệu video từ Camera Hikrobot tránh gây đơ/treo giao diện chính"""
    frame_received = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.cam = None
        self.mutex = QMutex()

    def run(self):
        # 1. Tìm kiếm thiết bị Hikrobot (GigE hoặc USB)
        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
        if ret != 0 or deviceList.nDeviceNum == 0:
            print("Không tìm thấy camera Hikrobot nào!")
            return

        # 2. Khởi tạo thực thể và mở Camera đầu tiên tìm thấy
        self.cam = MvCamera()
        stDeviceImg = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        
        ret = self.cam.MV_CC_CreateHandle(stDeviceImg)
        if ret != 0: return
        
        ret = self.cam.MV_CC_OpenDevice()
        if ret != 0: return

        # 3. Tối ưu Packet Size đối với dòng mạng GigE
        if stDeviceImg.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret == 0:
                    print(f"Đã tối ưu Packet Size lên: {nPacketSize}")
                else:
                    print(f"Cảnh báo: Không thể set Packet Size, lỗi: {hex(ret & 0xffffffff)}")
            else:
                # Nếu không lấy được giá trị tối ưu, ép về 1500 cho an toàn
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", 1500)
        # lấy exposure time và gain tự động
        self.cam.MV_CC_SetEnumValue("ExposureAuto", 2)
        self.cam.MV_CC_SetEnumValue("GainAuto", 2)
        # Ép về Mono8
        self.cam.MV_CC_SetEnumValue("PixelFormat", 0x01080001)
        # 4. Bắt đầu luồng bắt ảnh từ cảm biến (Grabbing)
        ret = self.cam.MV_CC_StopGrabbing()  # Đảm bảo tắt trạng thái cũ nếu có
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            self.cam.MV_CC_CloseDevice()
            return

        # Cấu hình biến đệm cấu trúc ảnh từ SDK
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        # Đánh dấu trạng thái đang chạy an toàn bằng Mutex
        with QMutexLocker(self.mutex):
            self.is_running = True
        print("Đã kết nối thành công với Camera Hikrobot. Đang Streaming...")

        while True:
            # Kiểm tra trạng thái dừng một cách an toàn
            with QMutexLocker(self.mutex):
                if not self.is_running:
                    break
            # Lấy khung hình từ bộ đệm của SDK (chờ tối đa 1000ms)
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if ret == 0:
                # Trích xuất dữ liệu ảnh thô (Pointer) từ cảm biến
                buf_size = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight
                img_ptr = cast(stOutFrame.pBufAddr, POINTER(c_ubyte * buf_size)).contents
                
                # Chuyển đổi vùng nhớ thô sang mảng NumPy tốc độ cao
                img_np = np.frombuffer(img_ptr, dtype=np.uint8)
                img_gray = img_np.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
                
                # Chỉ phát tín hiệu nếu luồng vẫn được phép chạy
                with QMutexLocker(self.mutex):
                    if self.is_running:
                        # Gửi ảnh xám (Mono8) trực tiếp về giao diện chính xử lý
                        self.frame_received.emit(img_gray)
                
                # Giải phóng bộ đệm hình ảnh để chuẩn bị nhận frame tiếp theo
                self.cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:
                # Nghỉ ngắn nếu chưa nhận được dữ liệu để tránh nghẽn CPU vòng lặp while
                self.msleep(10)

        # 5. Giải phóng tài nguyên phần cứng an toàn khi tắt luồng
        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        print("Đã giải phóng kết nối Camera Hikrobot.")

    def stop(self):
        with QMutexLocker(self.mutex):
            self.is_running = False
        self.wait()


class ImageCanvas(QWidget):
    """Khu vực hiển thị ảnh hoặc video, hỗ trợ bắt sự kiện Click chuột đo thông số"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_win = parent
        self.img_orig = None
        self.img_display = None
        self.points = []
        self.setMouseTracking(True)
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 300)

    def set_image(self, cv_img):
        self.img_orig = cv_img.copy()
        self.update_display()

    def update_display(self):
        if self.img_orig is None:
            return
        
        # Nếu là ảnh Mono8 (ảnh xám từ Hikrobot), convert sang BGR để vẽ màu UI
        if len(self.img_orig.shape) == 2:
            self.img_display = cv2.cvtColor(self.img_orig, cv2.COLOR_GRAY2BGR)
        else:
            self.img_display = self.img_orig.copy()
        h, w = self.img_display.shape[:2]
        cfg = self.main_win.config

        # Nếu đang ở chế độ xem trực tiếp camera, không vẽ đè các thông số đo cấu hình lên
        if self.main_win.is_live_preview:
            self.update()
            return

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
            if len(pts) >= 3:
                p3 = tuple(pts[2])
                cv2.line(self.img_display, p1, p3, (0, 255, 0), 2)
                cv2.putText(self.img_display, f"{dists[1]:.1f}px", p3, 1, 1, (0, 255, 0), 1)

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
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Bấm 'Mở camera' hoặc 'Mở ảnh nguồn' để bắt đầu")
            return

        h, w, ch = self.img_display.shape
        bytes_per_line = ch * w
        q_img = QImage(self.img_display.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)

    def mousePressEvent(self, event):
        # Nếu đang xem video trực tiếp, vô hiệu hóa các tính năng click đo thông số
        if self.main_win.is_live_preview or self.img_orig is None or event.button() != Qt.MouseButton.LeftButton:
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
            if len(self.points) >= 3: self.points = []
            self.points.append((x, y))
            state = mode
            if len(self.points) == 2:
                dists = math.sqrt((self.points[1][0]-self.points[0][0])**2 + (self.points[1][1]-self.points[0][1])**2)
                self.main_win.config["jaw_info"][state]["distance"].append(dists)
                self.main_win.config["jaw_info"][state]["points"].extend([self.points[0], self.points[1]])
            elif len(self.points) == 3:
                distl = math.sqrt((self.points[2][0]-self.points[0][0])**2 + (self.points[2][1]-self.points[0][1])**2)
                self.main_win.config["jaw_info"][state]["distance"].append(distl)
                self.main_win.config["jaw_info"][state]["points"].extend([self.points[2]])
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

                # 4. Tự động cắt và lưu Template (Hốc trống)
                # Lưu vào thư mục cùng cấp với file config
                wpc = self.main_win.config["workpiece_check"]
                tpl_dir = Path(f"{DATA_DIR}/templates")
                tpl_dir.mkdir(exist_ok=True)
                tpl_path = tpl_dir / "config_tpl.jpg"
                
                roi_tpl = self.img_orig[wpc["center"][1]-wpc["center_roi_radius_px"]:wpc["center"][1]+wpc["center_roi_radius_px"], wpc["center"][0]-wpc["center_roi_radius_px"]:wpc["center"][0]+wpc["center_roi_radius_px"]]
                if roi_tpl.size > 0:
                    cv2.imwrite(str(tpl_path), roi_tpl)
                    wpc["template_path"] = str(tpl_path)

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
        self.config = copy.deepcopy(self.default_config)
        self.history = []
        self.cv_img = None
        self.config_save_path = None
        
        # Biến quản lý trạng thái camera mẫu
        self.is_live_preview = False
        self.camera_thread = CameraWorker()
        self.latest_live_frame = None

        self.init_ui()

    def init_ui(self):
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(7)
        main_splitter.setStyleSheet("QSplitter::handle { background-color: #bdc3c7; }")
        self.setCentralWidget(main_splitter)

        # ---------------- KHU VỰC BÊN TRÁI (3/4) ----------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        top_bar = QHBoxLayout()
        
        # Thêm 2 nút phục vụ tính năng Camera mới theo yêu cầu
        self.btn_open_cam = QPushButton("Mở camera")
        self.btn_open_cam.clicked.connect(self.toggle_camera)
        self.btn_capture = QPushButton("Chụp ảnh mẫu")
        self.btn_capture.setEnabled(False)  # Chỉ cho bấm khi đã bật Cam
        self.btn_capture.clicked.connect(self.capture_sample_image)
        
        self.btn_open_img = QPushButton("Mở ảnh nguồn")
        self.btn_open_img.clicked.connect(self.open_image)
        self.btn_undo = QPushButton("↩Quay lại (Undo)")
        self.btn_undo.clicked.connect(self.undo_action)
        self.btn_clear = QPushButton("Xóa hết")
        self.btn_clear.clicked.connect(self.clear_all)
        
        # Sắp xếp đúng thứ tự: Cam -> Chụp -> Mở ảnh nguồn -> Undo -> Clear
        top_bar.addWidget(self.btn_open_cam)
        top_bar.addWidget(self.btn_capture)
        top_bar.addWidget(self.btn_open_img)
        top_bar.addWidget(self.btn_undo)
        top_bar.addWidget(self.btn_clear)
        top_bar.addStretch()
        left_layout.addLayout(top_bar)

        self.canvas = ImageCanvas(self)
        left_layout.addWidget(self.canvas, stretch=1)

        mode_group = QGroupBox("Chế độ lựa chọn thông số đo")
        mode_layout = QHBoxLayout(mode_group)
        mode_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        
        self.btn_group = QButtonGroup(self)
        modes = [
            ("roi", "Vùng ROI"),
            ("mask", "Vòng Mask"),
            ("circle", "Lỗ Chấu"),
            ("ref", "Đường tham chiếu"),
            ("open", "Chấu Mở"),
            ("close", "Chấu Đóng"),
            ("distance", "Khoảng cách"),
            ("wpc", "Vùng Phôi")
        ]
        
        for i, (mode_id, mode_name) in enumerate(modes):
            radio = QRadioButton(mode_name)
            if i == 0: radio.setChecked(True)
            mode_layout.addWidget(radio)
            self.btn_group.addButton(radio, i)
            radio.setProperty("mode_id", mode_id)

        # Kết nối sự kiện khi người dùng click đổi chế độ
        self.btn_group.idClicked.connect(self.on_mode_changed)

        left_layout.addWidget(mode_group)
        
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
        
        self.json_viewer.setMinimumWidth(150) 
        json_layout.addWidget(self.json_viewer)
        right_layout.addWidget(json_group, stretch=1)

        bottom_right_layout = QHBoxLayout()
        self.btn_open_config = QPushButton("Tải Config")
        self.btn_open_config.clicked.connect(self.load_config_file)
        self.btn_save_config = QPushButton("Lưu Config")
        self.btn_save_config.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
        self.btn_save_config.clicked.connect(self.save_config_file)
        
        bottom_right_layout.addWidget(self.btn_open_config)
        bottom_right_layout.addWidget(self.btn_save_config)
        right_layout.addLayout(bottom_right_layout)

        right_widget.setMinimumSize(250, 400)
        main_splitter.addWidget(right_widget)

        main_splitter.setStretchFactor(0, 3) 
        main_splitter.setStretchFactor(1, 1) 
        main_splitter.setSizes([960, 320])
        self.update_json_text()

    def current_mode(self):
        selected_button = self.btn_group.checkedButton()
        if selected_button:
            return selected_button.property("mode_id")
        return None
    
    # Bắt sự kiện chuyển chế độ Vùng phôi để hiển thị cảnh báo
    def on_mode_changed(self, btn_id):
        selected_button = self.btn_group.button(btn_id)
        if selected_button:
            mode_id = selected_button.property("mode_id")
            # Nếu người dùng chọn vào chế độ Vùng Phôi [W] (wpc)
            if mode_id == "wpc":
                QMessageBox.warning(
                    self, 
                    "Lưu ý cấu hình phôi", 
                    "Khi đo thông số này phải có phôi trong chấu để hệ thống tính toán ngưỡng độ sáng chính xác!"
                )

    def save_history(self):
        import copy
        self.history.append(copy.deepcopy(self.config))
        if len(self.history) > 20: self.history.pop(0)

    def undo_action(self):
        if self.is_live_preview: return
        if self.history:
            self.config = self.history.pop()
            self.canvas.points = []
            self.canvas.update_display()
            self.update_json_text()
        else:
            QMessageBox.information(self, "Thông báo", "Không có hành động nào để hoàn tác.")

    def clear_all(self):
        if self.is_live_preview: return
        self.save_history()
        self.config = copy.deepcopy(self.default_config)
        if self.cv_img is not None:
            self.config["roi"] = [0, 0, self.cv_img.shape[1], self.cv_img.shape[0]]
        self.canvas.points = []
        self.canvas.update_display()
        self.update_json_text()

    def open_image(self):
        # Nếu đang xem camera trực tiếp, yêu cầu tắt đi trước khi chọn file ảnh có sẵn
        if self.is_live_preview:
            self.toggle_camera()

        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh mâm cặp", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.cv_img = cv2.imread(file_path)
            if self.cv_img is None:
                QMessageBox.critical(self, "Lỗi", "Không thể đọc tệp ảnh này.")
                return
            self.config["roi"] = [0, 0, self.cv_img.shape[1], self.cv_img.shape[0]]
            self.canvas.set_image(self.cv_img)
            self.update_json_text()

    def toggle_camera(self):
        """Bật/Tắt chế độ Streaming live từ Camera"""
        if not self.is_live_preview:
            # Chuyển trạng thái sang bật camera
            self.btn_open_img.setEnabled(False)
            self.btn_open_cam.setText("Tắt camera")
            self.btn_open_cam.setStyleSheet("background-color: #e74c3c; color: white;")
            self.btn_capture.setEnabled(True)
            self.is_live_preview = True
            
            # kết nối camera 
            self.camera_thread.frame_received.connect(self.update_live_frame)
            self.camera_thread.start()
        else:
            # Tắt luồng camera
            self.is_live_preview = False
            self.btn_capture.setEnabled(False)
            self.btn_open_img.setEnabled(True)
            self.btn_open_cam.setText("camera")
            self.btn_open_cam.setStyleSheet("")
            
            if self.camera_thread:
                self.camera_thread.stop()
                # self.camera_thread = None
                
            # Trả lại màn hình hiển thị cũ nếu có
            if self.cv_img is not None:
                self.canvas.update_display()
            else:
                self.canvas.img_orig = None
                self.canvas.img_display = None
                self.canvas.update()

    def update_live_frame(self, frame):
        """Callback nhận khung hình trực tiếp từ luồng QThread đẩy về"""
        if self.is_live_preview:
            self.latest_live_frame = frame.copy()
            # Đưa frame vào canvas hiển thị trực tiếp lên UI
            self.canvas.img_orig = frame
            self.canvas.update_display()

    def capture_sample_image(self):
        """Chụp ảnh từ luồng trực tiếp, chốt khung hình tại đúng thời điểm ấn nút, lưu file và hiển thị"""
        if self.latest_live_frame is None:
            QMessageBox.warning(self, "Cảnh báo", "Không nhận được dữ liệu hình ảnh từ Camera để chụp.")
            return

        # SỬA LỖI: Chốt cứng (Snapshot) khung hình tại ĐÚNG thời điểm ấn nút vào biến tạm freeze_frame
        freeze_frame = self.latest_live_frame.copy()

        # Mở hộp thoại chọn vị trí và tên file muốn lưu (Lúc này luồng cam chạy tiếp cũng không ảnh hưởng)
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Lưu ảnh mẫu chụp từ Camera", 
            "sample_chuck.jpg", 
            "Image Files (*.jpg *.png *.bmp)"
        )
        
        if file_path:
            try:
                # Ghi file ảnh vật lý xuống ổ cứng bằng khung hình đã được đóng băng trước đó
                cv2.imwrite(file_path, freeze_frame)
                
                # Tắt chế độ Live stream tự động để chuyển sang chế độ đo trên ảnh vừa chụp
                self.toggle_camera()
                
                # Đọc lại chính file vừa lưu làm dữ liệu gốc của bộ công cụ đo
                self.cv_img = cv2.imread(file_path)
                self.config["roi"] = [0, 0, self.cv_img.shape[1], self.cv_img.shape[0]]
                
                # Cập nhật Canvas và JSON
                self.canvas.set_image(self.cv_img)
                self.update_json_text()
                
                QMessageBox.information(self, "Thành công", f"Đã chụp, lưu ảnh mẫu và nạp vào bộ cấu hình:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu hình ảnh:\n{str(e)}")

    def update_json_text(self):
        self.json_viewer.setText(json.dumps(self.config, indent=4, ensure_ascii=False))

    def load_config_file(self):
        if self.is_live_preview: return
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
        if self.is_live_preview: return
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

    def closeEvent(self, event):
        print("\n--- BẮT ĐẦU QUY TRÌNH ĐÓNG ỨNG DỤNG AN TOÀN ---")
        
        # 1. Ngắt kết nối tín hiệu PyQt để tránh nhận frame rác khi đang tắt
        try:
            self.camera_thread.frame_received.disconnect(self.update_live_frame)
        except Exception:
            pass

        # 2. Ra lệnh cho luồng phụ dừng và chặn luồng chính đợi luồng phụ tắt hẳn (QThread.wait)
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
            
        print("--- HỆ THỐNG ĐÃ ĐÓNG AN TOÀN ---")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())