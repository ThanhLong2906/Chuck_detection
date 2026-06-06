import time
import json
import socket
from pathlib import Path
import sys
import logging
path_to_src = Path(__file__).resolve().parent.parent
sys.path.append(str(path_to_src))
import cv2
import numpy as np
from RobotVision.MvImport.MvCameraControl_class import * 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RobotCommunication:
    """Quản lý kết nối Socket/Modbus gửi nhận dữ liệu với Robot"""
    def __init__(self, host='0.0.0.0', port=5000):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.conn = None
        logging.info(f"Socket Server đợi Robot kết nối tại port {port}...")

    def wait_for_robot(self):
        self.conn, addr = self.server.accept()
        logging.info(f"Robot đã kết nối từ IP: {addr}")

    def send_result(self, angle, workpiece_state):
        if not self.conn: return
        # Định dạng gói tin gửi Robot (ví dụ: "ANGLE:12.5;WORKPIECE:OPEN\n")
        data = f"ANGLE:{angle:.2f};WP:{workpiece_state}\n"
        try:
            self.conn.sendall(data.encode('utf-8'))
        except socket.error:
            logging.error("Mất kết nối với Robot, đang đợi kết nối lại...")
            self.wait_for_robot()

def init_hikrobot_hardware_trigger():
    """Khởi tạo camera ở chế độ kích hoạt cứng bằng dây từ Robot"""
    chassis = MvCameraControl()
    # (Phần chọn thiết bị Camera đầu tiên trong mạng GigE...)
    # Chuyển đổi chế độ Trigger
    chassis.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
    chassis.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_LINE1) # Pin 2 dây Hirose
    chassis.MV_CC_SetEnumValue("TriggerActivation", 0) # Rising Edge (Cạnh lên)
    
    # Bắt đầu luồng truyền ảnh (Nhưng cam sẽ đứng im đợi xung điện)
    chassis.MV_CC_StartGrabbing()
    return chassis

def main_detection_loop():
    # Load cấu hình JSON mới nhất từ Service 1
    config_path = Path("/opt/chuck_vision/config/active_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Khởi tạo kết nối truyền thông và Camera
    robot_com = RobotCommunication()
    robot_com.wait_for_robot()
    
    cam = init_hikrobot_hardware_trigger()
    logging.info("Hệ thống Vision chạy ngầm đã sẵn sàng. Đang đợi xung kích từ Robot...")

    st_frame_info = MV_FRAME_OUT_INFO_EX()
    memset(byref(st_frame_info), 0, sizeof(st_frame_info))
    
    while True:
        # HÀM NÀY SẼ BLOCK (SLEEP NGẦM) CHO ĐẾN KHI CÓ TÍN HIỆU ĐIỆN TỪ ROBOT
        # Timeout để vô hạn (hoặc vài phút để kiểm tra sống chết)
        ret = cam.MV_CC_GetOneFrameTimeout(byref(pData), nDataSize, byref(st_frame_info), 10000)
        
        if ret == 0:
            logging.info("-> Nhận được tín hiệu Trigger từ Robot! Đang xử lý ảnh...")
            # 1. Chuyển đổi con trỏ pData thành Numpy Array (Ảnh OpenCV)
            img_gray = np.array(pData).reshape(st_frame_info.nHeight, st_frame_info.nWidth)
            
            # angle = ChuckDetector.detect_angle(img_gray, config)
            # wp_state = check_chuck_state_by_distance(..., config)
            angle, wp_state = 45.2, "OPEN" # Giả lập kết quả
            
            # 3. Bắn kết quả về cho Robot qua TCP/IP
            robot_com.send_result(angle, wp_state)
            logging.info(f"-> Đã gửi kết quả cho Robot - Góc: {angle}, Trạng thái: {wp_state}")
        else:
            # Hết thời gian chờ (Timeout 10s) mà Robot chưa kích, tiếp tục vòng lặp đợi tiếp
            pass

if __name__ == "__main__":
    main_detection_loop()