"""
===========================================
Tự động detect, đo bằng 2 tâm của lỗ chấu 
===========================================
"""
import numpy as np
import cv2
import json
import time
from pathlib import Path
import logging
# Import thư viện SDK Hikrobot
from MvImport.MvCameraControl_class import *
from circle_detection.get_chuck_angle_circle import detect_angle_by_circles
from edge_detection.get_chuck_angle_edge import get_chuck_angle
from workpiece_detection.workpiece_check import WorkpieceDetection
from jaw_detection.jaw_check import JawDetection
from pathlib import Path
from datetime import datetime

def main_industrial(cfg, detect=str, edge=str, save = True, callback=None):
    """
    angle: góc xoay của mâm
    best_score: điểm số
    execution_time: thời gian xử lý ảnh
    """
    # Xác định phôi trong chấu
    wpc_detector = WorkpieceDetection(cfg)
    # xác định trạng thái đóng mở của chấu
    jaw_detector = JawDetection(cfg)
    angle = None
    best_score = None
    execution_time = None
    if save:
        # Lấy thời gian hiện tại
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
        save_path = f"{Path.home()}/data/images/image_{dt_string}.jpg" 
    while True:
        # Khởi tạo Camera Hikrobot
        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
        if ret != 0 or deviceList.nDeviceNum == 0:
            logging.warning("Không tìm thấy camera Hikrobot nào! Đang thử lại sau 2 giây...")
            print("Console: Đang tìm kiếm Camera...")
            time.sleep(2)
            continue
        
        cam = MvCamera()
        stDeviceImg = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        cam.MV_CC_CreateHandle(stDeviceImg)
        cam.MV_CC_OpenDevice()
        if ret != 0:
            logging.error(f"Không thể mở thiết bị! Mã lỗi: {ret}")
            time.sleep(2)
            continue
        
        # Cấu hình chế độ Hardware Trigger (Robot gửi tín hiệu vật lý)
        # Trong thực tế, chân tín hiệu thường là Line0 hoặc Line1 của camera
        cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
        cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_LINE0) 

        # Bắt đầu thu nhận ảnh
        cam.MV_CC_StartGrabbing()
        stOutFrame = MV_FRAME_OUT()
        
        logging.info("--- HỆ THỐNG SẴN SÀNG ---")
        logging.info("Đang đợi tín hiệu trigger từ Robot...")

        try:
            while True:
                # Đợi ảnh từ camera (Hàm sẽ dừng tại đây cho đến khi có tín hiệu điện từ Robot)
                # Timeout 5000ms (5 giây)
                ret = cam.MV_CC_GetImageBuffer(stOutFrame, 5000) 
                
                if ret == 0:
                    start_proc = time.perf_counter()
                    
                    # Chuyển đổi dữ liệu ảnh từ SDK sang định dạng Numpy/OpenCV
                    # Camera MV-CS060-10GM-PRO thường trả về Mono8 hoặc Bayer
                    pData = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                    memmove(byref(pData), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
                    
                    # Giả định camera đang để mode Mono8 (Trắng đen công nghiệp)
                    img = np.frombuffer(pData, dtype=np.uint8).reshape(
                        stOutFrame.stFrameInfo.nHeight, 
                        stOutFrame.stFrameInfo.nWidth
                    )

                    # Chuyển sang BGR để tương thích với các hàm vẽ trong detect_angle_by_circles
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if detect == "circle":
                        angle, best_score, best_triple, jaw_circles_full = detect_angle_by_circles(img_gray, cfg, save = save_path ,debug=False)
                    if detect == "edge":
                        angle, best_score = get_chuck_angle(img_gray, cfg, save = save_path, debug=False, mode = edge)

                    # has_workpiece, _ = wpc_detector.detect(img_gray)
                    has_workpiece, details = wpc_detector.check_by_brightness(img_gray, save=save_path)
                    # get_chuck_angle_local.main_local(image_rectified, cfg, detect = args.detect, edge = args.edge, save = args.save, debug = args.debug)
                    if has_workpiece:
                        logging.info("có phôi trong chấu")
                    else:
                        logging.info("chấu không có phôi")
                    # is_open = jaw_detector.detect(best_triple, save=save_path)
                    is_open = jaw_detector.detect_open(jaw_circles_full, save=save_path)
                    jaw_status = "OPEN" if is_open else "CLOSED" if is_open == False else "NG"
                    if is_open is not None:
                        if is_open:
                            logging.info("chấu mở")
                        else:
                            logging.info("chấu đóng")
                    else: 
                        logging.info("Không thể xác định đóng/mở")
                    end_proc = time.perf_counter()
                    execution_time = end_proc - start_proc
                    if callback:
                        callback(angle, best_score, execution_time, has_workpiece, jaw_status, save_path)
                    # Giải phóng buffer để nhận ảnh tiếp theo
                    cam.MV_CC_FreeImageBuffer(stOutFrame)
                else:
                    # Nếu quá 5s không có tín hiệu, in thông báo đợi
                    logging.info("Đang đợi tín hiệu trigger...")
                    pass

        except KeyboardInterrupt:
            logging.info("Dừng hệ thống...")
        finally:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    config_path = current_dir / "config.json"
    # load cấu hình
    try:
        with open(str(config_path), 'r') as f:
            cfg = json.load(f)
            main_industrial(cfg)
    except:
        logging.error("Lỗi đọc file config!")
    