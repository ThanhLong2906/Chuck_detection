"""
===========================================
detect thủ công, đo bằng tâm mâm đến tâm của lỗ chấu 
===========================================
"""
import logging
# Import thư viện SDK Hikrobot
from MvImport.MvCameraControl_class import *
from circle_detection.get_chuck_angle_circle import detect_angle_by_circles
from workpiece_detection.workpiece_check import WorkpieceDetection
from jaw_detection.jaw_check_v3 import JawDetection
from edge_detection.get_chuck_angle_edge import get_chuck_angle
from vision_system import VisionSystem
from datetime import datetime
from dotenv import load_dotenv
import time
from pathlib import Path
import cv2

# laod env
current_dir = Path(__file__).parent
env_path = f"{current_dir.parent.parent}/.env"
load_dotenv(dotenv_path=env_path, override=True)
RESULT_IMAGE_DIR = os.getenv("RESULT_IMAGE_DIR")

def main_local(image_rectified, cfg, detect: str = "circle", edge: str = "lsd", save = True, debug=False, demo=False):
    """
    angle: góc xoay của mâm
    best_score: điểm số
    execution_time: thời gian xử lý ảnh
    """
    # Xác định phôi trong chấu
    wpc_detector = WorkpieceDetection(cfg)
    jaw_detector = JawDetection(cfg)
    angle = None
    best_score = None
    execution_time = None
    status = "LOCAL"
    vs = VisionSystem(mode="local")
    if save:
        # Lấy thời gian hiện tại
        now = datetime.now()
        save_dir = f"{RESULT_IMAGE_DIR}/{now.strftime("%d-%m-%Y")}"
        save_path = f"{save_dir}/image_{now.strftime("%H-%M-%S")}.jpg" 
    # bắt đầu đo thời gian 
    start_proc = time.perf_counter()
    # detect góc
    if detect == "circle":
        angle, best_score, best_triple, jaw_circles_full = detect_angle_by_circles(image_rectified, cfg, save=save_path, debug = debug)
        # angle, best_score, best_triple = detect_angle_by_circles(image_rectified, cfg, save=save_path, debug = debug, demo=demo)
    if detect == "edge":
        angle, best_score = get_chuck_angle(image_rectified, cfg, save=save_path, mode = edge, debug = debug)
    # Kết thúc đo
    end_proc = time.perf_counter()
    # Tính thời gian
    execution_time = (end_proc - start_proc) * 1000
    if angle is not None:
        status += " - OK"
        logging.info(f"Góc lệch của mâm: {angle:.2f} độ")
        logging.info(f"Điểm số: {best_score:.2f}")
        logging.info(f"Thời gian xử lý ảnh: {execution_time:.5f} ms")
        print(f"--- KẾT QUẢ MỚI ---")
        print(f"Góc lệch: {angle:.2f} | Score: {best_score:.1f} | Time: {execution_time:.2f}ms")
        if angle < 5:
            logging.info("Chấu đúng vị trí")
            print("Chấu đúng vị trí")
        else:
            logging.info("Chấu không đúng vị trí")
            print("Chấu không đúng vị trí")
    else:
        logging.info("Không phát hiện được lỗ nào phù hợp để tính góc.")
        print("Không phát hiện được lỗ nào phù hợp để tính góc.")
        status += " - NOT FOUND ANGLE"

    has_workpiece, details = wpc_detector.check_by_brightness(cv2.cvtColor(image_rectified, cv2.COLOR_BGR2GRAY), save=save_path)
    # has_workpiece, details = wpc_detector.detect(cv2.cvtColor(image_rectified, cv2.COLOR_BGR2GRAY), save=save_path)
    # get_chuck_angle_local.main_local(image_rectified, cfg, detect = args.detect, edge = args.edge, save = args.save, debug = args.debug)
    print(f"details: {details}")
    if has_workpiece:
        # print("Chấu không có phôi")
        print("có phôi trong chấu")
    else:
        print("chấu không có phôi")
        # print("có phôi trong chấu")
    is_open = jaw_detector.detect(jaw_circles_full, save=save_path)
    # is_open = jaw_detector.detect_open(jaw_circles_full, save=save_path)
    jaw_status = "OPEN" if is_open else "CLOSED" if is_open == False else "NG"
    if is_open is not None:
        if is_open:
            print("chấu mở")
        else:
            print("chấu đóng")
    else: 
        print("Không thể xác định đóng/mở")
    
    # add vào db
    vs.db.insert_result(angle, best_score, execution_time, status, has_workpiece, jaw_status, str(save_path))
    return angle, best_score, best_triple, has_workpiece, is_open