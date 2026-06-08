"""
============================================
Detect thủ công, bằng 2 tâm của lỗ chấu
============================================
"""
import argparse
from pathlib import Path
import logging
import json
import cv2
from datetime import datetime
import get_chuck_angle_mvs
import get_chuck_angle_local
from workpiece_detection.workpiece_check import WorkpieceDetection
from jaw_detection.jaw_check import JawDetection
from vision_system import VisionSystem
from dotenv import load_dotenv
import os
import numpy as np
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Công cụ đo góc xoay của mâm 3 chấu")
    parser.add_argument("--mode", type=str, default="local", choices = ["local","robot"],help="Chế độ nhận đầu vào: \n + local: nhận ảnh tĩnh từ máy local\n + robot: nhận ảnh từ camera")
    parser.add_argument("--detect", type=str, choices= ["circle", "edge"], default = "circle", help="Cách xác định góc xoay\n + circle: sử dụng các lỗ trên chấu để xác định góc xoay.\n+ edge: Sử dụng cạnh của chấu để xác định góc xoay.")
    parser.add_argument("--image", type=str, help="Đường dẫn đến ảnh cần xử lý")
    parser.add_argument("--config", type=str, help="Đường dẫn đến file cấu hình")
    parser.add_argument("--save", type=str, default=True, help = "Lưu ảnh kết quả")
    parser.add_argument("--edge", type=str, choices=["lsd", "canny"], default="lsd", help="Phương pháp xác định cạnh: canny hoặc lsd.")
    parser.add_argument('--debug', type=bool, default=False, choices = [True, False], help='Bật chế độ debug để lưu ảnh debug')
    parser.add_argument('--verbose', action='store_true', help='Bật chế độ chi tiết')
    args = parser.parse_args()
    
    # lấy thông tin biến từ .env
    current_dir = Path(__file__).parent
    env_path = f"{current_dir.parent.parent}/.env"
    load_dotenv(dotenv_path=env_path, override=True)
    HOME_DIR = os.getenv("HOME_DIR")
    CONFIG_DIR = os.getenv("CONFIG_DIR")
    LOG_DIR = os.getenv("LOG_DIR")
    # Lấy thời gian hiện tại
    now = datetime.now()

    # # khởi tạo thư mục logs
    # log_path = f"{HOME_DIR}/logs"
    # Path(log_path).mkdir(parents=True, exist_ok = True)

    # Xác định level log dựa trên tham số verbose
    # Nếu không verbose: Chỉ lấy INFO từ test.py
    # Nếu verbose: Lấy DEBUG từ cả test.py và các module như circle_detection
    log_level = logging.DEBUG if args.verbose else logging.INFO
    # 3. Cấu hình logging
    # force=True để xóa các cấu hình mặc định (ngăn in ra terminal)
    logging.basicConfig(
        filename=f"{LOG_DIR}/{now.strftime("%d-%m-%Y")}.log",
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=log_level,
        force=True 
    )
    
    # Định dạng theo năm-tháng-ngày giờ-phút-giây
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    # 4. Nếu không ở chế độ verbose, tắt log của các module con
    if not args.verbose:
        # Tắt log từ các file được import (edge_detection, circle_detection)
        logging.getLogger('edge_detection').setLevel(logging.WARNING)
        logging.getLogger('circle_detection').setLevel(logging.WARNING)

    logging.info(f"--- CHƯƠNG TRÌNH BẮT ĐẦU {formatted_time}---")
    logging.info(f"Xác định góc xoay của chấu theo phương pháp {args.detect} detection")
    
    # load cấu hình
    config_path = args.config if Path(args.config).is_absolute() else f"{CONFIG_DIR}/{args.config}"
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
    except:
        logging.error("Lỗi đọc file config!")

    # Chế độ dọc ảnh từ local
    if args.mode == "local":
        try:
            # if not Path(args.image).is_absolute():
            #     image_path = current_dir / args.image
            # else: image_path = Path(args.image)
            TEST_IMAGE_DIR = os.getenv("TEST_IMAGE_DIR")
            image_path = args.image if Path(args.image).is_absolute() else f"{TEST_IMAGE_DIR}/{args.image}"
        except:
            logging.error("Bạn đang chọn chế độ lấy ảnh đầu vào từ local, hãy đưa đường dẫn file ảnh trong tham số --image.")
        logging.info(f"----------- Xử lý ảnh: {Path(image_path).name} -----------")
        # Đọc ảnh
        image_rectified = cv2.imread(str(image_path))
        get_chuck_angle_local.main_local(image_rectified, cfg, detect = args.detect, edge = args.edge, save = args.save, debug = args.debug)
        # has_workpiece, details = wpc_detector.detect(cv2.cvtColor(image_rectified, cv2.COLOR_BGR2GRAY))
        # get_chuck_angle_local.main_local(image_rectified, cfg, detect = args.detect, edge = args.edge, save = args.save, debug = args.debug)
        # is_open = jaw_detector.detect_open(best_triple)
        
    # Chế độ đọc ảnh từ camera
    if args.mode == "robot":
        #Khởi tạo vision_system
        vision_system = VisionSystem()
        get_chuck_angle_mvs.main_industrial(cfg, detect = args.detect, edge = args.edge, save = args.save, callback = vision_system.callback)

    print("ĐÃ HOÀN THÀNH CHƯƠNG TRÌNH!")