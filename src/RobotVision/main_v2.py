"""
===========================================
Tự động detect, đo bằng 2 tâm của lỗ chấu 
===========================================
"""
import argparse
from pathlib import Path
import logging
import json
import cv2
from datetime import datetime
import get_chuck_angle_mvs
import get_chuck_angle_local_v2
from vision_system import VisionSystem
from dotenv import load_dotenv
import os
import numpy as np
import copy
def center_detection(image_rectified, cfg):
    test_image = copy.deepcopy(image_rectified)
    # Tự động detect tâm chấu 
    p1, p2 = cfg["mask"].get("param1", 50), cfg.get("param2", 40)
    minR, maxR = cfg["mask"].get("minR", 650), cfg["mask"].get("maxR", 800)
    rw, rh = cfg["roi"][2], cfg["roi"][3]
    # Ép kiểu dữ liệu an toàn để OpenCV không bị crash
    if p1 < 1: p1 = 1
    if p2 < 1: p2 = 1
    if minR >= maxR: maxR = minR + 1
    print(f"p1: {p1}, p2: {p2}, minR: {minR}, maxR: {maxR}")
    blurred = cv2.GaussianBlur(image_rectified, (9, 9), 2)
    circles = cv2.HoughCircles(
        cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY),
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=1000, # Đặt minDist cực lớn để ép hệ thống chỉ lấy 1 đường tròn rõ nhất
        param1=p1, 
        param2=p2, 
        minRadius=minR, 
        maxRadius=maxR
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        c_x, c_y, r = circles[0, 0]
        best_center = (c_x, c_y, r)
        cfg["mask"]["center"] = [int(c_x), int(c_y)]
    else: 
        print("Không phát hiện được lỗ trên chấu, sử dụng tâm mặc định")
        cfg["mask"]["center"] = [1390, 1090]
        r = 700
    cv2.circle(test_image, (cfg["mask"]["center"][0], cfg["mask"]["center"][1]), r, (0,255,0), -1)
    cv2.imwrite("debug_center_detection.jpg", test_image)
    print(f"center sau khi detect: {cfg['mask']['center']}")
    # create roi from center
    cfg["roi"][0] = cfg["mask"]["center"][0] - rw//2 if cfg["mask"]["center"][0] - rw//2 > 0 else 0 
    cfg["roi"][1] = cfg["mask"]["center"][1] - rh//2 if cfg["mask"]["center"][1] - rh//2 > 0 else 0
    print(f"config roi sau khi detect center: {cfg['roi']}")
    # workpiece region
    p1, p2 = cfg["workpiece_check"].get("param1", 50), cfg["workpiece_check"].get("param2", 30)
    minR, maxR = cfg["workpiece_check"].get("minRadius", 300), cfg["workpiece_check"].get("maxRadius", 400)
    # Ép kiểu dữ liệu an toàn để OpenCV không bị crash
    if p1 < 1: p1 = 1
    if p2 < 1: p2 = 1
    if minR >= maxR: maxR = minR + 1
    # chỉ detect trong vùng roi
    img_roi = blurred[cfg["roi"][1]:cfg["roi"][1]+rh, cfg["roi"][0]:cfg["roi"][0]+rw]
    print(f"kích thước của img_roi: {img_roi.shape}")
    circles = cv2.HoughCircles(
        cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY),
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=1000, # Đặt minDist cực lớn để ép hệ thống chỉ lấy 1 đường tròn rõ nhất
        param1=p1, 
        param2=p2, 
        minRadius=minR, 
        maxRadius=maxR
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        c_x, c_y, r = circles[0, 0]
        best_center = (c_x, c_y, r)
        cfg["workpiece_check"]["center"] = [int(c_x) + cfg["roi"][0], int(c_y) + cfg["roi"][1]]
    else: 
        cfg["workpiece_check"]["center"] = [1390, 1090]
        r = 180
    # so sánh 2 tâm đã detect được, nếu quá xa nhau thì chọn giá trị mặc định cho cả 2 tâm
    dist = np.sqrt((cfg["mask"]["center"][0] - cfg["workpiece_check"]["center"][0])**2 + (cfg["mask"]["center"][1] - cfg["workpiece_check"]["center"][1])**2)
    print(f"Khoảng cách giữa 2 tâm: {dist}")
    if dist > 100:
        print("Khoảng cách giữa 2 tâm quá lớn, sử dụng giá trị mặc định cho cả 2 tâm")
        cfg["mask"]["center"] = [1390, 1090]
        cfg["workpiece_check"]["center"] = [1390, 1090]
        # create roi from center
        cfg["roi"][0] = cfg["mask"]["center"][0] - rw//2 if cfg["mask"]["center"][0] - rw//2 > 0 else 0 
        cfg["roi"][1] = cfg["mask"]["center"][1] - rh//2 if cfg["mask"]["center"][1] - rh//2 > 0 else 0
        print(f"config roi sau khi detect center: {cfg['roi']}")
        r = 180
    cfg["workpiece_check"]["center_roi_radius_px"] = int(r)
    cfg["workpiece_check"]["hough_wpc_min_radius"] = int(r * 0.7)
    cfg["workpiece_check"]["hough_wpc_max_radius"] = int(r * 1.3)
    cfg["workpiece_check"]["hough_wpc_min_dist"] = int(r * 0.5)
    mask = np.zeros(cv2.cvtColor(image_rectified, cv2.COLOR_BGR2GRAY).shape, dtype=np.uint8)
    cv2.circle(mask, tuple(cfg["workpiece_check"]["center"]), cfg["workpiece_check"]["center_roi_radius_px"], 255, -1)
    avg_brightness = cv2.mean(image_rectified, mask=mask)[0]
    cfg["workpiece_check"]["brightness_threshold"] = int(avg_brightness)
    # vẽ kết quả detect center và vùng roi lên ảnh để debug
    debug_image = copy.deepcopy(image_rectified)
    cv2.circle(debug_image, (cfg["mask"]["center"][0], cfg["mask"]["center"][1]), r, (0,255,0), 2)
    cv2.rectangle(debug_image, (cfg["roi"][0], cfg["roi"][1]), (cfg["roi"][0]+cfg["roi"][2], cfg["roi"][1]+cfg["roi"][3]), (255,0,0), 2)
    cv2.circle(debug_image, tuple(cfg["workpiece_check"]["center"]), cfg["workpiece_check"]["center_roi_radius_px"], (0,0,255), 2)
    cv2.imwrite("debug_center_and_roi.jpg", debug_image)    
    return best_center, cfg

    # bắt đầu thuật toán
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Công cụ đo góc xoay của mâm 3 chấu")
    parser.add_argument("--mode", type=str, default="robot", choices = ["local","robot"],help="Chế độ nhận đầu vào: \n + local: nhận ảnh tĩnh từ máy local\n + robot: nhận ảnh từ camera")
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
        # Detect center
        best_center, cfg = center_detection(image_rectified, cfg)
        print(f"config mới: {cfg}")
        get_chuck_angle_local_v2.main_local(image_rectified, cfg, detect = args.detect, edge = args.edge, save = args.save, debug = args.debug)
        # has_workpiece, details = wpc_detector.detect(cv2.cvtColor(image_rectified, cv2.COLOR_BGR2GRAY))
        # get_chuck_angle_local.main_local(image_rectified, cfg, detect = args.detect, edge = args.edge, save = args.save, debug = args.debug)
        # is_open = jaw_detector.detect_open(best_triple)
        
    # Chế độ đọc ảnh từ camera
    if args.mode == "robot":
        #Khởi tạo vision_system
        vision_system = VisionSystem()
        get_chuck_angle_mvs.main_industrial(cfg, detect = args.detect, edge = args.edge, save = args.save, callback = vision_system.callback)

    print("ĐÃ HOÀN THÀNH CHƯƠNG TRÌNH!")