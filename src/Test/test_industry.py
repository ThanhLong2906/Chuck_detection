import argparse
from pathlib import Path
import sys
import logging
import json
import cv2
from datetime import datetime
path_to_src = Path(__file__).resolve().parent.parent
sys.path.append(str(path_to_src))
from RobotVision.circle_detection.get_chuck_angle_circle import detect_angle_by_circles
from RobotVision.edge_detection.get_chuck_angle_edge import get_chuck_angle
from dotenv import load_dotenv
import os
import time

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Công cụ đo góc xoay của mâm 3 chấu")
    # parser.add_argument("--mode", type=str, default="robot", choices = ["local","robot"],help="Chế độ nhận đầu vào: \n + local: nhận ảnh tĩnh từ máy local\n + robot: nhận ảnh từ camera")
    parser.add_argument("--detect", type=str, choices= ["circle", "edge"], default = "circle", help="Cách xác định góc xoay\n + circle: sử dụng các lỗ trên chấu để xác định góc xoay.\n+ edge: Sử dụng cạnh của chấu để xác định góc xoay.")
    parser.add_argument("--image_dir", type=str, help="Đường dẫn đến thử mục ảnh cần test")
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
    RESULT_IMAGE_DIR = os.getenv("RESULT_IMAGE_DIR")
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
        filename=f"{LOG_DIR}/test_{now.strftime("%d-%m-%Y")}.log",
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

    # load cấu hình
    config_path = args.config if Path(args.config).is_absolute() else f"{CONFIG_DIR}/{args.config}"
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
    except:
        logging.error("Lỗi đọc file config!")

    # Đọc tất cả các file ảnh test
    # Folder chứa ảnh
    image_dir = Path(args.image_dir)

    # Các đuôi ảnh hỗ trợ
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    # Lấy danh sách ảnh
    image_paths = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions
    ]
    logging.info(f"--- CHƯƠNG TRÌNH BẮT ĐẦU {formatted_time}---")
    logging.info(f"Xác định góc xoay của chấu theo phương pháp {args.detect} detection")
    angles = []
    best_scores = []
    exe_time = []
    for path in image_paths:
        logging.info(f"----------- Xử lý ảnh: {Path(path).name} -----------")
        # Đọc ảnh
        image_rectified = cv2.imread(str(path))
        # get_chuck_angle_local.main_local(image_rectified, cfg, detect = args.detect, edge = args.edge, save = args.save, debug = args.debug)
        angle = None
        best_score = None
        execution_time = None
        status = "LOCAL"
        # vs = VisionSystem(mode="local")
        # Lấy thời gian hiện tại
        now = datetime.now()
        save_dir = f"{RESULT_IMAGE_DIR}/test_{now.strftime("%d-%m-%Y")}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = f"{save_dir}/image_{now.strftime("%H-%M-%S")}.jpg" 
        # bắt đầu đo thời gian 
        start_proc = time.perf_counter()
        # detect góc
        if args.detect == "circle":
            angle, best_score, _,_ = detect_angle_by_circles(image_rectified, cfg, save=save_path, debug = args.debug)
        if args.detect == "edge":
            angle, best_score = get_chuck_angle(image_rectified, cfg, save=save_path, mode = args.edge, debug = args.debug)
        # Kết thúc đo
        end_proc = time.perf_counter()
        # Tính thời gian
        execution_time = (end_proc - start_proc) * 1000
        angles.append(angle)
        best_scores.append(best_score)
        exe_time.append(execution_time)
        if angle is not None:
            status += " - OK"
            logging.info(f"Góc quay của mâm: {angle:.2f} độ")
            logging.info(f"Điểm số: {best_score:.2f}")
            logging.info(f"Thời gian xử lý ảnh: {execution_time:.5f} ms")
            print(f"--- KẾT QUẢ MỚI ---")
            print(f"Góc quay: {angle:.2f} | Score: {best_score:.1f} | Time: {execution_time:.2f}ms")
        else:
            logging.info("Không phát hiện được lỗ nào phù hợp để tính góc.")
            print("Không phát hiện được lỗ nào phù hợp để tính góc.")
            status += " - NOT FOUND ANGLE"
    results = [{
        "angle": angles[i],
        "best_score": best_scores[i],
        "time_exe": exe_time[i]
    } for i in range(len(angles))]
    # Lưu kết quả vào file result.json
    with open(f'{image_dir}/result.json', mode='a', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent = 2)

    print("ĐÃ HOÀN THÀNH CHƯƠNG TRÌNH!")