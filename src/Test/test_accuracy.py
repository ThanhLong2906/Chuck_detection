from edge_detection import get_chuck_angle_edge
from circle_detection import get_chuck_angle_circle
import json
from pathlib import Path
import argparse
import cv2
import os
import logging
import time
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Công cụ để đo trung bình thời gian tính toán và góc xoay của mâm.")
    parser.add_argument("--data_dir", type=str, default = "data/test", help="Thư mục chưa các ảnh để kiểm tra.")
    parser.add_argument("--mode", type=str, default = "circle", choices = ["circle", "edge"],help="Phương pháp xác định góc xoay: circle (lỗ chấu) hoặc edge (cạnh chấu).")
    parser.add_argument("--config_dir", type = str, required=True, help = "Thư mục chưa các file config, mỗi file config phải đặt tên theo ảnh tương ứng: [tên ảnh]_config.json")
    parser.add_argument("--edge", type=str, default="lsd", choices = ["lsd", "canny"], help="Phương pháp xác định cạnh nếu chọn chế độ cạnh chấu: lsd hoặc canny.")
    parser.add_argument("--save", type=str, default=None, help="Đường dẫn đến nơi lưu file ảnh kết quả (nếu muốn lưu).")
    parser.add_argument('--verbose', action='store_true', help='Bật chế độ chi tiết')
    args = parser.parse_args()
    current_dir = Path(__file__).parent
    # Xác định level log dựa trên tham số verbose
    # Nếu không verbose: Chỉ lấy INFO từ test.py
    # Nếu verbose: Lấy DEBUG từ cả test.py và các module như circle_detection
    log_level = logging.DEBUG if args.verbose else logging.INFO

    # 3. Cấu hình logging
    # force=True để xóa các cấu hình mặc định (ngăn in ra terminal)
    logging.basicConfig(
        filename=f'{current_dir}/detect.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=log_level,
        force=True 
    )
    # Lấy thời gian hiện tại
    now = datetime.now()
    # Định dạng theo năm-tháng-ngày giờ-phút-giây
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    # 4. Nếu không ở chế độ verbose, tắt log của các module con
    if not args.verbose:
        # Tắt log từ các file được import (edge_detection, circle_detection)
        logging.getLogger('edge_detection').setLevel(logging.WARNING)
        logging.getLogger('circle_detection').setLevel(logging.WARNING)
        # Chỉ giữ lại log của file chính:
        # logging.getLogger(__name__).setLevel(logging.INFO)

    logging.info(f"--- CHƯƠNG TRÌNH BẮT ĐẦU {formatted_time}---")

    # Cấu hình đường dẫn
    data_path = current_dir / args.data_dir if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    config_dir = current_dir / args.config_dir if not Path(args.config_dir).is_absolute() else Path(args.config_dir)
    save_dir = current_dir / args.save if args.save is not None and not Path(args.save).is_absolute() else args.save
    if save_dir and not Path(save_dir).exists():
        os.mkdir(save_dir)
    if not data_path.exists():
        logging.info(f"Thư mục dữ liệu không tồn tại: {data_path}")
        exit(1)
    # if not config_dir.exists():
    #     logging.info(f"Thư mục cấu hình không tồn tại: {config_dir}")
    #     exit(1)

    image_paths = sorted(
        [p for p in data_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    )
    if not image_paths:
        logging.info("Không tìm thấy ảnh trong thư mục data.")
        exit(1)

    angles = []
    exe_time = []
    best_scores = []
    image_names = []
    for image_path in image_paths:
        if save_dir:
            save_path = save_dir / f"{image_path.stem}.jpg"
        logging.info(f"Xử lý ảnh: {image_path.name}")

        if not config_dir.exists():
            logging.info(f"   Bỏ qua: không tìm thấy file config {config_dir}")
            angles.append(None)
            exe_time.append(None)
            continue

        start_time = time.perf_counter()
        image_rectified = cv2.imread(str(image_path))
        if image_rectified is None:
            logging.info(f"   Lỗi đọc ảnh: {image_path}")
            angles.append(None)
            exe_time.append(None)
            continue

        try:
            with open(config_dir, 'r') as f:
                cfg = json.load(f)
        except Exception as e:
            logging.info(f"   Lỗi đọc file config {config_dir}: {e}")
            angles.append(None)
            exe_time.append(None)
            continue
        
        if args.mode == "circle":
            # angle, best_score, _, _ = get_chuck_angle_circle.detect_angle_by_circles(image_rectified, cfg, save= save_path, debug=False)
            angle, best_score, best_triple, jaw_circles_full = get_chuck_angle_circle.detect_angle_by_circles(image_rectified, cfg, save=save_path)
        elif args.edge == "lsd":
            angle, best_score = get_chuck_angle_edge.get_chuck_angle(image_rectified, cfg, save=save_path, debug=False)
        else:
            angle, best_score = get_chuck_angle_edge.get_chuck_angle(image_rectified, cfg, mode = "canny", save=save_path, debug=False)
        end_time = time.perf_counter()
        if angle is not None:
            logging.info(f"  Góc quay của mâm: {angle:.2f} độ")
            logging.info(f"Điểm số: {best_score:.2f}")
        else:
            logging.info("  Không phát hiện được lỗ nào phù hợp để tính góc.")

        execution_time = (end_time - start_time) * 1000
        angles.append(angle)
        best_scores.append(best_score)
        exe_time.append(execution_time)
        image_names.append(Path(image_path).name)
        logging.info(f"  Thời gian xử lý: {execution_time:.5f} ms")

    results = [
        {
            "image_name": image_names[i],
            "angle": angles[i],
            "best_score": best_scores[i],
            "exe_time": exe_time[i],
        }
        for i in range(len(image_names))
    ]

    result_path = current_dir / f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("ĐÃ HOÀN THÀNH CHƯƠNG TRÌNH!")
    print(f"Kết quả đã lưu vào {result_path}")
    if save_dir:
        print(f"Hình ảnh minh họa kết quả lưu vào {save_dir}")
    