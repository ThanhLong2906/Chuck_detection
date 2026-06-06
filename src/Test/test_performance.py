import time
import cv2
import os
import glob
import json
from pathlib import Path
import sys
path_to_src = Path(__file__).resolve().parent.parent
sys.path.append(str(path_to_src))
from RobotVision.circle_detection.get_chuck_angle_circle import detect_angle_by_circles
from RobotVision.workpiece_detection.workpiece_check import WorkpieceDetection
from RobotVision.jaw_detection.jaw_check import JawDetection 
import argparse

class MockCamera:
    """Giả lập Camera đọc ảnh từ thư mục"""
    def __init__(self, folder_path):
        self.image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        self.current_idx = 0
        if not self.image_files:
            print(f"CẢNH BÁO: Không tìm thấy ảnh nào trong thư mục {folder_path}")

    def get_one_frame(self):
        """Hàm này mô phỏng việc camera chụp 1 bức ảnh"""
        if not self.image_files: return None
        
        img_path = self.image_files[self.current_idx]
        frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Tăng index để quay vòng ảnh
        self.current_idx = (self.current_idx + 1) % len(self.image_files)
        return frame

def run_performance_test(test_image_folder, config_path):
    camera = MockCamera(test_image_folder)
    with open(config_path, 'r') as f:
        config = json.load(f)
    wp_detector = WorkpieceDetection(config=config)
    jaw_detector = JawDetection(config=config)
    
    save_path = "/home/long/PROJECTS/AI/RobotVision/data/test/performance"

    print("=== BẮT ĐẦU BÀI TEST CPU/RAM ===")
    print("Hệ thống sẽ tự động load ảnh và xử lý mỗi 2 phút.")
    
    count = 1
    while True:
        frame = camera.get_one_frame()
        
        if frame is not None:
            print(f"\n[{count}] Đang xử lý ảnh lúc: {time.strftime('%H:%M:%S')}")
            
            # Đếm thời gian bắt đầu thuật toán
            start_time = time.time()
            
            angle, _, best_triple, jaw_circle_full = detect_angle_by_circles(frame, config, save=save_path)
            has_workpiece, _ = WorkpieceDetection.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), save=save_path)
            # ----------------------------------------------------
            is_open = JawDetection.detect_open(jaw_circle_full, save=save_path)
            # Đếm thời gian kết thúc
            process_time = time.time() - start_time
            print(f" -> Hoàn thành! Thuật toán chạy mất: {process_time:.4f} giây")
        else:
            print("Không có ảnh để test.")
            break

        # Chờ 2 phút (120 giây) trước khi lấy ảnh tiếp theo
        print(" -> Đang chờ 120s cho chu kỳ tiếp theo...")
        time.sleep(120)
        count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test hệ thống")
    parser.add_argument("--image_dir", type=str, required=True, help="Đường dẫn thư mục chứa ảnh test")
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn file config")
    parser.add_argument("--log", type=str, default=None, help="Đường dẫn file log")
    run_performance_test()