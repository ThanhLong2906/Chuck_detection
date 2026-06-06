import cv2
from pathlib import Path
import time
import argparse
from dotenv import load_dotenv
import os

def main(video_path, output_dir="/home/long/PROJECTS/AI/RobotVision/data/config"):
    if not Path(video_path).exists():
        print(f"không tồn tại video tại đường dẫn: {video_path}")
        return
    
    # Tạo thư mục output nếu chưa tồn tại
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Hiển thị frame
            cv2.imshow("Video", frame)
            
            # Chờ phím nhấn (1ms timeout)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Nhấn 's' để lưu frame
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"{output_dir}/frame_{timestamp}_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Lưu frame: {filename}")
            
            elif key == ord('q'):  # Nhấn 'q' để thoát
                print("Thoát chương trình")
                break
            
            frame_count += 1
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    load_dotenv(dotenv_path = f"{current_dir.parent.parent.parent}/.env", override=True)
    data_dir = os.getenv("DATA_DIR")
    parser = argparse.ArgumentParser(description="extract any frame from any video.")
    parser.add_argument("--video_path", type=str, required=True, help="Đường dẫn tới file video")
    parser.add_argument("--frame_dir", type=str, default = f"{data_dir}/test")
    args = parser.parse_args()
    main(args.video_path, args.frame_dir)
