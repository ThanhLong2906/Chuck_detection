import numpy as np
import cv2
from pathlib import Path
from dotenv import load_dotenv    
import os

# --- THÔNG SỐ CẤU HÌNH ---
# Số góc trong của bàn cờ (Ví dụ bàn cờ 9x7 ô -> góc trong là 8x6)
CHECKERBOARD = (10, 7) 
# Kích thước thực tế của 1 cạnh ô vuông (đơn vị mm hoặc cm)
SQUARE_SIZE = 25 
current_dir = Path(__file__).parent

def calibrate(url: str, checkerboard: tuple = CHECKERBOARD, square_size: float = SQUARE_SIZE, save_img: bool = False, save_path: Path = "calibration_results"):
    if Path(save_path).is_dir():
        save_dir = save_path
    else:
        save_dir = current_dir / save_path
        save_dir.mkdir(exist_ok=True)
    if save_img:
        img_save_dir = current_dir / "calibration_images"
        img_save_dir.mkdir(exist_ok=True)
    # Điều kiện dừng cho thuật toán tinh chỉnh góc
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Chuẩn bị tọa độ thế giới thực cho các điểm (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp = objp * square_size

    # Các mảng để lưu trữ điểm thế giới thực và điểm trên ảnh
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    window_name = "Calibration Lab"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    print("Nhấn 's' để chụp ảnh bàn cờ, nhấn 'q' để bắt đầu tính toán.")

    count = 0
    while True:
        for _ in range(5):
            cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Tìm các góc của bàn cờ
        ret_found, corners = cv2.findChessboardCorners(gray, checkerboard, None)

        display_frame = frame.copy()
        if ret_found:
            # Tinh chỉnh tọa độ góc chính xác hơn
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # Vẽ các góc lên ảnh để xem thử
            cv2.drawChessboardCorners(display_frame, checkerboard, corners2, ret_found)

        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and ret_found:
            objpoints.append(objp)
            imgpoints.append(corners2)
            count += 1
            if save_img:
                filename = f"{img_save_dir}/calib_image_{count}.jpg"
                cv2.imwrite(filename, frame)
            print(f"Đã lưu ảnh thứ {count}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- TÍNH TOÁN CALIBRATION ---
    if len(objpoints) > 0:
        print("Đang tính toán thông số camera... vui lòng đợi.")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Lưu kết quả vào file để sau này nạp vào pipeline
        np.savez(save_dir / "camera_params.npz", mtx=mtx, dist=dist)
        
        print("\n--- KẾT QUẢ CALIBRATION ---")
        print("Camera Matrix (Nội thông):\n", mtx)
        print("\nDistortion Coefficients (Hệ số biến dạng):\n", dist)
        print("\nĐã lưu thông số vào file 'camera_params.npz'")
    else:
        print("Chưa chụp được ảnh nào hợp lệ!")

if __name__ == "__main__":
    load_dotenv()

    CAM_USER = os.getenv("CAM_USER")
    CAM_PASS = os.getenv("CAM_PASS")
    CAM_IP = os.getenv("CAM_IP")
    CAM_PORT = os.getenv("CAM_PORT")
    CAM_CHANNEL = os.getenv("CAM_CHANNEL")
    RTSP_URL = f'rtsp://{CAM_USER}:{CAM_PASS}@{CAM_IP}:{CAM_PORT}/Streaming/Channels/{CAM_CHANNEL}'
    calibrate(RTSP_URL, save_img=True)