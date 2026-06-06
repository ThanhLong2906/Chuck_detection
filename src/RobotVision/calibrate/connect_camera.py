from dotenv import load_dotenv
import os
import cv2
import numpy as np
load_dotenv()

CAM_USER = os.getenv("CAM_USER")
CAM_PASS = os.getenv("CAM_PASS")
CAM_IP = os.getenv("CAM_IP")
CAM_PORT = os.getenv("CAM_PORT")
CAM_CHANNEL = os.getenv("CAM_CHANNEL")
RTSP_URL = f'rtsp://{CAM_USER}:{CAM_PASS}@{CAM_IP}:{CAM_PORT}/Streaming/Channels/{CAM_CHANNEL}'

if __name__ == "__main__":
    window_name = "Camera Feed"
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm độ trễ bằng cách giảm kích thước bộ đệm
    # Load thông số đã calibrate
    data = np.load("/home/long/PROJECTS/AI/Z131_Robot_Thai-Nguyen/calibrate/calibration_results/camera_params.npz")
    mtx = data['mtx']
    dist = data['dist']

    

    if not cap.isOpened():
        print("Không thể kết nối đến camera.")
        exit()  
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình từ camera.")
            break

        # Khử méo cho frame mới
        h, w = frame.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)

        # Cắt ảnh theo ROI để bỏ phần rìa đen (nếu cần)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imshow(window_name, dst)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()