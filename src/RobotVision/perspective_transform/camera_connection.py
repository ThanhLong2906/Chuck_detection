import cv2
from dotenv import load_dotenv
import os

window_name = 'Camera'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# 1. load variables from .env file
load_dotenv()

CAM_USER = os.getenv('CAM_USERNAME')
CAM_PASS = os.getenv('CAM_PASSWORD')
CAM_IP = os.getenv('CAM_IP')
CAM_PORT = os.getenv('CAM_PORT')
CAM_CHANNEL = os.getenv('CAM_CHANNEL')

# connect to camera
RTSP_URL = f'rtsp://{CAM_USER}:{CAM_PASS}@{CAM_IP}:{CAM_PORT}/Streaming/Channels/{CAM_CHANNEL}'
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 