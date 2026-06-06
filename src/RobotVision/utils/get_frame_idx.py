import cv2
cap = cv2.VideoCapture("/home/long/PROJECTS/AI/RobotVision/src/RobotVision/workpiece_detection/Video_20260519185303521.mp4")
cv2.namedWindow("video", cv2.WINDOW_NORMAL)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("video", frame)

    key = cv2.waitKey(25) & 0xFF
    if key == ord("d"):
        current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        print(f"frame hiện tại: {current_frame_index}")
    elif key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
