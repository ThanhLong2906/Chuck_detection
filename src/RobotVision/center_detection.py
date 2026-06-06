import cv2
import numpy as np
import json

def nothing(x):
    pass

def tune_hough_circles(image_path):
    img = cv2.imread(image_path)
    # # đọc file config
    # try:
    #     with open("/home/long/PROJECTS/AI/RobotVision/config/on_dinh_1.json", 'r') as f:
    #         cfg = json.load(f)
    # except:
    #     print("không đọc được file json")
    # xi = cfg["roi"][0]
    # yi = cfg["roi"][1]
    # rw = cfg["roi"][2]
    # rh = cfg["roi"][3]
    # img = img_ori[yi:yi+rh, xi:xi+rw]
    if img is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return
        
    # Thay đổi kích thước nếu ảnh camera GigE quá to (ví dụ 5MP) để dễ nhìn trên màn hình
    # h, w = img.shape[:2]
    # if w > 1200:
    #     img = cv2.resize(img, (int(w/2), int(h/2)))
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Dùng GaussianBlur thay cho MedianBlur đối với ảnh kim loại để làm mềm các viền lóa sáng
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    window_name = 'Tune Chuck Center'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Khởi tạo 4 thanh trượt (Trackbar) quan trọng nhất của thuật toán HoughCircles
    cv2.createTrackbar('param1 (Canny)', 'Tune Chuck Center', 50, 300, nothing)
    cv2.createTrackbar('param2 (Threshold)', 'Tune Chuck Center', 40, 150, nothing)
    cv2.createTrackbar('minRadius', 'Tune Chuck Center', 650, 1000, nothing)
    cv2.createTrackbar('maxRadius', 'Tune Chuck Center', 800, 1000, nothing)

    print("--- CÔNG CỤ TÌM TÂM MÂM CẶP ---")
    print("Kéo các thanh trượt trên cửa sổ ảnh để điều chỉnh.")
    print("Nhấn phím 'q' hoặc 'ESC' để thoát và chốt tọa độ cuối cùng.")

    best_center = None
    print(f"Ảnh gốc đã được cắt ROI: {img.shape[1]}x{img.shape[0]} (W x H)")
    while True:
        display_img = img.copy()
        
        # Lấy giá trị hiện tại từ thanh trượt
        p1 = cv2.getTrackbarPos('param1 (Canny)', 'Tune Chuck Center')
        p2 = cv2.getTrackbarPos('param2 (Threshold)', 'Tune Chuck Center')
        minR = cv2.getTrackbarPos('minRadius', 'Tune Chuck Center')
        maxR = cv2.getTrackbarPos('maxRadius', 'Tune Chuck Center')
        
        # Ép kiểu dữ liệu an toàn để OpenCV không bị crash
        if p1 < 1: p1 = 1
        if p2 < 1: p2 = 1
        if minR >= maxR: maxR = minR + 1
        print(f"Đang thử: param1={p1}, param2={p2}, minRadius={minR}, maxRadius={maxR}", end='\r')
        # Chạy thuật toán tìm đường tròn
        circles = cv2.HoughCircles(
            gray_blurred, 
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
            
            # Vẽ tâm (đỏ) và viền đường tròn (xanh lá)
            cv2.circle(display_img, (c_x, c_y), 3, (0, 0, 255), -1)
            cv2.circle(display_img, (c_x, c_y), r, (0, 255, 0), 2)
            
            # In text tọa độ góc trên cùng
            cv2.putText(display_img, f"Center: ({c_x}, {c_y}) R: {r}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Cập nhật hình ảnh lên UI
        cv2.imshow(window_name, display_img)
        cv2.imwrite('debug_hough_circles.jpg', display_img)  # Lưu ảnh debug tạm thời để xem lại sau khi thoát
        # Lắng nghe phím bấm (1ms)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()
    
    if best_center:
        print(f"\nTọa độ tâm chốt cuối cùng: X={best_center[0]}, Y={best_center[1]}, Bán kính={best_center[2]}")
        # Nếu ở đầu hàm bạn đã resize ảnh chia 2, nhớ nhân 2 lại ở đây để ra tọa độ chuẩn của ảnh gốc
        return best_center
    else:
        print("\nKhông tìm thấy tâm nào phù hợp.")
        return None

if __name__ == "__main__":
    # Đưa đúng tên file ảnh bạn vừa upload vào đây
    tune_hough_circles("/home/long/PROJECTS/AI/RobotVision/data/test/chau_mo/frame_20260525-102510_98.jpg")