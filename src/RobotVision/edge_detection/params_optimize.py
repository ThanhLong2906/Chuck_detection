from typing import Literal
from pathlib import Path
import cv2
import numpy as np
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"
from get_chuck_angle_edge import *

def get_chuck_angle(image_rectified, cfg, dp, p1, p2, save=None, mode: Literal['canny', 'lsd'] = 'canny', verbose=False):
    """
    image_rectified: Ảnh đã được nắn thẳng (top-down view)
    """
    # Xử lý ROI
    rx, ry, rw, rh = cfg["roi"]
    roi_img = image_rectified[ry:ry+rh, rx:rx+rw]
    # 1. Tiền xử lý và Canny
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    # Áp dụng CLAHE để tăng cường độ tương phản cục bộ
    # clipLimit: Ngưỡng giới hạn tương phản (thường từ 2.0 - 5.0)
    # tileGridSize: Kích thước vùng tính toán (thường là 8x8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.bilateralFilter(gray_clahe, 9, 75, 75) 
    if mode == 'lsd':
        # 3. Chạy LSD
        lsd = cv2.createLineSegmentDetector(0)
        lines, _, _, _ = lsd.detect(blur)
    else:
        edges = cv2.Canny(blur, 50, 150)

        # 5. Morphology (Phép toán CLOSE) để nối liền các đoạn cạnh đứt quãng
        # kernel: Kích thước vùng ảnh hưởng (3x3 hoặc 5x5)
        kernel = np.ones((5, 5), np.uint8)
        # Closing = Dilate (giãn) sau đó Erode (xói mòn) giúp lấp đầy lỗ hổng
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        # # edges_closed = cv2.medianBlur(edges_closed, 3)
        # 2. Dùng HoughLinesP để tìm các đoạn thẳng
        # threshold: số điểm giao tối thiểu để tạo đường thẳng
        # minLineLength: độ dài tối thiểu của đoạn thẳng
        # maxLineGap: khoảng cách tối đa giữa các đoạn để nối lại
        lines = cv2.HoughLinesP(edges_closed, 1, np.pi / 180, threshold=20, 
                                minLineLength=70, maxLineGap=20)
    
    if lines is None:
        print("Không tìm thấy đường thẳng nào trong ROI!")
        return None
    
    if verbose:
        debug_img_1 = roi_img.copy()
        path = Path(save)
        debug_1 = path.with_name(path.stem + "_debug_1" + path.suffix)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_img_1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imwrite(debug_1, debug_img_1)
    
    # 4. Lấy thông số lọc từ Config
    center = cfg["mask"]["center"]
    center[0] -= rx
    center[1] -= ry
    center[0] = int(center[0])
    center[1] = int(center[1])
    ref_point_1 = cfg['ref_points'][0]
    ref_point_2 = cfg['ref_points'][1]  
    if cfg["measurements"]:
        ref_dist = cfg["measurements"][0]["dist"]
        # Cho phép sai số khoảng 10-15% so với khoảng cách đã đo
        min_d = ref_dist * 0.85
        max_d = ref_dist * 1.15
    else:
        # Nếu không có đo đạc, dùng tạm bán kính trong/ngoài của mask làm ngưỡng
        min_d = cfg["mask"]["min_radius"]
        max_d = cfg["mask"]["max_radius"]
    
    line_list = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < 80: continue
        
        # TÍNH KHOẢNG CÁCH TỪ TÂM ĐẾN ĐƯỜNG THẲNG (Point-to-Line Distance)
        # Công thức: d = |(y2-y1)x0 - (x2-x1)y0 + x2y1 - y2x1| / length
        dist_to_center = distance_point_to_infinite_line(center, (x1, y1), (x2, y2))
        # BỘ LỌC CHÍNH: Chỉ giữ lại các đường có khoảng cách xấp xỉ giá trị đo
        if dist_to_center <= max_d:
            angle = calculate_angle_with_two_ref_points((x1, y1), (x2, y2), center, ref_point_1, ref_point_2)
            if angle < 0: angle += 360
            line_list.append({'coords': (int(x1), int(y1), int(x2), int(y2)), 'angle': angle, 'length': length})
    if verbose:
        debug_img_2 = roi_img.copy()
        debug_2 = path.with_name(path.stem + "_debug_2" + path.suffix)
        for line in line_list:
            x1, y1, x2, y2 = line['coords']
            cv2.line(debug_img_2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(debug_2, debug_img_2)

    best_triple = line_list
    jaw_edges, scores = find_triplets_120_degrees(line_list, tolerance=5.0)
    print(f"Tìm thấy {len(jaw_edges)} bộ ba thỏa mãn")
    final_angle = None
    if len(jaw_edges) > 0:
        best_index = np.argmax(scores)
        best_triple = jaw_edges[best_index]
        angles = [line['angle'] for line in best_triple]
        # Mâm có 3 chấu cách nhau 120 độ, tất cả về góc cơ sở (modulo 120)
        base_angles = [a % 120 for a in angles]
        final_angle = np.median(base_angles) # Dùng trung vị
    if save is not None:
        # Vẽ các đường thẳng và angle của chúng lên roi_img
        debug_img = roi_img.copy()
        for line in best_triple:
            x1, y1, x2, y2 = line['coords']
            angle = line['angle']
            # Vẽ đường thẳng
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Vẽ angle ở giữa đường thẳng
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(debug_img, f"{angle:.1f}°", (mid_x, mid_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if final_angle is not None:
            cv2.putText(debug_img, f"Goc quay: {final_angle:.2f} deg", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(debug_img, f"Score: {scores[best_index]:.2f}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(save, debug_img)
        cv2.imshow("Edge Detection", debug_img)
    return final_angle