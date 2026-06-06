import os
from typing import Literal
from pathlib import Path
import cv2
import numpy as np
import math
import itertools
import logging
from utils.pair_angle import average_angles_mod
# Không in ra warning
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"

def check_parallel(target_line, all_lines, angle_tol = 2.0):
    """
    Kiểm tra xem một đường thẳng có song song với bất kỳ đường thẳng nào trong danh sách hay không.
    
    target_line: Đường thẳng cần kiểm tra
    all_lines: List các dict {'coords': (...), 'angle': ..., 'length': ...}
    angle_tol: Sai số góc để xác định song song (độ)
    
    Trả về True nếu có ít nhất một đường thẳng song song, ngược lại trả về False.
    """
    for line in all_lines:
        if line == target_line:
            continue
        diff = abs(target_line['angle'] - line['angle'])
        if diff <= angle_tol or abs(diff - 360) <= angle_tol:
            return True
    return False

def find_triplets_120_degrees(lines, angle_tolerance=2.0 ,tolerance=5.0):
    """
    Tìm tất cả các bộ ba đường thẳng có góc cách nhau 120 độ.
    
    lines: List các dict {'coords': (...), 'angle': ..., 'length': ...}
    angle_tolerance: Sai số góc cho phép (độ)
    tolerance: Sai số khoảng cách cho phép (độ)
    """
    if len(lines) < 2:
        logging.info("Không đủ đường thẳng để kiểm tra (cần ít nhất 2 đường).")
        return [], []
    # Sắp xếp danh sách theo góc để việc kiểm tra logic hơn (không bắt buộc nhưng giúp dễ kiểm soát)
    sorted_lines = sorted(lines, key=lambda x: x['angle'])

    for line in lines:
        has_parallel = check_parallel(line, lines, angle_tol=angle_tolerance)
        line['has_parallel'] = has_parallel
    final_candidates = []
    scores = []
    # Duyệt qua tất cả các tổ hợp 3 đường thẳng khác nhau
    for triple in itertools.combinations(sorted_lines, 3):
        # Lấy góc của 3 đường thẳng trong bộ đang xét
        a, b, c = triple[0]['angle'], triple[1]['angle'], triple[2]['angle']
        
        # Vì đã sắp xếp, nên có a < b < c
        diff1 = b - a
        diff2 = c - b
        diff3 = 360 - (c - a) # Khoảng cách vòng từ góc cuối về góc đầu

        # Kiểm tra điều kiện mỗi khoảng cách gần bằng 120 độ
        if (abs(diff1 - 120) <= tolerance and 
            abs(diff2 - 120) <= tolerance and 
            abs(diff3 - 120) <= tolerance):
            
            base_score = 0.5
            
            # Kiểm tra điều kiện song song
            parallels = sum(line['has_parallel'] for line in triple)
            
            if parallels > 0:
                total_score = base_score + 0.1*parallels
            else:
                total_score = base_score - 0.1
            if triple not in final_candidates:
                final_candidates.append(triple)
                scores.append(total_score)
    if len(final_candidates) == 0:
        for pair in itertools.combinations(sorted_lines, 2):
            # Lấy góc của 2 đường thẳng trong bộ đang xét
            a, b = pair[0]['angle'], pair[1]['angle']
            
            # Vì đã sắp xếp, nên có a < b
            angle_diff = b - a

            # Kiểm tra điều kiện mỗi khoảng cách gần bằng 120 độ
            if (abs(angle_diff - 120) <= tolerance or 
                abs(angle_diff - 240) <= tolerance):
                
                # Kiểm tra điều kiện song song
                parallels = sum(line['has_parallel'] for line in pair)
                
                if parallels >= 1:
                    total_score = 0.2 if parallels == 2 else 0.1
                    if pair not in final_candidates:
                        final_candidates.append(pair)
                        scores.append(total_score)
    return final_candidates, scores

def calculate_angle_with_two_ref_points(point_1, point_2, center, ref_p1, ref_p2):
    """
    point_1: Điểm thứ nhất (cx, cy)
    point_2: Điểm thứ hai (px, py)
    center: Tâm mâm (cx, cy)
    ref_p1, ref_p2: Hai điểm tạo thành đường thẳng tham chiếu
    """
    # 1. Tính bình phương khoảng cách từ center đến point_1 và point_2
    # (Không cần căn bậc hai để tối ưu tốc độ)
    dist_NM_sq = (point_1[0] - center[0])**2 + (point_1[1] - center[1])**2
    dist_NO_sq = (point_2[0] - center[0])**2 + (point_2[1] - center[1])**2

    # 2. Xác định điểm đầu (Start) và điểm cuối (End) của vector
    # Điểm gần N hơn là điểm đầu
    if dist_NM_sq <= dist_NO_sq:
        start_pt = point_1
        end_pt = point_2
    else:
        start_pt = point_2
        end_pt = point_1

    # 1. Tính góc tuyệt đối của đường thẳng tham chiếu (Vector P1 -> P2)
    dx_ref = int(ref_p2[0]) - int(ref_p1[0])
    dy_ref = int(ref_p2[1]) - int(ref_p1[1])
    # Góc tuyệt đối của trục tham chiếu so với trục hoành OpenCV
    abs_angle_ref = math.degrees(math.atan2(dy_ref, dx_ref))
    
    # 2. Tính góc tuyệt đối của cạnh so với trục hoành OpenCV
    dx_pt = int(end_pt[0]) - int(start_pt[0])
    dy_pt = int(end_pt[1]) - int(start_pt[1])
    abs_angle_pt = math.degrees(math.atan2(dy_pt, dx_pt))
    
    # 3. Tính góc tương đối (Cùng chiều kim đồng hồ)
    relative_angle = abs_angle_pt - abs_angle_ref
    
    # 4. Chuẩn hóa về dải [0, 360)
    if relative_angle < 0:
        relative_angle += 360
        
    return relative_angle

def distance_point_to_infinite_line(M, N, O):
    # Tọa độ điểm M(a, b), N(x1, y1), O(x2, y2)
    a, b = M
    x1, y1 = N
    x2, y2 = O

    # Vector NO: (dx, dy)
    dx = x2 - x1
    dy = y2 - y1

    # Nếu N và O trùng nhau, đường thẳng không xác định, trả về khoảng cách MN
    if dx == 0 and dy == 0:
        return math.sqrt((a - x1)**2 + (b - y1)**2)

    # Công thức khoảng cách từ điểm đến đường thẳng tổng quát:
    # d = |(x2 - x1)(y1 - b) - (x1 - a)(y2 - y1)| / sqrt(dx^2 + dy^2)
    numerator = abs(dx * (y1 - b) - (x1 - a) * dy)
    denominator = math.sqrt(dx**2 + dy**2)
    
    return numerator / denominator

def get_chuck_angle(image_rectified, cfg, save=None, mode: Literal['canny', 'lsd'] = 'lsd', debug=False):
    """
    image_rectified: Ảnh đã được nắn thẳng (top-down view)
    output: 
    + final_angle: góc xoay của mâm
    + best_score: điểm số của bộ cạnh/lỗ chấu xác định ra góc xoay
    """
    final_angle = None
    best_score = None

    # 2. Xử lý ROI
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
        logging.info("Không tìm thấy đường thẳng nào trong ROI!")
        return final_angle, best_score
    
    if debug:
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
        
        # TÍNH KHOẢNG CÁCH TỪ TÂM ĐẾN ĐƯỜNG THẲNG
        # Công thức: d = |(y2-y1)x0 - (x2-x1)y0 + x2y1 - y2x1| / length
        dist_to_center = distance_point_to_infinite_line(center, (x1, y1), (x2, y2))
        # BỘ LỌC CHÍNH: Chỉ giữ lại các đường có khoảng cách xấp xỉ giá trị đo
        if dist_to_center <= max_d:
            angle = calculate_angle_with_two_ref_points((x1, y1), (x2, y2), center, ref_point_1, ref_point_2)
            if angle < 0: angle += 360
            line_list.append({'coords': (int(x1), int(y1), int(x2), int(y2)), 'angle': angle, 'length': length})
    if debug:
        debug_img_2 = roi_img.copy()
        debug_2 = path.with_name(path.stem + "_debug_2" + path.suffix)
        for line in line_list:
            x1, y1, x2, y2 = line['coords']
            cv2.line(debug_img_2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(debug_2, debug_img_2)

    best_triple = line_list
    jaw_edges, scores = find_triplets_120_degrees(line_list, tolerance=5.0)
    logging.info(f"Tìm thấy {len(jaw_edges)} bộ ba thỏa mãn")
    
    if len(jaw_edges) > 0:
        best_index = np.argmax(scores)
        best_triple = jaw_edges[best_index]
        angles = [line['angle'] for line in best_triple]
        # Mâm có 3 chấu cách nhau 120 độ, tất cả về góc cơ sở (modulo 120)
        base_angles = [a % 120 for a in angles]
        if len(angles) <3: 
            final_angle = average_angles_mod(base_angles)
        else:
            final_angle = np.median(base_angles) # Dùng trung vị
        best_score = scores[best_index]
        logging.info("   Bộ ba có điểm số cao nhất là:")
        for idx, line in enumerate(best_triple, start=1):
            logging.info(f"   Đoạn thẳng {idx} có điểm đầu ({line['coords'][0]},{line['coords'][1]}), điểm cuối ({line['coords'][2]},{line['coords'][3]})")
        logging.info(f"   Điểm số: {best_score}")
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
            cv2.putText(debug_img, f"Score: {best_score:.2f}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        os.makedirs(Path(save).parent, exist_ok=True)
        cv2.imwrite(save, debug_img)
    return final_angle, best_score