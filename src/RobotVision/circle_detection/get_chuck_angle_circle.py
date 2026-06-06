import json
import cv2
import numpy as np
import math
import itertools
import argparse
from pathlib import Path
import sys
vision_path = Path(__file__).resolve().parent.parent
sys.path.append(str(vision_path))
import time 
import os
import logging
import copy
from utils.pair_angle import average_angles_mod
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"

def refine_subpixel_center(gray_img, x, y, r):
    """Tinh chỉnh tâm lỗ xuống mức sub-pixel bằng trọng tâm cường độ"""
    win_size = int(r * 0.5) # Kích thước vùng khảo sát dựa trên bán kính
    x_int, y_int = int(x), int(y)
    
    # Cắt vùng ROI nhỏ quanh tâm
    y1, y2 = max(0, y_int-win_size), min(gray_img.shape[0], y_int+win_size+1)
    x1, x2 = max(0, x_int-win_size), min(gray_img.shape[1], x_int+win_size+1)
    roi = gray_img[y1:y2, x1:x2]
    
    # Đảo ngược ảnh vì lỗ màu tối (pixel giá trị thấp)
    # Ta cần lỗ thành vùng có trọng số cao để tính trọng tâm
    roi_inv = 255 - roi
    
    # Tính Moments để tìm centroid
    M = cv2.moments(roi_inv)
    if M["m00"] != 0:
        local_x = M["m10"] / M["m00"]
        local_y = M["m01"] / M["m00"]
        # Chuyển về tọa độ global
        return x1 + local_x, y1 + local_y
    return x, y

def calculate_angle_with_two_ref_points(center, point, ref_p1, ref_p2):
    """
    center: Tâm mâm (cx, cy)
    point: Tọa độ lỗ cần tính góc (px, py)
    ref_p1, ref_p2: Hai điểm tạo thành đường thẳng tham chiếu
    """
    # 1. Tính góc tuyệt đối của đường thẳng tham chiếu (Vector P1 -> P2)
    dx_ref = int(ref_p2[0]) - int(ref_p1[0])
    dy_ref = int(ref_p2[1]) - int(ref_p1[1])
    # Góc tuyệt đối của trục tham chiếu so với trục hoành OpenCV
    abs_angle_ref = math.degrees(math.atan2(dy_ref, dx_ref))
    
    # 2. Tính góc tuyệt đối của lỗ so với tâm mâm
    dx_pt = int(point[0]) - int(center[0])
    dy_pt = int(point[1]) - int(center[1])
    abs_angle_pt = math.degrees(math.atan2(dy_pt, dx_pt))
    
    # 3. Tính góc tương đối (Cùng chiều kim đồng hồ)
    relative_angle = abs_angle_pt - abs_angle_ref
    
    # 4. Chuẩn hóa về dải [0, 360)
    if relative_angle < 0:
        relative_angle += 360
    
    # # Tính góc giữa 2 vector
    # a = (point[0]-center[0], point[1]-center[1])
    # b = (ref_p2[0] - ref_p1[0], ref_p2[1] - ref_p1[1])
    # vector_angle = angle_between_vectors(a,b)
    return relative_angle #, vector_angle

def is_real_hole(gray_img, x, y, r):
    # Tạo một mặt nạ nhỏ tại tâm lỗ
    mask = np.zeros(gray_img.shape, dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(r*0.5), 255, -1)
    
    # Tính độ sáng trung bình trong lòng lỗ
    mean_val = cv2.mean(gray_img, mask=mask)[0]
    
    # Nếu lòng lỗ quá sáng (> 100/255), khả năng cao là nhiễu bề mặt kim loại
    if mean_val > 50: 
        return False
    return True

def check_collinear(target_circle, all_circles, angle_tol=2.0):
    """Kiểm tra một lỗ có cặp đồng chấu hay không"""
    for c_all in all_circles:
        # Khoảng cách tâm giữa 2 lỗ
        dist_centers = np.linalg.norm(np.array(target_circle['pos']) - np.array(c_all['pos']))
        
        if dist_centers > 5: # Không phải chính nó
            angle_diff = abs(target_circle['angle'] - c_all['angle'])
            # Xử lý góc xoay vòng 360
            angle_diff = min(angle_diff, 360 - angle_diff) # Xử lý vòng tròn 360
            
            if angle_diff <= angle_tol:
                return True, c_all
    return False, None

def calculate_confidence_score(triple, all_circles, angle_tol=2.0):
    """
    triple: Bộ 3 lỗ đang xét (mỗi lỗ là dict {'pos', 'angle', 'dist'})
    all_circles: Tất cả các lỗ bắt được để tìm cặp đồng chấu
    """
    collinear_count = 0
    triple_full = list(copy.deepcopy(triple))
    for c_triple in triple:
        is_collinear, c_collinear = check_collinear(c_triple, all_circles, angle_tol)
        if is_collinear:
            triple_full.append(c_collinear)
            collinear_count += 1

    # Tính điểm theo yêu cầu
    if collinear_count > 0:
        # Trường hợp 1: Có ít nhất một góc có đồng chấu
        score = 0.5 + (collinear_count * 0.1) 
    else:
        # Trường hợp 2: Đáp ứng bộ 3 (120 độ) nhưng không đồng chấu
        score = 0.3
        
    return score, triple_full

def filter_symmetry_circles(jaw_circles, tolerance=5):
    """
    jaw_circles: Danh sách các lỗ đã lọc qua bán kính và độ tối lòng lỗ.
    tolerance: Sai số góc cho phép (mặc định là 5 độ).
    """
    if len(jaw_circles) < 2:
        logging.info("Không đủ lỗ để kiểm tra đối xứng (cần ít nhất 2 lỗ).")
        return [], [], []
    final_candidates = []
    jaw_circles_full = []
    scores = []
    # Tạo tất cả các tổ hợp bộ 3 từ danh sách các lỗ tìm được
    for triple in itertools.combinations(jaw_circles, 3):
        # Lấy góc của 3 lỗ và sắp xếp tăng dần
        angles = sorted([c['angle'] for c in triple])
        
        # Tính khoảng cách góc giữa các lỗ
        diff1 = angles[1] - angles[0]
        diff2 = angles[2] - angles[1]
        diff3 = 360 - (angles[2] - angles[0])
        
        # Kiểm tra xem cả 3 khoảng cách góc có xấp xỉ 120 độ không
        is_symmetric = (abs(diff1 - 120) <= tolerance and 
                        abs(diff2 - 120) <= tolerance and 
                        abs(diff3 - 120) <= tolerance)
        
        if is_symmetric:
            #Tính điểm tin cậy dựa trên tính chất đồng chấu
            score, triple_full = calculate_confidence_score(triple, jaw_circles)
            if triple not in final_candidates:
                final_candidates.append(triple)
                jaw_circles_full.append(triple_full)
                scores.append(score)
    
    if len(final_candidates) == 0:
        for pair in itertools.combinations(jaw_circles, 2):
            angle_diff = abs(pair[0]['angle'] - pair[1]['angle'])
            angle_diff = min(angle_diff, 360 - angle_diff)
            
            # Kiểm tra lệch 120 hoặc 240 độ (tương đương với việc thiếu 1 chấu)
            if (abs(angle_diff - 120) <= tolerance or 
                abs(angle_diff - 240) <= tolerance):
                
                # Đếm số lỗ đồng chấu trong cặp
                col_p1 = check_collinear(pair[0], jaw_circles)
                col_p2 = check_collinear(pair[1], jaw_circles)
                collinear_in_pair = sum([col_p1[0], col_p2[0]]) # True = 1, False = 0

                if collinear_in_pair >= 1:
                    score = 0.2 if collinear_in_pair == 2 else 0.1
                    if pair not in final_candidates:
                        final_candidates.append(pair)
                        jaw_circles_full.append(pair)
                        scores.append(score)    
                    
    return final_candidates, scores, jaw_circles_full

def detect_angle_by_circles(image_rectified, cfg, save = None, debug = False):
    """
    output: 
    + final_angle: góc xoay của mâm
    + best_score: điểm số của bộ cạnh/lỗ chấu xác định ra góc xoay
    """
    final_angle = None
    best_score = None

    rx, ry, rw, rh = cfg["roi"]
    roi_img = image_rectified[ry:ry+rh, rx:rx+rw]
    # 1. Tiền xử lý tập trung vào làm rõ các hốc/lỗ
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    # Sử dụng CLAHE mạnh để làm nổi bật lòng các lỗ tròn
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Làm mờ để giảm nhiễu hạt kim loại
    blur = cv2.medianBlur(enhanced, 5)

    # 2. Tìm các hình tròn (HoughCircles)
    # Các tham số cần tinh chỉnh dựa trên kích thước lỗ thực tế trong ảnh
    circles = cv2.HoughCircles(
        blur, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=cfg['circles_info']['min_dist'],      # Khoảng cách tối thiểu giữa các tâm lỗ
        param1=40,      # Ngưỡng Canny
        param2=40,       # Ngưỡng tích lũy (càng thấp càng bắt được nhiều lỗ mờ)
        minRadius=cfg['circles_info']['min_radius'],    # Bán kính lỗ nhỏ nhất (đo bằng tool)
        maxRadius=cfg['circles_info']['max_radius']     # Bán kính lỗ lớn nhất
    )

    if circles is None:
        logging.info("Không tìm thấy hình tròn nào trong ROI!")
        return final_angle, best_score, []
    if debug:
        debug_img_1 = roi_img.copy()
        path = Path(save)
        debug_1 = path.with_name(path.stem + "_debug_1" + path.suffix)
        for circle in circles[0, :]:
            x, y, r = circle
            cv2.circle(debug_img_1, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.imwrite(debug_1, debug_img_1)
    circles = np.uint16(np.around(circles))
    cx_mam, cy_mam = cfg["mask"]["center"]
    cx_mam -= rx
    cy_mam -= ry
    cx_mam = int(cx_mam)
    cy_mam = int(cy_mam)
    ref_point_1 = cfg['ref_points'][0]
    ref_point_2 = cfg['ref_points'][1]  
    jaw_circles = []
    
    for i in circles[0, :]:
        xi, yi, ri = i[0], i[1], i[2]
        xi_sub, yi_sub = refine_subpixel_center(gray, xi, yi, ri)
        # Tính khoảng cách từ tâm mâm đến tâm lỗ
        dist = math.sqrt((xi_sub - cx_mam)**2 + (yi_sub - cy_mam)**2)
        # logging.info(f"   Phát hiện lỗ tròn tại ({xi}, {yi}) với bán kính {ri} và khoảng cách đến tâm mâm {dist:.1f}")
        # Lọc lỗ nằm trong dải bán kính của chấu kẹp (sử dụng config mask)
        if cfg["mask"]["min_radius"] < dist < cfg["mask"]["max_radius"]:
            if is_real_hole(gray, xi, yi, ri):
                # Tính góc của lỗ này so với tâm mâm
                angle = calculate_angle_with_two_ref_points((cx_mam, cy_mam), (xi_sub, yi_sub), ref_point_1, ref_point_2)
                jaw_circles.append({'pos': (xi_sub, yi_sub), 'rad':ri,'dist': dist, 'angle': angle})
    if debug:
        debug_img_2 = roi_img.copy()
        path = Path(save)
        debug_2 = path.with_name(path.stem + "_debug_2" + path.suffix)
        for circle in jaw_circles:
            x, y = circle['pos']
            r = circle['rad']
            cv2.circle(debug_img_2, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.imwrite(debug_2, debug_img_2)
    best_triple = jaw_circles
    # Lọc tiếp các lỗ dựa trên tính đối xứng 120 độ của mâm 3 chấu
    jaw_circles, scores, jaw_circles_full = filter_symmetry_circles(jaw_circles, tolerance=15)
    logging.info(f"   Số bộ 3 lỗ tròn phù hợp sau khi lọc: {len(jaw_circles)}")

    # Tính toán góc quay cuối cùng
    final_angle = None
    
    if len(jaw_circles) > 0:
        # Chọn bộ 3 lỗ có điểm tin cậy cao nhất
        best_index = np.argmax(scores)
        best_triple = jaw_circles[best_index]
        jaw_circles_full = jaw_circles_full[best_index]
        angles = [c['angle'] for c in best_triple]

        # Nếu chỉ detect được 2 góc -> suy diễn ra góc thứ 3
        if len(angles) == 2:
            a1, a2 = sorted(angles)
            diff = a2 - a1
            
            # Nếu khoảng cách ~120 độ, góc còn lại nằm ngoài khoảng a1, a2
            if 110 < diff < 130:
                a3 = (a2 + 120) % 360
                angles.append(a3)
            # Nếu khoảng cách ~240 độ, góc còn lại nằm giữa a1, a2
            elif 230 < diff < 250:
                a3 = (a1 + 120) % 360
                angles.append(a3)    
        # Mâm có 3 chấu cách nhau 120 độ, tất cả về góc cơ sở (modulo 120)
        # base_angles = [a % 120 for a in angles]
        base_angle = [min(a, 360-a) for a in angles]
        final_angle = min(base_angle)
        # if len(angles) <3: 
        #     final_angle = average_angles_mod(base_angles)
        # else:
        #     final_angle = np.median(base_angles)
            # final_angle = min(angles_vector)
        best_score = scores[best_index]
        logging.info("   Bộ ba có điểm số cao nhất là:")
        for idx, c in enumerate(best_triple, start=1):
            xi, yi, rad = c['pos'][0], c['pos'][1], c['rad']
            logging.info(f"   Lỗ tròn thứ {idx} tại ({xi}, {yi}) với bán kính {rad}")
        logging.info(f"   Điểm số: {best_score}")
    if save is not None:
        # Vẽ minh họa
        debug_img = roi_img.copy()
        for c in best_triple:
            xi, yi, rad = c['pos'][0], c['pos'][1], c['rad']
            angle = c['angle']
            cv2.circle(debug_img, (int(xi), int(yi)), int(rad), (0, 255, 0), 2)
            cv2.line(debug_img, (cx_mam, cy_mam), (int(xi), int(yi)), (255, 0, 0), 1)
            # 4. Hiển thị giá trị góc ngay tại vị trí lỗ
            # Text sẽ hiển thị dạng "45.0°"
            label = f"{angle:.1f}"
            # Đặt tọa độ text hơi lệch ra một chút để không đè lên lỗ
            text_pos = (int(xi) + 10, int(yi) - 10)    
            cv2.putText(debug_img, label, text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) # Viền chữ đen
            cv2.putText(debug_img, label, text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)     # Chữ đỏ nội dung
        if final_angle is not None:
            cv2.putText(debug_img, f"Goc lech: {final_angle:.2f} deg", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
            cv2.putText(debug_img, f"Score: {best_score:.1f}", (20, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX,2.5, (0, 0, 255), 2)
            if final_angle <5:
                cv2.putText(debug_img, f"Chau dung vi tri", (20, 230), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)
            else:
                cv2.putText(debug_img, f"Chau sai vi tri", (20, 230), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
        os.makedirs(Path(save).parent, exist_ok=True)
        cv2.imwrite(save, debug_img)
    return final_angle, best_score, best_triple, jaw_circles_full

if __name__ == "__main__":
    # Bắt đầu đo
    start_time = time.perf_counter()    
    parser = argparse.ArgumentParser(description="Công cụ phát hiện lỗ tròn và đo góc quay mâm 3 chấu")
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến ảnh cần xử lý")
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn đến file cấu hình")
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='Bật chế độ verbose để lưu ảnh debug')
    args = parser.parse_args()
    current_dir = Path(__file__).parent
    if not Path(args.image).is_absolute():
        image_path = current_dir / args.image
    else:    image_path = Path(args.image)
    if not Path(args.config).is_absolute():
        config_path = current_dir / args.config
    else:    config_path = Path(args.config)
    # Load ảnh và config
    image_rectified = cv2.imread(str(image_path))
    # load cấu hình
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
    except:
        print("Lỗi đọc file config!")
    save_path = current_dir / "spc_debug" / f"{image_path.stem}_circle.jpg"
    angle,_,_,_ = detect_angle_by_circles(image_rectified, cfg, save=save_path, debug=args.verbose)
    if angle is not None:
        print(f"Góc quay của mâm: {angle:.2f} độ")
    else:
        print("Không phát hiện được lỗ nào phù hợp để tính góc.")
    # Kết thúc đo
    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1000
    print(f"Thời gian xử lý ảnh: {execution_time:.5f} ms")
    cv2.waitKey(0)
    cv2.destroyAllWindows()