import cv2
import numpy as np
from pathlib import Path
import argparse

current_dir = Path(__file__).parent

parser = argparse.ArgumentParser(description='Template Matching để tìm góc quay của mâm')
parser.add_argument('--template', type=str, default='imgs/thang_cut.jpg', help='Đường dẫn đến ảnh template (ảnh xám)')
parser.add_argument('--frame', type=str, default='imgs/result_1_rotated_40.png', help='Đường dẫn đến ảnh frame (ảnh xám)')
args = parser.parse_args()
if not Path(args.template).is_absolute():
    args.template = current_dir / args.template
if not Path(args.frame).is_absolute():
    args.frame = current_dir / args.frame

template_path = args.template
frame_path = args.frame

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

def create_template(template):
    pass

def find_angle_by_template(frame_gray, template_gray):
    """
    frame_gray: Ảnh mâm hiện tại (đã nắn thẳng, ảnh xám)
    template_gray: Ảnh mẫu đặc trưng (ảnh xám)
    center: Tâm của mâm (x, y)
    """
    best_max_val = -1
    best_angle = 0
    res_ls = []
    
    # Thử xoay mẫu từ 0 đến 360 độ (bước nhảy 1 độ để chính xác)
    for angle in range(0, 360, 1):
        # Xoay template
        (h, w) = template_gray.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated_template = cv2.warpAffine(template_gray, M, (w, h))
        
        # Thực hiện khớp mẫu
        res = cv2.matchTemplate(frame_gray, rotated_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        # Lưu lại góc có độ khớp cao nhất
        if max_val > best_max_val:
            best_max_val = max_val
            best_angle = angle
            best_match_loc = max_loc
        res_ls.append((angle, max_val))
    return best_angle, best_max_val, res_ls

template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
print(f"Đã tải template, kích thước: {template.shape}")
frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)

# Áp dụng Canny edge detection
template = cv2.Canny(template, 100, 150)
frame = cv2.Canny(frame, 100, 150)
angle, score, res_ls = find_angle_by_template(frame, template)
print(f"Góc quay hiện tại: {angle} độ (Độ tin cậy: {score:.2f}), chi tiết góc đã thử: \n{res_ls}") 
cv2.imshow("Template Edges", template)
cv2.imshow("Frame Edges", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
