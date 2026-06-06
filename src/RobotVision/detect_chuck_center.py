"""
Xác định vòng tròn tâm mâm 3 chấu bằng xử lý ảnh thuần (OpenCV)
Pipeline: Grayscale → Blur → Threshold → Morphology → Contour → Hough
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

IMAGE_PATH = "your_image.jpg" 
OUTPUT_PATH = "result.png"

# Tỉ lệ vùng ROI tìm tâm (0.0-1.0, mặc định 30% trung tâm ảnh)
ROI_RATIO = 0.30

# Ngưỡng circularity tối thiểu (0=bất kỳ, 1=tròn hoàn hảo)
MIN_CIRCULARITY = 0.55

# Diện tích tối thiểu của vòng tròn tâm (px²)
MIN_AREA = 300
# ============================================================


def detect_center_circle(image_path, output_path=None, roi_ratio=0.30,
                          min_circularity=0.55, min_area=300):
    """
    Xác định vòng tròn tối ở tâm mâm 3 chấu.

    Trả về: dict với cx, cy, radius, circularity hoặc None nếu không tìm thấy.
    """
    # --- Load & chuyển grayscale ---
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cx_img, cy_img = w // 2, h // 2
    roi_r = int(min(h, w) * roi_ratio)

    # --- BƯỚC 1: Gaussian Blur ---
    # Giảm nhiễu, giúp threshold và Hough ổn định hơn
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # --- BƯỚC 2: Adaptive Threshold ---
    # Tách vùng tối (lỗ tâm) ra khỏi nền kim loại sáng bóng
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,   # Kích thước vùng lân cận
        C=10            # Hằng số trừ đi
    )

    # --- BƯỚC 3: Giới hạn ROI vùng tâm ---
    # Tránh phát hiện các lỗ bulong xung quanh
    mask = np.zeros_like(gray)
    cv2.circle(mask, (cx_img, cy_img), roi_r, 255, -1)
    thresh_roi = cv2.bitwise_and(thresh, mask)

    # --- BƯỚC 4: Morphology ---
    # Close: lấp các lỗ nhỏ bên trong vùng tối
    # Open: loại bỏ nhiễu nhỏ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(thresh_roi, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel, iterations=1)

    # --- BƯỚC 5: Tìm Contours ---
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)

        # Circularity: 4π·A / P²  (1.0 = tròn hoàn hảo)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        # Khoảng cách tâm contour đến tâm ảnh
        dist = np.hypot(cx - cx_img, cy - cy_img)

        # Chỉ chấp nhận contour tròn + gần tâm
        if circularity >= min_circularity and dist < roi_r * 0.65:
            # Score ưu tiên: tròn + lớn + gần tâm
            score = circularity * area / (dist + 1)
            if score > best_score:
                best_score = score
                best = {
                    "cx": int(cx), "cy": int(cy),
                    "radius": int(radius),
                    "circularity": round(circularity, 4),
                    "area": int(area),
                    "dist_to_center": round(dist, 1)
                }

    # --- BƯỚC 6 (phụ): Hough Circle để xác nhận / bổ sung ---
    if best is None:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=80,
            param1=80, param2=35,
            minRadius=20, maxRadius=roi_r
        )
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            for (cx, cy, r) in circles:
                dist = np.hypot(cx - cx_img, cy - cy_img)
                if dist < roi_r * 0.5:
                    best = {
                        "cx": int(cx), "cy": int(cy),
                        "radius": int(r),
                        "circularity": None,
                        "area": None,
                        "dist_to_center": round(dist, 1),
                        "method": "HoughCircles"
                    }
                    break

    # --- Visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Xác định vòng tròn tâm mâm 3 chấu — Xử lý ảnh thuần',
                 fontsize=15, fontweight='bold', color='white', y=0.98)

    panels = [
        (gray,       '1. Grayscale',           'gray'),
        (blurred,    '2. Gaussian Blur',        'gray'),
        (thresh,     '3. Adaptive Threshold',   'gray'),
        (thresh_roi, '4. ROI tìm kiếm',         'gray'),
        (cleaned,    '5. Morphology (đã làm sạch)', 'gray'),
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), '6. Kết quả', None),
    ]

    for ax, (data, title, cmap) in zip(axes.flat, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, color='white', fontsize=10, pad=4)
        ax.axis('off')
        ax.set_facecolor('#0f0f1a')

    # Vẽ vùng ROI
    roi_patch = plt.Circle((cx_img, cy_img), roi_r,
                            color='cyan', fill=False, lw=1.5, ls='--')
    axes[1, 2].add_patch(roi_patch)
    axes[1, 2].text(cx_img, cy_img - roi_r - 12, 'ROI',
                    color='cyan', ha='center', fontsize=8)

    if best:
        cx, cy, r = best["cx"], best["cy"], best["radius"]
        # Vòng tròn phát hiện
        c_patch = plt.Circle((cx, cy), r, color='#00ff88', fill=False, lw=3)
        axes[1, 2].add_patch(c_patch)
        # Dấu thập tâm
        axes[1, 2].plot(cx, cy, '+', color='red', ms=24, mew=3)
        # Chú thích
        circ_str = f"{best['circularity']:.3f}" if best['circularity'] else "N/A"
        label = (f"Tâm: ({cx}, {cy})\n"
                 f"R = {r} px\n"
                 f"Circularity = {circ_str}\n"
                 f"∆ tâm ảnh = {best['dist_to_center']} px")
        axes[1, 2].annotate(
            label,
            xy=(cx, cy - r), xytext=(cx + r + 30, cy - r - 30),
            color='#ffff00', fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#ffff00', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.4', fc='#000000cc', ec='#ffff00', lw=1)
        )
        axes[1, 2].set_title('6. KẾT QUẢ ✅', color='#00ff88', fontsize=10,
                              fontweight='bold', pad=4)
    else:
        axes[1, 2].set_title('6. Không tìm thấy ❌', color='red', fontsize=10, pad=4)

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Đã lưu kết quả: {output_path}")

    plt.close()
    return best


# ============================================================
# CHẠY TRỰC TIẾP
# ============================================================
if __name__ == "__main__":
    # img_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    # out_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PATH
    img_path = f"/home/long/PROJECTS/AI/RobotVision/data/test/chau_mo/frame_20260525-102514_261.jpg"
    out_path = f"/home/long/PROJECTS/AI/RobotVision/data/test"

    print(f"Đang xử lý: {img_path}")
    result = detect_center_circle(
        img_path, out_path,
        roi_ratio=ROI_RATIO,
        min_circularity=MIN_CIRCULARITY,
        min_area=MIN_AREA
    )

    if result:
        print("\n✅ Tìm thấy vòng tròn tâm:")
        for k, v in result.items():
            print(f"   {k:20s} = {v}")
    else:
        print("\n❌ Không tìm thấy — thử giảm MIN_CIRCULARITY hoặc tăng ROI_RATIO")
