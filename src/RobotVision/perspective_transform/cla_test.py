"""
Pipeline: Biến đổi ảnh nghiêng → thẳng, có xử lý depth 3D
Setup: vật thể 3D (hộp + cốc) đặt trên bàn, camera cố định

CHIẾN LƯỢC:
  Layer 0 – Bàn / nền:      H_box (xử lý đúng, nằm trên mặt phẳng hộp)
  Layer 1 – Mặt hộp (z=0):  H_box
  Layer 2 – Cốc (z>0):      H_box + affine correction (scale+translate)

CÁCH DÙNG:
  1. Đặt camera cố định
  2. Chụp ảnh thẳng (ground truth) và ảnh nghiêng
  3. Chỉnh sửa 4 biến CONFIGURATION bên dưới
  4. Chạy script
"""

import cv2
import numpy as np
import os

# ══════════════════════════════════════════
# CONFIGURATION – chỉnh theo ảnh
# ══════════════════════════════════════════
FRONT_IMG  = '/home/long/PROJECTS/AI/Z131_Robot_Thai-Nguyen/perspective_tranform/imgs/anh_thang.png'
TILTED_IMG = '/home/long/PROJECTS/AI/Z131_Robot_Thai-Nguyen/perspective_tranform/imgs/anh_nghieng.png'
OUTPUT_DIR = 'output'

# Bước A: Pick 4 góc của object PHẲNG (hộp/mâm) trong 2 ảnh
# → Dùng công cụ xem ảnh, ghi lại tọa độ pixel
# Thứ tự: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
PTS_FRONT_BOX = np.float32([
    [242, 191],   # TL trong ảnh thẳng
    [962, 181],   # TR
    [970, 666],   # BR
    [220, 672],   # BL
])
PTS_TILTED_BOX = np.float32([
    [ 95, 273],   # TL trong ảnh nghiêng
    [1090, 195],  # TR
    [1100, 500],  # BR
    [  75, 540],  # BL
])

# Bước B: Detect hình tròn của vật thể cao (cốc/mâm)
# Nếu không có: đặt None để bỏ qua bước này
#   (cx, cy, r) trong ảnh SIMPLE WARP (chạy lần đầu để xem)
CUP_IN_SIMPLE_WARP = (740, 422, 244)
#   (cx, cy, r) trong ảnh THẲNG (ground truth)
CUP_IN_FRONT       = (820, 428, 299)

# ══════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════

def pick_corners_interactive(img, n_points=4, window_name='Pick corners'):
    """
    Mở cửa sổ để click chọn điểm.
    Trả về: array shape (n_points, 2)
    """
    points = []
    img_copy = img.copy()

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n_points:
            points.append([x, y])
            cv2.circle(img_copy, (x,y), 8, (0,255,0), -1)
            cv2.putText(img_copy, str(len(points)), (x+10,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow(window_name, img_copy)
            if len(points) == n_points:
                print(f"  {n_points} points picked. Press any key to continue.")

    cv2.imshow(window_name, img_copy)
    cv2.setMouseCallback(window_name, mouse_cb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.float32(points)


def detect_circle(img, min_r=80, max_r=400):
    """Detect circle lớn nhất trong ảnh (dùng cho cốc/mâm tròn)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 200,
        param1=50, param2=30,
        minRadius=min_r, maxRadius=max_r
    )
    if circles is not None:
        return tuple(circles[0][0].astype(int))  # (cx, cy, r)
    return None


def compute_H_box(pts_front, pts_tilted):
    H, _ = cv2.findHomography(pts_tilted, pts_front)
    return H


def cup_affine_correction(warp_box, cup_warp, cup_front):
    """
    cup_warp:  (cx,cy,r) vị trí cốc trong simple warp
    cup_front: (cx,cy,r) vị trí cốc trong ảnh thẳng GT
    """
    cx_w, cy_w, r_w = cup_warp
    cx_f, cy_f, r_f = cup_front

    pts_src = np.float32([
        [cx_w, cy_w], [cx_w+r_w, cy_w],
        [cx_w, cy_w-r_w], [cx_w-r_w, cy_w]
    ])
    pts_dst = np.float32([
        [cx_f, cy_f], [cx_f+r_f, cy_f],
        [cx_f, cy_f-r_f], [cx_f-r_f, cy_f]
    ])

    M, _ = cv2.estimateAffinePartial2D(pts_src, pts_dst)
    h, w = warp_box.shape[:2]

    # Warp toàn ảnh với M
    warp_cup = cv2.warpAffine(warp_box, M, (w, h))

    # Tạo blend mask (hình tròn tại vị trí cốc)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx_w, cy_w), int(r_w * 1.15), 255, -1)
    mask_out  = cv2.warpAffine(mask, M, (w, h))
    mask_soft = cv2.GaussianBlur(
        mask_out.astype(np.float32), (51, 51), 0) / 255.0

    # Blend
    a = mask_soft[:, :, np.newaxis]
    result = (warp_cup * a + warp_box * (1 - a)).astype(np.uint8)
    return result, M


def run(front_path, tilted_path,
        pts_front_box, pts_tilted_box,
        cup_in_warp=None, cup_in_front=None,
        output_dir='output'):

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 55)
    print("  3D-Aware Rectification Pipeline")
    print("=" * 55)

    img_f = cv2.imread(front_path)
    img_t = cv2.imread(tilted_path)
    h, w  = img_f.shape[:2]

    # ── STEP 1: Box homography ──
    print("\n[1] Computing box homography ...")
    H_box  = compute_H_box(pts_front_box, pts_tilted_box)
    result = cv2.warpPerspective(img_t, H_box, (w, h))
    cv2.imwrite(f'{output_dir}/step1_simple_warp.jpg', result)
    print(f"    Saved → {output_dir}/step1_simple_warp.jpg")

    # ── STEP 2: Cup correction (optional) ──
    if cup_in_warp is not None and cup_in_front is not None:
        print("\n[2] Applying cup depth correction ...")
        result, M = cup_affine_correction(result.copy(), cup_in_warp, cup_in_front)
        
        # Nếu chưa biết cup_in_warp, auto-detect
        detected = detect_circle(result)
        if detected:
            print(f"    Cup in result: center=({detected[0]},{detected[1]}), r={detected[2]}")
            print(f"    Cup in GT:     center=({cup_in_front[0]},{cup_in_front[1]}), r={cup_in_front[2]}")
            err = np.array(detected[:2]) - np.array(cup_in_front[:2])
            print(f"    Position error: {np.linalg.norm(err[:2]):.1f} px")

    cv2.imwrite(f'{output_dir}/step2_depth_corrected.jpg', result)
    print(f"    Saved → {output_dir}/step2_depth_corrected.jpg")

    # ── STEP 3: Comparison ──
    print("\n[3] Saving comparison ...")
    def rsz(x): return cv2.resize(x, (w//2, h//2))

    comp_top = np.hstack([rsz(img_t), rsz(img_f)])
    simple   = cv2.warpPerspective(img_t, H_box, (w, h))
    comp_bot = np.hstack([rsz(simple), rsz(result)])

    def lbl(arr, txt, x=10):
        cv2.putText(arr, txt, (x, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    lbl(comp_top, 'INPUT (nghieng)')
    lbl(comp_top, 'GROUND TRUTH (thang)', w//2 + 10)
    lbl(comp_bot, 'Simple warpPerspective')
    lbl(comp_bot, 'Multi-H depth-aware', w//2 + 10)

    sep = np.full((6, comp_top.shape[1], 3), 200, np.uint8)
    comparison = np.vstack([comp_top, sep, comp_bot])
    cv2.imwrite(f'{output_dir}/step3_comparison.jpg', comparison)
    print(f"    Saved → {output_dir}/step3_comparison.jpg")

    print("\n✓ Done!")
    return result


if __name__ == '__main__':
    run(
        front_path    = FRONT_IMG,
        tilted_path   = TILTED_IMG,
        pts_front_box = PTS_FRONT_BOX,
        pts_tilted_box= PTS_TILTED_BOX,
        cup_in_warp   = CUP_IN_SIMPLE_WARP,
        cup_in_front  = CUP_IN_FRONT,
        output_dir    = OUTPUT_DIR,
    )
