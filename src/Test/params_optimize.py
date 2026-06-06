import json
import cv2
import numpy as np
import math
import itertools
from pathlib import Path
from get_chuck_angle_circle import *

def detect_angle_with_params(image_rectified, cfg, dp, p1, p2):
    """Phiên bản tùy chỉnh để nhận tham số Hough trực tiếp"""
    rx, ry, rw, rh = cfg["roi"]
    roi_img = image_rectified[ry:ry+rh, rx:rx+rw]
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.medianBlur(enhanced, 5)

    circles = cv2.HoughCircles(
        blur, 
        cv2.HOUGH_GRADIENT, 
        dp=dp, 
        minDist=cfg['circles_info']['min_dist'],
        param1=p1,
        param2=p2,
        minRadius=cfg['circles_info']['min_radius'],
        maxRadius=cfg['circles_info']['max_radius']
    )

    if circles is None:
        return None, 0

    circles = np.uint16(np.around(circles))
    cx_mam, cy_mam = cfg["mask"]["center"]
    cx_mam -= rx; cy_mam -= ry
    ref_point_1, ref_point_2 = cfg['ref_points'][0], cfg['ref_points'][1]
    
    jaw_circles = []
    for i in circles[0, :]:
        xi, yi, ri = int(i[0]), int(i[1]), int(i[2])
        dist = math.sqrt((xi - cx_mam)**2 + (yi - cy_mam)**2)
        if cfg["mask"]["min_radius"] < dist < cfg["mask"]["max_radius"]:
            if is_real_hole(gray, xi, yi, ri):
                angle = calculate_angle_with_two_ref_points((cx_mam, cy_mam), (xi, yi), ref_point_1, ref_point_2)
                jaw_circles.append({'pos': (xi, yi), 'dist': dist, 'angle': angle})

    candidates, scores, _ = filter_symmetry_circles(jaw_circles, tolerance=15)
    
    if len(candidates) > 0:
        best_idx = np.argmax(scores)
        best_triple = candidates[best_idx]
        angles = [c['angle'] for c in best_triple]
        base_angles = [a % 120 for a in angles]
        return np.median(base_angles), scores[best_idx]
    
    return None, 0

def run_test_grid():
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"
    conf_dir = current_dir / "conf"

    # Định nghĩa các dải tham số  test
    dp_values = [1.0, 1.2, 1.4]
    p1_values = [30, 40, 50]
    p2_values = [20, 25, 30, 35]

    print(f"{'Image':<12} | {'DP':<4} | {'P1':<4} | {'P2':<4} | {'Angle':<8} | {'Score':<6}")
    print("-" * 60)

    for i in range(5, 11):
        img_name = f"chuck_{i}.jpg"
        cfg_name = f"chuck_{i}_config.json"
        
        img_path = data_dir / img_name
        cfg_path = conf_dir / cfg_name

        if not img_path.exists() or not cfg_path.exists():
            continue

        image = cv2.imread(str(img_path))
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)

        best_params_for_img = {"score": 0}

        for dp, p1, p2 in itertools.product(dp_values, p1_values, p2_values):
            angle, score = detect_angle_with_params(image, cfg, dp, p1, p2)
            
            if score > 0:
                print(f"{img_name:<12} | {dp:<4} | {p1:<4} | {p2:<4} | {angle:>8.2f} | {score:>6.1f}")
                
                if score > best_params_for_img["score"]:
                    best_params_for_img = {"dp": dp, "p1": p1, "p2": p2, "score": score, "angle": angle}

        if best_params_for_img["score"] > 0:
            print(f"==> Best for {img_name}: DP={best_params_for_img['dp']}, P1={best_params_for_img['p1']}, P2={best_params_for_img['p2']} (Score: {best_params_for_img['score']})")
        print("=" * 60)

if __name__ == "__main__":
    run_test_grid()