import cv2
import numpy as np
import logging
from pathlib import Path

class WorkpieceDetection:
    def __init__(self, config):
        """
        config: Dictionary chứa các tham số từ file config.json
        """
        self.cfg = config.get('workpiece_check', {})
        self.template = None
        # Load template ngay khi khởi tạo để tăng tốc độ xử lý
        template_path = self.cfg.get('template_path', 'template_khongphoi.jpg')
        if Path(template_path).exists():
            self.template = cv2.imread(template_path, 0)
            self.temp_thres = 0.5
        else:
            logging.warning(f"Template không tồn tại tại: {template_path}. Cách 3 sẽ bị bỏ qua.")
            self.temp_thres = None
        self.br_thres = None

    def check_by_brightness(self, img_gray, save=None):
        """CÁCH 1: Phân tích độ sáng vùng tâm"""
        roi_img = img_gray.copy()
        radius = self.cfg.get('center_roi_radius_px', 40)
        threshold = self.cfg.get('brightness_threshold', 60)
        mask = np.zeros(roi_img.shape, dtype=np.uint8)
        cv2.circle(mask, (self.cfg["center"][0], self.cfg["center"][1]), radius, 255, -1)
        mean_val = cv2.mean(roi_img, mask=mask)[0]
        # print(f"self.center: {self.cfg["center"]}")
        self.br_thres = 80
        has_workpiece = mean_val > self.br_thres
        #vẽ hình
        if save is not None:
            frame  = cv2.imread(save)
            if has_workpiece:
                cv2.putText(frame, f"Co phoi", (20, 310), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"khong co phoi", (20, 310), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
            cv2.imwrite(save, frame)
        return  has_workpiece, mean_val

    def check_by_circles(self, img_gray):
        """CÁCH 2: Phát hiện vòng tròn của phôi"""
        # Cắt ROI tâm
        roi = img_gray.copy()[self.cfg["center"][1]-self.cfg["center_roi_radius_px"]:self.cfg["center"][1]+self.cfg["center_roi_radius_px"], self.cfg["center"][0]-self.cfg["center_roi_radius_px"]:self.cfg["center"][0]+self.cfg["center_roi_radius_px"]]
        # if roi.size == 0: return False, 0
        cv2.imwrite("/home/long/PROJECTS/AI/RobotVision/data/test_temp/circle_1.jpg", roi)
        circles = cv2.HoughCircles(
            roi, cv2.HOUGH_GRADIENT,
            dp=self.cfg.get('hough_wpc_dp', 1),
            minDist=self.cfg.get('hough_wpc_min_dist', 20),
            param1=self.cfg.get('hough_wpc_param1', 50),
            param2=self.cfg.get('hough_wpc_param2', 20),
            minRadius=self.cfg.get('hough_wpc_min_radius', 10),
            maxRadius=self.cfg.get('hough_wpc_max_radius', 35)
        )
        # vẻ hình debug
        for circle in circles[0]:
            cv2.circle(roi,(int(circle[0]), int(circle[1])),int(circle[2]),(0,255,0),2)
        cv2.imwrite("/home/long/PROJECTS/AI/RobotVision/data/test_temp/circle_2.jpg", roi)
        count = len(circles[0]) if circles is not None else 0
        return count > 0, count

    def check_by_template(self, img_gray):
        """CÁCH 3: So khớp với hốc trống (Template Matching)"""
        if self.template is None:
            return False, 0
        h, w = self.template.shape
        x, y = self.cfg["center"][0], self.cfg["center"][1]
        # Cắt vùng tìm kiếm rộng hơn template một chút
        search_size = int(max(h, w) * 0.8)
        roi_template = img_gray.copy()[self.cfg["center"][1]-self.cfg["center_roi_radius_px"]:self.cfg["center"][1]+self.cfg["center_roi_radius_px"], self.cfg["center"][0]-self.cfg["center_roi_radius_px"]:self.cfg["center"][0]+self.cfg["center_roi_radius_px"]]
        cv2.imwrite("/home/long/PROJECTS/AI/RobotVision/data/test_temp/circle_3.jpg", roi_template)
        if roi_template.shape[0] < h or roi_template.shape[1] < w: return False, 0
        
        res = cv2.matchTemplate(roi_template, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        return max_val > self.temp_thres, max_val

    def detect(self, img_gray, save = None):
        """
        Hợp nhất cả 2 phương pháp bằng cơ chế biểu quyết
        """
        res_br, val_br = self.check_by_brightness(img_gray)
        # res2, val2 = self.check_by_circles(img_gray)
        res_tpl, val_tpl = self.check_by_template(img_gray)
        # votes = [res1, res2, res3]
        # final_result = sum(votes) >= 2 # Ít nhất 2/3 phương pháp đồng ý
        conf_br = abs(val_br - self.br_thres) / max(self.br_thres, 255 - self.br_thres)
        conf_tpl = abs(val_tpl - self.temp_thres) / max(self.temp_thres, 1 - self.temp_thres)
        # --- 3. CHIẾN LƯỢC TRỌNG SỐ ĐỘNG ---
        total_conf = conf_br + conf_tpl
        if total_conf < 0.1:
            # Nếu cả 2 đều sát ngưỡng, ưu tiên Template (thường ổn định hơn)
            w_br, w_tpl = 0.6, 0.4
        else:
            # Phương pháp nào tự tin hơn (xa ngưỡng hơn) sẽ có trọng số cao hơn
            w_br = conf_br / total_conf
            w_tpl = conf_tpl / total_conf
        final_score = (int(res_br) * w_br) + (int(res_tpl) * w_tpl)
    
        # Quyết định cuối cùng: Nếu điểm tổng hợp > 0.5 là CÓ PHÔI
        has_workpiece = final_score > 0.5
        details = {
            "brightness": {"present": res_br, "value": round(val_br, 2), "conf": conf_br, "weight": w_br},
            # "circles": {"present": res2, "value": val2},
            "template": {"present": res_tpl, "value": round(val_tpl, 2), "conf": conf_tpl, "weight": w_tpl}
        }
        
        #vẽ hình
        if save is not None:
            frame  = cv2.imread(save)
            if has_workpiece:
                cv2.putText(frame, f"Co phoi", (20, 310), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Khong co phoi", (20, 310), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
            cv2.imwrite(save, frame)
        return has_workpiece, details