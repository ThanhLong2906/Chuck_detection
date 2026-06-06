import cv2
import json
import math
import argparse
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
current_dir = Path(__file__).parent
load_dotenv(f"{current_dir.parent.parent}/.env", override=True)
DATA_DIR = os.getenv("DATA_DIR")
class VisionConfigToolCircle:
    def __init__(self, img_path, config_path):
        self.img_orig = cv2.imread(str(img_path))
        if self.img_orig is None:
            raise ValueError("Không thể tải ảnh. Kiểm tra lại đường dẫn!")
        
        self.config_path = Path(config_path)
        
        self.img_display = self.img_orig.copy()
        # Khởi tạo cấu hình mặc định
        self.default_config = {
            "roi": [0, 0, self.img_orig.shape[1], self.img_orig.shape[0]],
            "mask": {"center": [0, 0], "min_radius": 0, "max_radius": 0},
            "circles_info": {
                "detected_circles": [], 
                "min_radius": 0,
                "max_radius": 0,
                "min_dist": 0
            },
            "ref_points": [],
            "jaw_info": {
                "open":{"points":[],
                        "distance":[]},
                "close": {"points":[],
                        "distance":[]}
            },
            "distance": {"points":[],
                "dist1": 0,
                "dist2": 0
            },
            "workpiece_check": {
                "center": [],
                "center_roi_radius_px": 0,
                "brightness_threshold": 0,
                "hough_wpc_min_radius": 0,
                "hough_wpc_max_radius": 0,
                "template_path": ""
            }
        }
        self.config = self.default_config.copy()
        
        self.points = []
        self.mode = None 
        self.window_name = "Vision Comprehensive Tool"
        self.history = [] # Lưu lịch sử để Back (Undo)

    def save_history(self):
        """Lưu bản sao của config hiện tại vào lịch sử trước mỗi hành động quan trọng."""
        import copy
        self.history.append(copy.deepcopy(self.config))
        if len(self.history) > 20: # Giới hạn 20 bước quay lại
            self.history.pop(0)

    def load_config(self):
        """Ấn 'l' để load lại config cũ."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            print(f"--- Đã load cấu hình từ {self.config_path} ---")
        else:
            print(f"--- Không tìm thấy file {self.config_path} để load ---")

    def save_config(self):
        circles = self.config["circles_info"]["detected_circles"]
        if circles:
            radii = [c['radius'] for c in circles]
            self.config["circles_info"]["min_radius"] = int(min(radii) - 2)
            self.config["circles_info"]["max_radius"] = int(max(radii) + 2)
            
            if len(circles) >= 2:
                dists = []
                for i in range(len(circles)):
                    for j in range(i + 1, len(circles)):
                        p1, p2 = circles[i]['center'], circles[j]['center']
                        dists.append(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))
                self.config["circles_info"]["min_dist"] = int(min(dists) - 5) if min(dists) > 5 else 5

        wpc = self.config["workpiece_check"]
        if wpc["center_roi_radius_px"] > 0:
            r = wpc["center_roi_radius_px"]
            wpc["hough_wpc_min_radius"] = int(r * 0.7)
            wpc["hough_wpc_max_radius"] = int(r * 1.3)
            wpc["hough_wpc_min_dist"] = int(r * 0.5)

        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"--- Đã lưu cấu hình vào  {self.config_path} ---")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.save_history() # Lưu lại trạng thái trước khi thay đổi

            # Chế độ ROI
            if self.mode == 'roi':
                if len(self.points) >= 2: self.points = [] # Reset nếu đã đủ 2 điểm mà vẫn click
                self.points.append((x, y))
                if len(self.points) == 2:
                    x1, y1 = self.points[0]
                    x2, y2 = self.points[1]
                    self.config["roi"] = [min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)]
                    self.points = []

            # Chế độ distance (khoảng cách 2 lỗ ở 2 chấu khác nhau)
            elif self.mode == "open":
                if len(self.points) >= 4: self.points = []
                self.points.append((x,y))
                if len(self.points) == 2:
                    dists = math.sqrt((self.points[1][0]-self.points[0][0])**2 + (self.points[1][1]-self.points[0][1])**2)
                    self.config["jaw_info"]["open"]["distance"].append(dists)
                    self.config["jaw_info"]["open"]["points"].append(self.points[0])
                    self.config["jaw_info"]["open"]["points"].append(self.points[1])
                elif len(self.points) ==4:
                    distl = math.sqrt((self.points[3][0]-self.points[2][0])**2 + (self.points[3][1]-self.points[2][1])**2)
                    self.config["jaw_info"]["open"]["distance"].append(distl)
                    self.config["jaw_info"]["open"]["points"].append(self.points[2])
                    self.config["jaw_info"]["open"]["points"].append(self.points[3])
                    self.points = []

            elif self.mode == "close":
                if len(self.points) >= 4: self.points = []
                self.points.append((x,y))
                if len(self.points) == 2:
                    dists = math.sqrt((self.points[1][0]-self.points[0][0])**2 + (self.points[1][1]-self.points[0][1])**2)
                    self.config["jaw_info"]["close"]["distance"].append(dists)
                    self.config["jaw_info"]["close"]["points"].append(self.points[0])
                    self.config["jaw_info"]["close"]["points"].append(self.points[1])
                elif len(self.points) ==4:
                    distl = math.sqrt((self.points[3][0]-self.points[2][0])**2 + (self.points[3][1]-self.points[2][1])**2)
                    self.config["jaw_info"]["close"]["distance"].append(distl)
                    self.config["jaw_info"]["close"]["points"].append(self.points[2])
                    self.config["jaw_info"]["close"]["points"].append(self.points[3])
                    self.points = []
            
            # # Chế độ distance (khoảng cách 2 lỗ ở 2 chấu khác nhau)
            # elif self.mode == "open":
            #     if len(self.points) >= 3: self.points = []
            #     self.points.append((x,y))
            #     if len(self.points) == 2:
            #         dist = math.sqrt((self.points[1][0]-self.points[0][0])**2 + (self.points[1][1]-self.points[0][1])**2)
            #         self.config["jaw_info"]["open"]["distance"].append(dist)
            #         self.config["jaw_info"]["open"]["points"].append(self.points[0])
            #         self.config["jaw_info"]["open"]["points"].append(self.points[1])
            #     elif len(self.points) ==3:
            #         dist = math.sqrt((self.points[2][0]-self.points[0][0])**2 + (self.points[2][1]-self.points[0][1])**2)
            #         self.config["jaw_info"]["open"]["distance"].append(dist)
            #         self.config["jaw_info"]["open"]["points"].append(self.points[2])
            #         self.points = []

            # elif self.mode == "close":
            #     if len(self.points) >= 3: self.points = []
            #     self.points.append((x,y))
            #     if len(self.points) == 2:
            #         dist = math.sqrt((self.points[1][0]-self.points[0][0])**2 + (self.points[1][1]-self.points[0][1])**2)
            #         self.config["jaw_info"]["close"]["distance"].append(dist)
            #         self.config["jaw_info"]["close"]["points"].append(self.points[0])
            #         self.config["jaw_info"]["close"]["points"].append(self.points[1])
            #     elif len(self.points) ==3:
            #         dist = math.sqrt((self.points[2][0]-self.points[0][0])**2 + (self.points[2][1]-self.points[0][1])**2)
            #         self.config["jaw_info"]["close"]["distance"].append(dist)
            #         self.config["jaw_info"]["close"]["points"].append(self.points[2])
            #         self.points = []

            # Chế độ distance
            elif self.mode == "distance":
                if len(self.points) >= 3: self.points = []
                self.points.append((x,y))
                if len(self.points) == 2:
                    dist1 = math.sqrt((self.points[1][0]-self.points[0][0])**2 + (self.points[1][1]-self.points[0][1])**2)
                    self.config["distance"]["dist1"] = dist1
                    self.config["distance"]["points"].append(self.points[0])
                    self.config["distance"]["points"].append(self.points[1])
                elif len(self.points) ==3:
                    dist2 = math.sqrt((self.points[2][0]-self.points[0][0])**2 + (self.points[2][1]-self.points[0][1])**2)
                    self.config["distance"]["points"].append(self.points[2])
                    self.config["distance"]["dist2"] = dist2
                    self.points = []

            # Chế độ Mask
            elif self.mode == 'mask':
                if len(self.points) >= 3: 
                    self.points = [] # Xóa hết bắt đầu lại
                    self.config["mask"] = {"center": [0, 0], "min_radius": 0, "max_radius": 0}
                
                self.points.append((x, y))
                if len(self.points) == 1:
                    self.config["mask"]["center"] = [x, y]
                elif len(self.points) == 2:
                    dist = math.sqrt((x-self.points[0][0])**2 + (y-self.points[0][1])**2)
                    self.config["mask"]["min_radius"] = int(dist)
                elif len(self.points) == 3:
                    dist = math.sqrt((x-self.points[0][0])**2 + (y-self.points[0][1])**2)
                    self.config["mask"]["max_radius"] = int(dist)
                    self.points = []

            # Chế độ Circle
            elif self.mode == 'circle':
                self.points.append((x, y))
                if len(self.points) == 2:
                    r = math.sqrt((x - self.points[0][0])**2 + (y - self.points[0][1])**2)
                    self.config["circles_info"]["detected_circles"].append({'center': self.points[0], 'radius': r})
                    self.points = []

            # Chế độ Reference Point
            elif self.mode == 'ref':
                if len(self.config["ref_points"]) >= 2: 
                    self.config["ref_points"] = [] # Nếu đã đủ 2 điểm mà click tiếp thì xóa hết làm lại
                self.config["ref_points"].append((x, y))

            # --- CHẾ ĐỘ MỚI Workpiece (Phím 'W') ---
            elif self.mode == 'wpc':
                self.points.append((x, y))
                if len(self.points) == 2:
                    # 1. Tính bán kính từ tâm (điểm 1) đến điểm click (điểm 2)
                    p1, p2 = self.points[0], self.points[1]
                    r = int(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))
                    self.config["workpiece_check"]["center"] = self.points[0]
                    # 2. Cập nhật config
                    self.config["workpiece_check"]["center_roi_radius_px"] = r
                    
                    # 3. Tính độ sáng trung bình tại vùng vừa vẽ để gợi ý Threshold
                    img_gray = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2GRAY)
                    mask = np.zeros(img_gray.shape, dtype=np.uint8)
                    cv2.circle(mask, p1, r, 255, -1)
                    avg_brightness = cv2.mean(img_gray, mask=mask)[0]
                    self.config["workpiece_check"]["brightness_threshold"] = int(avg_brightness)
                    
                    # 4. Tự động cắt và lưu Template (Hốc trống)
                    # Lưu vào thư mục cùng cấp với file config
                    tpl_dir = DATA_DIR / "templates"
                    tpl_dir.mkdir(exist_ok=True)
                    tpl_path = tpl_dir / f"{self.config_path.stem}_tpl.jpg"
                    
                    roi_tpl = self.img_orig[p1[1]-r:p1[1]+r, p1[0]-r:p1[0]+r]
                    if roi_tpl.size > 0:
                        cv2.imwrite(str(tpl_path), roi_tpl)
                        self.config["workpiece_check"]["template_path"] = str(tpl_path)
                        print(f"--- Đã lưu template tại: {tpl_path} ---")

                    print(f"--- Config WPC: Radius {r}, Brightness {int(avg_brightness)} ---")
                    self.points = []

    def draw_overlay(self):
        self.img_display = self.img_orig.copy()
        h, w = self.img_display.shape[:2]

        # Hiển thị ROI
        r = self.config["roi"]
        cv2.rectangle(self.img_display, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (255, 0, 0), 2)
        
        # # hiển thị jaw_info
        # if len(self.config["jaw_info"]["open"]["points"]) != 0:
        #     if len(self.config["jaw_info"]["open"]["points"]) >=2:
        #         p1, p2 = tuple(self.config["jaw_info"]["open"]["points"][0]), tuple(self.config["jaw_info"]["open"]["points"][1])#, tuple(self.config["distance"]["points"][2])
        #         cv2.line(self.img_display, p1, p2, (0, 255, 0), 2)
        #         cv2.putText(self.img_display, f"{self.config["jaw_info"]["open"]['distance'][0]:.1f}px", p2, 1, 1, (0, 255, 0), 1)
        #     if len(self.config["jaw_info"]["open"]["points"]) >= 3:
        #         p3 = tuple(self.config["jaw_info"]["open"]["points"][2])#, tuple(self.config["jaw_info"]["open"]["points"][3])
        #         cv2.line(self.img_display, p1, p3, (0, 255, 0), 2)
        #         cv2.putText(self.img_display, f"{self.config["jaw_info"]["open"]['distance'][1]:.1f}px", p3, 1, 1, (0, 255, 0), 1)
    
        # if len(self.config["jaw_info"]["close"]["points"]) != 0:
        #     if len(self.config["jaw_info"]["close"]["points"]) >=2:
        #         p1, p2 = tuple(self.config["jaw_info"]["close"]["points"][0]), tuple(self.config["jaw_info"]["close"]["points"][1])#, tuple(self.config["distance"]["points"][2])
        #         cv2.line(self.img_display, p1, p2, (0, 255, 0), 2)
        #         cv2.putText(self.img_display, f"{self.config["jaw_info"]["close"]['distance'][0]:.1f}px", p2, 1, 1, (0, 255, 0), 1)
        #     if len(self.config["jaw_info"]["close"]["points"]) >= 3:
        #         p3 = tuple(self.config["jaw_info"]["close"]["points"][2])#, tuple(self.config["jaw_info"]["close"]["points"][3])
        #         cv2.line(self.img_display, p1, p3, (0, 255, 0), 2)
        #         cv2.putText(self.img_display, f"{self.config["jaw_info"]["close"]['distance'][1]:.1f}px", p3, 1, 1, (0, 255, 0), 1)

        # hiển thị jaw_info
        if len(self.config["jaw_info"]["open"]["points"]) != 0:
            if len(self.config["jaw_info"]["open"]["points"]) >=2:
                p1, p2 = tuple(self.config["jaw_info"]["open"]["points"][0]), tuple(self.config["jaw_info"]["open"]["points"][1])#, tuple(self.config["distance"]["points"][2])
                cv2.line(self.img_display, p1, p2, (0, 255, 0), 2)
                cv2.putText(self.img_display, f"{self.config["jaw_info"]["open"]['distance'][0]:.1f}px", p2, 1, 1, (0, 255, 0), 1)
            if len(self.config["jaw_info"]["open"]["points"]) >= 4:
                p3,p4 = tuple(self.config["jaw_info"]["open"]["points"][2]), tuple(self.config["jaw_info"]["open"]["points"][3])
                cv2.line(self.img_display, p3, p4, (0, 255, 0), 2)
                cv2.putText(self.img_display, f"{self.config["jaw_info"]["open"]['distance'][1]:.1f}px", p4, 1, 1, (0, 255, 0), 1)
    
        if len(self.config["jaw_info"]["close"]["points"]) != 0:
            if len(self.config["jaw_info"]["close"]["points"]) >=2:
                p1, p2 = tuple(self.config["jaw_info"]["close"]["points"][0]), tuple(self.config["jaw_info"]["close"]["points"][1])#, tuple(self.config["distance"]["points"][2])
                cv2.line(self.img_display, p1, p2, (0, 255, 0), 2)
                cv2.putText(self.img_display, f"{self.config["jaw_info"]["close"]['distance'][0]:.1f}px", p2, 1, 1, (0, 255, 0), 1)
            if len(self.config["jaw_info"]["close"]["points"]) >= 4:
                p3,p4 = tuple(self.config["jaw_info"]["close"]["points"][2]), tuple(self.config["jaw_info"]["close"]["points"][3])
                cv2.line(self.img_display, p3, p4, (0, 255, 0), 2)
                cv2.putText(self.img_display, f"{self.config["jaw_info"]["close"]['distance'][1]:.1f}px", p4, 1, 1, (0, 255, 0), 1)

        # hiển thị đoạn thẳng distance
        if len(self.config["distance"]["points"]) != 0:
            if len(self.config["distance"]["points"]) >=2:
                p1, p2 = tuple(self.config["distance"]["points"][0]), tuple(self.config["distance"]["points"][1])#, tuple(self.config["distance"]["points"][2])
                cv2.line(self.img_display, p1, p2, (0, 255, 0), 2)
                cv2.putText(self.img_display, f"{self.config["distance"]['dist1']:.1f}px", p2, 1, 1, (0, 255, 0), 1)
            if len(self.config["distance"]["points"]) >= 3:
                p3 = tuple(self.config["distance"]["points"][2])
                cv2.line(self.img_display, p1, p3, (0, 255, 0), 2)
                cv2.putText(self.img_display, f"{self.config["distance"]['dist2']:.1f}px", p3, 1, 1, (0, 255, 0), 1)
        
        # Hiển thị điểm đang click dở dang
        for p in self.points:
            cv2.circle(self.img_display, p, 3, (0, 0, 255), -1)

        # Hiển thị Mask
        m = self.config["mask"]
        if m["center"] != [0, 0]:
            cv2.circle(self.img_display, tuple(m["center"]), 5, (0, 255, 255), -1)
            if m["min_radius"] > 0:
                cv2.circle(self.img_display, tuple(m["center"]), m["min_radius"], (0, 255, 255), 2)
            if m["max_radius"] > 0:
                cv2.circle(self.img_display, tuple(m["center"]), m["max_radius"], (0, 255, 255), 2)

        # Hiển thị Circles
        for circ in self.config["circles_info"]["detected_circles"]:
            cv2.circle(self.img_display, tuple(circ['center']), int(circ['radius']), (0, 165, 255), 2)
            cv2.circle(self.img_display, tuple(circ['center']), 2, (0, 0, 255), -1)

        # Hiển thị Reference
        ref_pts = self.config["ref_points"]
        for p in ref_pts: cv2.circle(self.img_display, p, 4, (255, 255, 255), -1)
        if len(ref_pts) == 2:
            cv2.arrowedLine(self.img_display, ref_pts[0], ref_pts[1], (255, 255, 255), 2, tipLength=0.1)

        # HIỂN THỊ WORKPIECE CHECK (Màu tím hồng)
        wpc = self.config["workpiece_check"]
        mask_center = self.config["workpiece_check"]["center"]
        if mask_center != [0, 0] and wpc["center_roi_radius_px"] > 0:
            cv2.circle(self.img_display, tuple(mask_center), wpc["center_roi_radius_px"], (255, 0, 255), 2)
            cv2.putText(self.img_display, "WPC ROI", (mask_center[0]+5, mask_center[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
        # Menu và trạng thái
        menu = "[R]:ROI [M]:Mask [D]:Dist [O]:Circle [F]:Ref [W]:Workpiece [L]:Load [B]:Back [C]:Clear All [S]:Save [Q]:Quit"
        curr_mode = f"MODE: {str(self.mode).upper()} | Points in queue: {len(self.points)}"
        cv2.putText(self.img_display, menu, (10, h - 45), 1, 1.2, (255, 255, 255), 2)
        cv2.putText(self.img_display, curr_mode, (10, h - 15), 1, 1.2, (0, 255, 0), 2)

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        print("Bắt đầu công cụ cấu hình tổng hợp...")

        while True:
            self.draw_overlay()
            cv2.imshow(self.window_name, self.img_display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'): self.mode = 'roi'; self.points = []
            elif key == ord('m'): self.mode = 'mask'; self.points = []
            elif key == ord('o'): self.mode = 'circle'; self.points = []
            elif key == ord('f'): self.mode = 'ref'; self.points = []
            elif key == ord('w'): self.mode = 'wpc'; self.points = [] # 
            elif key == ord('k'): self.mode = 'open'; self.points = [] # 
            elif key == ord('h'): self.mode = 'close'; self.points = [] # 
            elif key == ord('l'): self.load_config() # Load config cũ
            elif key == ord('b'): # Back (Undo) hành động vừa thực hiện
                if self.history:
                    self.config = self.history.pop()
                    self.points = []
                    print("--- Đã quay lại bước trước ---")
            elif key == ord('c'): # Xóa hết toàn bộ
                self.config = self.default_config.copy()
                self.config["roi"] = [0, 0, self.img_orig.shape[1], self.img_orig.shape[0]]
                self.points = []
                self.history = []
                print("--- Đã xóa hết tham số ---")
            elif key == ord('s'): self.save_config()
            elif key == ord('q'): break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    config_path = current_dir / "config.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    image_path = Path(args.image) if Path(args.image).is_absolute() else current_dir / args.image
    tool = VisionConfigToolCircle(image_path, config_path)
    tool.run()