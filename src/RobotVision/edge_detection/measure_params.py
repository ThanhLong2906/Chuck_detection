import cv2
import json
import math
import argparse
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"
import copy
from pathlib import Path

class VisionConfigToolEdge:
    def __init__(self, img_path, config_path):
        self.img_orig = cv2.imread(img_path)
        if self.img_orig is None:
            raise ValueError("Không thể tải ảnh. Kiểm tra lại đường dẫn!")
        
        self.config_path = Path(config_path)

        self.img_display = self.img_orig.copy()
        
        # Cấu hình mặc định
        self.default_config = {
            "roi": [0, 0, self.img_orig.shape[1], self.img_orig.shape[0]],
            "mask": {"center": [0, 0], "min_radius": 0, "max_radius": 0},
            "measurements": [],
            "ref_points": []
        }
        self.config = copy.deepcopy(self.default_config)
        
        self.points = []
        self.mode = None # 'roi', 'mask', 'measure', 'ref'
        self.window_name = "Vision Config Tool Enhanced"
        self.history = [] # Lưu lịch sử để Back (Undo)

    def save_history(self):
        """Lưu trạng thái trước khi thực hiện thay đổi."""
        self.history.append(copy.deepcopy(self.config))
        if len(self.history) > 30: self.history.pop(0)

    def load_config(self):
        """Ấn 'l' để load lại file config cũ."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            print(f"--- Đã load cấu hình từ {self.config_path} ---")
        else:
            print(f"--- Không tìm thấy file {self.config_path} ---")

    def save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"--- Đã lưu cấu hình vào {self.config_path} ---")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.save_history()
            
            # Chế độ ROI: 2 điểm
            if self.mode == 'roi':
                if len(self.points) >= 2: self.points = [] # Reset nếu click tiếp sau khi xong
                self.points.append((x, y))
                if len(self.points) == 2:
                    x1, y1, x2, y2 = self.points[0][0], self.points[0][1], self.points[1][0], self.points[1][1]
                    self.config["roi"] = [min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)]
                    self.points = []

            # Chế độ Mask: 3 điểm (Tâm -> R_min -> R_max)
            elif self.mode == 'mask':
                if len(self.points) >= 3:
                    self.points = []
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

            # Chế độ Measure: 2 điểm (Thêm vào danh sách measurements)
            elif self.mode == 'measure':
                self.points.append((x, y))
                if len(self.points) == 2:
                    dist = math.sqrt((x-self.points[0][0])**2 + (y-self.points[0][1])**2)
                    self.config["measurements"].append({"p1": self.points[0], "p2": self.points[1], "dist": dist})
                    self.points = []

            # Chế độ Ref: 2 điểm
            elif self.mode == 'ref':
                if len(self.config["ref_points"]) >= 2:
                    self.config["ref_points"] = [] # Reset nếu click tiếp sau khi xong
                self.config["ref_points"].append((x, y))

    def draw_overlay(self):
        self.img_display = self.img_orig.copy()
        h, w = self.img_display.shape[:2]

        # Vẽ ROI (Xanh dương)
        r = self.config["roi"]
        cv2.rectangle(self.img_display, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (255, 0, 0), 2)

        # Vẽ Mask (Vàng)
        m = self.config["mask"]
        if m["center"] != [0, 0]:
            center = tuple(m["center"])
            cv2.circle(self.img_display, center, 4, (0, 255, 255), -1)
            if m["min_radius"] > 0:
                cv2.circle(self.img_display, center, m["min_radius"], (0, 255, 255), 2)
            if m["max_radius"] > 0:
                cv2.circle(self.img_display, center, m["max_radius"], (0, 255, 255), 2)

        # Vẽ Measurements (Xanh lá)
        for meas in self.config["measurements"]:
            p1, p2 = tuple(meas["p1"]), tuple(meas["p2"])
            cv2.line(self.img_display, p1, p2, (0, 255, 0), 2)
            cv2.putText(self.img_display, f"{meas['dist']:.1f}px", p2, 1, 1, (0, 255, 0), 1)

        # Vẽ Reference Points (Trắng)
        ref_pts = self.config["ref_points"]
        for p in ref_pts: cv2.circle(self.img_display, p, 4, (255, 255, 255), -1)
        if len(ref_pts) == 2:
            cv2.arrowedLine(self.img_display, ref_pts[0], ref_pts[1], (255, 255, 255), 2, tipLength=0.1)

        # Hiển thị điểm đang click dở (Đỏ)
        for p in self.points:
            cv2.circle(self.img_display, p, 3, (0, 0, 255), -1)

        # Menu hướng dẫn
        menu = "[R]:ROI [M]:Mask [D]:Dist [F]:Ref [L]:Load [B]:Back [C]:Clear All [S]:Save [Q]:Quit"
        status = f"MODE: {str(self.mode).upper()} | Points: {len(self.points)}"
        cv2.putText(self.img_display, menu, (10, h - 40), 1, 1.2, (255, 255, 255), 2)
        cv2.putText(self.img_display, status, (10, h - 15), 1, 1.2, (0, 255, 0), 2)

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            self.draw_overlay()
            cv2.imshow(self.window_name, self.img_display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'): self.mode = 'roi'; self.points = []
            elif key == ord('m'): self.mode = 'mask'; self.points = []
            elif key == ord('d'): self.mode = 'measure'; self.points = []
            elif key == ord('f'): self.mode = 'ref'; self.points = []
            elif key == ord('l'): self.load_config() # Tải lại config cũ
            elif key == ord('b'): # Undo hành động gần nhất
                if self.history:
                    self.config = self.history.pop()
                    self.points = []
                    print("--- Đã quay lại bước trước ---")
            elif key == ord('c'): # Xóa hết tham số
                self.config = copy.deepcopy(self.default_config)
                self.config["roi"] = [0, 0, self.img_orig.shape[1], self.img_orig.shape[0]]
                self.points = []
                self.history = []
                print("--- Đã xóa hết tham số đo ---")
            elif key == ord('s'): self.save_config()
            elif key == ord('q'): break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    current_dir = Path(__file__).parent
    img_path = args.image if Path(args.image).is_absolute() else current_dir / args.image
    cfg_path = args.config if Path(args.config).is_absolute() else current_dir / args.config

    tool = VisionConfigToolEdge(str(img_path), str(cfg_path))
    tool.run()