import cv2
import numpy as np
import logging
from pathlib import Path
import math
# import json
import itertools 

class JawDetection:
    def __init__(self, config):
        """
        config: Dictionary chứa các tham số từ file config.json
        """
        self.config = config
        self.center = self.config["mask"]["center"]
        self.config_jaw = self.config.get('jaw_info', {})
        # # DETECT
        # self.thres1 = np.mean([self.config_jaw["open"]["distance"][0], self.config_jaw["close"]["distance"][0]])
        # self.thres2 = np.mean([self.config_jaw["open"]["distance"][1], self.config_jaw["close"]["distance"][1]])
        # DETECT_OPEN
        # khoảng cách từ tâm tới các điểm tâm đường tròn nhỏ, tâm đường tròn to (khi mở)
        self.dist1 = self.distance(self.config_jaw["open"]["points"][0], self.center)
        self.dist2 = self.distance(self.config_jaw["open"]["points"][2], self.center)
        # khoảng cách từ tâm tới các điểm tâm đường tròn nhỏ, tâm đường tròn to (khi đóng)
        self.dist3 = self.distance(self.config_jaw["close"]["points"][0], self.center)
        self.dist4 = self.distance(self.config_jaw["close"]["points"][2], self.center)
        self.dists = [np.mean([self.dist1, self.dist3]), np.mean([self.dist2, self.dist4])]
        # threshold
        self.thres_large = np.mean([self.config_jaw["open"]["distance"][1],self.config_jaw["close"]["distance"][1]])
        print(f"thres_large: {self.thres_large}")
        self.thres_small = np.mean([self.config_jaw["open"]["distance"][0],self.config_jaw["close"]["distance"][0]])
        print(f"thres_small: {self.thres_small}")
    def detect(self, best_triple, save=None):
        """
            Xác định trạng thái chấu bằng phương pháp tính khoảng cách từ tâm mâm tới tâm lỗ chấu
        """
        # load ảnh để save
        if save is not None:
            frame = cv2.imread(save)

        results = []
        for c in best_triple:
            xi, yi, dist = c['pos'][0], c['pos'][1], c['dist']
            # dis_ = math.sqrt((xi-self.center[0])**2 + (yi-self.center[1])**2)
            # xác định xem nó gần thằng nào
            # Lỗ gần
            if np.abs(dist-self.thres1) < np.abs(dist-self.thres2):
                print(f"Khoảng cách từ lỗ gần đến tâm: {dist} | Threshold {self.thres1}")
                if dist < self.thres1:
                    # chấu đóng
                    results.append(False)
                else:
                    # chấu mở 
                    results.append(True)
            # lỗ xa 
            elif np.abs(dist-self.thres1) > np.abs(dist-self.thres2):
                print(f"Khoảng cách từ lỗ xa đến tâm: {dist} | Threshold {self.thres2}")
                if dist < self.thres2:
                    # chấu đóng
                    results.append(False)
                else:
                    # chấu mở 
                    results.append(True)
            else:
                return None
        true_count = results.count(True)
        false_count = results.count(False)
        if true_count > false_count:
            if save is not None:
                cv2.putText(frame, f"Chau mo", (20, 390),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)
                cv2.imwrite(save, frame)
            return True
        elif true_count < false_count:
            if save is not None:
                cv2.putText(frame, f"Chau dong", (20, 390),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
                cv2.imwrite(save, frame)
            return False
        else:
            if save is not None:
                cv2.putText(frame, f"Không xac dinh", (20, 390),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
                cv2.imwrite(save, frame)
            return None
        
    def distance(self, p1, p2):
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

    def detect_open(self, jaw_circles_full, save=None):
        # load ảnh để save
        if save is not None:
            frame = cv2.imread(save)
        small_c = []
        large_c = []
        result = []
        for c in jaw_circles_full:
            xi, yi, dist_c = c['pos'][0], c['pos'][1], c['dist']
            # xác định xem nó gần điểm -> nó là điểm lớn hay nhỏ
            if np.argmin([np.abs(dist_c - dist) for dist in self.dists]) == 0:
                # lỗ nhỏ 
                small_c.append(c)
            elif np.argmin([np.abs(dist_c - dist) for dist in self.dists]) == 1:
                # lỗ lớn
                large_c.append(c)
        # tính khoảng cách của từng lỗ trong nhóm small_c/large_c -> so sánh nó với threshold
        # True: mở, False: đóng
        # nhóm lớn
        if len(large_c) < 2 and len(small_c) < 2:
            return None
        for pair in itertools.combinations(large_c, 2):
            print(f"Khoảng cách giữa cặp lỗ lớn: {self.distance(pair[0]['pos'], pair[1]['pos']):.1f} | Threshold: {self.thres_large}")
            if self.distance(pair[0]['pos'], pair[1]['pos']) > self.thres_large:
                result.append(True)
            else: result.append(False)
        # nhóm nhỏ
        for pair in itertools.combinations(small_c, 2):
            print(f"Khoảng cách giữa cặp lỗ nhỏ: {self.distance(pair[0]['pos'], pair[1]['pos']):.1f} | Threshold: {self.thres_small}")
            if self.distance(pair[0]['pos'], pair[1]['pos']) > self.thres_small:
                result.append(True)
            else: result.append(False) 
        
        true_count = result.count(True)
        false_count = result.count(False)
        if true_count > false_count:
            if save is not None:
                cv2.putText(frame, f"Chau mo", (20, 390),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)
                cv2.imwrite(save, frame)
            return True
        elif true_count < false_count:
            if save is not None:
                cv2.putText(frame, f"Chau dong", (20, 390),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
                cv2.imwrite(save, frame)
            return False
        else:
            if save is not None:
                cv2.putText(frame, f"Không xac dinh", (20, 390),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
                cv2.imwrite(save, frame)
            return None
        

# class JawDetection:
#     def __init__(self, config):
#         """
#         config: Dictionary chứa các tham số từ file config.json
#         """
#         self.config = config
#         self.center = self.config["mask"]["center"]
#         self.dist1 = self.config["distance"]["dist1"]
#         self.dist2 = self.config["distance"]["dist2"]

#     def detect_open(self, best_triple):
#         """
#         Hợp nhất cả 2 phương pháp bằng cơ chế biểu quyết
#         """
#         distances = []
#         for c in best_triple:
#             xi, yi, rad = c['pos'][0], c['pos'][1], c['rad']
#             dis_ = math.sqrt((xi-self.center[0])**2 + (yi-self.center[1])**2)
#             # xác định xem nó gần thằng nào
#             if dis_ < min(self.dist1, self.dist2):
#                 distances.append(False)
#             elif dis_> max(self.dist1, self.dist2):
#                 distances.append(True)
#             elif abs(dis_ - self.dist1) < abs(dis_ - self.dist2):
#                 distances.append(True)
#             else: distances.append(False)
#         if sum(distances) >=2: return True
#         else: return False

# class JawDetection:
#     def __init__(self, config):
#         """
#         config: Dictionary chứa các tham số từ file config.json
#         """
#         self.config = config.get('jaw_info', {})
#         self.thres_near = (self.config["open"]["distance"][0] + self.config["close"]["distance"][0])/ 2
#         self.thres_far = (self.config["open"]["distance"][1] + self.config["close"]["distance"][1]) / 2
#         print(f"thres_near: {self.thres_near}, thres_far: {self.thres_far}")
#     def detect_open(self, best_triple):
#         """
#         Hợp nhất cả 2 phương pháp bằng cơ chế biểu quyết
#         """
#         results = []
#         for c in best_triple:
#             xi, yi, dist = c['pos'][0], c['pos'][1], c['dist']
#             print(f"Circle at ({xi}, {yi}) with distance {dist:.1f}")
#             # xác định xem nó gần thằng nào
#             # lỗ gần
#             if np.abs(dist-self.thres_near) < np.abs(dist-self.thres_far):
#                 # chấu đóng
#                 if dist < self.thres_near:
#                     results.append(False)
#                 # chấu mở
#                 else: results.append(True)
#             # lỗ xa
#             elif np.abs(dist-self.thres_near) > np.abs(dist-self.thres_far):
#                 # chấu đóng
#                 if dist < self.thres_far:
#                     results.append(False)
#                 # chấu mở
#                 else: results.append(True)
#         true_count = results.count(True)
#         false_count = results.count(False)
#         if true_count > false_count:
#             return True
#         elif true_count < false_count:
#             return False
#         else:
#             return None
        
# import cv2
# import numpy as np
# from collections import deque


# class ChuckStateDetector:

#     def __init__(
#         self,
#         open_area_thresh=12000,
#         history_size=7,
#         stable_count=5
#     ):

#         self.roi = (1036, 759, 740, 660)
#         self.open_area_thresh = open_area_thresh

#         self.history = deque(maxlen=history_size)
#         self.stable_count = stable_count

#         self.last_state = "UNKNOWN"

#     # =====================================================
#     # MAIN
#     # =====================================================
#     def detect(self, frame):

#         debug = frame.copy()

#         gray = self.preprocess(frame)

#         roi, roi_rect = self.extract_roi(gray)

#         binary = self.segment_dark_region(roi)

#         binary = self.postprocess(binary)

#         area, contour = self.measure_center_area(binary)

#         raw_state = self.classify(area)

#         stable_state = self.temporal_filter(raw_state)

#         confidence = abs(area - self.open_area_thresh)

#         self.draw_debug(
#             debug,
#             roi_rect,
#             contour,
#             area,
#             stable_state,
#             confidence
#         )

#         return {
#             "state": stable_state,
#             "area": area,
#             "confidence": confidence,
#             "debug": debug,
#             "binary": binary
#         }

#     # =====================================================
#     # PREPROCESS
#     # =====================================================
#     def preprocess(self, frame):

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         clahe = cv2.createCLAHE(
#             clipLimit=2.0,
#             tileGridSize=(8, 8)
#         )

#         gray = clahe.apply(gray)

#         gray = cv2.bilateralFilter(gray, 7, 40, 40)

#         return gray

#     # =====================================================
#     # ROI
#     # =====================================================
#     def extract_roi(self, gray):

#         h, w = gray.shape

#         # nếu chưa set ROI
#         if self.roi is None:

#             size = 300

#             cx = w // 2
#             cy = h // 2

#             x = cx - size // 2
#             y = cy - size // 2

#             roi_rect = (x, y, size, size)

#         else:

#             roi_rect = self.roi

#         x, y, rw, rh = roi_rect

#         roi = gray[y:y+rh, x:x+rw]

#         return roi, roi_rect

#     # =====================================================
#     # THRESHOLD
#     # =====================================================
#     def segment_dark_region(self, roi):

#         binary = cv2.adaptiveThreshold(
#             roi,
#             255,
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY_INV,
#             31,
#             7
#         )

#         return binary

#     # =====================================================
#     # MORPHOLOGY
#     # =====================================================
#     def postprocess(self, binary):

#         kernel = cv2.getStructuringElement(
#             cv2.MORPH_ELLIPSE,
#             (5, 5)
#         )

#         binary = cv2.morphologyEx(
#             binary,
#             cv2.MORPH_OPEN,
#             kernel
#         )

#         binary = cv2.morphologyEx(
#             binary,
#             cv2.MORPH_CLOSE,
#             kernel
#         )

#         return binary

#     # =====================================================
#     # MEASURE
#     # =====================================================
#     def measure_center_area(self, binary):

#         h, w = binary.shape

#         cx = w // 2
#         cy = h // 2

#         contours, _ = cv2.findContours(
#             binary,
#             cv2.RETR_EXTERNAL,
#             cv2.CHAIN_APPROX_SIMPLE
#         )

#         best_contour = None
#         best_area = 0

#         for cnt in contours:

#             area = cv2.contourArea(cnt)

#             if area < 100:
#                 continue

#             inside = cv2.pointPolygonTest(
#                 cnt,
#                 (cx, cy),
#                 False
#             )

#             if inside >= 0:

#                 if area > best_area:
#                     best_area = area
#                     best_contour = cnt

#         return best_area, best_contour

#     # =====================================================
#     # CLASSIFY
#     # =====================================================
#     def classify(self, area):

#         if area > self.open_area_thresh:
#             return "OPEN"

#         return "CLOSE"

#     # =====================================================
#     # TEMPORAL FILTER
#     # =====================================================
#     def temporal_filter(self, state):

#         self.history.append(state)

#         if len(self.history) < self.stable_count:
#             return self.last_state

#         open_count = self.history.count("OPEN")
#         close_count = self.history.count("CLOSE")

#         if open_count >= self.stable_count:
#             self.last_state = "OPEN"

#         elif close_count >= self.stable_count:
#             self.last_state = "CLOSE"

#         return self.last_state

#     # =====================================================
#     # DRAW DEBUG
#     # =====================================================
#     def draw_debug(
#         self,
#         frame,
#         roi_rect,
#         contour,
#         area,
#         state,
#         confidence
#     ):

#         x1, y1, rw, rh = roi_rect
#         x2 = x1 + rw
#         y2 = y1 + rh
#         # ROI
#         cv2.rectangle(
#             frame,
#             (x1, y1),
#             (x2, y2),
#             (255, 0, 0),
#             2
#         )

#         # contour
#         if contour is not None:

#             contour_global = contour.copy()
#             contour_global[:, :, 0] += x1
#             contour_global[:, :, 1] += y1

#             cv2.drawContours(
#                 frame,
#                 [contour_global],
#                 -1,
#                 (0, 255, 0),
#                 2
#             )

#         # center
#         h, w = frame.shape[:2]

#         cx = w // 2
#         cy = h // 2

#         cv2.circle(
#             frame,
#             (cx, cy),
#             5,
#             (0, 0, 255),
#             -1
#         )

#         # state color
#         color = (0, 255, 0)

#         if state == "CLOSE":
#             color = (0, 0, 255)

#         # text
#         cv2.putText(
#             frame,
#             f"STATE: {state}",
#             (30, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             color,
#             2
#         )

#         cv2.putText(
#             frame,
#             f"AREA: {int(area)}",
#             (30, 80),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (255, 255, 255),
#             2
#         )

#         cv2.putText(
#             frame,
#             f"CONF: {int(confidence)}",
#             (30, 120),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (255, 255, 255),
#             2
#         )

# if __name__ == "__main__":

#     input_video = "/home/long/PROJECTS/AI/RobotVision/data/video/Video_20260520134049678.avi"
#     output_video = "result.mp4"

#     cap = cv2.VideoCapture(input_video)

#     fps = cap.get(cv2.CAP_PROP_FPS)

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     print(f"Video FPS: {fps}, Width: {width}, Height: {height}")
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")

#     writer = cv2.VideoWriter(
#         output_video,
#         fourcc,
#         fps,
#         (width, height)
#     )
#     window_name = "Chuck State Detection"
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#     detector = ChuckStateDetector()

#     while True:

#         ret, frame = cap.read()

#         if not ret:
#             break

#         result = detector.detect(frame)

#         debug_frame = result["debug"]

#         writer.write(debug_frame)

#         cv2.imshow(window_name, debug_frame)

#         if cv2.waitKey(1) == 27:
#             break

#     cap.release()
#     writer.release()

#     cv2.destroyAllWindows()

#     print("DONE")