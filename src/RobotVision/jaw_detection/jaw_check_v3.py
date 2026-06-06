"""
===========================================
detect thủ công, đo bằng khoảng cách từ tâm mâm đến tâm của lỗ chấu 
===========================================
"""
import cv2
import numpy as np
import math
import itertools 

class JawDetection:
    def __init__(self, config):
        """
        config: Dictionary chứa các tham số từ file config.json
        """
        self.config = config
        self.center = self.config["mask"]["center"]
        self.config_jaw = self.config.get('jaw_info', {})
        # DETECT threshold
        self.thres_gan = np.mean([self.config_jaw["open"]["distance"][0], self.config_jaw["close"]["distance"][0]])
        self.thres_xa = np.mean([self.config_jaw["open"]["distance"][1], self.config_jaw["close"]["distance"][1]])
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
            # Lỗ gần
            if np.abs(dist-self.thres_gan) < np.abs(dist-self.thres_xa):
                print(f"Khoảng cách từ lỗ gần đến tâm: {dist} | Threshold {self.thres_gan}")
                if dist < self.thres_gan:
                    # chấu đóng
                    results.append(False)
                else:
                    # chấu mở 
                    results.append(True)
            # lỗ xa 
            elif np.abs(dist-self.thres_gan) > np.abs(dist-self.thres_xa):
                print(f"Khoảng cách từ lỗ xa đến tâm: {dist} | Threshold {self.thres_xa}")
                if dist < self.thres_xa:
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