import cv2
import json
import logging
from pathlib import Path
from circle_detection.get_chuck_angle_circle import detect_angle_by_circles
from workpiece_detection.workpiece_check import WorkpieceDetection
from jaw_detection.jaw_check import JawDetection
from MvImport.MvCameraControl_class import *
import time
import numpy as np

class LiveDemoSystem:
    def __init__(self, config_path, camera_idx=0):
        self.config_path = Path(config_path)
        self.camera_idx = camera_idx
        
        # Tải cấu hình JSON
        self.config = self.load_config()
        
        # Trạng thái điều khiển (Flags)
        self.show_alignment = False  # Trạng thái phím 'l'
        self.run_detection = False   # Trạng thái phím 'd'
        
        
        self.wpc_detector = WorkpieceDetection(self.config)
        self.jaw_detector = JawDetection(self.config) 

    def load_config(self):
        if not self.config_path.exists():
            print(f"CẢNH BÁO: Không tìm thấy file cấu hình tại {self.config_path}")
            return {}
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def draw_alignment_guide(self, frame):
        """ VẼ CẤU HÌNH CŨ LÊN VIDEO ĐỂ CĂN CHỈNH CAMERA """
        # 1. Vẽ vùng ROI chữ nhật (nếu có)
        if "roi" in self.config:
            x, y, w, h = self.config["roi"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 191, 0), 2) # Màu xanh cyan
            cv2.putText(frame, "ALIGN ROI", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 191, 0), 1)

        # 2. Vẽ vòng tròn Mask/Tâm mâm đã xác định từ trước
        if "mask" in self.config:
            mask_cfg = self.config["mask"]
            center = tuple(mask_cfg.get("center", [0, 0]))
            max_r = mask_cfg.get("max_radius", 0)
            if center != (0, 0) and max_r > 0:
                # Vẽ tâm hình chữ thập
                cv2.drawMarker(frame, center, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                # Vẽ vòng tròn biên mâm mẫu
                cv2.circle(frame, center, max_r, (0, 255, 255), 2) # Màu vàng

        # 3. Vẽ vòng tròn vùng check phôi mẫu
        if "workpiece_check" in self.config:
            wpc_cfg = self.config["workpiece_check"]
            center = tuple(self.config.get("mask", {}).get("center", [0, 0]))
            r_wpc = wpc_cfg.get("center_roi_radius_px", 0)
            if center != (0, 0) and r_wpc > 0:
                cv2.circle(frame, center, r_wpc, (255, 0, 255), 1, cv2.LINE_AA) # Màu tím hồng

        # Gợi ý chữ trên màn hình
        cv2.putText(frame, "STATUS: ALIGNMENT MODE (Move cam to match overlays)", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame
    
    def run_inference(self, frame):
        rx, ry, w, h = self.config["roi"]
        cx_mam = self.config["mask"]["center"][0]
        cy_mam = self.config["mask"]["center"][1]
        """ CHẠY THUẬT TOÁN DETECT GÓC VÀ PHÔI THỰC TẾ """
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- PHẦN 1: TÍNH GÓC ---
        final_angle, best_score, best_triple, has_workpiece, is_open = detect_angle_by_circles(img_gray, self.config)
        for c in best_triple:
            xi, yi, rad = c['pos'][0], c['pos'][1], c['rad']
            xi += rx
            yi += ry
            angle = c['angle']
            cv2.circle(frame, (int(xi), int(yi)), int(rad), (0, 255, 0), 2)
            cv2.line(frame, (cx_mam, cy_mam), (int(xi), int(yi)), (255, 0, 0), 1)
            # 4. Hiển thị giá trị góc ngay tại vị trí lỗ
            # Text sẽ hiển thị dạng "45.0°"
            label = f"{angle:.1f}"
            # Đặt tọa độ text hơi lệch ra một chút để không đè lên lỗ
            text_pos = (int(xi) + 10, int(yi) - 10)    
            cv2.putText(frame, label, text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2) # Viền chữ đen
            cv2.putText(frame, label, text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 1)     # Chữ đỏ nội dung
        # vẽ trục
        # ref_point1 = self.config["ref_points"][0]
        # ref_point2 = self.config["ref_points"][1]
        # new_ref1, new_ref2 = convert_coods(ref_point1, ref_point2, frame) 
        # cv2.line(frame, (new_ref1[0], new_ref1[1]), (new_ref2[0], new_ref2[1]), (255, 0, 0), 1) 
        if final_angle is not None:
            cv2.putText(frame, f"Goc lech: {final_angle:.2f} deg", (20, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Score: {best_score:.1f}", (20, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX,2.5, (0, 0, 255), 2)
            
            if final_angle <5:
                cv2.putText(frame, f"Chau dung vi tri", (20, 230), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Chau sai vi tri", (20, 230), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Không xác định được góc lệch", (20, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
            
        if has_workpiece:
            cv2.putText(frame, f"Co phoi", (20, 310), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Khong co phoi", (20, 310), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
        if is_open:
            cv2.putText(frame, f"Chau mo", (2500, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Chau dong", (2500, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
        return frame
    
    def start(self):
        window_name = "live_demo"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            # Khởi tạo Camera Hikrobot
            deviceList = MV_CC_DEVICE_INFO_LIST()
            ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
            if ret != 0 or deviceList.nDeviceNum == 0:
                logging.warning("Không tìm thấy camera Hikrobot nào! Đang thử lại sau 2 giây...")
                print("Console: Đang tìm kiếm Camera...")
                time.sleep(2)
                continue
            
            cam = MvCamera()
            stDeviceImg = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
            cam.MV_CC_CreateHandle(stDeviceImg)
            cam.MV_CC_OpenDevice()
            if ret != 0:
                logging.error(f"Không thể mở thiết bị! Mã lỗi: {ret}")
                time.sleep(2)
                continue
            
            # # Cấu hình chế độ Hardware Trigger (Robot gửi tín hiệu vật lý)
            # # Trong thực tế, chân tín hiệu thường là Line0 hoặc Line1 của camera
            # cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
            # cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_LINE0) 

            # Bắt đầu thu nhận ảnh
            cam.MV_CC_StartGrabbing()
            stOutFrame = MV_FRAME_OUT()
            
            # logging.info("--- HỆ THỐNG SẴN SÀNG ---")
            # logging.info("Đang đợi tín hiệu trigger từ Robot...")

            try:
                while True:
                    # Đợi ảnh từ camera (Hàm sẽ dừng tại đây cho đến khi có tín hiệu điện từ Robot)
                    # Timeout 5000ms (5 giây)
                    ret = cam.MV_CC_GetImageBuffer(stOutFrame, 5000) 
                    
                    if ret == 0:
                        # start_proc = time.perf_counter()
                        
                        # Chuyển đổi dữ liệu ảnh từ SDK sang định dạng Numpy/OpenCV
                        # Camera MV-CS060-10GM-PRO thường trả về Mono8 hoặc Bayer
                        pData = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                        memmove(byref(pData), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
                        
                        # Giả định camera đang để mode Mono8 (Trắng đen công nghiệp)
                        img = np.frombuffer(pData, dtype=np.uint8).reshape(
                            stOutFrame.stFrameInfo.nHeight, 
                            stOutFrame.stFrameInfo.nWidth
                        )

                        # Chuyển sang BGR để tương thích với các hàm vẽ trong detect_angle_by_circles
                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        # Bản sao frame để xử lý tránh nhiễu nét vẽ
                        display_frame = img_gray.copy()

                        # Luồng 1: Nếu bật chế độ Căn chỉnh 'L'
                        if self.show_alignment:
                            display_frame = self.draw_alignment_guide(display_frame)
                            
                        # Luồng 2: Nếu bật chế độ Detect 'D'
                        elif self.run_detection:
                            display_frame = self.run_inference(display_frame)
                        
                        else:
                            # Trạng thái chờ mặc định
                            cv2.putText(display_frame, "STATUS: STANDBY (Press L to align or D to detect)", (15, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        # Hiển thị ảnh ra màn hình
                        cv2.imshow(window_name, display_frame)

                        # Bắt sự kiện bàn phím
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('l'):
                            self.show_alignment = not self.show_alignment
                            if self.show_alignment: 
                                self.run_detection = False # Tắt detect nếu đang căn chỉnh
                            print(f" -> Chế độ Căn chỉnh hiển thị: {self.show_alignment}")
                            
                        elif key == ord('d'):
                            self.run_detection = not self.run_detection
                            if self.run_detection: 
                                self.show_alignment = False # Tắt căn chỉnh nếu đang detect
                            print(f" -> Chế độ Thuật toán (Inference): {self.run_detection}")
                            
                        elif key == ord('q'):
                            print("Đang đóng ứng dụng demo...")
                            break

                        # Giải phóng buffer để nhận ảnh tiếp theo
                        cam.MV_CC_FreeImageBuffer(stOutFrame)
                    else:
                        # Nếu quá 5s không có tín hiệu, in thông báo đợi
                        logging.info("Đang đợi tín hiệu trigger...")
                        pass

            except KeyboardInterrupt:
                logging.info("Dừng hệ thống...")
            finally:
                cam.MV_CC_StopGrabbing()
                cam.MV_CC_CloseDevice()
                cam.MV_CC_DestroyHandle()

if __name__ == "__main__":
    CONFIG_FILE = "/home/long/PROJECTS/AI/RobotVision/config/circle_conf/frame_20260519-195351_871_config.json"
    
    demo = LiveDemoSystem(config_path=CONFIG_FILE, camera_idx=0)
    demo.start()