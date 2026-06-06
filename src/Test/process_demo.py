import cv2
import time
from workpiece_detection.workpiece_check import WorkpieceDetection
import get_chuck_angle_local
import json
import numpy as np
import argparse
from datetime import datetime
def convert_coods(x:list,y:list, img):
    a = (1.0*(y[1]-y[0]))/(1.0*(x[1]-x[0]))
    b = -a*x[0]+y[0]
    y1_ = (int)(0)
    y2_ = (int)(np.shape(img)[1])
    x1_ = (int)((y1_-b)/a)
    x2_ = (int)((y2_-b)/a)
    new_point1 = (x1_, x2_)
    new_point2 = (y1_, y2_)
    return new_point1, new_point2

class DemoProcessor:
    def __init__(self, detector, wpc_detector):
        self.detector = detector          # Đối tượng tính góc
        self.wpc_detector = wpc_detector  # Đối tượng check phôi

    def process_frame(self, frame):
        # 1. Tiền xử lý
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Detect góc và tâm mâm
        # Giả sử hàm detect trả về: angle, center, circles
        angle, center, circles = self.detector.detect_angle(gray)
        
        # 3. Check phôi tại tâm vừa tìm được
        has_wpc, wpc_details = self.wpc_detector.detect(gray, center)
        
        # 4. Vẽ kết quả lên Frame để demo
        self.visualize(frame, angle, center, circles, has_wpc, wpc_details)
        
        return frame

    def visualize(self, frame, angle, center, circles, has_wpc, wpc_details):
        # Vẽ tâm mâm
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        
        # Vẽ các chấu đã detect được
        for c in circles:
            cv2.circle(frame, (int(c[0]), int(c[1])), int(c[2]), (255, 255, 0), 2)
            
        # Hiển thị trạng thái phôi
        color_wpc = (0, 255, 0) if has_wpc else (0, 0, 255)
        status_text = "WORKPIECE: YES" if has_wpc else "WORKPIECE: NO"
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_wpc, 2)
        
        # Hiển thị góc xoay
        cv2.putText(frame, f"ANGLE: {angle:.2f} deg", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Vẽ vạch 0 độ (Reference)
        cv2.line(frame, center, (center[0], center[1] - 100), (0, 0, 255), 2)

def run_demo(video_path, output_path, config, segments):
    rx, ry, _, _ = config["roi"]
    cap = cv2.VideoCapture(video_path)
    
    # Thiết lập ghi video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    # Khởi tạo các detector
    wpc_detector = WorkpieceDetection(config)
    cx_mam = config["mask"]["center"][0]
    cy_mam = config["mask"]["center"][1]
    current_frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if len(segments) != 0:
            is_inside = any(start <= current_frame_idx <= end for start, end in segments)
        else: is_inside=True
        # Xử lý từng frame
        # processed_frame = processor.process_frame(frame)
        if is_inside:
            final_angle, best_score, best_triple, has_workpiece, is_open = get_chuck_angle_local.main_local(frame, config, demo=True)
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
            ref_point1 = config["ref_points"][0]
            ref_point2 = config["ref_points"][1]
            new_ref1, new_ref2 = convert_coods(ref_point1, ref_point2, frame) 
            cv2.line(frame, (new_ref1[0], new_ref1[1]), (new_ref2[0], new_ref2[1]), (255, 0, 0), 1) 
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
            
            if is_open is not None:
                if is_open:
                    cv2.putText(frame, f"Chau mo", (20, 390),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Chau dong", (20, 390),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Khong the xac dinh dong/mo", (20, 390),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
        out.write(frame) # Lưu lại để gửi khách hàng
        print(f"Processed frame {current_frame_idx}")
        current_frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="tạo demo trên video")
    parser.add_argument("--video_path", type=str, required=True, help="Đường dẫn file video")
    parser.add_argument("--config_path", type=str, required=True, help ="Đường dẫn file config")
    parser.add_argument("--output_dir", type=str, default= "/home/long/PROJECTS/AI/RobotVision/data/video/create_demo", help ="Đường dẫn file output")
    args = parser.parse_args()
    # load cấu hình
    # segments = [(8,60), (136,174), (231, 269), (320,350), (486,516), (556, 581), (636,814), (873, 889), (1020, 1100)]
    # segments = [(182,441), (529,568), (605, 634), (737,779), (859,896), (949, 987), (1028,1068), (1109, 1143), (1193, 1221), (1379,1431), (1463,1500), (1536, 1590), (1644, 1683), (1712, 1873)]
    segments = []
    # config_path = "/home/long/PROJECTS/AI/RobotVision/config/circle_conf/frame_20260519-123910_1037_config.json"
    # config_path = "/home/long/PROJECTS/AI/RobotVision/config/circle_conf/frame_20260519-195351_871_config.json"
    try:
        with open(args.config_path, 'r') as f:
            cfg = json.load(f)
    except:
        print("Lỗi đọc file config!")

    # video_path = "/home/long/PROJECTS/AI/RobotVision/src/RobotVision/workpiece_detection/Video_20260519104454379.mp4"
    # video_path = "/home/long/PROJECTS/AI/RobotVision/src/RobotVision/workpiece_detection/Video_20260519185303521.mp4"
    # output_path = "/home/long/PROJECTS/AI/RobotVision/data/video/create_demo/demo_6.mp4"
    output_path = f"{args.output_dir}/demo_{datetime.now().strftime("%d%m%Y-%H%M%S")}.mp4"
    run_demo(args.video_path, output_path, cfg, segments)