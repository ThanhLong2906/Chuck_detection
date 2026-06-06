import cv2
import numpy as np
import argparse
from pathlib import Path

# ===== CẤU HÌNH =====
parser = argparse.ArgumentParser(description="Chương trình chọn điểm và tính homography")
parser.add_argument("--src", type=str, default=None, help="Đường dẫn ảnh nguồn (nghiêng)")
parser.add_argument("--dst", type=str, default=None, help="Đường dẫn ảnh đích (phẳng)")
parser.add_argument("--save_config", type=str, default="config.py", help="Đường dẫn file để lưu cấu hình (mặc định: config.py)") 
parser.add_argument("--warped", type = str, default="warped.jpeg", help="Đường dẫn file để lưu ảnh kết quả (mặc định: warped.jpeg)")  
args = parser.parse_args()
current_dir = Path(__file__).parent

PATH_IMAGE_SRC = args.src if Path(args.src).is_absolute() else Path(current_dir/"imgs"/args.src)
PATH_IMAGE_DST = args.dst if Path(args.dst).is_absolute() else Path(current_dir/"imgs"/args.dst)
PATH_WARPED = args.warped if Path(args.warped).is_absolute() else Path(current_dir/"imgs"/args.warped) 
CONFIG_PATH = args.save_config if Path(args.save_config).is_absolute() else Path(current_dir / args.save_config)
# Danh sách lưu tọa độ các điểm đã chọn
src_points = []
dst_points = []

# Màu sắc và độ dày nét vẽ
COLOR_POINT = (0, 0, 255) # Màu đỏ (BGR)
COLOR_TEXT = (0, 255, 0)   # Màu xanh lá (BGR)

# ===== CÁC HÀM XỬ LÝ SỰ KIỆN CHUỘT =====

def click_event_src(event, x, y, flags, param):
    """Xử lý click chuột trên ảnh nguồn."""
    global src_points, img_src_display
    
    # Khi nhấn chuột trái
    if event == cv2.EVENT_LBUTTONDOWN:
        # Lưu tọa độ
        src_points.append((x, y))
        print(f"Nguồn: Đã chọn điểm {len(src_points)} tại ({x}, {y})")
        
        # Vẽ chấm tròn và số thứ tự lên ảnh hiển thị
        cv2.circle(img_src_display, (x, y), 5, COLOR_POINT, -1)
        cv2.putText(img_src_display, str(len(src_points)-1), (x + 10, y + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        
        # Cập nhật lại cửa sổ
        cv2.imshow("1. SOURCE - Anh Nghieng", img_src_display)

def click_event_dst(event, x, y, flags, param):
    """Xử lý click chuột trên ảnh đích."""
    global dst_points, img_dst_display
    
    # Khi nhấn chuột trái
    if event == cv2.EVENT_LBUTTONDOWN:
        # Lưu tọa độ
        dst_points.append((x, y))
        print(f"Đích: Đã chọn điểm {len(dst_points)} tại ({x}, {y})")
        
        # Vẽ chấm tròn và số thứ tự lên ảnh hiển thị
        cv2.circle(img_dst_display, (x, y), 5, COLOR_POINT, -1)
        cv2.putText(img_dst_display, str(len(dst_points)-1), (x + 10, y + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        
        # Cập nhật lại cửa sổ
        cv2.imshow("2. DESTINATION - Anh Phang", img_dst_display)
def save_config(H, width, height, save_path=args.save_config):
    with open(save_path, "w") as f:
        f.write("import numpy as np\n\n")

        f.write("H_MATRIX = np.array([\n")
        for row in H:
            f.write(f"    {list(row)},\n")
        f.write("])\n")

        f.write(f"WARP_SIZE = ({width}, {height})\n")
    print(f"[SAVED] Config saved to {save_path}")

if __name__ == "__main__":
    # 1. Đọc ảnh
    img_src = cv2.imread(PATH_IMAGE_SRC)
    img_dst = cv2.imread(PATH_IMAGE_DST)
    height, width = img_dst.shape[:2]
    # Kiểm tra xem ảnh có tồn tại không
    if img_src is None or img_dst is None:
        print("Lỗi: Không thể đọc ảnh. Vui lòng kiểm tra lại đường dẫn.")
        exit()

    # Tạo bản sao để vẽ (tránh làm hỏng ảnh gốc)
    img_src_display = img_src.copy()
    img_dst_display = img_dst.copy()

    # 2. Tạo cửa sổ và gắn hàm callback chuột
    cv2.namedWindow("1. SOURCE - Anh Nghieng")
    cv2.setMouseCallback("1. SOURCE - Anh Nghieng", click_event_src)

    cv2.namedWindow("2. DESTINATION - Anh Phang")
    cv2.setMouseCallback("2. DESTINATION - Anh Phang", click_event_dst)

    # 3. Hiển thị và hướng dẫn
    print("\n=== HƯỚNG DẪN SỬ DỤNG ===")
    print("1. Click chuột trái vào ảnh 'Nghieng' để chọn 1 điểm.")
    print("2. Click chuột trái vào ảnh 'Phang' tại vị trí TƯƠNG ỨNG.")
    print("3. Nhấn phím 'ESC' để THOÁT và in kết quả tọa độ.")
    print("4. Nhấn phím 'C' để XÓA TẤT CẢ các điểm đã chọn và làm lại.")
    print("---------------------------\n")

    # Hiển thị ảnh ban đầu
    cv2.imshow("1. SOURCE - Anh Nghieng", img_src_display)
    cv2.imshow("2. DESTINATION - Anh Phang", img_dst_display)

    # 4. Vòng lặp chờ sự kiện phím
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Nhấn 'ESC' để thoát
        if key == 27:
            break
        
        # Nhấn 'C' để xóa và làm lại
        elif key == ord('c'):
            src_points = []
            dst_points = []
            img_src_display = img_src.copy()
            img_dst_display = img_dst.copy()
            cv2.imshow("1. SOURCE - Anh Nghieng", img_src_display)
            cv2.imshow("2. DESTINATION - Anh Phang", img_dst_display)
            print("\n=== ĐÃ XÓA TẤT CẢ DIỂM. VUI LÒNG CHỌN LẠI ===\n")

    cv2.destroyAllWindows()
    if len(src_points) != len(dst_points) or len(src_points) < 4:
        print("\nLưu ý: Số lượng điểm trên 2 ảnh không bằng nhau hoặc chưa đủ 4 điểm.")
        print("Tọa độ Nguồn:", src_points)
        print("Tọa độ Đích:", dst_points)
        print("Vui lòng chạy lại và chọn đúng số lượng điểm.")
    else:
        # 2. Tính toán ma trận Homography với độ chính xác tối đa
        # cv2.RANSAC giúp loại bỏ các cặp điểm sai lệch
        # 2.0 là ngưỡng sai số (reprojection error) cho phép (đơn vị pixel)
        H, mask = cv2.findHomography(np.array(src_points), np.array(dst_points), cv2.RANSAC, 2.0)
        # 4. Thực hiện biến đổi ảnh
        result = cv2.warpPerspective(img_src, H, (width, height))
        cv2.imshow("Warped Result", result)
        destination_frame = cv2.imwrite(PATH_WARPED, result)
        # In ra số lượng điểm "xịn" đã được sử dụng
        # Chuyển mask về dạng mảng phẳng để dễ đọc
        status = mask.ravel().tolist()

        print("--- PHÂN TÍCH ĐIỂM ---")
        for i, s in enumerate(status):
            label = "TỐT (Inlier)" if s == 1 else "XẤU (Outlier) <--- Cần xem lại!"
            print(f"Cặp điểm số {i}: {label}")

        # Tạo ảnh để kiểm tra
        debug_img = img_src.copy()

        for i, p in enumerate(src_points):
            # Nếu là Outlier (mask == 0)
            if mask[i] == 0:
                color = (0, 0, 255) # Màu đỏ cho điểm xấu
                cv2.circle(debug_img, (int(p[0]), int(p[1])), 10, color, 2)
                cv2.putText(debug_img, "ERR", (int(p[0])+10, int(p[1])), 1, 1, color, 2)
            else:
                color = (0, 255, 0) # Màu xanh cho điểm tốt
                cv2.circle(debug_img, (int(p[0]), int(p[1])), 5, color, -1)

        cv2.imshow("Kiem tra diem loi", debug_img)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow("Kiem tra diem loi")
        # Lưu cấu hình ra file
        save_config(H, width, height, save_path=CONFIG_PATH)