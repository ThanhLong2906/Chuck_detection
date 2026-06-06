# BƯỚC 1: Tạo và in checkerboard chuẩn
# Chạy code này 1 lần để tạo file PNG, rồi in ra giấy A4
import cv2
import numpy as np

def create_checkerboard(rows=7, cols=10, square_mm=25, dpi=96):
    """
    rows, cols: số ô vuông nội (inner corners)
    square_mm: kích thước mỗi ô (mm) — phải đo lại sau khi in!
    """
    px_per_mm = dpi / 25.4
    sq = int(square_mm * px_per_mm)

    img_h = (rows + 1) * sq
    img_w = (cols + 1) * sq
    board = np.ones((img_h, img_w), dtype=np.uint8) * 255

    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y1, y2 = r * sq, (r + 1) * sq
                x1, x2 = c * sq, (c + 1) * sq
                board[y1:y2, x1:x2] = 0

    cv2.imwrite('checkerboard_7x10.png', board)
    print(f"Đã tạo checkerboard {rows}x{cols}, mỗi ô {square_mm}mm")
    print(f"Kích thước ảnh: {img_w}x{img_h} px")
    print(">>> IN RA GIẤY A4, ĐO LẠI Ô VUÔNG THỰC TẾ SAU KHI IN!")

create_checkerboard()