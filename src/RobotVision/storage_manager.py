"""
Trình quản lý nơi lưu trữ:
Cấu trúc thư mục:
data/
  ├── images/            # Chứa ảnh thô/ảnh debug
  │     ├── 20260514/    # Chia theo ngày để truy xuất cực nhanh
  │     └── 20260515/
database/                # Chứa file db
logs/                    # Chứa các file log
"""
import os
import shutil
import psutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
CURRENT_DIR = Path(__file__).parent
env_path = f"{CURRENT_DIR.parent.parent}/.env"
load_dotenv(dotenv_path=env_path, override=True)
RESULT_IMAGE_DIR = os.getenv("RESULT_IMAGE_DIR")
DATABASE_DIR = os.getenv("DATABASE_DIR")
LOG_DIR = os.getenv("LOG_DIR")

class StorageManager:
    def __init__(self, max_usage_percent=97):
        # Kiểm tra và xóa bớt file kết quả
        self.max_usage_percent = max_usage_percent
        self.data_dir = Path(RESULT_IMAGE_DIR)
        self.check_and_cleanup()

        # tạo các thư mục cần thiết
        now = datetime.now()
        Path(f"{self.data_dir}/{now.strftime("%d-%m-%Y")}").mkdir(parents=True, exist_ok=True)
        Path(DATABASE_DIR).mkdir(parents=True, exist_ok=True)
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    def check_and_cleanup(self):
        """Kiểm tra ổ cứng và xóa thư mục cũ nhất nếu đầy"""
        usage = psutil.disk_usage(self.data_dir.anchor)
        
        if usage.percent > self.max_usage_percent:
            print(f"Cảnh báo: Ổ cứng đầy ({usage.percent}%). Đang dọn dẹp...")
            
            # Lấy danh sách các thư mục con (các ngày) và sắp xếp theo thời gian tạo
            subdirs = sorted(
                [d for d in self.data_dir.iterdir() if d.is_dir()],
                key=os.path.getctime
            )
            
            if subdirs:
                oldest_dir = subdirs[0]
                print(f"Đang xóa dữ liệu cũ nhất: {oldest_dir}")
                shutil.rmtree(oldest_dir)
                
                # Gọi đệ quy cho đến khi dung lượng dưới ngưỡng
                self.check_and_cleanup()

    # def get_save_path(self):
    #     """Tạo thư mục theo ngày và trả về đường dẫn lưu ảnh"""
    #     today = datetime.now().strftime("%Y%m%d")
    #     day_dir = self.data_dir / today
    #     day_dir.mkdir(exist_ok=True)
    #     return day_dir