from pathlib import Path
import logging
from robot_communication import RobotClient
from database_manager import VisionDB
from storage_manager import StorageManager
from dotenv import load_dotenv    
import os

current_dir = Path(__file__).parent
env_path = f"{current_dir.parent.parent}/.env"
load_dotenv(dotenv_path=env_path, override=True)
IP_ROBOT = os.getenv("IP_ROBOT")
PORT_ROBOT = os.getenv("PORT_ROBOT")
DB_NAME = os.getenv("DB_NAME")
DATABASE_DIR = os.getenv("DATABASE_DIR")

class VisionSystem:
    def __init__(self, db_name =None, mode = "robot"):
        self.storage = StorageManager()
        self.db_path = f"{DATABASE_DIR}/{DB_NAME}" if db_name is None else db_name
        self.db = VisionDB(self.db_path)
        self.mode = mode
        if self.mode != "local":
            self.robot = RobotClient(ip=IP_ROBOT, port=PORT_ROBOT)
            self.robot.connect()

    def callback(self, angle, score, exe_time, has_workpiece, jaw_status, image_path):
        """Hàm này sẽ được gọi mỗi khi camera chụp được ảnh và xử lý xong"""
        status = "NOT FOUND ANGLE"
        if angle is not None:
            status = "OK"
            print(f"--- KẾT QUẢ MỚI ---")
            print(f"Góc lệch: {angle:.2f} | Score: {score:.1f} | Time: {exe_time:.2f}ms")
            logging.info(f"Góc lệch của mâm: {angle:.2f} độ")
            logging.info(f"Điểm số: {score:.1f}")
            logging.info(f"Thời gian xử lý ảnh: {exe_time:.5f} ms")

            if self.mode != "local":
                # Gửi góc tới robot
                print(f"Đang gửi góc {angle:.2f} tới Robot...")
                success = self.robot.send_angle(angle)
                if success:
                    print("Gửi thành công!")
                else:
                    print("Gửi thất bại!")

            #Lưu kết quả vào db
            self.db.insert_result(angle, score, exe_time, status, has_workpiece, jaw_status, image_path)
        else:
            print("Ảnh vừa chụp không detect được góc.")    
            logging.warning("Ảnh vừa chụp không detect được góc.")    