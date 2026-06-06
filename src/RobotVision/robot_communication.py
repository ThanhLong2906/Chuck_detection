import socket
import logging

class RobotClient:
    def __init__(self, ip="192.168.1.100", port=5000):
        self.addr = (ip, port)
        self.client = None

    def connect(self):
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(2.0) # Không để đợi quá lâu làm treo Vision
            self.client.connect(self.addr)
            return True
        except Exception as e:
            logging.error(f"Không thể kết nối tới Robot: {e}")
            return False

    def send_angle(self, angle):
        if not self.client:
            if not self.connect(): return
        
        try:
            # Gửi dữ liệu dưới dạng chuỗi, ví dụ: "ANGLE:12.34\n"
            message = f"{angle:.2f}\n".encode('utf-8')
            self.client.sendall(message)
            return True
        except Exception as e:
            logging.error(f"Lỗi khi gửi dữ liệu tới Robot: {e}")
            self.client = None # Reset kết nối để lần sau thử lại
            return False