import sqlite3
from datetime import datetime

class VisionDB:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS inspection_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            angle FLOAT,
            score FLOAT,
            execution_time FLOAT,
            status TEXT,
            has_workpiece INTEGER CHECK (has_workpiece IN (0,1)),
            jaw_status TEXT,
            image_path TEXT
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def insert_result(self, angle, score, exe_time, status, has_workpiece, is_open, img_path):
        query = "INSERT INTO inspection_results (timestamp, angle, score, execution_time, status, has_workpiece, jaw_status, image_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        self.conn.execute(query, (datetime.now(), angle, score, exe_time, status, has_workpiece, is_open, img_path))
        self.conn.commit()