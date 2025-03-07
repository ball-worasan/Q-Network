import time
import logging


class CurriculumManager:
    def __init__(self):
        self.current_stage = 1
        self.success_count = 0
        self.success_threshold = 5  # สำเร็จ N ครั้งเพื่อเลื่อนไปด่านถัดไป

    def get_current_stage(self):
        return self.current_stage

    def update_stage(self, success):
        """
        ถ้า success เป็น True แสดงว่า agent ทำภารกิจของ Stage สำเร็จหนึ่งครั้ง
        นับจำนวนจนถึง threshold แล้วเลื่อนไป Stage ถัดไป
        """
        if success:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.current_stage += 1
                self.success_count = 0
                logging.info(f"=== Stage updated to {self.current_stage} ===")

    def reset(self):
        """รีเซ็ตค่า ถ้าต้องเริ่มใหม่"""
        self.current_stage = 1
        self.success_count = 0
