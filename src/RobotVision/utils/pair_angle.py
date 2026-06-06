import math

def average_angles_mod(angles, mod_value=120.0):
    """
    Tính trung bình các góc có tính đến sự gián đoạn tại điểm 0/mod_value
    angles: list các góc (đã tính mod 120)
    """
    if not angles:
        return None
    if len(angles) == 1:
        return angles[0]
        
    # Chuyển đổi sang vòng tròn 360 độ để dùng lượng giác
    factor = 360.0 / mod_value
    
    sin_sum = 0
    cos_sum = 0
    
    for a in angles:
        rad = math.radians(a * factor)
        sin_sum += math.sin(rad)
        cos_sum += math.cos(rad)
    
    # Tính góc trung bình trong không gian vector
    avg_rad = math.atan2(sin_sum, cos_sum)
    avg_deg = math.degrees(avg_rad)
    
    # Đưa về lại không gian mod_value Ban đầu
    final_angle = (avg_deg / factor) % mod_value
    return final_angle