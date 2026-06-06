import cv2
import numpy as np
from MvImport.MvCameraControl_class import *
from pathlib import Path
from datetime import datetime 
from dotenv import load_dotenv
import os
# ngăn không cho thư viện hiển thị warning
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["QT_LOGGING_RULES"] = "*.warning=false"

def capture_for_config(save_dir="data"):
    """
    Hàm mở camera, hiển thị luồng ảnh trực tiếp và lưu ảnh khi nhấn phím 's'.
    Dùng để lấy ảnh mẫu phục vụ việc tạo file config.
    """
    # 1. Khởi tạo thư mục lưu
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 2. Tìm kiếm và mở Camera
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print("Không tìm thấy camera Hikrobot nào!")
        return

    cam = MvCamera()
    stDeviceImg = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    cam.MV_CC_CreateHandle(stDeviceImg)
    cam.MV_CC_OpenDevice()
    #Tối ưu packetsize
    if stDeviceImg.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if nPacketSize > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret == 0:
                print(f"Đã tối ưu Packet Size lên: {nPacketSize}")
            else:
                print(f"Cảnh báo: Không thể set Packet Size, lỗi: {hex(ret & 0xffffffff)}")
        else:
            # Nếu không lấy được giá trị tối ưu, ép về 1500 cho an toàn
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", 1500)
    # lấy exposure time và gain tự động
    cam.MV_CC_SetEnumValue("ExposureAuto", 2)
    cam.MV_CC_SetEnumValue("GainAuto", 2)
    # Ép về Mono8
    cam.MV_CC_SetEnumValue("PixelFormat", 0x01080001)
    try:
        # 3. Cấu hình chế độ chụp liên tục (Off Trigger) để xem live
        cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        
        # Bắt đầu truyền ảnh
        cam.MV_CC_StartGrabbing()
        stOutFrame = MV_FRAME_OUT()
        
        print("--- CHẾ ĐỘ CHỤP ẢNH CONFIG ---")
        print("Phím 's': Lưu ảnh mẫu")
        print("Phím 'q': Thoát")

        while True:
            # Lấy ảnh từ buffer (Timeout ngắn để loop mượt)
            ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            
            if ret == 0:
                # Chuyển đổi sang OpenCV format
                pData = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                memmove(byref(pData), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
                
                img = np.frombuffer(pData, dtype=np.uint8).reshape(
                    stOutFrame.stFrameInfo.nHeight, 
                    stOutFrame.stFrameInfo.nWidth
                )
                
                # Nếu là ảnh Mono, chuyển sang BGR để hiển thị
                img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                # Hiển thị lên màn hình
                cv2.namedWindow("Capture Config", cv2.WINDOW_NORMAL)
                cv2.imshow("Capture Config", img_display)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):
                    # Lưu ảnh với timestamp
                    now = datetime.now()
                    filename = f"config_sample_{now.strftime("%d%m%Y-%H%M%S")}.jpg"
                    save_path = Path(save_dir) / filename
                    cv2.imwrite(str(save_path), img)
                    print(f"Đã lưu ảnh mẫu tại: {save_path}")
                    
                elif key == ord('q'):
                    break
                
                cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:
                print(f"Lỗi lấy ảnh! Mã lỗi: {hex(ret & 0xffffffff)}") 
                # nguyên nhân: do xung đột giữa phần mềm mvs và code - tranh nhau đọc ảnh -> tắt hẳn 1 trong 2 đi

    finally:
        # Giải phóng tài nguyên
        cv2.destroyAllWindows()
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    env_path = f"{current_dir.parent.parent.parent}/.env"
    load_dotenv(dotenv_path=env_path, override=True)
    CONFIG_IMAGE_DIR = os.getenv("CONFIG_IMAGE_DIR")
    capture_for_config(CONFIG_IMAGE_DIR) 