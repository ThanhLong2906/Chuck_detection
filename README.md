# Xác định mâm 3 chấu trong máy CNC

## Features
- Xác định vùng an toàn trên mâm 3 chấu để  robot gắp phôi kim loại ra.

## Tech Stack
- Python 3.12+
- OpenCV
- NumPy

## Project Structure
```
z131_robot_Thai-Nguyen/
    ├── src/
        ├── RobotVision/
            ├── circle_detection/
            ├── edge_detection/
            ├── jaw_detection/
            ├── workpiece_detection/
            ├── data/
            ├── utils/
            ├── MvImport/
            ├── center_detection.py
            ├── databse_manager.py
            ├── storage_manager.py
            ├── vision_system.py
            ├── robot_communication.py
            ├── create_config.py
            ├── get_chuck_angle_mvs.py
            ├── main.py
        ├── MeasureParams/
            ├── UI.py
    ├── data/
    ├── config/
    ├── database/
    ├── logs/
    ├── .env
    ├── .gitignore
    ├── requirement.txt
    └── README.md
```

## Installation

### 1. Tạo môi trường ảo
python -m venv .venv  
source .venv/bin/activate             

### 2. Cài đặt thư viện
pip install -r requirements.txt

## Usage

Quá trình sử dụng bao gồm 2 giai đoạn:
### Giai đoạn 1: 
Đo các thông số cho thuật toán. Các thông số bao gồm: ROI (vùng ảnh mâm
xoay mà thuật toán xử lý), Mask (vùng ảnh thuật toán tìm lỗ chấu), Circle (các thống số
lỗ chấu), Reference (Đường tham chiếu để tính góc xoay), Workpiece (Vùng xác định
phôi), open/close (Khoảng cách từ tâm mâm đến tâm lỗ chấu khi chấu đóng/mở). Để đo
các thông số này, người dùng làm theo những bước sau:
#### Bước 1:
Ấn vào icon để mở giao diện đo tham số
#### Bước 2: 
Ấn vào nút ”camera”(ở trên cùng góc trái) để kết nối với camera. Phần màn hình
lớn ở giữa giao diện hiện lên camera, người dùng căn chỉnh mâm vào đúng vị trí, có phôi,
và chấu đóng.
#### Bước 3: 
Sau khi căn chuẩn, ấn vào nút ”Chụp ảnh mẫu”(bên cạnh nút ”camera”) để chụp
ảnh mẫu thứ nhất.
#### Bước 4: 
Mở lại camera, điều chỉnh phôi để chấu mở (có thể bỏ phôi ra), rồi chụp ảnh mẫu thứ hai.
#### Bước 5: 
Ấn nút ”Mở ảnh nguồn”, chọn ảnh mẫu thứ nhất
#### Bước 6: 
Vị trí các nút chế độ đo thông số ở dòng dưới màn hình bên trái. Đo các thông số
trên ảnh mẫu như sau:
+ Ấn nút ”Vùng ROI”. Ấn 1 điểm ở góc trên bên trái và 1 điểm góc dưới bên phải
tạo thành hình chữ nhật vừa đủ để bao quanh vùng mâm trên ảnh.
+ Ấn nút ”Vòng mặt nạ”. Ấn 1 điểm chọn làm tâm của mâm. Chọn điểm thứ nhất
trên vòng tròn phôi, chọn điểm thứ hai là điểm xa tâm nhất vẫn thuộc mâm sao cho vùng
giữa 2 vòng tròn này chưa các lỗ trên chấu.
+ Ấn nút ”Lỗ chấu”. Ấn một điểm là tâm của lỗ chấu thứ nhất, ấn một điểm nữa
phía trong vành lỗ chấu đó để thiết lập bán kính nhỏ nhất mà lỗ chấu có thể có. Ấn một
điểm là tâm của lỗ chấu thứ hai (cùng trên chấu với lỗ thứ nhất), ấn một điểm nữa bên
ngoài lỗ chấu đó để thiết lập bán kính lớn nhất mà lỗ chấu có thể có.
+ Ấn nút ”Đường tham chiếu”. chọn 2 điểm để tạo một vector từ điểm đầu tiên ->
điểm thứ hai. Góc của mâm xoay được tính là góc giữa các lỗ/cạnh chấu với vector này.
+ Ấn nút ”Chấu đóng”. Ấn một điểm vào tâm tâm, điểm thứ hai vào tâm của lỗ
chấu gần tâm mâm, điểm thứ ba vào tâm của lỗ chấu xa tâm mâm. Bước này để đo khoảng
cách từ tâm mâm đến tâm các lỗ chấu khi chấu đóng.
+ Ấn nút ”Vùng phôi”. Ấn một điểm vào tâm mâm, điểm thứ hai nằm trên vòng
tròn trong cùng vùng lõi mâm. Bước này để thiết lập vùng mà thuật toán sẽ dùng để xác
định có phôi hay không.
+ Sau khi thiết lập xong ảnh thứ nhất, ấn lưu file config.
#### Bước 7: 
Mở ảnh mẫu thứ hai, giữ nguyên file config. Ấn vào nút ”Chấu mở”. Ấn một điểm vào tâm tâm, điểm thứ hai vào tâm của lỗ chấu gần tâm mâm, điểm thứ ba vào tâm của lỗ chấu xa tâm mâm. Bước này để đo khoảng cách từ tâm mâm đến tâm các lỗ chấu khi chấu mở. Sau đó lưu lại thay đổi
### Giai đoạn 2: 
Xác định chấu an toàn để robot để đưa phôi vào
```
python src/RobotVision/main.py --image [đường dẫn ảnh chấu] --config [đường dẫn file config]
```