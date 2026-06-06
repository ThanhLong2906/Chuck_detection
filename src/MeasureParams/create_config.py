import argparse
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
from pathlib import Path
from MeasureParams.measure_params import VisionConfigToolCircle
from edge_detection.measure_params import VisionConfigToolEdge
from measure_params import VisionConfigToolWorkpiece
from dotenv import load_dotenv

if __name__ == "__main__":
    # Đọc file .env
    current_dir = Path(__file__).parent
    env_path = f"{current_dir.parent.parent}/.env"
    load_dotenv(dotenv_path=env_path, override=True)
    CONFIG_DIR = os.getenv("CONFIG_DIR")
    CONFIG_IMAGE_DIR = os.getenv("CONFIG_IMAGE_DIR")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn ảnh chụp mâm")
    parser.add_argument("--config", type=str, help = "Đường dẫn vị trí lưu file config")
    parser.add_argument("--detect", type=str, default="circle", choices=["circle", "edge", "workpiece"],help="Phương pháp xác định góc xoay của mâm: circle hoặc edge.")
    args = parser.parse_args()
    image_path = Path(args.image) if Path(args.image).is_absolute() else f"{CONFIG_IMAGE_DIR}/{args.image}"
    filename = Path(image_path).stem
    config_dir = f"{CONFIG_DIR}/circle_conf" if args.detect == "circle" else f"{CONFIG_DIR}/edge_conf"
    config_path_ = f"{config_dir}/{filename}_config.json" 
    config_path = args.config if args.config and Path(args.config).is_absolute() else config_path_ 
    if args.detect == "circle":
        tool = VisionConfigToolCircle(image_path, config_path)
    elif args.detect == "edge":
        tool = VisionConfigToolEdge(image_path, config_path)
    else:
        tool = VisionConfigToolWorkpiece(image_path,config_path)
    tool.run()