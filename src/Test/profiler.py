import cv2
import time
import json
import os
import sys
import csv
import argparse
import threading
import numpy as np
import psutil
import sys
from pathlib import Path
src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))
from RobotVision.circle_detection.get_chuck_angle_circle import detect_angle_by_circles
from RobotVision.workpiece_detection.workpiece_check import WorkpieceDetection
from RobotVision.jaw_detection.jaw_check import JawDetection 
# Kiểm tra matplotlib (optional — chỉ cần khi vẽ biểu đồ)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# CẤU HÌNH
SAMPLE_INTERVAL = 0.05   # giây — tần suất lấy mẫu CPU/RAM
OUTPUT_CSV      = "profiling_report.csv"
OUTPUT_CHART    = "profiling_chart.png"

# LỚP SAMPLER — chạy ngầm để lấy mẫu CPU/RAM liên tục
class ResourceSampler:
    """Chạy thread ngầm, lấy mẫu CPU% và RAM mỗi SAMPLE_INTERVAL giây."""

    def __init__(self, interval=SAMPLE_INTERVAL):
        self.interval   = interval
        self.process    = psutil.Process(os.getpid())
        self.cpu_log    = []   # list[(timestamp, cpu_percent)]
        self.ram_log    = []   # list[(timestamp, ram_mb)]
        self._stop_evt  = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def stop(self):
        self._stop_evt.set()
        self._thread.join()

    def _run(self):
        while not self._stop_evt.is_set():
            t = time.perf_counter() - self._t0
            try:
                cpu = self.process.cpu_percent(interval=None)
                ram = self.process.memory_info().rss / 1024 / 1024
                self.cpu_log.append((t, cpu))
                self.ram_log.append((t, ram))
            except psutil.NoSuchProcess:
                break
            time.sleep(self.interval)

    #Thống kê tổng hợp 
    @property
    def cpu_avg(self):
        vals = [v for _, v in self.cpu_log]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def cpu_peak(self):
        return max((v for _, v in self.cpu_log), default=0.0)

    @property
    def ram_peak(self):
        return max((v for _, v in self.ram_log), default=0.0)

    @property
    def ram_start(self):
        return self.ram_log[0][1] if self.ram_log else 0.0

    @property
    def ram_delta(self):
        return self.ram_peak - self.ram_start

# HÀM TIỆN ÍCH
def get_system_info():
    """Lấy thông tin hệ thống một lần khi khởi động."""
    cpu_freq = psutil.cpu_freq()
    ram_total = psutil.virtual_memory().total / 1024 / 1024 / 1024

    info = {
        "os"          : f"Ubuntu 24.04 (Linux {os.uname().release})",
        "cpu_cores"   : psutil.cpu_count(logical=False),
        "cpu_threads" : psutil.cpu_count(logical=True),
        "cpu_freq_mhz": f"{cpu_freq.max:.0f}" if cpu_freq else "N/A",
        "ram_total_gb": f"{ram_total:.1f}",
        "python"      : sys.version.split()[0],
        "opencv"      : cv2.__version__,
    }
    return info

def print_header(title: str):
    width = 62
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)

def print_result(label: str, value: str, unit: str = ""):
    print(f"  {label:<30} {value:>12} {unit}")

def print_separator():
    print("  " + "─" * 58)

def image_pipeline(frame: np.ndarray, config: dict) -> dict:
    """
    Pipeline xử lý ảnh mẫu với OpenCV.

    Trả về dict chứa các kết quả trung gian nếu cần.
    """
    save_path = None #"/home/long/PROJECTS/AI/RobotVision/data/test/performance"

    wp_detector = WorkpieceDetection(config=config)
    jaw_detector = JawDetection(config=config)

    angle, _, best_triple, jaw_circle_full = detect_angle_by_circles(frame, config, save=save_path)
    chuck_position = True if np.abs(angle)< 7 else False
    has_workpiece, _ = wp_detector.detect(img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), save=save_path)
    
    is_open = jaw_detector.detect_open(jaw_circle_full, save=save_path)
    return {"chuck_postition": chuck_position, "has_workpiece": has_workpiece, "is_open": is_open}

# PROFILE MỘT ẢNH
def profile_single(image_path: str | None = None, config: dict = None) -> dict:
    """
    Profile một ảnh đơn. Nếu image_path=None thì tạo ảnh giả để demo.
    Trả về dict chứa tất cả metrics.
    """
    #Chuẩn bị ảnh 
    if image_path and os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is None:
            
            raise ValueError(f"Không đọc được ảnh: {image_path}")
        label = os.path.basename(image_path)
    else:
        # Tạo ảnh ngẫu nhiên 1920×1080 để demo
        img = np.random.randint(0, 256, (2048, 3072, 3), dtype=np.uint8)
        label = "demo_3072x2048.png"
        print("  [Demo] Không có ảnh thật → dùng ảnh ngẫu nhiên 2048×3072")

    h, w = img.shape[:2]
    file_size_kb = os.path.getsize(image_path) / 1024 if image_path else 0
    # chuẩn bị file config

    # Khởi động sampler
    sampler = ResourceSampler()
    sampler.start()

    # Chạy pipeline
    t_start = time.perf_counter()
    pipeline_result = image_pipeline(img, config)
    t_end   = time.perf_counter()

    # Dừng sampler
    sampler.stop()
    wall_time = t_end - t_start

    # Tổng hợp metrics
    metrics = {
        "file"          : label,
        "resolution"    : f"{w}x{h}",
        "file_size_kb"  : f"{file_size_kb:.1f}" if file_size_kb else "N/A",
        "wall_time_s"   : wall_time,
        "cpu_avg_pct"   : sampler.cpu_avg,
        "cpu_peak_pct"  : sampler.cpu_peak,
        "ram_start_mb"  : sampler.ram_start,
        "ram_peak_mb"   : sampler.ram_peak,
        "ram_delta_mb"  : sampler.ram_delta,
        "num_contours"  : pipeline_result.get("num_contours", "N/A"),
        "cpu_log"       : sampler.cpu_log,
        "ram_log"       : sampler.ram_log,
    }
    return metrics


# IN KẾT QUẢ ĐẸP RA TERMINAL
def print_metrics(metrics: dict, sys_info: dict):
    print_header(f"KẾT QUẢ PROFILING — {metrics['file']}")

    print(f"\n  File       : {metrics['file']}")
    print(f"  Độ phân giải: {metrics['resolution']} px")
    if metrics["file_size_kb"] != "N/A":
        print(f"  Kích thước : {metrics['file_size_kb']} KB")

    print_separator()

    print_result("Wall time (thời gian thực)",
                 f"{metrics['wall_time_s']*1000:.2f}", "ms")
    print_result("Số contour tìm được",
                 str(metrics["num_contours"]))

    print_separator()
    print("  CPU")
    print_result("    Trung bình",  f"{metrics['cpu_avg_pct']:.1f}",  "%")
    print_result("    Đỉnh (peak)", f"{metrics['cpu_peak_pct']:.1f}", "%")

    print_separator()
    print("  RAM")
    print_result("    Trước khi chạy",     f"{metrics['ram_start_mb']:.1f}", "MB")
    print_result("    Đỉnh trong quá trình",f"{metrics['ram_peak_mb']:.1f}", "MB")
    print_result("    Tăng thêm (delta)",  f"{metrics['ram_delta_mb']:.1f}", "MB")

    print_separator()
    print("  Hệ thống")
    print_result("    CPU cores / threads",
                 f"{sys_info['cpu_cores']} / {sys_info['cpu_threads']}")
    print_result("    CPU tần số tối đa",  f"{sys_info['cpu_freq_mhz']}", "MHz")
    print_result("    RAM tổng",           f"{sys_info['ram_total_gb']}", "GB")
    print_result("    OpenCV version",     sys_info["opencv"])
    print("═" * 62 + "\n")


# BATCH MODE — chạy nhiều ảnh
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def profile_batch(folder: str, config: str, sys_info: dict) -> list[dict]:
    image_files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ]

    if not image_files:
        print(f"  Không tìm thấy ảnh nào trong: {folder}")
        return []

    print_header(f"BATCH MODE — {len(image_files)} ảnh trong '{folder}'")

    with open(config, "r") as f:
        cfg = json.load(f)
    all_metrics = []
    for i, path in enumerate(image_files, 1):
        print(f"\n  [{i}/{len(image_files)}] Đang xử lý: {os.path.basename(path)}")
        try:
            m = profile_single(path, cfg)
            print_metrics(m, sys_info)
            all_metrics.append(m)
        except Exception as e:
            print(f"  Lỗi: {e}")

    #Tổng kết batch
    if all_metrics:
        total_time   = sum(m["wall_time_s"] for m in all_metrics)
        avg_time     = total_time / len(all_metrics)
        throughput   = len(all_metrics) / total_time
        avg_cpu      = sum(m["cpu_avg_pct"] for m in all_metrics) / len(all_metrics)
        peak_ram     = max(m["ram_peak_mb"] for m in all_metrics)

        print_header("TỔNG KẾT BATCH")
        print_result("  Tổng số ảnh",         str(len(all_metrics)))
        print_result("  Tổng thời gian",       f"{total_time:.3f}", "s")
        print_result("  Trung bình / ảnh",     f"{avg_time*1000:.1f}", "ms")
        print_result("  Throughput",           f"{throughput:.2f}", "ảnh/giây")
        print_result("  CPU trung bình",       f"{avg_cpu:.1f}", "%")
        print_result("  RAM peak (cao nhất)",  f"{peak_ram:.1f}", "MB")
        print("═" * 62 + "\n")

    return all_metrics


# XUẤT BÁO CÁO CSV
def save_csv(all_metrics: list[dict], path: str = OUTPUT_CSV):
    if not all_metrics:
        return
    fields = ["file", "resolution", "file_size_kb",
              "wall_time_s", "cpu_avg_pct", "cpu_peak_pct",
              "ram_start_mb", "ram_peak_mb", "ram_delta_mb", "num_contours"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"  CSV đã lưu: {path}")

# VẼ BIỂU ĐỒ

def plot_timeline(metrics: dict, out_path: str = OUTPUT_CHART):
    """Vẽ biểu đồ CPU% và RAM MB theo thời gian trong một lần chạy."""
    if not HAS_MATPLOTLIB:
        print("  matplotlib chưa cài → bỏ qua vẽ biểu đồ.")
        print("  pip install matplotlib")
        return

    cpu_log = metrics["cpu_log"]
    ram_log = metrics["ram_log"]
    if not cpu_log:
        return

    t_cpu, v_cpu = zip(*cpu_log)
    t_ram, v_ram = zip(*ram_log)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Profiling timeline — {metrics['file']}", fontsize=13, fontweight="bold")

    # CPU
    ax1.fill_between(t_cpu, v_cpu, alpha=0.4, color="#e74c3c")
    ax1.plot(t_cpu, v_cpu, color="#c0392b", linewidth=1.5)
    ax1.set_ylabel("CPU (%)", fontsize=10)
    ax1.set_ylim(0, max(max(v_cpu) * 1.3, 10))
    ax1.axhline(y=sum(v_cpu)/len(v_cpu), linestyle="--",
                color="#e74c3c", alpha=0.7, label=f"Avg {sum(v_cpu)/len(v_cpu):.1f}%")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # RAM
    ax2.fill_between(t_ram, v_ram, alpha=0.4, color="#3498db")
    ax2.plot(t_ram, v_ram, color="#2980b9", linewidth=1.5)
    ax2.set_ylabel("RAM (MB)", fontsize=10)
    ax2.set_xlabel("Thời gian (giây)", fontsize=10)
    ax2.axhline(y=max(v_ram), linestyle="--",
                color="#2980b9", alpha=0.7, label=f"Peak {max(v_ram):.1f} MB")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Biểu đồ đã lưu: {out_path}")


def plot_batch_summary(all_metrics: list[dict], out_path: str = "batch_summary.png"):
    """Vẽ biểu đồ tổng hợp cho batch."""
    if not HAS_MATPLOTLIB or not all_metrics:
        return

    labels    = [m["file"][:20] for m in all_metrics]
    times     = [m["wall_time_s"] * 1000 for m in all_metrics]
    ram_peaks = [m["ram_peak_mb"] for m in all_metrics]
    cpu_avgs  = [m["cpu_avg_pct"] for m in all_metrics]
    x         = range(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle("Batch Profiling Summary", fontsize=14, fontweight="bold")

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    data   = [times, ram_peaks, cpu_avgs]
    ylabels= ["Wall time (ms)", "RAM peak (MB)", "CPU avg (%)"]

    for ax, vals, ylabel, color in zip(axes, data, ylabels, colors):
        bars = ax.bar(x, vals, color=color, alpha=0.8, edgecolor="white")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Biểu đồ batch đã lưu: {out_path}")

    plt.close()
    print(f"  Biểu đồ batch đã lưu: {out_path}")



# WATCH MODE — chạy liên tục, mỗi N giây load 1 ảnh xoay vòng

import signal
import datetime
import itertools

def _format_countdown(seconds_left: float) -> str:
    m, s = divmod(int(seconds_left), 60)
    return f"{m:02d}:{s:02d}"

def _print_watch_status(round_no: int, file: str, next_in: float, total: int):
    """In dòng trạng thái gọn, ghi đè lên dòng cũ."""
    bar_len = 20
    elapsed_ratio = 1 - (next_in / max(next_in, 1))
    filled = int(bar_len * elapsed_ratio)
    bar = "█" * filled + "░" * (bar_len - filled)
    line = (f"\r  ⏳ [{bar}] {_format_countdown(next_in)} "
            f"| Lần #{round_no} | Tổng: {total} ảnh | {file[:30]:<30}")
    print(line, end="", flush=True)

def profile_watch(folder: str, interval_s: float, config:str, sys_info: dict) -> list[dict]:
    """
    Chạy liên tục: mỗi `interval_s` giây xử lý một ảnh (xoay vòng).
    Dừng khi người dùng nhấn Ctrl+C.
    Trả về toàn bộ danh sách metrics đã thu.
    """
    with open(config, "r") as f:
        cfg = json.load(f)
    # ── Lấy danh sách ảnh ──────────────────────────────────
    image_files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ])
    if not image_files:
        print(f"\n  Không tìm thấy ảnh nào trong: {folder}")
        return []

    all_metrics: list[dict] = []
    session_start = time.perf_counter()

    print_header("WATCH MODE — chạy liên tục đến Ctrl+C")
    print(f"  Thư mục   : {folder}")
    print(f"  Số ảnh tìm thấy : {len(image_files)}")
    print(f"  Khoảng cách : {interval_s:.0f}s ({interval_s/60:.1f} phút)")
    print(f"  Bắt đầu   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  Nhấn Ctrl+C bất cứ lúc nào để dừng và xem báo cáo tổng.\n")

    # Xoay vòng qua danh sách ảnh vô hạn
    file_cycle = itertools.cycle(image_files)
    round_no   = 0
    stopped    = False

    def _handle_sigint(sig, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        while not stopped:
            round_no += 1
            current_file = next(file_cycle)
            fname = os.path.basename(current_file)

            # Xử lý ảnh 
            print(f"\n  ┌─ Lần #{round_no} ── {datetime.datetime.now().strftime('%H:%M:%S')} ─────────────────────────")
            print(f"  │  {fname}")
            try:
                m = profile_single(current_file, cfg)
                m["round"]      = round_no
                m["timestamp"]  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                all_metrics.append(m)

                # In kết quả gọn
                print(f"  │  Wall time : {m['wall_time_s']*1000:.2f} ms")
                print(f"  │  CPU avg   : {m['cpu_avg_pct']:.1f}%  peak: {m['cpu_peak_pct']:.1f}%")
                print(f"  │  RAM peak  : {m['ram_peak_mb']:.1f} MB  (Δ {m['ram_delta_mb']:+.1f} MB)")
                print(f"  └──────────────────────────────────────────────────────")
            except Exception as e:
                print(f"  │  Lỗi: {e}")
                print(f"  └──────────────────────────────────────────────────────")

            if stopped:
                break

            # ── Đếm ngược chờ ──────────────────────────────
            t_wait_start = time.perf_counter()
            next_file    = os.path.basename(next(itertools.islice(
                               itertools.cycle(image_files),
                               round_no % len(image_files),
                               round_no % len(image_files) + 1)))
            print(f"  💤  Nghỉ {interval_s:.0f}s → ảnh tiếp: {next_file}", end="")

            while not stopped:
                elapsed   = time.perf_counter() - t_wait_start
                remaining = interval_s - elapsed
                if remaining <= 0:
                    break
                _print_watch_status(round_no, fname, remaining, len(all_metrics))
                time.sleep(0.5)

            print()  # xuống dòng sau countdown

    except Exception as e:
        print(f"\n  Lỗi không mong đợi: {e}")
    finally:
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    return all_metrics


def print_watch_summary(all_metrics: list[dict], session_start: float):
    """In báo cáo tổng kết sau khi Ctrl+C."""
    if not all_metrics:
        print("\n Chưa có dữ liệu để tổng kết.")
        return

    total_rounds   = len(all_metrics)
    total_elapsed  = time.perf_counter() - session_start
    h, rem         = divmod(int(total_elapsed), 3600)
    m, s           = divmod(rem, 60)

    avg_wall   = sum(x["wall_time_s"]  for x in all_metrics) / total_rounds * 1000
    avg_cpu    = sum(x["cpu_avg_pct"]  for x in all_metrics) / total_rounds
    peak_cpu   = max(x["cpu_peak_pct"] for x in all_metrics)
    avg_ram    = sum(x["ram_peak_mb"]  for x in all_metrics) / total_rounds
    peak_ram   = max(x["ram_peak_mb"]  for x in all_metrics)
    max_delta  = max(x["ram_delta_mb"] for x in all_metrics)

    # Tìm ảnh chậm nhất / nhanh nhất
    slowest = max(all_metrics, key=lambda x: x["wall_time_s"])
    fastest = min(all_metrics, key=lambda x: x["wall_time_s"])

    print_header("BÁO CÁO TỔNG KẾT — WATCH SESSION")
    print(f"\n  Thời gian chạy    : {h:02d}h {m:02d}m {s:02d}s")
    print(f"  Tổng số lần chạy  : {total_rounds}")
    print_separator()
    print("  Thời gian xử lý / ảnh")
    print_result("    Trung bình",   f"{avg_wall:.2f}",                         "ms")
    print_result("    Nhanh nhất",
                 f"{fastest['wall_time_s']*1000:.2f}",
                 f"ms  ({fastest['file']})")
    print_result("    Chậm nhất",
                 f"{slowest['wall_time_s']*1000:.2f}",
                 f"ms  ({slowest['file']})")
    print_separator()
    print("  CPU (toàn session)")
    print_result("    Avg trung bình",  f"{avg_cpu:.1f}",  "%")
    print_result("    Peak cao nhất",   f"{peak_cpu:.1f}", "%")
    print_separator()
    print("  RAM (toàn session)")
    print_result("    Peak trung bình", f"{avg_ram:.1f}",   "MB")
    print_result("    Peak cao nhất",   f"{peak_ram:.1f}",  "MB")
    print_result("    Delta cao nhất",  f"{max_delta:.1f}", "MB")
    print_separator()

    # Cảnh báo memory leak
    if len(all_metrics) >= 3:
        first3_ram = [x["ram_peak_mb"] for x in all_metrics[:3]]
        last3_ram  = [x["ram_peak_mb"] for x in all_metrics[-3:]]
        if sum(last3_ram)/3 - sum(first3_ram)/3 > 50:
            print("  CẢNH BÁO: RAM tăng dần qua các lần chạy — có thể memory leak!")
        else:
            print("  RAM ổn định, không phát hiện memory leak.")

    print("═" * 62 + "\n")


# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Profiling CPU/RAM cho pipeline OpenCV — Ubuntu 24.04"
    )
    parser.add_argument("--image",  type=str, default=None,
                        help="Đường dẫn ảnh đơn cần profile")
    parser.add_argument("--batch",  type=str, default=None,
                        help="Thư mục chứa nhiều ảnh để chạy batch")
    parser.add_argument("--config", type=str, default=None,
                        help="Đường dẫn file config")
    parser.add_argument("--watch",    type=str,   default=None,
                        help="Thư mục ảnh — chạy liên tục đến Ctrl+C")
    parser.add_argument("--interval", type=float, default=120,
                        help="Số giây chờ giữa 2 lần chạy trong watch mode (mặc định: 120)")
    parser.add_argument("--report", action="store_true",
                        help="Xuất báo cáo CSV + biểu đồ PNG")
    args = parser.parse_args()

    # Thông tin hệ thống
    sys_info = get_system_info()
    print_header("THÔNG TIN HỆ THỐNG")
    for k, v in sys_info.items():
        print_result(f"  {k}", str(v))

    session_start = time.perf_counter()

    #Chạy theo mode
    if args.watch:
        all_metrics = profile_watch(args.watch, args.interval, args.config, sys_info)
        print_watch_summary(all_metrics, session_start)
        if args.report and all_metrics:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_csv(all_metrics, f"watch_report_{ts}.csv")
            plot_batch_summary(all_metrics, f"watch_chart_{ts}.png")
    elif args.batch:
        all_metrics = profile_batch(args.batch, args.config, sys_info)
        if args.report:
            save_csv(all_metrics)
            plot_batch_summary(all_metrics)

    else:
        m = profile_single(args.image)
        print_metrics(m, sys_info)
        if args.report:
            save_csv([m])
            plot_timeline(m)


if __name__ == "__main__":
    main()