#!/usr/bin/env python3
"""
Train YOLOv8n-seg cho 4 vật gắp (dataset từ autolabel_objects.py), rồi export ONNX cho Pi.

    ~/venvs/new_rl_ros2/bin/python scripts/train_objects.py                 # train + export
    ~/venvs/new_rl_ros2/bin/python scripts/train_objects.py --epochs 5      # thử nhanh
    ~/venvs/new_rl_ros2/bin/python scripts/train_objects.py --export-only runs/objects4/<tên>/weights/best.pt

imgsz 640 (ảnh quay 1280x720 bị co lại; Pi chạy 640x360 nên cũng ở tỉ lệ này).
Augment: lật ngang, xoay ±10°, đổi màu/sáng nhẹ (giữ màu vật: hue nhỏ), mosaic, scale rộng
để học vật nhỏ/xa. Không lật dọc (UAV không nhìn ngược).
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "objects4" / "data.yaml"


def data_yaml():
    """data.yaml ghi đường dẫn tuyệt đối của máy đã gán nhãn; chép sang máy khác (zip) thì sai.
    Ghi bản data_local.yaml trỏ đúng thư mục hiện tại."""
    lines = [l for l in DATA.read_text(encoding="utf-8").splitlines() if not l.startswith("path:")]
    out = DATA.parent / "data_local.yaml"
    out.write_text(f"path: {DATA.parent}\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return str(out)


def export(best, imgsz):
    from ultralytics import YOLO
    m = YOLO(best)
    p = m.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=False)
    print(f"ONNX: {p}")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--device", default="0", help="GPU số mấy, hoặc cpu")
    ap.add_argument("--export-imgsz", type=int, nargs=2, default=[384, 640], metavar=("H", "W"),
                    help="kích thước ONNX cho Pi (H W), 640x360 làm tròn lên bội 32")
    ap.add_argument("--export-only", default=None, help="đường dẫn best.pt, bỏ qua train")
    a = ap.parse_args()

    if a.export_only:
        export(a.export_only, a.export_imgsz)
        return
    from ultralytics import YOLO
    m = YOLO(str(ROOT / "weights" / a.model) if (ROOT / "weights" / a.model).exists() else a.model)
    data = data_yaml()
    m.train(data=data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
            project=str(ROOT / "runs" / "objects4"), name=a.name, exist_ok=True,
            patience=20, workers=4, cache=False,
            fliplr=0.5, flipud=0.0, degrees=10, hsv_h=0.01, hsv_s=0.5, hsv_v=0.4,
            scale=0.6, translate=0.15, mosaic=1.0, close_mosaic=10)
    best = ROOT / "runs" / "objects4" / a.name / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=data, imgsz=a.imgsz, split="val", device=a.device, verbose=True)
    print(f"mAP50 box {metrics.box.map50:.3f}, mask {metrics.seg.map50:.3f}; mAP50-95 mask {metrics.seg.map:.3f}")
    export(str(best), a.export_imgsz)


if __name__ == "__main__":
    main()
