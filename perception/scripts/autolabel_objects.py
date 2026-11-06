#!/usr/bin/env python3
"""
Gán nhãn tự động (mặt nạ YOLO-seg) cho 4 vật gắp bằng YOLOE gợi ý bằng ẢNH — không cần CLIP.

Cách làm:
  1. Lấy đặc trưng mẫu (visual prompt embedding) của mỗi vật từ vài khung đã khoanh tay (REFS),
     trung bình theo lớp -> YOLOE thành detector 4 lớp cố định.
  2. Chạy trên mọi ảnh trong datasets/frames/<clip>/ (imgsz 1280 để bắt vật xa).
  3. Lọc theo ràng buộc "mỗi vật chỉ có MỘT cái":
       - clip một vật (tên clip cho biết vật, camera đứng yên): loại phát hiện đứng yên cùng chỗ qua
         nhiều khung (nền bị nhận nhầm), giữ phát hiện tốt nhất, gán lớp theo clip;
       - clip nhiều vật: mỗi lớp giữ phát hiện tốt nhất >= --keep, cái yếu hơn hẳn coi là nền;
       - ảnh có vật lưng chừng (--drop-lo .. --keep) hoặc 2 ứng viên cùng lớp sát điểm (< --margin)
         là MƠ HỒ -> bỏ cả ảnh (thà thiếu ảnh còn hơn nhãn sai).
  4. Chia train/val theo CLIP (VAL_CLIPS), ghi datasets/objects4/{images,labels}/{train,val}, data.yaml,
     và ảnh xem trước để duyệt bằng mắt (datasets/objects4/preview/).

    ~/venvs/new_rl_ros2/bin/python scripts/autolabel_objects.py
    ~/venvs/new_rl_ros2/bin/python scripts/autolabel_objects.py --clips "nhieu_vat_*" --limit 50   # thử nhanh

Sau đó: train bằng scripts/train_objects.py.
"""
import argparse
import fnmatch
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "datasets" / "frames"
OUT = ROOT / "datasets" / "objects4"

NAMES = ["hop_nau", "tru_xanh", "hop_pin", "lo_nap_do"]

# Tiền tố tên clip -> lớp (clip chỉ có 1 vật). Clip không khớp = clip nhiều vật.
SINGLE = {"hop_nau": 0, "san_hop_nau": 0, "tru": 1, "san_tru": 1,
          "ovonic": 2, "san_ovonic": 2, "yellow_bottle": 3, "san_yellow_bottle": 3}
MULTI = ("nhieu_vat", "nhieu_vat_2_3", "di_chuyen_vat", "xa")
# Val tách theo clip: 1 clip đủ 4 vật trên bàn + 1 clip camera di chuyển
VAL_CLIPS = ("nhieu_vat_20261028_165504", "di_chuyen_vat_20261028_165912")

# Khung mẫu khoanh tay (pixel 1280x720): (clip/ảnh, [x1 y1 x2 y2], lớp)
REFS = [
    ("nhieu_vat_20261028_165504/nhieu_vat_20261028_165504_000150.jpg",
     [[755, 252, 877, 442], [922, 337, 1075, 585], [280, 195, 465, 495], [522, 360, 658, 553]], [0, 1, 2, 3]),
    ("nhieu_vat_2_3_20261028_170105/nhieu_vat_2_3_20261028_170105_001760.jpg",
     [[40, 225, 420, 365], [415, 250, 580, 505], [680, 155, 770, 360]], [2, 3, 1]),
    ("nhieu_vat_2_3_20261028_170105/nhieu_vat_2_3_20261028_170105_005190.jpg",
     [[470, 213, 925, 400], [153, 227, 347, 433]], [2, 3]),
    ("ovonic_20261028_163006/ovonic_20261028_163006_000000.jpg", [[440, 347, 733, 467]], [2]),
    ("san_ovonic_20261028_164015/san_ovonic_20261028_164015_001668.jpg", [[633, 333, 827, 427]], [2]),
]


def clip_kind(clip):
    """-> lớp (int) nếu clip một vật, None nếu clip nhiều vật, False nếu không dùng."""
    tag = clip.rsplit("_", 2)[0]  # bỏ _YYYYMMDD_HHMMSS
    if tag in SINGLE:
        return SINGLE[tag]
    if tag in MULTI:
        return None
    return False


def build_model(weights):
    import torch
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    m = YOLOE(weights)
    f0, b0, c0 = REFS[0]
    m.predict(str(FRAMES / f0), visual_prompts=dict(bboxes=np.array(b0, float), cls=np.arange(len(c0))),
              predictor=YOLOEVPSegPredictor, verbose=False)
    pred = m.predictor
    acc = [[] for _ in NAMES]
    for f, boxes, cls in REFS:
        pred.set_prompts(dict(bboxes=np.array(boxes, float), cls=np.arange(len(cls))))
        v = pred.get_vpe(str(FRAMES / f))[0]
        for i, k in enumerate(cls):
            acc[k].append(torch.nn.functional.normalize(v[i], dim=-1))
    pe = torch.stack([torch.nn.functional.normalize(torch.stack(a).mean(0), dim=-1) for a in acc])[None]
    det = YOLOE(weights)
    det.set_classes(NAMES, pe.cpu())
    return det


def select(dets, single_cls, keep, keep_single, drop_lo, margin):
    """dets: list (cls, conf, poly, box). -> (danh sách giữ, lý do bỏ ảnh hoặc None).

    Mỗi vật chỉ có MỘT cái: phát hiện cùng lớp yếu hơn hẳn (kém > margin) chắc chắn là nhầm -> coi
    là nền. Chỉ bỏ ảnh khi không biết cái nào thật (2 ứng viên sát điểm) hoặc vật thật lưng chừng.
    """
    if single_cls is not None:
        # clip một vật: bỏ qua lớp YOLOE đoán, chỉ tin vị trí; lớp lấy theo clip
        dets = sorted(dets, key=lambda d: -d[1])
        if not dets or dets[0][1] < keep_single:
            return [], "không thấy vật"
        if len(dets) > 1 and dets[0][1] - dets[1][1] < margin:
            return [], "2 ứng viên"
        return [(single_cls, dets[0][1], dets[0][2])], None
    kept = []
    for c in range(len(NAMES)):
        dc = sorted([d for d in dets if d[0] == c], key=lambda d: -d[1])
        if not dc:
            continue
        if dc[0][1] < keep:
            return [], f"{NAMES[c]} lưng chừng {dc[0][1]:.2f}"
        if len(dc) > 1 and dc[0][1] - dc[1][1] < margin:
            return [], f"2 {NAMES[c]}"
        kept.append(dc[0])
    if not kept:
        # clip nhiều vật: khung nào cũng có vật, không thấy gì = vật nhỏ/khuất bị sót, không phải nền
        return [], "không thấy vật"
    return kept, None


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def static_mask(all_dets, frac, iou=0.7):
    """Phát hiện đứng yên một chỗ trong >= frac số khung (camera cố định, clip cầm tay) là nền
    bị nhận nhầm (thảm, thùng ở góc) -> True để loại. Vật cầm tay thì di chuyển nên không bị loại."""
    boxes = [d[3] for dets in all_dets for d in dets]
    n = len(all_dets)
    out = []
    for dets in all_dets:
        flags = []
        for d in dets:
            hits = sum(box_iou(d[3], b) >= iou for b in boxes)
            flags.append(hits >= frac * n)
        out.append(flags)
    return out


def to_yolo_line(cls, poly, w, h, eps_px=1.5):
    p = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    p = cv2.approxPolyDP(p, eps_px, True).reshape(-1, 2)
    if len(p) < 3:
        return None
    xy = np.clip(p / [w, h], 0, 1)
    return f"{cls} " + " ".join(f"{v:.5f}" for v in xy.ravel())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "weights" / "yoloe-11l-seg.pt"))
    ap.add_argument("--clips", default="*", help="mẫu tên thư mục trong datasets/frames")
    ap.add_argument("--keep", type=float, default=0.6, help="độ tin tối thiểu để giữ một vật")
    ap.add_argument("--keep-single", type=float, default=0.3,
                    help="như --keep cho clip một vật (lớp đã biết theo clip, chỉ cần đúng vị trí)")
    ap.add_argument("--drop-lo", type=float, default=0.25, help="dưới mức này coi như không có")
    ap.add_argument("--margin", type=float, default=0.25,
                    help="2 ứng viên cùng lớp chênh độ tin ít hơn mức này -> không biết cái nào thật, bỏ ảnh")
    ap.add_argument("--static-frac", type=float, default=0.15,
                    help="clip một vật: phát hiện đứng yên cùng chỗ trong >= tỉ lệ khung này là nền, loại")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N ảnh mỗi clip (thử nhanh)")
    ap.add_argument("--preview-every", type=int, default=25)
    a = ap.parse_args()

    clips = [d for d in sorted(FRAMES.iterdir()) if d.is_dir() and fnmatch.fnmatch(d.name, a.clips)
             and clip_kind(d.name) is not False]
    if not clips:
        raise SystemExit("Không có clip nào khớp.")
    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ("train", "val"):
        (OUT / "images" / s).mkdir(parents=True)
        (OUT / "labels" / s).mkdir(parents=True)
    (OUT / "preview").mkdir()

    det = build_model(a.weights)
    stats = {}
    for clip in clips:
        single = clip_kind(clip.name)
        split = "val" if clip.name in VAL_CLIPS else "train"
        imgs = sorted(clip.glob("*.jpg"))
        if a.limit:
            imgs = imgs[:: max(1, len(imgs) // a.limit)][: a.limit]
        st = dict(split=split, total=len(imgs), kept=0, objs=[0] * len(NAMES), dropped={}, static=0)
        all_dets, shapes = [], []
        for f in imgs:
            r = det.predict(str(f), conf=a.drop_lo, imgsz=a.imgsz, retina_masks=True, verbose=False)[0]
            dets = []
            if r.masks is not None:
                for c, s, poly, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.masks.xy,
                                         r.boxes.xyxy.tolist()):
                    if len(poly) >= 3:
                        dets.append((int(c), float(s), poly, b))
            all_dets.append(dets)
            shapes.append(r.orig_shape)
        if single is not None:
            flags = static_mask(all_dets, a.static_frac)
            st["static"] = sum(map(sum, flags))
            all_dets = [[d for d, s in zip(ds, fl) if not s] for ds, fl in zip(all_dets, flags)]
        for i, (f, dets, (h, w)) in enumerate(zip(imgs, all_dets, shapes)):
            kept, why = select(dets, single, a.keep, a.keep_single, a.drop_lo, a.margin)
            if why:
                key = why.split(" ")[0] if "lưng" in why else why
                st["dropped"][key] = st["dropped"].get(key, 0) + 1
                continue
            lines = [ln for c, _, poly, *_ in kept if (ln := to_yolo_line(c, poly, w, h))]
            shutil.copy2(f, OUT / "images" / split / f.name)
            (OUT / "labels" / split / (f.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            st["kept"] += 1
            for c, *_ in kept:
                st["objs"][c] += 1
            if i % a.preview_every == 0:
                im = cv2.imread(str(f))
                for c, s, poly, *_ in kept:
                    col = [(40, 90, 160), (255, 120, 0), (200, 200, 200), (0, 200, 255)][c]
                    cv2.polylines(im, [np.int32(poly)], True, col, 2)
                    x, y = np.int32(poly).min(0)
                    cv2.putText(im, f"{NAMES[c]} {s:.2f}", (int(x), max(15, int(y) - 5)), 0, 0.7, col, 2)
                cv2.imwrite(str(OUT / "preview" / f.name), cv2.resize(im, (640, 360)))
        stats[clip.name] = st
        print(f"{clip.name:40s} {split:5s} giữ {st['kept']:4d}/{st['total']:4d}  vật {st['objs']}  "
              f"nền đứng yên {st['static']}  bỏ {st['dropped']}", flush=True)

    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    (OUT / "autolabel_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    tot = {s: sum(v["kept"] for v in stats.values() if v["split"] == s) for s in ("train", "val")}
    objs = np.sum([v["objs"] for v in stats.values()], axis=0).tolist()
    print(f"Tổng: train {tot['train']} ảnh, val {tot['val']} ảnh; số vật theo lớp {dict(zip(NAMES, objs))}")
    print(f"Duyệt: {OUT / 'preview'}")


if __name__ == "__main__":
    main()
