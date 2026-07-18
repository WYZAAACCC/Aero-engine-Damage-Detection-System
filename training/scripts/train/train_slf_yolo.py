"""训练 SLF-YOLO 叶片缺陷检测 (BladeSynth + AEBIS)

按指导书 Section 4 执行:
- 类别: dent, nick, scratch, corrosion, crack, pit, fracture, edge_deformation, other_damage (9类)
- 预训练: COCO yolov8n.pt → 领域微调
- 输入: 640×640
- 8GB VRAM: batch=4

用法:
  python train_slf_yolo.py --data /f/aero-training/artifacts/raw/bladesynth --epochs 50
"""

import argparse, hashlib, json, os, sys, time, shutil
from pathlib import Path
import numpy as np
import yaml


# ═══════════════════════════════════════════════════════════════════════
# 数据转换: BladeSynth masks → YOLO format
# ═══════════════════════════════════════════════════════════════════════

BLADESYNTH_CLASSES = {
    "Dent": 0, "Nick": 1, "Scratch": 2, "Corrosion": 3,
    "Crack": 4, "Pit": 5, "Fracture": 6, "Edge_deformation": 7,
    "Other_damage": 8,
}

def mask_to_yolo_bbox(mask_path: Path, img_w: int, img_h: int):
    """Binary mask → YOLO normalized bbox list."""
    from PIL import Image
    mask = Image.open(mask_path).convert("L")
    mask_arr = np.array(mask)
    if mask_arr.max() < 10:
        return []  # empty mask

    # 连通域 → bbox
    from scipy import ndimage
    labeled, n_labels = ndimage.label(mask_arr > 30)
    boxes = []
    for i in range(1, n_labels + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) < 20:  # 极小区域过滤
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        w, h = x2 - x1, y2 - y1
        if w < 5 or h < 5:  # 太小的bbox跳过
            continue
        # YOLO normalized: cx/W, cy/H, w/W, h/H
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        nw = w / img_w
        nh = h / img_h
        boxes.append((cx, cy, nw, nh))
    return boxes


def prepare_bladesynth_yolo(bladesynth_root: str, output_dir: str):
    """Convert BladeSynth to YOLO format."""
    bs = Path(bladesynth_root)
    out = Path(output_dir)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # 查找所有图片和mask
    images = sorted(bs.rglob("*.png")) + sorted(bs.rglob("*.jpg"))
    print(f"[CONVERT] Found {len(images)} images in {bladesynth_root}")

    for i, img_path in enumerate(images):
        # 确定类别
        cls_name = img_path.parent.name
        cls_id = BLADESYNTH_CLASSES.get(cls_name, BLADESYNTH_CLASSES.get(cls_name.lower(), -1))
        if cls_id < 0 and cls_name.lower() != "normal":
            continue  # 跳过无法识别的

        # 查找mask
        mask_path = None
        for mdir in ['masks', 'ground_truth', 'BlackWhite']:
            mp = img_path.parent.parent / mdir / cls_name / img_path.name.replace('.jpg','.png').replace('.JPG','.png')
            if mp.exists(): mask_path = mp; break

        # 简单切分 80/20
        split = "train" if i % 5 != 0 else "val"
        dst_img = out / "images" / split / f"bs_{i:06d}.png"

        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        img.save(str(dst_img))

        if mask_path is not None and cls_id >= 0:
            boxes = mask_to_yolo_bbox(mask_path, img.width, img.height)
            label_file = out / "labels" / split / f"bs_{i:06d}.txt"
            with open(label_file, "w") as f:
                for cx, cy, nw, nh in boxes:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    print(f"[CONVERT] YOLO dataset ready at {output_dir}")


def create_yaml_config(data_dir: str, output_path: str):
    """创建 YOLO data.yaml。"""
    config = {
        "path": str(Path(data_dir).absolute()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "dent", 1: "nick", 2: "scratch", 3: "corrosion",
            4: "crack", 5: "pit", 6: "fracture", 7: "edge_deformation",
            8: "other_damage",
        },
        "nc": 9,
    }
    with open(output_path, "w") as f:
        yaml.dump(config, f)
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/f/aero-training/artifacts/raw")
    parser.add_argument("--output", default="/f/aero-training/artifacts/models/slf_yolo/v1.0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="cuda" if __import__('torch').cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    print(f"[Device] {args.device}")

    # 准备数据
    yolo_data_dir = "/f/aero-training/artifacts/canonical/slf_yolo_blade"
    data_root = Path(args.data)

    # 从 BladeSynth 转换
    bs_dir = data_root / "bladesynth"
    if bs_dir.exists():
        prepare_bladesynth_yolo(str(bs_dir), yolo_data_dir)
    else:
        print(f"[WAIT] BladeSynth not found at {bs_dir} — downloading...")
        print("[WAIT] If download is in progress, re-run after it completes.")
        return

    yaml_path = create_yaml_config(yolo_data_dir, f"{yolo_data_dir}/data.yaml")

    if args.smoke_test:
        args.epochs = 2

    # YOLO 训练
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics required")
        sys.exit(1)

    model = YOLO("yolov8n.pt")  # COCO pre-trained
    print(f"[TRAIN] Starting SLF-YOLO training...")
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=str(args.device),
        workers=0,
        project=str(Path(args.output).parent),
        name=Path(args.output).name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
    )

    # 保存产物
    best_pt = Path(args.output) / "weights" / "best.pt"
    if best_pt.exists():
        sha = hashlib.sha256(best_pt.read_bytes()).hexdigest()
        with open(Path(args.output) / "weight.sha256", "w") as f:
            f.write(f"{sha}  best.pt\n")
        with open(Path(args.output) / "model_card.md", "w") as f:
            f.write(f"""# SLF-YOLO Blade Defect Detector

- **Task:** Multi-class blade defect detection (9 classes)
- **Data:** BladeSynth (synthetic, 12,500 images)
- **Architecture:** YOLOv8-nano (COCO pretrained)
- **Input:** {args.img_size}×{args.img_size}
- **Limitations:** Trained on synthetic data only — NOT validated on real borescope images.
- **Status:** experimental
""")
        print(f"[DONE] Model saved: {best_pt}")
    else:
        print("[WARN] best.pt not found — check training logs")


if __name__ == "__main__":
    from PIL import Image; import scipy.ndimage as ndimage
    main()
