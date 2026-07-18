"""训练 PatchCore 异常检测 (AeBAD 航空发动机叶片)

按照指导书 Section 3 (方案B) 执行:
- WideResNet-50 预训练特征提取
- Coreset 子采样减小 memory bank
- KNN 距离→异常分数
- 8GB VRAM 可运行

用法:
  python train_patchcore_aebad.py --data /f/aero-training/artifacts/raw/AeBAD/AeBAD --output /f/aero-training/artifacts/models/patchcore_aebad/v1.0
"""

import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from collections import defaultdict
# sklearn metrics imported at use site


# ═══════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════

class AeBADDataset(Dataset):
    """加载 AeBAD 叶片图像。"""
    def __init__(self, root: str, split: str = "train", subset: str = "AeBAD_S",
                 transform=None):
        root = Path(root) / subset
        self.samples = []
        self.labels = []  # 0=good, 1=anomaly
        self.paths = []

        if split == "train":
            good_dir = root / "train" / "good"
            if good_dir.exists():
                for subdir in good_dir.iterdir():
                    if subdir.is_dir():
                        for img in subdir.glob("*"):
                            if not img.suffix.lower() in ('.png', '.jpg', '.jpeg'): continue
                            if img.name.startswith('._'): continue
                            self.samples.append(str(img))
                            self.labels.append(0)
                            self.paths.append(str(img))
        else:  # test
            test_dir = root / "test"
            if test_dir.exists():
                for anomaly_type in test_dir.iterdir():
                    if anomaly_type.is_dir() and anomaly_type.name != "good":
                        for subdir in anomaly_type.iterdir():
                            if subdir.is_dir():
                                for img in subdir.glob("*"):
                                    self.samples.append(str(img))
                                    self.labels.append(1)
                                    self.paths.append(str(img))
                    elif anomaly_type.is_dir() and anomaly_type.name == "good":
                        for subdir in anomaly_type.iterdir():
                            if subdir.is_dir():
                                for img in subdir.glob("*"):
                                    self.samples.append(str(img))
                                    self.labels.append(0)
                                    self.paths.append(str(img))

        if not self.samples:
            raise FileNotFoundError(f"No images found in {root}/{'train/good' if split=='train' else 'test'}")

        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        try:
            img = Image.open(self.samples[idx]).convert("RGB")
        except Exception:
            # 跳过损坏文件/资源分叉，返回空白图
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return self.transform(img), self.labels[idx], self.paths[idx]


# ═══════════════════════════════════════════════════════════════════════
# PatchCore 特征提取与 Memory Bank
# ═══════════════════════════════════════════════════════════════════════

class WideResNetFeatureExtractor(nn.Module):
    """WideResNet-50 中层特征提取 (layer2+layer3)。"""
    def __init__(self):
        super().__init__()
        wrn = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.layer1 = nn.Sequential(wrn.conv1, wrn.bn1, wrn.relu, wrn.maxpool, wrn.layer1)
        self.layer2 = wrn.layer2
        self.layer3 = wrn.layer3

    def forward(self, x):
        with torch.no_grad():
            x = self.layer1(x)
            f2 = self.layer2(x)   # (B, 512, 28, 28)
            f3 = self.layer3(f2)  # (B, 1024, 14, 14)
            # 拼接并平均池化到 patch 级别
            f2_up = F.interpolate(f2, size=(14, 14), mode='bilinear')
            features = torch.cat([f2_up, f3], dim=1)  # (B, 1536, 14, 14)
            features = F.adaptive_avg_pool2d(features, (7, 7))  # (B, 1536, 7, 7)
            return features.flatten(2).transpose(1, 2)  # (B, 49, 1536)


def build_memory_bank(extractor, dataloader, device, coreset_ratio=0.1):
    """构建正常样本特征库（coreset采样）。"""
    extractor.eval()
    all_features = []
    with torch.no_grad():
        for x, _, _ in dataloader:
            feat = extractor(x.to(device))  # (B, 49, 1536)
            all_features.append(feat.cpu().numpy())

    all_feat = np.concatenate(all_features, axis=0)  # (N_total, 49, 1536)
    n_patches, n_dim = all_feat.shape[0] * all_feat.shape[1], all_feat.shape[2]
    flat = all_feat.reshape(-1, n_dim)  # (N_patches, 1536)

    # Coreset 子采样
    n_coreset = max(100, int(len(flat) * coreset_ratio))
    indices = np.random.choice(len(flat), min(n_coreset, len(flat)), replace=False)
    coreset = flat[indices]

    print(f"[MemoryBank] Full: {len(flat)} patches → Coreset: {len(coreset)} patches "
          f"(ratio={coreset_ratio})")

    return torch.tensor(coreset, dtype=torch.float32), all_feat.shape[1]


def anomaly_score_batch(features, memory_bank, k=5):
    """计算每个patch到memory bank中k近邻的平均距离→异常分数。"""
    # features: (B, 49, D), memory_bank: (M, D)
    B, P, D = features.shape
    features_flat = features.reshape(-1, D).cuda()
    memory_bank_gpu = memory_bank.cuda()

    # 分批计算距离避免OOM
    BATCH = 512
    distances = []
    for i in range(0, len(features_flat), BATCH):
        chunk = features_flat[i:i+BATCH]
        dist = torch.cdist(chunk, memory_bank_gpu)  # (chunk, M)
        topk, _ = torch.topk(dist, k, dim=1, largest=False)
        distances.append(topk.mean(dim=1))  # (chunk,)
    anomaly_scores = torch.cat(distances).reshape(B, P).max(dim=1)[0]  # (B,)
    return anomaly_scores


# ═══════════════════════════════════════════════════════════════════════
# 训练入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/f/aero-training/artifacts/raw/AeBAD/AeBAD")
    parser.add_argument("--output", default="/f/aero-training/artifacts/models/patchcore_aebad/v1.0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--coreset-ratio", type=float, default=0.1)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory // 1024**3} GB VRAM")

    # ── 加载数据 ──
    train_ds = AeBADDataset(args.data, "train", "AeBAD_S")
    test_ds = AeBADDataset(args.data, "test", "AeBAD_S")

    if args.smoke_test:
        indices = np.random.choice(len(train_ds), min(50, len(train_ds)), replace=False)
        train_ds.samples = [train_ds.samples[i] for i in indices]
        indices = np.random.choice(len(test_ds), min(30, len(test_ds)), replace=False)
        test_ds.samples = [test_ds.samples[i] for i in indices]

    print(f"[DATA] Train (normal): {len(train_ds)} images")
    print(f"[DATA] Test: {len(test_ds)} images "
          f"(good={sum(1 for l in test_ds.labels if l==0)}, "
          f"anomaly={sum(1 for l in test_ds.labels if l==1)})")

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=0)

    # ── 构建 Memory Bank ──
    print(f"[INFO] Building memory bank...")
    extractor = WideResNetFeatureExtractor().to(device)
    memory_bank, feat_dim = build_memory_bank(extractor, train_loader, device, args.coreset_ratio)

    # ── 评估 ──
    print(f"[INFO] Evaluating...")
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for x, labels, _ in test_loader:
            features = extractor(x.to(device))
            scores = anomaly_score_batch(features, memory_bank, args.k_neighbors)
            all_scores.append(scores.cpu().numpy())
            all_labels.extend(labels)

    scores = np.concatenate(all_scores)
    labels = np.array(all_labels)

    # ── 指标 ──
    from sklearn.metrics import roc_auc_score
    try:
        image_auroc = roc_auc_score(labels, scores)
    except Exception:
        image_auroc = 0.0

    # 最佳F1阈值
    best_f1, best_thresh = 0.0, np.median(scores)
    for t in np.percentile(scores, np.linspace(50, 99, 50)):
        preds = (scores > t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    # 误报率
    normal_scores = scores[labels == 0]
    fpr = np.mean(normal_scores > best_thresh) if len(normal_scores) > 0 else 0

    print(f"\n[RESULTS]")
    print(f"  Image AUROC: {image_auroc:.4f}")
    print(f"  Best F1: {best_f1:.4f} (threshold={best_thresh:.4f})")
    print(f"  FPR@bestF1: {fpr:.4f}")

    # ── 保存产物 ──
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"memory_bank": memory_bank, "feat_dim": feat_dim,
                "threshold": float(best_thresh), "image_auroc": float(image_auroc),
                "coreset_ratio": args.coreset_ratio, "k_neighbors": args.k_neighbors},
               output_dir / "model.pth")

    config = {"model_id": "patchcore_aebad_s", "image_auroc": round(float(image_auroc), 4),
              "best_f1": round(float(best_f1), 4), "threshold": round(float(best_thresh), 4),
              "fpr": round(float(fpr), 4), "coreset_ratio": args.coreset_ratio}
    with open(output_dir / "config.yaml", "w") as f:
        import yaml; yaml.dump(config, f)

    sha = hashlib.sha256((output_dir / "model.pth").read_bytes()).hexdigest()
    with open(output_dir / "weight.sha256", "w") as f:
        f.write(f"{sha}  model.pth\n")

    with open(output_dir / "model_card.md", "w") as f:
        f.write(f"""# PatchCore — AeBAD-S Anomaly Detection

- **Task:** Unsupervised anomaly detection on aero-engine blades
- **Data:** AeBAD-S (single blade images)
- **Architecture:** WideResNet-50 (layer2+layer3) + Coreset Memory Bank
- **Image AUROC:** {image_auroc:.4f}
- **Best F1:** {best_f1:.4f}
- **Limitations:** AeBAD has domain shift (illumination/view). Coreset sampling loses minority patterns.
- **Status:** experimental — validated_public_domain (AeBAD only)
""")

    print(f"\n[DONE] Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
