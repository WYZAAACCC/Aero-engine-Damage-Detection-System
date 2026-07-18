"""训练 UNet 叶片缺陷分割 (AEBIS 数据集, ResNet-18 编码器)

按指导书 Section 5 执行:
- 二值缺陷分割 (0=背景, 255=缺陷)
- ResNet-18 encoder + UNet decoder (8GB VRAM 友好)
- Dice + BCE 损失
- 输入 256×256

用法:
  python train_unet_aebis.py --data /f/aero-training/artifacts/raw/AEBIS
"""

import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════
# AEBIS 数据集
# ═══════════════════════════════════════════════════════════════════════

class AEBISDataset(Dataset):
    def __init__(self, root: str, split: str = "train", size: int = 256):
        root = Path(root) / split
        self.img_dir = root / "JPEGImages"
        self.mask_dir = root / "BlackWhite"
        self.edge_dir = root / "Edge"

        self.samples = sorted(
            [p.stem for p in self.img_dir.glob("*.jpg")] +
            [p.stem for p in self.img_dir.glob("*.png")]
        )
        if not self.samples:
            raise FileNotFoundError(f"No images in {self.img_dir}")

        self.img_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        stem = self.samples[idx]
        # 尝试 jpg/png
        for ext in ['.jpg', '.png', '.JPG', '.PNG']:
            img_path = self.img_dir / (stem + ext)
            if img_path.exists(): break
        else:
            img_path = self.img_dir / (stem + '.jpg')

        mask_path = self.mask_dir / (stem + '.png')
        edge_path = self.edge_dir / (stem + '.png')

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") if mask_path.exists() else Image.new("L", img.size, 0)
        edge = Image.open(edge_path).convert("L") if edge_path.exists() else Image.new("L", img.size, 0)

        return self.img_transform(img), (self.mask_transform(mask) > 0.5).float(), \
               (self.mask_transform(edge) > 0.5).float()


# ═══════════════════════════════════════════════════════════════════════
# UNet with ResNet-18 encoder
# ═══════════════════════════════════════════════════════════════════════

class UNetResNet18(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        rn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.enc0 = nn.Sequential(rn.conv1, rn.bn1, rn.relu)  # 64, H/2
        self.pool0 = rn.maxpool  # H/4
        self.enc1 = rn.layer1  # 64, H/4
        self.enc2 = rn.layer2  # 128, H/8
        self.enc3 = rn.layer3  # 256, H/16
        self.enc4 = rn.layer4  # 512, H/32
        self.bridge = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU())

        # Decoder
        self.up4 = self._up(512, 256); self.dec4 = self._conv(512, 256)
        self.up3 = self._up(256, 128); self.dec3 = self._conv(256, 128)
        self.up2 = self._up(128, 64);  self.dec2 = self._conv(128, 64)
        self.up1 = self._up(64, 64);   self.dec1 = self._conv(128, 64)
        self.up0 = self._up(64, 64)  # 128→256
        self.out = nn.Conv2d(64, n_classes, 1)

    def _up(self, in_c, out_c):
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                             nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU())

    def _conv(self, in_c, out_c):
        return nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
                             nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU())

    def forward(self, x):
        e0 = self.enc0(x)      # 64, H/2
        e1 = self.enc1(self.pool0(e0))  # 64, H/4
        e2 = self.enc2(e1)     # 128, H/8
        e3 = self.enc3(e2)     # 256, H/16
        e4 = self.enc4(e3)     # 512, H/32
        b = self.bridge(e4)

        d4 = self.dec4(torch.cat([self.up4(b), e3], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e0], 1))
        d0 = self.up0(d1)  # 128→256
        return torch.sigmoid(self.out(d0))


def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(pred, target):
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice, bce.item(), dice.item()


# 训练函数
def train_epoch(model, loader, opt, device):
    model.train()
    total, n = 0.0, 0
    for x, mask, _ in loader:
        x, mask = x.to(device), mask.to(device)
        opt.zero_grad()
        pred = model(x)
        loss, _, _ = combined_loss(pred, mask)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0); n += x.size(0)
    return total / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_dice, n = 0.0, 0
    for x, mask, _ in loader:
        x, mask = x.to(device), mask.to(device)
        pred = model(x)
        pred_bin = (pred > 0.5).float()
        dice = 1 - dice_loss(pred_bin, mask)  # Dice coefficient
        total_dice += dice.item() * x.size(0); n += x.size(0)
    return total_dice / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/f/aero-training/artifacts/raw/AEBIS")
    parser.add_argument("--output", default="/f/aero-training/artifacts/models/unet_aebis/v1.0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device(args.device)
    print(f"[Device] {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory//1024**3}GB VRAM")

    train_ds = AEBISDataset(args.data, "Train", args.size)
    test_ds = AEBISDataset(args.data, "Test", args.size)

    if args.smoke_test:
        train_ds.samples = train_ds.samples[:30]
        test_ds.samples = test_ds.samples[:20]
        args.epochs = 3

    # 从训练集划出验证集 (80/20) — 必须在截断train之前创建val
    n_train = int(len(train_ds.samples) * 0.8)
    # 复制部分样本给验证集
    val_ds = AEBISDataset(args.data, "Train", args.size)
    val_ds.samples = train_ds.samples[n_train:]
    train_ds.samples = train_ds.samples[:n_train]

    print(f"[DATA] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=0)

    model = UNetResNet18().to(device)
    print(f"[MODEL] Params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_dice, best_ep = 0.0, 0

    for ep in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, opt, device)
        val_dice = evaluate(model, val_loader, device)
        if val_dice > best_dice:
            best_dice, best_ep = val_dice, ep
            out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "input_size": args.size,
                        "val_dice": float(val_dice)}, out_dir / "best_model.pth")
        print(f"Ep {ep+1:3d}/{args.epochs} | loss={train_loss:.4f} | val_dice={val_dice:.4f} | {time.time()-t0:.1f}s" +
              (" *" if ep == best_ep else ""))

    # Final test
    ckpt = torch.load(Path(args.output) / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_dice = evaluate(model, test_loader, device)
    print(f"\n[FINAL] Test Dice: {test_dice:.4f} (best ep={best_ep+1})")

    out_dir = Path(args.output)
    with open(out_dir / "config.yaml", "w") as f:
        import yaml; yaml.dump({"model": "unet_resnet18", "test_dice": round(float(test_dice),4),
                                "input_size": args.size}, f)
    sha = hashlib.sha256((out_dir / "best_model.pth").read_bytes()).hexdigest()
    with open(out_dir / "weight.sha256", "w") as f: f.write(f"{sha}  best_model.pth\n")
    print(f"[DONE] {out_dir}")


if __name__ == "__main__":
    main()
