"""训练 FaultSense LSTM-AE + RUL (PyTorch移植版)

按照指导书 Section 9 + 上游仓库 momo-2609/FaultSense 复现:
- 架构: LSTM Encoder→Decoder (重构) + RUL Head (回归)
- 异常检测: 正常期重构误差阈值 (k-sigma, k=2.5)
- RUL标签: piecewise linear, cap=130
- 单阶段训练: AE重构损失 + RUL回归损失
- 输入窗口: 30 cycles, 14 传感器

用法:
  python train_faultsense.py --subset FD001 --epochs 100
  python train_faultsense.py --subset FD001 --smoke-test
"""

import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════════
# C-MAPSS 数据 (复用 cnn_lstm 的数据加载逻辑)
# ═══════════════════════════════════════════════════════════════════════

USEFUL_SENSOR_INDICES = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
N_FEATURES = len(USEFUL_SENSOR_INDICES)
SEQ_LEN = 30  # FaultSense 上游窗口
RUL_CAP = 130


def load_cmapss_raw(data_dir: str, subset: str) -> dict:
    data_dir = Path(data_dir)
    train_file = data_dir / f"train_{subset}.txt"
    test_file = data_dir / f"test_{subset}.txt"
    rul_file = data_dir / f"RUL_{subset}.txt"

    def _parse(path):
        data = np.loadtxt(str(path))
        units = {}
        for row in data:
            uid = int(row[0])
            if uid not in units:
                units[uid] = []
            units[uid].append(row[1:])
        return {uid: np.array(v) for uid, v in units.items()}

    return {
        "train": _parse(train_file),
        "test": _parse(test_file),
        "test_rul": np.loadtxt(str(rul_file)).flatten(),
    }


def compute_norm_stats(units: dict):
    """从训练数据计算每个传感器的均值和标准差。"""
    all_vals = []
    for data in units.values():
        features = data[:, 1 + np.array(USEFUL_SENSOR_INDICES)]
        all_vals.append(features)
    all_data = np.concatenate(all_vals, axis=0)
    mean = all_data.mean(axis=0, keepdims=True)
    std = all_data.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0  # 防止除零
    return mean.astype(np.float32), std.astype(np.float32)


def build_sequences(units: dict, seq_len: int = SEQ_LEN,
                    stride: int = 1, rul_cap: int = RUL_CAP,
                    normal_only: bool = False, normal_frac: float = 0.4,
                    norm_stats: tuple = None):
    """构建 (seq, rul_label) 对。"""
    mean, std = norm_stats if norm_stats else (0.0, 1.0)
    X, y_rul, uids = [], [], []
    for uid, data in units.items():
        cycles = data[:, 0]
        features = data[:, 1 + np.array(USEFUL_SENSOR_INDICES)]
        features = (features - mean) / std  # z-score归一化
        max_cycle = cycles[-1]
        for start in range(0, len(cycles) - seq_len, stride):
            end = start + seq_len
            raw_rul = max_cycle - cycles[end - 1]
            if normal_only and raw_rul / max_cycle < normal_frac:
                continue
            seq = features[start:end].astype(np.float32)
            X.append(seq)
            y_rul.append(min(raw_rul, rul_cap) / rul_cap)
            uids.append(uid)
    return np.array(X), np.array(y_rul, dtype=np.float32), uids


class FaultSenseDataset(Dataset):
    def __init__(self, X, y_rul, uids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_rul = torch.tensor(y_rul, dtype=torch.float32).unsqueeze(-1)
        self.uids = uids
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        uid = self.uids[idx] if idx < len(self.uids) else -1
        return self.X[idx], self.y_rul[idx], uid


# ═══════════════════════════════════════════════════════════════════════
# FaultSenseModel (PyTorch 移植)
# ═══════════════════════════════════════════════════════════════════════

class FaultSenseModel(nn.Module):
    """LSTM Encoder-Decoder + RUL Head。

    上游: momo-2609/FaultSense-LSTM-Anomaly-Detection-on-NASA-CMAPSS
    原实现为 PyTorch (非TF), hidden=32, dropout=0.5.
    """

    def __init__(self, n_features=N_FEATURES, hidden=32, dropout=0.5, rul_cap=RUL_CAP):
        super().__init__()
        self.hidden = hidden
        self.rul_cap = rul_cap

        # Encoder LSTM
        self.encoder = nn.LSTM(n_features, hidden, num_layers=2,
                               batch_first=True, dropout=dropout, bidirectional=False)
        # Decoder LSTM
        self.decoder = nn.LSTM(hidden, hidden, num_layers=1,
                               batch_first=True, bidirectional=False)
        self.reconstruct = nn.Linear(hidden, n_features)

        # RUL prediction head (从encoder最后hidden state)
        self.rul_fc = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),  # 输出 [0,1]
        )

        # 异常阈值 (训练后校准)
        self.register_buffer('threshold', torch.tensor(0.0))
        self._threshold_calibrated = False

    def encode(self, x):
        """x: (B, seq_len, n_features) → (B, hidden)"""
        _, (h_n, _) = self.encoder(x)
        return h_n[-1]  # 最后一层hidden state

    def decode(self, h, seq_len):
        """h: (B, hidden) → (B, seq_len, n_features)"""
        h = h.unsqueeze(1).repeat(1, seq_len, 1)  # (B, seq_len, hidden)
        dec_out, _ = self.decoder(h)
        return self.reconstruct(dec_out)

    def forward(self, x):
        """返回: (reconstruction, rul_pred)"""
        seq_len = x.size(1)
        h = self.encode(x)
        recon = self.decode(h, seq_len)
        rul_pred = self.rul_fc(h)
        return recon, rul_pred

    def anomaly_score(self, x):
        """对输入序列计算重构误差→异常分数。"""
        seq_len = x.size(1)
        h = self.encode(x)
        recon = self.decode(h, seq_len)
        mse = torch.mean((x - recon) ** 2, dim=(1, 2))
        return mse

    def calibrate_threshold(self, normal_loader, k=2.5):
        """用正常样本校准异常阈值 (k-sigma)。"""
        self.eval()
        device = next(self.parameters()).device
        scores = []
        with torch.no_grad():
            for x, _, _ in normal_loader:
                scores.append(self.anomaly_score(x.to(device)))
        all_scores = torch.cat(scores)
        mean_s = all_scores.mean()
        std_s = all_scores.std()
        self.threshold = mean_s + k * std_s
        self._threshold_calibrated = True
        return float(self.threshold)

    def is_anomaly(self, x):
        """判断输入是否异常。"""
        return self.anomaly_score(x) > self.threshold


# ═══════════════════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device, ae_weight=0.3):
    """联合训练: AE重构损失 + RUL回归损失"""
    model.train()
    total_loss, n = 0.0, 0
    mse_ae, mse_rul = 0.0, 0.0

    for x, y_rul, _ in loader:
        x, y_rul = x.to(device), y_rul.to(device)
        optimizer.zero_grad()
        recon, rul_pred = model(x)
        loss_ae = nn.functional.mse_loss(recon, x)
        loss_rul = nn.functional.mse_loss(rul_pred, y_rul)
        loss = ae_weight * loss_ae + (1 - ae_weight) * loss_rul
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        mse_ae += loss_ae.item() * x.size(0)
        mse_rul += loss_rul.item() * x.size(0)
        n += x.size(0)

    return total_loss / n, mse_ae / n, mse_rul / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    for x, y_rul, _ in loader:
        x, y_rul = x.to(device), y_rul.to(device)
        recon, rul_pred = model(x)
        loss = nn.functional.mse_loss(rul_pred, y_rul)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_preds.append(rul_pred.cpu().numpy())
        all_labels.append(y_rul.cpu().numpy())

    preds = np.concatenate(all_preds).flatten() * RUL_CAP
    labels = np.concatenate(all_labels).flatten() * RUL_CAP

    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    mae = np.mean(np.abs(preds - labels))
    d = preds - labels
    nasa = np.mean(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))

    return total_loss / n, rmse, mae, nasa, preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/raw/cmapss")
    parser.add_argument("--subset", default="FD001", choices=["FD001","FD002","FD003","FD004"])
    parser.add_argument("--output", default="artifacts/models/faultsense/v1.0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory // 1024**3} GB VRAM")

    # ── 加载数据 ──
    print(f"[INFO] Loading C-MAPSS {args.subset}...")
    data = load_cmapss_raw(args.data, args.subset)
    train_units, test_units, test_rul = data["train"], data["test"], data["test_rul"]

    # 划分 train/val (按unit_id)
    all_uids = sorted(train_units.keys())
    np.random.shuffle(all_uids)
    n = len(all_uids)
    train_split = {uid: train_units[uid] for uid in all_uids[:int(n * 0.7)]}
    val_split = {uid: train_units[uid] for uid in all_uids[int(n * 0.7):int(n * 0.85)]}
    normal_split = {uid: train_units[uid] for uid in all_uids[:int(n * 0.5)]}  # 校准用

    if args.smoke_test:
        train_split = {uid: train_units[uid] for uid in list(all_uids)[:3]}
        val_split = {uid: train_units[uid] for uid in list(all_uids)[3:5]}
        normal_split = train_split
        args.epochs = 3

    # ── z-score归一化（仅从训练集计算）──
    norm_stats = compute_norm_stats(train_split)

    # 正常样本（校准阈值用）
    X_normal, _, _ = build_sequences(normal_split, SEQ_LEN, stride=5, normal_only=True, normal_frac=0.4, norm_stats=norm_stats)
    X_train, y_train, _ = build_sequences(train_split, SEQ_LEN, stride=1, norm_stats=norm_stats)
    X_val, y_val, _ = build_sequences(val_split, SEQ_LEN, stride=5, norm_stats=norm_stats)

    print(f"[DATA] Normal: {len(X_normal)}, Train: {len(X_train)}, Val: {len(X_val)}")

    normal_ds = FaultSenseDataset(X_normal, np.zeros(len(X_normal)), [])
    train_ds = FaultSenseDataset(X_train, y_train, [])
    val_ds = FaultSenseDataset(X_val, y_val, [])

    normal_loader = DataLoader(normal_ds, min(256, len(normal_ds)), shuffle=False)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, min(256, len(val_ds)), shuffle=False)

    # ── 模型 ──
    model = FaultSenseModel(n_features=N_FEATURES, hidden=args.hidden).to(device)
    print(f"[MODEL] Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    best_val_rmse = float('inf')
    best_epoch = 0
    patience = 20
    no_improve = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, ae_loss, rul_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_rmse, val_mae, val_nasa, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            no_improve = 0
            output_dir = Path(args.output) / args.subset
            output_dir.mkdir(parents=True, exist_ok=True)
            # 校准异常阈值
            threshold = model.calibrate_threshold(normal_loader)
            torch.save({
                "model_state_dict": model.state_dict(),
                "hidden": args.hidden,
                "n_features": N_FEATURES,
                "seq_len": SEQ_LEN,
                "rul_cap": RUL_CAP,
                "threshold": threshold,
                "sensor_indices": USEFUL_SENSOR_INDICES,
                "val_rmse": float(val_rmse),
            }, output_dir / "best_model.pth")
        else:
            no_improve += 1

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"loss={train_loss:.4f} (ae={ae_loss:.4f} rul={rul_loss:.4f}) | "
              f"val_rmse={val_rmse:.1f} val_nasa={val_nasa:.1f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | "
              f"time={elapsed:.1f}s" + (" *" if epoch == best_epoch else ""))

        if no_improve >= patience:
            print(f"[INFO] Early stopping at epoch {epoch+1}")
            break

    # ── 最终评估 ──
    print(f"\n[FINAL] Best epoch: {best_epoch+1}, loading checkpoint...")
    checkpoint = torch.load(Path(args.output) / args.subset / "best_model.pth",
                           map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 测试集RUL预测
    predictions = []
    for uid, data_arr in sorted(test_units.items()):
        cycles = data_arr[:, 0]
        mean_np = norm_stats[0].flatten() if isinstance(norm_stats[0], np.ndarray) else np.array(norm_stats[0]).flatten()
        std_np = norm_stats[1].flatten() if isinstance(norm_stats[1], np.ndarray) else np.array(norm_stats[1]).flatten()
        features = (data_arr[:, 1 + np.array(USEFUL_SENSOR_INDICES)] - mean_np) / std_np
        features = features.astype(np.float32)
        if len(cycles) >= SEQ_LEN:
            seq = features[-SEQ_LEN:]
        else:
            seq = np.pad(features, ((SEQ_LEN - len(features), 0), (0, 0)), mode='edge')
        x = torch.tensor(seq).float().unsqueeze(0).to(device)
        _, rul_pred = model(x)
        predictions.append(max(0, rul_pred.item() * RUL_CAP))

    predictions = np.array(predictions)
    rmse = np.sqrt(np.mean((predictions - test_rul) ** 2))
    mae = np.mean(np.abs(predictions - test_rul))
    d = predictions - test_rul
    nasa = np.mean(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))

    print(f"  Test RMSE: {rmse:.1f} cycles")
    print(f"  Test MAE:  {mae:.1f} cycles")
    print(f"  Test NASA Score: {nasa:.0f}")
    print(f"  Threshold: {float(model.threshold):.6f}")

    # ── 产物 ──
    output_dir = Path(args.output) / args.subset
    config = {
        "model_id": f"faultsense_lstm_ae_{args.subset.lower()}",
        "subset": args.subset, "seq_len": SEQ_LEN, "rul_cap": RUL_CAP,
        "n_features": N_FEATURES, "hidden": args.hidden,
        "test_rmse": round(float(rmse), 1),
        "test_nasa": round(float(nasa), 0),
        "threshold": round(float(model.threshold), 6),
        "val_rmse": round(float(best_val_rmse), 1),
        "best_epoch": best_epoch + 1,
    }
    with open(output_dir / "config.yaml", "w") as f:
        import yaml; yaml.dump(config, f)
    with open(output_dir / "preprocessing.json", "w") as f:
        json.dump({
            "sensor_indices": USEFUL_SENSOR_INDICES,
            "seq_len": SEQ_LEN, "rul_cap": RUL_CAP,
            "normal_frac": 0.4, "k_sigma": 2.5,
        }, f, indent=2)

    sha = hashlib.sha256((output_dir / "best_model.pth").read_bytes()).hexdigest()
    with open(output_dir / "weight.sha256", "w") as f:
        f.write(f"{sha}  best_model.pth\n")
    with open(output_dir / "model_card.md", "w") as f:
        f.write(f"""# FaultSense LSTM-Autoencoder — C-MAPSS {args.subset}

- **Task:** Anomaly detection + RUL prediction (LSTM Autoencoder)
- **Data:** NASA C-MAPSS {args.subset}
- **Architecture:** 2-layer LSTM Encoder (hidden={args.hidden}) + 1-layer Decoder + RUL MLP
- **Input:** {SEQ_LEN}×{N_FEATURES} sensor sequence
- **Anomaly:** Reconstruction error threshold (k-sigma, k=2.5, calibrated on early normal cycles)
- **RUL Cap:** {RUL_CAP}
- **Test RMSE:** {rmse:.1f}, **NASA:** {nasa:.0f}
- **Status:** experimental — validated_public_domain (C-MAPSS only)
""")

    print(f"\n[DONE] Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
