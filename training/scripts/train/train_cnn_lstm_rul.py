"""训练 CNN-LSTM RUL 预测模型 (C-MAPSS FD001-004)

按照指导书 Section 11 执行:
- 架构: Conv1d→BN→ReLU→Conv1d→ReLU→LSTM→MLP→RUL
- 输入: [batch, seq_len=50, num_features]
- RUL标签: piecewise linear, cap=130
- 划分: 按unit_id分组, 70/15/15
- 输出: RMSE, MAE, NASA asymmetric score

用法:
  python train_cnn_lstm_rul.py --subset FD001 --epochs 50
  python train_cnn_lstm_rul.py --subset FD001 --smoke-test
"""

import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════════
# C-MAPSS 数据加载器
# ═══════════════════════════════════════════════════════════════════════

SENSOR_COLS = [
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21",
]
# 通常移除近常数列: s1, s5, s6, s10, s16, s18, s19 (保留14个有意义的传感器)
USEFUL_SENSOR_INDICES = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]  # 0-based in sensors
N_FEATURES = len(USEFUL_SENSOR_INDICES)


def load_cmapss(data_dir: str, subset: str) -> dict:
    """加载C-MAPSS子集的所有轨迹。

    Returns:
        {unit_id: np.array(cycles, features)}
    """
    data_dir = Path(data_dir)
    train_file = data_dir / f"train_{subset}.txt"
    test_file = data_dir / f"test_{subset}.txt"
    rul_file = data_dir / f"RUL_{subset}.txt"

    if not train_file.exists():
        raise FileNotFoundError(f"{train_file} not found. Clone from github.com/edwardzjl/CMAPSSData")

    def _parse(path):
        data = np.loadtxt(str(path))
        units = {}
        for row in data:
            uid = int(row[0])
            if uid not in units:
                units[uid] = []
            units[uid].append(row[1:])  # cycle + op3 + s1-s21
        return {uid: np.array(v) for uid, v in units.items()}

    train_units = _parse(train_file)
    test_units = _parse(test_file)
    test_rul = np.loadtxt(str(rul_file)).flatten()

    return {"train": train_units, "test": test_units, "test_rul": test_rul}


def build_sequences(units: dict, seq_len: int = 50, stride: int = 1,
                    rul_cap: int = 130) -> tuple[np.ndarray, np.ndarray, list]:
    """从原始轨迹构建(seq, label)对。"""
    X, y, uids = [], [], []

    for uid, data in units.items():
        cycles = data[:, 0]
        features = data[:, 1 + np.array(USEFUL_SENSOR_INDICES)]  # skip cycle + op3
        max_cycle = cycles[-1]

        for start in range(0, len(cycles) - seq_len, stride):
            end = start + seq_len
            seq = features[start:end].astype(np.float32)
            raw_rul = max_cycle - cycles[end - 1]
            rul = min(raw_rul, rul_cap)  # piecewise linear cap
            X.append(seq)
            y.append(rul / rul_cap)  # normalize to [0, 1]
            uids.append(uid)

    return np.array(X), np.array(y, dtype=np.float32), uids


class CMAPSSDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, uids: list,
                 seq_len: int = 50, n_features: int = N_FEATURES):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        self.uids = uids

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.uids[idx]


# ═══════════════════════════════════════════════════════════════════════
# CNN-LSTM 模型
# ═══════════════════════════════════════════════════════════════════════

class CNNLSTMRUL(nn.Module):
    """Conv1d + LSTM RUL 预测器。"""
    def __init__(self, n_features=N_FEATURES, seq_len=50,
                 conv_channels=32, lstm_hidden=64, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, conv_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_channels * 2)
        self.lstm = nn.LSTM(conv_channels * 2, lstm_hidden, num_layers=2,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, seq_len, n_features) → transpose for Conv1d
        x = x.transpose(1, 2)  # (B, n_features, seq_len)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)  # (B, seq_len, conv_channels*2)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # 取最后时间步
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════
# 训练与评估
# ═══════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, rul_cap=130):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    preds = np.concatenate(all_preds).flatten() * rul_cap
    labels = np.concatenate(all_labels).flatten() * rul_cap

    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    mae = np.mean(np.abs(preds - labels))
    # NASA asymmetric score: penalizes late predictions more
    d = preds - labels
    nasa = np.mean(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))

    return total_loss / n, rmse, mae, nasa, preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/raw/cmapss")
    parser.add_argument("--subset", default="FD001", choices=["FD001","FD002","FD003","FD004"])
    parser.add_argument("--output", default="artifacts/models/cnn_lstm_rul/v1.0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--rul-cap", type=int, default=130)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory // 1024**3} GB VRAM")

    # ── 加载数据 ──
    print(f"[INFO] Loading C-MAPSS {args.subset}...")
    data = load_cmapss(args.data, args.subset)
    train_units = data["train"]
    test_units = data["test"]
    test_rul = data["test_rul"]

    print(f"[INFO] Train units: {len(train_units)}, Test units: {len(test_units)}")

    # 按unit_id划分 train/val
    all_uids = sorted(train_units.keys())
    np.random.shuffle(all_uids)
    n = len(all_uids)
    train_uids = {uid: train_units[uid] for uid in all_uids[:int(n * 0.7)]}
    val_uids = {uid: train_units[uid] for uid in all_uids[int(n * 0.7):int(n * 0.85)]}

    if args.smoke_test:
        train_uids = {uid: train_units[uid] for uid in list(all_uids)[:5]}
        val_uids = {uid: train_units[uid] for uid in list(all_uids)[5:10]}
        args.epochs = 3

    X_train, y_train, uids_train = build_sequences(train_uids, args.seq_len, stride=1, rul_cap=args.rul_cap)
    X_val, y_val, uids_val = build_sequences(val_uids, args.seq_len, stride=5, rul_cap=args.rul_cap)

    print(f"[DATA] Train seqs: {len(X_train)}, Val seqs: {len(X_val)}")

    train_ds = CMAPSSDataset(X_train, y_train, uids_train, args.seq_len)
    val_ds = CMAPSSDataset(X_val, y_val, uids_val, args.seq_len)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0)

    # ── 模型 ──
    model = CNNLSTMRUL(n_features=N_FEATURES, seq_len=args.seq_len).to(device)
    print(f"[MODEL] Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_rmse, val_mae, val_nasa, _, _ = evaluate(
            model, val_loader, criterion, device, args.rul_cap)
        scheduler.step(val_loss)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            output_dir = Path(args.output) / args.subset
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "seq_len": args.seq_len,
                "n_features": N_FEATURES,
                "rul_cap": args.rul_cap,
                "sensor_indices": USEFUL_SENSOR_INDICES,
                "val_rmse": float(val_rmse),
                "val_nasa": float(val_nasa),
            }, output_dir / "best_model.pth")

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_rmse={val_rmse:.1f} val_mae={val_mae:.1f} val_nasa={val_nasa:.1f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | "
              f"time={elapsed:.1f}s" + (" *" if epoch == best_epoch else ""))

    # ── 最终评估 (测试集) ──
    print(f"\n[FINAL EVALUATION] Best epoch: {best_epoch+1}")

    checkpoint = torch.load(Path(args.output) / args.subset / "best_model.pth",
                           map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 测试集: 用每个engine的最后一个窗口预测RUL
    print(f"[INFO] Predicting RUL for {len(test_units)} test engines...")
    predictions = []
    for i, (uid, data_arr) in enumerate(sorted(test_units.items())):
        cycles = data_arr[:, 0]
        features = data_arr[:, 1 + np.array(USEFUL_SENSOR_INDICES)].astype(np.float32)
        # 取最后seq_len个时间步
        if len(cycles) >= args.seq_len:
            seq = features[-args.seq_len:]
        else:
            seq = np.pad(features, ((args.seq_len - len(features), 0), (0, 0)), mode='edge')
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(x).item() * args.rul_cap
        predictions.append(max(0, pred))

    predictions = np.array(predictions)
    rmse = np.sqrt(np.mean((predictions - test_rul) ** 2))
    mae = np.mean(np.abs(predictions - test_rul))
    d = predictions - test_rul
    nasa = np.mean(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))

    print(f"  Test RMSE: {rmse:.1f} cycles")
    print(f"  Test MAE:  {mae:.1f} cycles")
    print(f"  Test NASA Score: {nasa:.0f}")

    # ── 导出产物 ──
    output_dir = Path(args.output) / args.subset
    config = {
        "model_id": f"cnn_lstm_rul_cmapss_{args.subset.lower()}",
        "subset": args.subset,
        "seq_len": args.seq_len,
        "rul_cap": args.rul_cap,
        "n_features": N_FEATURES,
        "sensor_indices": USEFUL_SENSOR_INDICES,
        "test_rmse": round(float(rmse), 1),
        "test_mae": round(float(mae), 1),
        "test_nasa_score": round(float(nasa), 0),
        "val_rmse": round(float(best_val_rmse), 1),
        "best_epoch": best_epoch + 1,
        "pytorch_version": torch.__version__,
        "device": str(device),
    }
    with open(output_dir / "config.yaml", "w") as f:
        import yaml; yaml.dump(config, f)

    with open(output_dir / "preprocessing.json", "w") as f:
        json.dump({
            "sensor_indices": USEFUL_SENSOR_INDICES,
            "sensor_cols": SENSOR_COLS,
            "seq_len": args.seq_len,
            "rul_cap": args.rul_cap,
            "normalization": "none (raw sensor values, no z-score applied)",
            "note": "C-MAPSS sensors are anonymous; do NOT map to real physical quantities",
        }, f, indent=2)

    sha = hashlib.sha256((output_dir / "best_model.pth").read_bytes()).hexdigest()
    with open(output_dir / "weight.sha256", "w") as f:
        f.write(f"{sha}  best_model.pth\n")

    with open(output_dir / "model_card.md", "w") as f:
        f.write(f"""# CNN-LSTM RUL Predictor — C-MAPSS {args.subset}

- **Task:** Remaining Useful Life (RUL) prediction for turbofan engines
- **Data:** NASA C-MAPSS {args.subset}
- **Architecture:** Conv1d(32→64) + BiLSTM(64×2) + MLP(32→1)
- **Input:** {args.seq_len}×{N_FEATURES} sensor sequence
- **RUL Cap:** {args.rul_cap} cycles (piecewise linear)
- **Test RMSE:** {rmse:.1f} cycles
- **Test NASA Score:** {nasa:.0f}
- **Weight SHA256:** {sha}
- **Limitations:**
  - C-MAPSS is SIMULATED data — NOT real engine telemetry
  - Anonymous sensors (s1-s21) — do NOT map to physical quantities
  - Only FD001 has C=1 operating condition; others have C=6
  - Uncertainty: ensemble of 5 seeds recommended for production
- **Status:** experimental — validated_public_domain (C-MAPSS only)
""")

    print(f"\n[DONE] Model saved to: {output_dir}")
    print(f"  best_model.pth")
    print(f"  RMSE={rmse:.1f}, NASA={nasa:.0f}")
    print(f"  SHA256: {sha}")


if __name__ == "__main__":
    main()
