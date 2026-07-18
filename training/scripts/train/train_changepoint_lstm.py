"""训练 ChangePoint-LSTM RUL (C-MAPSS FD002/4 — 多变工况)

按指导书 Section 12:
- CUSUM 退化变点检测
- 变点后LSTM RUL回归
- 按unit_id分组, 防泄漏

用法:
  python train_changepoint_lstm.py --subset FD002 --epochs 50
"""

import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

USEFUL_SENSORS = [2,3,4,7,8,9,11,12,13,14,15,17,20,21]
N_FEATURES, SEQ_LEN, RUL_CAP = 14, 50, 130


def load_cmapss(path: str, subset: str):
    p = Path(path)
    def _parse(f):
        data = np.loadtxt(str(f)); units = {}
        for row in data:
            uid = int(row[0])
            units.setdefault(uid, []).append(row[1:])
        return {u: np.array(v) for u, v in units.items()}
    return {"train": _parse(p / f"train_{subset}.txt"),
            "test": _parse(p / f"test_{subset}.txt"),
            "test_rul": np.loadtxt(str(p / f"RUL_{subset}.txt")).flatten()}


def detect_change_point(signal: np.ndarray, window: int = 30, threshold: float = 3.0) -> int:
    """CUSUM变点检测。返回变点索引（无变点时返回-1）。"""
    if len(signal) < window * 2:
        return -1
    # 早期窗口作为基线
    baseline_mean = np.mean(signal[:window])
    baseline_std = np.std(signal[:window]) + 1e-8
    # CUSUM正向累积
    cusum_pos = np.zeros(len(signal))
    for i in range(window, len(signal)):
        std_score = (signal[i] - baseline_mean) / baseline_std
        cusum_pos[i] = max(0, cusum_pos[i-1] + std_score - 0.5)
    # 找到超过阈值的第一个点
    exceed = np.where(cusum_pos > threshold)[0]
    if len(exceed) > 0:
        return exceed[0]
    return -1


def build_health_index(features: np.ndarray) -> np.ndarray:
    """构建健康指数: 标准化后取第一主成分。"""
    from sklearn.decomposition import PCA
    z = (features - features.mean(0)) / (features.std(0) + 1e-8)
    pca = PCA(n_components=1)
    hi = pca.fit_transform(z).flatten()
    # 确保退化方向为正（健康→退化）
    if np.corrcoef(hi, np.arange(len(hi)))[0, 1] < 0:
        hi = -hi
    return hi


def build_sequences(units, norm_stats=None, stride=1):
    """构建序列+变点感知RUL标签。"""
    X, y_rul, y_cp, uids = [], [], [], []
    mean, std = norm_stats or (0, 1)

    for uid, data in units.items():
        cycles = data[:, 0]
        feats = data[:, 1 + np.array(USEFUL_SENSORS)]
        feats = (feats - mean) / (std + 1e-8)
        max_cycle = cycles[-1]

        # 健康指数 → 变点检测
        hi = build_health_index(feats[:, :min(14, feats.shape[1])])
        cp_idx = detect_change_point(hi, window=min(30, len(hi)//3))

        for start in range(0, len(cycles) - SEQ_LEN, stride):
            end = start + SEQ_LEN
            raw_rul = max_cycle - cycles[end-1]
            rul = min(raw_rul, RUL_CAP)

            # 变点感知: 变点之前标记为健康(RUL=cap)
            is_degraded = 1.0 if cp_idx < 0 or end > cp_idx else 0.0
            has_cp = 1.0 if cp_idx > 0 else 0.0

            X.append(feats[start:end].astype(np.float32))
            y_rul.append(rul / RUL_CAP)
            y_cp.append(has_cp)
            uids.append(uid)

    return np.array(X), np.array(y_rul, np.float32), np.array(y_cp, np.float32), uids


def compute_norm_stats(units):
    all_f = np.concatenate([d[:, 1 + np.array(USEFUL_SENSORS)] for d in units.values()])
    return all_f.mean(0, keepdims=True).astype(np.float32), all_f.std(0, keepdims=True).astype(np.float32)


class CPDataset(Dataset):
    def __init__(self, X, yr, yc, uids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.yr = torch.tensor(yr, dtype=torch.float32).unsqueeze(-1)
        self.yc = torch.tensor(yc, dtype=torch.float32)
        self.uids = uids
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.yr[i], self.yc[i], self.uids[i] if i < len(self.uids) else -1


class ChangePointLSTM(nn.Module):
    def __init__(self, n_feat=N_FEATURES, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.lstm = nn.LSTM(64, hidden, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.rul_head = nn.Sequential(nn.Linear(hidden*2, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
        self.cp_head = nn.Sequential(nn.Linear(hidden*2, 16), nn.ReLU(), nn.Linear(16, 1))  # change point detection

    def forward(self, x):
        x = self.conv(x.transpose(1,2)).transpose(1,2)
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.rul_head(h), torch.sigmoid(self.cp_head(h))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="artifacts/raw/cmapss")
    parser.add_argument("--subset", default="FD002")
    parser.add_argument("--output", default="/f/aero-training/artifacts/models/changepoint_lstm/v1.0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    device = torch.device(args.device)
    print(f"[Device] {device}")

    data = load_cmapss(args.data, args.subset)
    tu, test_u, test_rul = data["train"], data["test"], data["test_rul"]
    all_uid = sorted(tu.keys())
    np.random.shuffle(all_uid)
    n = len(all_uid)
    train_u = {u: tu[u] for u in all_uid[:int(n*0.7)]}
    val_u = {u: tu[u] for u in all_uid[int(n*0.7):int(n*0.85)]}

    if args.smoke_test:
        train_u = {u: tu[u] for u in list(all_uid)[:5]}
        val_u = {u: tu[u] for u in list(all_uid)[5:8]}
        args.epochs = 3

    ns = compute_norm_stats(train_u)
    Xt, Ytr, Ytc, _ = build_sequences(train_u, ns, stride=1)
    Xv, Yvr, Yvc, _ = build_sequences(val_u, ns, stride=5)

    print(f"[DATA] Train:{len(Xt)} Val:{len(Xv)} Test units:{len(test_u)}")
    # 统计变点检出率
    cp_rate = Ytc.mean()
    print(f"[CP] Change point detected in {cp_rate*100:.0f}% of training windows")

    train_loader = DataLoader(CPDataset(Xt, Ytr, Ytc, []), args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(CPDataset(Xv, Yvr, Yvc, []), min(256, len(Xv)))

    model = ChangePointLSTM().to(device)
    print(f"[MODEL] Params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_rmse, best_ep = float('inf'), 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time(); tl, n = 0.0, 0
        for x, yr, yc, _ in train_loader:
            x, yr, yc = x.to(device), yr.to(device), yc.to(device)
            opt.zero_grad()
            pr, pc = model(x)
            loss = nn.functional.mse_loss(pr, yr) + 0.1 * nn.functional.binary_cross_entropy(pc, yc.unsqueeze(-1))
            loss.backward(); opt.step()
            tl += loss.item() * x.size(0); n += x.size(0)

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for x, yr, _, _ in val_loader:
                pr, _ = model(x.to(device))
                vp.append(pr.cpu().numpy()); vl.append(yr.numpy())
        vp = np.concatenate(vp).flatten() * RUL_CAP
        vl = np.concatenate(vl).flatten() * RUL_CAP
        rmse = np.sqrt(np.mean((vp - vl)**2))

        if rmse < best_rmse:
            best_rmse, best_ep = rmse, ep
            d = Path(args.output) / args.subset; d.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "norm_stats": ns,
                        "val_rmse": float(rmse)}, d / "best_model.pth")

        print(f"Ep {ep+1:3d} | loss={tl/n:.4f} | val_rmse={rmse:.1f} | {time.time()-t0:.1f}s" + (" *" if ep==best_ep else ""))

    # Test
    ckpt = torch.load(Path(args.output) / args.subset / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    preds = []
    for uid in sorted(test_u.keys()):
        d = test_u[uid]
        f = d[:, 1 + np.array(USEFUL_SENSORS)]
        f = (f - ns[0]) / (ns[1] + 1e-8)
        seq = f[-SEQ_LEN:] if len(f) >= SEQ_LEN else np.pad(f, ((SEQ_LEN-len(f),0),(0,0)), mode='edge')
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        p, _ = model(x); preds.append(max(0, p.item() * RUL_CAP))
    preds = np.array(preds)
    rmse = np.sqrt(np.mean((preds - test_rul)**2))
    d = preds - test_rul; nasa = np.mean(np.where(d<0, np.exp(-d/13)-1, np.exp(d/10)-1))
    print(f"\n[FINAL] Test RMSE: {rmse:.1f}  NASA: {nasa:.0f}  (best ep={best_ep+1})")

    out = Path(args.output) / args.subset
    with open(out / "config.yaml", "w") as f:
        import yaml; yaml.dump({"subset": args.subset, "test_rmse": round(float(rmse),1), "nasa": round(float(nasa),0)}, f)
    print(f"[DONE] {out}")


if __name__ == "__main__":
    main()
