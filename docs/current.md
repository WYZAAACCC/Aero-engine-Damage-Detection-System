# 模型训练状态 — 2026-07-19 最新

## Phase 1（6个）

| 模型 | 状态 | 说明 |
|------|:---:|------|
| **WCamba** 轴承故障 | ✅ | CWRU 100% 准确率，已接入适配器 |
| **CNN-LSTM** RUL | ✅ | FD001 RMSE=15.0，已接入适配器 |
| **FaultSense** 异常+RUL | ✅ | FD001 RMSE=16.1，PyTorch移植完成，已接入适配器 |
| **CA²/PatchCore** 异常检测 | ✅ | AeBAD AUROC=0.60，PatchCore+Memory Bank(2552特征)，已接入适配器 |
| **UNet/EGCIENet** 叶片分割 | ✅ | AEBIS Dice=0.79，UNet+ResNet18替代SegFormer，已接入适配器 |
| **SLF-YOLO** 缺陷检测 | ⏳ | BladeSynth 25.7GB 下载中，训练脚本已就绪 |

## Phase 2（5个）

| 模型 | 状态 | 原因 |
|------|:---:|------|
| **ChangePoint-LSTM** | ✅ | FD002 RMSE=31.7，已接入适配器 |
| SAM-Adapter | ❌ | 8GB VRAM 不够（需≥12GB） |
| TS-SAM | ❌ | 8GB VRAM 远远不够（需≥24GB） |
| Paderborn | ❌ | 数据需大学官网申请 |
| Isolation Forest | ❌ | 需真实发动机遥测数据 |

## Phase 3（2个）

| 模型 | 状态 |
|------|:---:|
| PINN | ⏸️ 需N-CMAPSS+物理参数 |
| Paris/S-N | ⏸️ 需材料试验数据 |

## 总结

```
已训练+已接入: 6/13 (WCamba, CNN-LSTM, FaultSense, PatchCore, UNet, ChangePoint)
下载中:        1/13 (SLF-YOLO/BladeSynth)
VRAM不足:       2/13 (SAM/TS-SAM)
缺外部数据:     2/13 (Paderborn, Isolation Forest)
延后:           2/13 (PINN, Paris/S-N)
```
