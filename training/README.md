# Aero-Engine Damage Detection — Model Training

按照《航空发动机损伤检测系统：模型训练数据与自动化实施指导书》执行。

## 环境

- GPU: NVIDIA GeForce RTX 4060 Laptop, 8 GB VRAM
- CUDA: 12.4
- PyTorch: 2.6.0+cu124
- TensorFlow: NOT AVAILABLE (Python 3.13 incompatibility)
  - FaultSense upstream is PyTorch, not TF — will retrain with .pt weights

## 训练顺序

Phase 1（8GB VRAM 可行）:
1. WCamba — CWRU 轴承故障分类 (2-4 GB)
2. CNN-LSTM RUL — C-MAPSS 寿命预测 (2-4 GB)
3. FaultSense — C-MAPSS 异常检测 (2-4 GB, PyTorch版)
4. SLF-YOLO — BladeSynth 缺陷检测 (8 GB, batch=4)
5. EGCIENet — AEBIS 缺陷分割 (8 GB, batch=2)

Phase 2（需更多VRAM或轻量配置）:
6. PatchCore — AeBAD 异常检测 (8-16 GB)
7. SAM-Adapter — 裂纹分割 (12-16 GB, ❌ 8GB不够)

## 目录结构

```
training/
├── configs/        YAML配置
├── scripts/        下载/转换/训练/评估/导出脚本
├── src/            训练库代码
├── tests/          数据/模型/集成测试
└── reports/        训练报告

artifacts/
├── raw/            原始下载（只读）
├── canonical/      统一格式
├── splits/         划分清单
└── models/         训练好的权重
```
