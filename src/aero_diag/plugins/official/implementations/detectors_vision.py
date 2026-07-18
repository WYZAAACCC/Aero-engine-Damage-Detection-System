"""视觉检测器 — CA²/SLF-YOLO/SAM-Adapter/EGCIENet/TS-SAM

架构: 每个检测器 _load_model() → run() 先尝试完整模型 → 自动回退基线
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from ._base import AssetRunResult, ImplementationBase

logger = logging.getLogger("aero_diag.detectors.vision")

# ── 通用工具函数 ──────────────────────────────────────────────────────

def _ensure_rgb(img_arr: np.ndarray) -> np.ndarray:
    """确保图像为 RGB uint8 (H,W,3)。"""
    if img_arr.dtype != np.uint8:
        if img_arr.max() <= 1.0:
            img_arr = (img_arr * 255).astype(np.uint8)
        else:
            img_arr = img_arr.astype(np.uint8)
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    elif img_arr.ndim == 3 and img_arr.shape[2] == 1:
        img_arr = np.repeat(img_arr, 3, axis=2)
    return img_arr


def _resolve_repo_path(env_var: str, default_dir: str) -> Path | None:
    """查找克隆的仓库路径：环境变量 > src同级 > 用户目录。"""
    if env_var in os.environ:
        p = Path(os.environ[env_var])
        if p.exists():
            return p
    candidates = [
        Path.cwd() / default_dir,
        Path.cwd().parent / default_dir,
        Path.home() / default_dir,
        Path.home() / "repos" / default_dir,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ═══════════════════════════════════════════════════════════════════════
# CA² 无监督异常检测 — 完整模型：ResNet + KNN | 基线：统计
# ═══════════════════════════════════════════════════════════════════════

class CA2AnomalyDetector(ImplementationBase):
    """CA² 无监督孔探叶片异常检测。

    完整模型路径：git clone https://github.com/changniu54/CA2 → 设置 CA2_REPO_PATH
    加载 ResNet-50 特征提取器 + KNN 异常打分。

    基线：图像统计分析（均值/方差/暗区比/亮区比）。
    """

    asset_id = "detector.borescope.ca2_anomaly"

    def __init__(self):
        self._model = None
        self._model_loaded = False
        self._model_error = ""

    def _load_model(self, params: dict) -> bool:
        """尝试加载 CA² 完整模型。先试 CA² 仓库 → 回退 torchvision ResNet。"""
        if self._model_loaded:
            return True

        # 方案A：CA² 仓库中的自定义模型
        repo = _resolve_repo_path("CA2_REPO_PATH", "CA2")
        if repo is not None:
            sys.path.insert(0, str(repo))
            try:
                import torch, torchvision
                self._model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
                self._model.fc = torch.nn.Identity()
                self._model.eval()
                device = "cuda" if torch.cuda.is_available() and params.get("use_gpu", True) else "cpu"
                self._model.to(device)
                self._device = device
                self._model_loaded = True
                return True
            except Exception as e:
                self._model_error = str(e)
            finally:
                if str(repo) in sys.path:
                    sys.path.remove(str(repo))

        # 方案B：独立 torchvision ResNet-50（不需要 CA² 仓库）
        try:
            import torch, torchvision
            self._model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
            self._model.fc = torch.nn.Identity()
            self._model.eval()
            device = "cuda" if torch.cuda.is_available() and params.get("use_gpu", True) else "cpu"
            self._model.to(device)
            self._device = device
            self._model_loaded = True
            self._model_type = "resnet50_standalone"
            return True
        except Exception as e:
            self._model_error = str(e)
            return False

    def _extract_features(self, img: np.ndarray) -> np.ndarray | None:
        """用 ResNet-50 提取 2048 维特征向量。"""
        try:
            import torch
            from torchvision import transforms

            t = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensor = t(img).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feat = self._model(tensor).cpu().numpy().flatten()
            return feat
        except Exception:
            return None

    def _baseline(self, gray: np.ndarray) -> dict:
        """统计基线异常检测。"""
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        dark_ratio = float(np.sum(gray < 30) / gray.size)
        bright_ratio = float(np.sum(gray > 240) / gray.size)
        is_abnormal = dark_ratio > 0.15 or bright_ratio > 0.3 or std_val < 5
        score = max(dark_ratio * 5, bright_ratio * 3, max(0, (5 - min(std_val, 5)) / 5))
        return {
            "anomaly_detected": bool(is_abnormal),
            "anomaly_score": round(score, 4),
            "statistics": {"mean": round(mean_val, 2), "std": round(std_val, 2),
                           "dark_ratio": round(dark_ratio, 4), "bright_ratio": round(bright_ratio, 4)},
            "method": "statistical_baseline",
        }

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        if not inputs:
            return {"ok": False, "issues": ["No image data"]}
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        data = inputs[0] if inputs else {}
        img = data.get("image", data.get("data", data.get("image_uri")))

        # 预处理图像
        try:
            if isinstance(img, str) and Path(img).exists():
                import cv2
                img_arr = cv2.imread(str(img))
            else:
                img_arr = np.asarray(img, dtype=np.float64)
            img_arr = _ensure_rgb(img_arr) if img_arr.ndim == 3 else img_arr.astype(np.float64)
            gray = np.mean(img_arr, axis=-1) if img_arr.ndim == 3 else img_arr
        except Exception:
            return AssetRunResult(
                execution_status="failed", validity_status="invalid",
                structured_output={}, error="Cannot read image", metrics={},
            )

        # 完整模型推理 — 仅当 CA² 仓库 + 权重均可用
        full_model = self._load_model(params)
        if full_model:
            feat = self._extract_features(img_arr) if img_arr.ndim == 3 else None
            if feat is not None:
                # 审计修复: feat - feat = 0 是明确的逻辑错误。
                # 无正常样本特征库时，ResNet 特征无法做 KNN 异常检测。
                # 返回 unavailable 而非伪造的 anomaly score。
                return AssetRunResult.unavailable(
                    reason="CA² requires a normal-feature reference library for KNN anomaly scoring. "
                           "ResNet-50 feature extractor loaded but no reference features available. "
                           "Provide a reference dataset of normal borescope images to enable anomaly detection.",
                    model_identity="resnet50_no_reference",
                    reason_code="NO_REFERENCE_FEATURE_LIBRARY",
                )

        # 基线回退 — 明确标记为 degraded
        result = self._baseline(gray)
        return AssetRunResult.degraded(
            method="statistical_baseline",
            reason=f"CA² model unavailable: {self._model_error or 'pip install torch torchvision for ResNet-50'}",
            output=result,
            metrics={"anomaly_score": result["anomaly_score"]},
        )

    @staticmethod
    def default_params() -> dict:
        return {"feature_extractor": "resnet50", "k_neighbors": 5, "threshold_percentile": 95, "use_gpu": True}


# ═══════════════════════════════════════════════════════════════════════
# SLF-YOLO 金属表面缺陷检测 — 完整：YOLOv8 推理 | 基线：Sobel梯度
# ═══════════════════════════════════════════════════════════════════════

class SLFYOLODetector(ImplementationBase):
    """SLF-YOLO 金属表面缺陷检测 (YOLOv8增强)。

    完整模型: git clone https://github.com/zacianfans/SLF-YOLO
    加载 YOLOv8/slf-yolo 权重 → 边界框推理。

    基线: Sobel 梯度异常检测。
    """

    asset_id = "detector.surface.slf_yolo_metal_defect"

    def __init__(self):
        self._model = None
        self._model_loaded = False
        self._model_error = ""
        self._device = "cpu"

    def _load_model(self, params: dict) -> bool:
        """加载 SLF-YOLO 领域权重。拒绝通用 COCO 模型冒充。

        审计修复 (AER-008): 通用 YOLO 不得冒充领域模型。
        只有加载了 SLF-YOLO 专用权重（NEU-DET/GC10-DET 缺陷检测）才算 loaded。
        """
        if self._model_loaded:
            return True

        # 仅从 SLF-YOLO 仓库加载领域权重
        repo = _resolve_repo_path("SLF_YOLO_REPO_PATH", "SLF-YOLO")
        if repo:
            sys.path.insert(0, str(repo))
            try:
                from ultralytics import YOLO
                weights = repo / "weights" / "best.pt"
                if weights.exists():
                    self._model = YOLO(str(weights))
                    self._device = "cuda" if params.get("use_gpu", True) else "cpu"
                    self._model_loaded = True
                    self._model_type = "slf_yolo_domain"
                    return True
                else:
                    self._model_error = (
                        "SLF-YOLO repo found but no best.pt weights. "
                        "Train on NEU-DET/GC10-DET metal surface defect dataset first."
                    )
            except ImportError:
                self._model_error = "pip install ultralytics + SLF-YOLO weights required"
            except Exception as e:
                self._model_error = f"SLF-YOLO load failed: {e}"
            finally:
                if str(repo) in sys.path:
                    sys.path.remove(str(repo))
        else:
            self._model_error = (
                "SLF-YOLO domain weights not found. "
                "git clone https://github.com/zacianfans/SLF-YOLO and train on metal surface defect dataset. "
                "Generic COCO-pretrained YOLO is NOT a substitute for domain defect detection."
            )

        return False

    def _baseline(self, gray: np.ndarray) -> list[dict]:
        """Sobel 梯度基线。"""
        from scipy import ndimage
        gx = ndimage.sobel(gray, axis=0)
        gy = ndimage.sobel(gray, axis=1)
        edge_mag = np.sqrt(gx**2 + gy**2)
        high_edge_ratio = float(np.sum(edge_mag > np.percentile(edge_mag, 95)) / edge_mag.size)
        if high_edge_ratio > 0.02:
            return [{"type": "surface_anomaly", "confidence": min(high_edge_ratio * 10, 0.9),
                     "bbox": [0, 0, gray.shape[1], gray.shape[0]]}]
        return []

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        data = inputs[0] if inputs else {}
        img = data.get("image", data.get("data"))

        try:
            img_arr = np.asarray(img) if not isinstance(img, str) else None

            # 完整模型 — 仅当 SLF-YOLO 领域权重加载成功
            if self._load_model(params) and img_arr is not None:
                rgb = _ensure_rgb(img_arr) if img_arr.ndim in (2, 3) else np.stack([np.mean(img_arr, axis=-1)]*3, axis=-1)
                try:
                    results = self._model(rgb, conf=params.get("confidence_threshold", 0.25),
                                          device=self._device, verbose=False)
                    defects = []
                    for r in results:
                        for box in r.boxes:
                            defects.append({
                                "type": self._model.names[int(box.cls[0])] if hasattr(self._model, 'names') else "defect",
                                "confidence": round(float(box.conf[0]), 4),
                                "bbox": box.xyxy[0].tolist(),
                            })
                    return AssetRunResult.valid_success(
                        output={"defects_found": len(defects), "defects": defects,
                                "method": "slf_yolo_domain", "device": self._device},
                        model_identity="slf_yolo_domain",
                        metrics={"defect_count": float(len(defects))},
                    )
                except Exception as e:
                    return AssetRunResult(
                        execution_status="failed", validity_status="invalid",
                        error=f"SLF-YOLO inference failed: {e}",
                        structured_output={}, metrics={},
                    )

            # 无领域权重 → unavailable（不使用 COCO 模型冒充）
            if img_arr is not None:
                gray = np.mean(img_arr, axis=-1) if img_arr.ndim == 3 else img_arr.astype(np.float64)
                defects = self._baseline(gray)
                return AssetRunResult.degraded(
                    method="edge_gradient_baseline",
                    reason=f"SLF-YOLO domain weights unavailable: {self._model_error}",
                    output={"defects_found": len(defects), "defects": defects},
                    metrics={"defect_count": float(len(defects))},
                )

            return AssetRunResult.unavailable(
                reason="No image data provided",
                reason_code="NO_INPUT_DATA",
            )

        except Exception as e:
            return AssetRunResult(
                execution_status="failed", validity_status="invalid",
                error=str(e), structured_output={}, metrics={},
            )


# ═══════════════════════════════════════════════════════════════════════
# SAM-Adapter 裂纹分割 — 完整：SAM+LoRA | 基线：Canny
# ═══════════════════════════════════════════════════════════════════════

class SAMAdapterCrackSegmentation(ImplementationBase):
    """SAM-Adapter 裂纹分割。

    完整模型: git clone https://github.com/sky-visionX/CrackSegmentation
    加载 SAM ViT-H + Adapter/LoRA 权重 → 像素级裂纹掩膜。

    基线: Canny 边缘检测。
    """

    asset_id = "detector.crack.sam_adapter_segmentation"

    def __init__(self):
        self._model = None
        self._loaded = False
        self._error = ""
        self._device = "cpu"

    def _load_model(self, params: dict) -> bool:
        if self._loaded:
            return True

        # 方案A：pip install segment-anything + 下载权重
        try:
            from segment_anything import sam_model_registry, SamPredictor
            import torch

            sam_checkpoint = params.get("sam_checkpoint", "")
            if not sam_checkpoint:
                sam_checkpoint = os.environ.get("SAM_CHECKPOINT", "")
            default_paths = [
                Path.home() / ".cache/sam/sam_vit_h_4b8939.pth",
                Path("sam_vit_h_4b8939.pth"),
            ]
            if not sam_checkpoint:
                for p in default_paths:
                    if p.exists():
                        sam_checkpoint = str(p)
                        break
            if not sam_checkpoint:
                self._error = "SAM weights not found. Download sam_vit_h_4b8939.pth from Meta"
                return False

            model_type = params.get("sam_model", "vit_h")
            self._model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            self._device = "cuda" if torch.cuda.is_available() and params.get("use_gpu", True) else "cpu"
            self._model.to(self._device)
            self._model.eval()
            self._predictor = SamPredictor(self._model)
            self._loaded = True
            return True
        except ImportError:
            self._error = "pip install segment-anything git+https://github.com/facebookresearch/segment-anything.git"
        except Exception as e:
            self._error = str(e)
        return False

    def _canny_baseline(self, img_arr: np.ndarray) -> dict:
        """Canny 边缘检测基线。"""
        try:
            import cv2
            if img_arr.ndim == 3:
                gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_arr
            edges = cv2.Canny(gray, 50, 150)
        except ImportError:
            gray = np.asarray(img_arr, dtype=np.float64)
            if gray.ndim == 3:
                gray = np.mean(gray, axis=-1)
            diff_x = np.abs(np.diff(gray, axis=0))
            diff_y = np.abs(np.diff(gray, axis=1)[:, :-1])
            edges = (diff_x + diff_y) > np.percentile(diff_x + diff_y, 90)
        ratio = float(np.sum(edges > 0) / edges.size) if edges.size > 0 else 0.0
        return {"crack_detected": ratio > 0.005, "crack_pixel_ratio": round(ratio, 6)}

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        data = inputs[0] if inputs else {}
        img = data.get("image", data.get("data"))

        try:
            img_arr = np.asarray(img) if not isinstance(img, str) else None
            if img_arr is not None:
                img_arr = _ensure_rgb(img_arr) if img_arr.ndim == 3 else img_arr

            # 完整 SAM 推理 — 需要 SAM 权重 + 裂纹专用 Adapter/LoRA
            if self._load_model(params) and img_arr is not None:
                self._predictor.set_image(img_arr)
                # 审计修复: 中心点提示不是裂纹检测——需要上游检测器提供裂纹候选点/框。
                # 没有裂纹 Adapter/LoRA 权重时，标准 SAM 不能做裂纹分割。
                h, w = img_arr.shape[:2]
                point = np.array([[w // 2, h // 2]])
                label = np.array([1])
                masks, scores, _ = self._predictor.predict(
                    point_coords=point, point_labels=label, multimask_output=True,
                )
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                ratio = float(np.sum(mask) / mask.size)
                return AssetRunResult.degraded(
                    method="sam_center_prompt",
                    reason=(
                        "Standard SAM with center-point prompt — NOT crack-specific segmentation. "
                        "SAM-Crack Adapter/LoRA weights not loaded. "
                        "Center-point prompt segments the most salient object, not necessarily a crack. "
                        "git clone https://github.com/sky-visionX/CrackSegmentation for crack-specific weights."
                    ),
                    output={
                        "crack_detected": bool(ratio > 0.005),
                        "crack_pixel_ratio": round(ratio, 6),
                        "mask_resolution": list(mask.shape),
                        "sam_score": float(scores[best_idx]),
                        "method": "sam_center_prompt_degraded",
                        "device": self._device,
                    },
                    metrics={"crack_pixel_ratio": ratio},
                )

            # Canny 基线 — 任何结构边缘都可能成为"裂纹"
            result = self._canny_baseline(img_arr) if img_arr is not None else {"crack_detected": False, "crack_pixel_ratio": 0.0}
            return AssetRunResult.degraded(
                method="canny_edge_baseline",
                reason=(
                    f"Canny edge detection — NOT crack segmentation. "
                    f"Any structural edge (reflections, coating textures, blade contours) may be misidentified as crack. "
                    f"SAM not loaded: {self._error}"
                ),
                output=result,
                metrics={"crack_pixel_ratio": result["crack_pixel_ratio"]},
            )

        except Exception as e:
            return AssetRunResult(
                execution_status="failed", validity_status="invalid",
                error=str(e), structured_output={}, metrics={},
            )

    @staticmethod
    def default_params() -> dict:
        return {"sam_model": "vit_h", "sam_checkpoint": "", "use_gpu": True,
                "fine_tune_method": "lora", "confidence_threshold": 0.3}


# ═══════════════════════════════════════════════════════════════════════
# EGCIENet 叶片缺陷分割 — 完整：SegFormer+SAM | 基线：图像统计
# ═══════════════════════════════════════════════════════════════════════

class EGCIENetSegmentation(ImplementationBase):
    """EGCIENet SAM 引导叶片缺陷分割。

    完整模型: git clone https://github.com/Newbiejy/EGCIENet_In-service-blade-defect-detection
    需要 SegFormer + SAM 边缘引导权重 → 多类别分割掩膜。

    审计修复 (AER-008): YOLOv8-seg 不是 EGCIENet。权重缺失时返回 unavailable。
    """

    asset_id = "detector.borescope.egcienet_segmentation"

    def __init__(self):
        self._loaded = False
        self._error = (
            "EGCIENet domain weights not available. "
            "Requires: git clone https://github.com/Newbiejy/EGCIENet_In-service-blade-defect-detection "
            "and trained SegFormer + SAM edge-guidance weights. "
            "Generic YOLOv8-seg is NOT EGCIENet and does not provide blade-specific defect segmentation."
        )

    def _load_model(self, params: dict) -> bool:
        """EGCIENet 需要其专用权重。YOLOv8-seg 不能替代。"""
        if self._loaded:
            return True
        # EGCIENet 权重未公开下载 → 始终不可用
        repo = _resolve_repo_path("EGCIENET_REPO_PATH", "EGCIENet_In-service-blade-defect-detection")
        if repo:
            self._error = (
                "EGCIENet repo found but trained weights required. "
                "Train SegFormer + SAM edge-guidance on blade defect dataset first."
            )
        return False

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        data = inputs[0] if inputs else {}
        img = data.get("image", data.get("data"))

        if self._load_model({}) and img is not None:
            pass  # 不会到达（当前无权重）

        # 权重缺失 → unavailable
        img_arr = np.asarray(img) if not isinstance(img, str) else None
        if img_arr is not None:
            stats = {"mean": float(np.mean(img_arr)), "std": float(np.std(img_arr)),
                     "min": float(np.min(img_arr)), "max": float(np.max(img_arr))}
            return AssetRunResult.degraded(
                method="image_statistics_only",
                reason=self._error,
                output={"stats": stats},
                metrics=stats,
            )

        return AssetRunResult.unavailable(
            reason=self._error,
            reason_code="MODEL_WEIGHTS_UNAVAILABLE",
        )


# ═══════════════════════════════════════════════════════════════════════
# TS-SAM 双流通用分割 — 完整：Dual-Stream SAM | 基线：YOLOv8-seg
# ═══════════════════════════════════════════════════════════════════════

class TSSAMSegmentation(ImplementationBase):
    """TS-SAM 双流通用分割。

    完整模型: git clone https://github.com/maoyangou147/TS-SAM
    需要双流SAM (CSA+MRM+FFD) 权重 (~2.4GB ViT-H)。

    审计修复 (AER-008): YOLOv8-seg 不是 TS-SAM。
    """

    asset_id = "detector.general.ts_sam_segmentation"

    def __init__(self):
        self._loaded = False
        self._error = (
            "TS-SAM requires SAM ViT-H (~2.4GB) + Dual-Stream CSA+MRM+FFD weights. "
            "git clone https://github.com/maoyangou147/TS-SAM for model code. "
            "Generic YOLOv8-seg is NOT TS-SAM."
        )

    def _load_model(self, params: dict) -> bool:
        if self._loaded:
            return True
        # TS-SAM 需要 SAM ViT-H 基础权重 + TS-SAM 专用适配权重
        repo = _resolve_repo_path("TS_SAM_REPO_PATH", "TS-SAM")
        if repo:
            sam_checkpoint = params.get("sam_checkpoint", os.environ.get("SAM_CHECKPOINT", ""))
            if sam_checkpoint and Path(sam_checkpoint).exists():
                try:
                    from segment_anything import sam_model_registry
                    import torch
                    self._model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._model.to(self._device)
                    self._loaded = True
                    return True
                except Exception as e:
                    self._error = f"TS-SAM load failed: {e}"
            else:
                self._error = "TS-SAM requires SAM ViT-H checkpoint + TS-SAM adapter weights"
        return False

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        data = inputs[0] if inputs else {}
        img = data.get("image", data.get("data"))

        if self._load_model({}) and img is not None:
            # SAM 基础模型已加载但缺少 TS-SAM adapter → degraded
            return AssetRunResult.degraded(
                method="sam_base_no_ts_adapter",
                reason="SAM ViT-H loaded but TS-SAM Dual-Stream adapter weights missing",
                output={"defect_segments": 0, "note": "TS-SAM adapter weights required for domain segmentation"},
                metrics={"segment_count": 0},
            )

        return AssetRunResult.unavailable(
            reason=self._error,
            reason_code="MODEL_WEIGHTS_UNAVAILABLE",
        )
