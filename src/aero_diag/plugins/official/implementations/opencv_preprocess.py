"""OpenCV 孔探图像预处理 — P1_HIGH"""

import numpy as np

from ._base import AssetRunResult, ImplementationBase


class OpenCVPreprocessor(ImplementationBase):
    """OpenCV 孔探图像标准化预处理。

    光照校正(CLAHE)、去噪、锐化、缩放。
    """

    asset_id = "preprocessor.image.opencv_preprocess"

    def validate_inputs(self, inputs: list, parameters: dict, context: dict) -> dict:
        if not inputs:
            return {"ok": False, "issues": ["No image data provided"]}
        return {"ok": True, "issues": []}

    def run(self, inputs: list, parameters: dict, context: dict) -> AssetRunResult:
        params = {**self.default_params(), **parameters}
        data = inputs[0] if inputs else {}

        # 获取图像
        img = data.get("image", data.get("data", data.get("image_uri")))
        if img is None:
            return AssetRunResult(
                status="failed", structured_output={},
                warnings=["No image data found"], metrics={},
            )

        processed = False
        try:
            import cv2
        except ImportError:
            # fallback: numpy-only basic processing
            img_arr = np.asarray(img, dtype=np.float64)
            if img_arr.ndim == 3:
                # normalize
                img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8) * 255
                img_arr = img_arr.astype(np.uint8)
            return AssetRunResult(
                status="partial",
                structured_output={
                    "processed": True, "method": "numpy_fallback",
                    "shape": list(img_arr.shape), "dtype": str(img_arr.dtype),
                },
                warnings=["OpenCV not installed — using numpy fallback (no CLAHE/denoising)"],
                metrics={"original_range": [float(np.min(img_arr)), float(np.max(img_arr))]},
            )

        # OpenCV 路径
        if isinstance(img, str):
            img_arr = cv2.imread(img, cv2.IMREAD_COLOR)
            if img_arr is None:
                return AssetRunResult(status="failed", structured_output={}, error=f"Cannot read image: {img}")
        else:
            img_arr = np.asarray(img, dtype=np.uint8)

        original_shape = img_arr.shape

        # CLAHE 光照校正
        if params.get("clahe", True) and img_arr.ndim >= 2:
            if img_arr.ndim == 3:
                lab = cv2.cvtColor(img_arr, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(
                    clipLimit=params.get("clahe_clip_limit", 2.0),
                    tileGridSize=tuple(params.get("clahe_tile_size", [8, 8])),
                )
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                img_arr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_arr = clahe.apply(img_arr)

        # 去噪
        if params.get("denoise", True):
            h = params.get("denoise_strength", 10)
            if img_arr.ndim == 3:
                img_arr = cv2.fastNlMeansDenoisingColored(img_arr, None, h, h, 7, 21)
            else:
                img_arr = cv2.fastNlMeansDenoising(img_arr, None, h, 7, 21)

        # 缩放
        target_size = params.get("target_size", [512, 512])
        if target_size:
            img_arr = cv2.resize(img_arr, tuple(target_size))

        # 锐化
        if params.get("sharpen", False):
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 9
            img_arr = cv2.filter2D(img_arr, -1, kernel)

        return AssetRunResult(
            status="success",
            structured_output={
                "processed": True,
                "method": "opencv_clahe_denoise",
                "original_shape": list(original_shape),
                "output_shape": list(img_arr.shape),
                "steps_applied": [s for s, flag in
                    [("clahe", params.get("clahe")), ("denoise", params.get("denoise")),
                     ("resize", bool(target_size)), ("sharpen", params.get("sharpen"))]
                    if flag],
            },
            metrics={
                "original_mean": float(np.mean(img_arr)),
                "original_std": float(np.std(img_arr)),
            },
        )

    @staticmethod
    def default_params() -> dict:
        return {
            "clahe": True, "clahe_clip_limit": 2.0, "clahe_tile_size": [8, 8],
            "denoise": True, "denoise_strength": 10,
            "target_size": [512, 512], "sharpen": False,
        }
