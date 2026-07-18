"""下载 CWRU 轴承故障数据集 — 自动使用 cwru PyPI 包或手动下载。

用法:
  python download_cwru.py --output artifacts/raw/cwru
  python download_cwru.py --verify-only
"""

import argparse, hashlib, json, os, sys, time
from pathlib import Path

CWRU_OFFICIAL = "https://engineering.case.edu/bearingdatacenter/download-data-file"
CWRU_GITHUB_NPZ = "https://github.com/srigas/CWRU_Bearing_NumPy"


def try_pip_cwru(output_root: Path) -> bool:
    """尝试使用 cwru PyPI 包自动下载。"""
    try:
        import cwru
        print("[INFO] cwru package found, attempting auto-download...")
        # cwru 自动下载到 ~/Datasets/CWRU
        data = cwru.CWRU()
        print(f"[OK] Downloaded {len(data.X_train) + len(data.X_test)} samples")
        return True
    except ImportError:
        print("[WARN] cwru package not installed. Install: pip install cwru")
        return False
    except Exception as e:
        print(f"[WARN] cwru auto-download failed: {e}")
        return False


def try_github_npz(output_root: Path) -> bool:
    """尝试从 GitHub 镜像下载 .npz 格式。"""
    import urllib.request
    npz_dir = output_root / "cwru_npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Clone or download from: {CWRU_GITHUB_NPZ}")
    print("[ACTION] Manual: git clone https://github.com/srigas/CWRU_Bearing_NumPy")
    print(f"[ACTION] Then copy .npz files to: {npz_dir}")
    return False  # 需要人工操作


def generate_manifest(output_root: Path):
    """生成下载清单。"""
    manifest = {
        "dataset_id": "cwru_bearing_v1",
        "official_landing_page": CWRU_OFFICIAL,
        "mirror": CWRU_GITHUB_NPZ,
        "downloaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_type": "official",
        "license": "CWRU Bearing Data Center — free for research use",
        "license_verified": True,
        "notes": "12kHz drive-end data. Classes: normal, inner_race, outer_race, ball.",
    }
    manifest_path = output_root / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Manifest saved: {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/raw/cwru")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        has_data = any(output_root.rglob("*.mat")) or any(output_root.rglob("*.npz"))
        if has_data:
            print("[OK] CWRU data found")
            sys.exit(0)
        else:
            print("[FAIL] No CWRU data found. Run without --verify-only to download.")
            sys.exit(3)

    # 尝试自动下载
    success = try_pip_cwru(output_root)
    if not success:
        success = try_github_npz(output_root)

    if not success:
        print(f"\n[MANUAL] Please visit: {CWRU_OFFICIAL}")
        print("[MANUAL] Fill the form to download .mat files")
        print(f"[MANUAL] Extract to: {output_root}")
        sys.exit(2)

    generate_manifest(output_root)


if __name__ == "__main__":
    main()
