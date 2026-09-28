# TransXAI — public Colab environment and project-layout setup
# This release helper creates only directories and runtime metadata. It does not
# modify the frozen scientific configurations distributed in this repository.

import importlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path("/content/gdrive/MyDrive/Colab Notebooks/CPET")
BASE_SEED = 20260906
NUM_WORKERS = 0

try:
    from google.colab import drive
    drive.mount("/content/gdrive", force_remount=False)
except Exception as exc:
    raise RuntimeError("Run this script in Google Colab with Drive access.") from exc

dependency_map = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
    "sklearn": "scikit-learn", "skimage": "scikit-image",
    "matplotlib": "matplotlib", "seaborn": "seaborn", "tqdm": "tqdm",
    "yaml": "pyyaml", "captum": "captum", "torchcam": "torchcam",
    "pydicom": "pydicom", "albumentations": "albumentations",
    "cv2": "opencv-python-headless", "statsmodels": "statsmodels",
    "kagglehub": "kagglehub", "torchmetrics": "torchmetrics",
    "pyarrow": "pyarrow", "h5py": "h5py",
}
missing = []
for module, package in dependency_map.items():
    try:
        importlib.import_module(module)
    except Exception:
        missing.append(package)
if missing:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *sorted(set(missing))],
        check=True,
    )
    importlib.invalidate_caches()

import numpy as np
import torch
import torchvision

os.environ["PYTHONHASHSEED"] = str(BASE_SEED)
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

directories = [
    "configs", "data/raw", "data/manifests", "data/splits",
    "checkpoints/classifiers/resnet18", "checkpoints/classifiers/regnet_x_400mf",
    "results/classifiers/resnet18", "results/classifiers/regnet_x_400mf",
    "results/explanations/resnet18", "results/explanations/regnet_x_400mf",
    "results/transport_refinement/resnet18",
    "results/transport_refinement/regnet_x_400mf", "runs",
]
for relative in directories:
    (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)

runtime = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "project_root": str(PROJECT_ROOT), "base_seed": BASE_SEED,
    "num_workers": NUM_WORKERS, "python": sys.version.replace("\n", " "),
    "platform": platform.platform(), "torch": torch.__version__,
    "torchvision": torchvision.__version__, "cuda_available": torch.cuda.is_available(),
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
runtime_path = PROJECT_ROOT / "runs/public_release_runtime.json"
tmp_path = runtime_path.with_suffix(".tmp")
tmp_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp_path, runtime_path)

print("TransXAI public environment: PASS")
print(json.dumps(runtime, indent=2, sort_keys=True))
print("Copy the frozen JSON files from this repository's configs/ directory to")
print(PROJECT_ROOT / "configs")
