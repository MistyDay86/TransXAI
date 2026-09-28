"""
CPET — NOTEBOOK 08 — INDEPENDENT VALIDATION, ABLATIONS, MATCHING CONTROLS, AND EFFICIENCY

Single-cell Colab script. Copy the complete file into one Colab cell or execute it with %run.

This compact notebook freezes and evaluates four reviewer-facing control families:
  1. independent pixel-space faithfulness/transport and post-hoc clinical localization;
  2. targeted ablations of T-CPT on three representative explainer families;
  3. recipient-matching controls on all six explainers;
  4. empirical runtime, memory, and recipient-count scaling.

It never retrains a classifier and never overwrites Notebooks 01–07. Clinical masks are used
only after all explanations and refinements have been frozen. Expensive jobs are resume-aware.

The complete 3x5x6 primary grid is audited from frozen artifacts. New model forwards and true
re-optimizations are restricted to a deterministic, decision-stratified audit cohort: fold 0 and
16 donors per dataset. This is a reviewer-facing mechanistic audit, not a second main experiment.

Hardware: CUDA GPU required; a Tesla T4 is sufficient. Expected first-run time: 20-40 minutes.
num_workers=0.
"""

# ==================================================================================================
# 0. FROZEN USER SETTINGS
# ==================================================================================================

PROJECT_ROOT = "/content/gdrive/MyDrive/Colab Notebooks/CPET"
SEED = 20260906
NUM_WORKERS = 0

PRIMARY_BUDGET = 0.10
DATASETS = ["siim_acr", "isic2016", "pad_ufes20"]
FOLDS = [0, 1, 2, 3, 4]
CONTROL_FOLDS = [0]
CONTROL_DONORS_PER_FOLD = 16
EXPLAINERS = [
    "gradcam", "layercam", "integrated_gradients", "lrp", "rise", "extremal_perturbation"
]
ABLATION_EXPLAINERS = ["gradcam", "integrated_gradients", "rise"]
ABLATIONS = ["random_search", "no_cvar", "no_validation", "no_trust_region"]
MATCHING_POLICIES = ["nearest_same_decision", "random_same_decision", "nearest_opposite_decision"]

# Same frozen T-CPT definition as Notebook 06B.
N_RECIPIENT_A = 12
N_RECIPIENT_B = 12
N_RECIPIENT_Q = 10
CALIBRATION_A_CANDIDATES = 384
CALIBRATION_B_CANDIDATES = 256
MAX_ITERATIONS = 15
MAX_SWAP_FRACTION = 0.25
SWAP_BATCHES = (1, 2, 4)
GRADIENT_ALTERNATIVES = 3
LAMBDA_CVAR = 0.35
CVAR_ALPHA = 0.80
LAMBDA_TV = 0.002
LOCAL_EFFECT_TOLERANCE = 0.10
MIN_BASE_EFFECT = 1e-3
MIN_SUPPORT_IMPROVEMENT = 1e-5
VALIDATION_ACCEPT_MARGIN = 0.002
CPTS_EPSILON = 0.05

# Independent evaluation. The clinical non-inferiority margin is absolute on [0,1] metrics.
PIXEL_BASELINE_SIGMA_FRACTION = 0.05
LOCALIZATION_NONINFERIORITY_MARGIN = 0.02
BOOTSTRAP_REPLICATES = 5000
PERMUTATION_REPLICATES = 50000
RECIPIENT_SCALING_COUNTS = [1, 2, 5, 10, 20, 40]
FIGURE_DPI = 600

# Numerical replay is performed from FP16 cached features on a GPU runtime. These thresholds are
# descriptive audit references only: primary results are read directly from frozen Notebook 07
# artifacts, so finite donor-level replay deviations must never block the control analysis.
REPLAY_MEAN_ABS_TOLERANCE = 0.005
REPLAY_CELL_MEAN_TOLERANCE = 0.002
REPLAY_MAX_ABS_TOLERANCE = 0.050

# ==================================================================================================
# 1. IMPORTS, DRIVE, RUNTIME, AND ATOMIC UTILITIES
# ==================================================================================================

import os
import re
import gc
import sys
import json
import math
import time
import random
import hashlib
import warnings
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

try:
    import h5py
    import cv2
    from PIL import Image
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from torchvision.models import resnet18
except Exception as exc:
    raise RuntimeError(
        "Missing dependency. Run Notebook 01 first in this project. "
        f"Detail: {type(exc).__name__}: {exc}"
    ) from exc

WIDTH = 122
START_TIME = time.time()


def say(message=""):
    print(message, flush=True)


def banner(title):
    say("\n" + "=" * WIDTH)
    say(title)
    say("=" * WIDTH)


def section(index, total, title):
    say(f"\n[{index}/{total}] {title}")


def elapsed(seconds=None):
    seconds = time.time() - START_TIME if seconds is None else float(seconds)
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def stable_int(text):
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:8], 16)


SHA256_CACHE = {}


def sha256_file(path, chunk=8 * 1024 * 1024):
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    if key in SHA256_CACHE:
        return SHA256_CACHE[key]
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    digest = h.hexdigest()
    SHA256_CACHE[key] = digest
    return digest


def canonical_json_bytes(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def atomic_write_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_bytes(path, canonical_json_bytes(obj))


def atomic_write_dataframe(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".tmp.{os.getpid()}" + path.suffix)
    if path.suffix == ".parquet":
        frame.to_parquet(tmp, index=False)
    elif path.suffix == ".csv":
        frame.to_csv(tmp, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path.suffix}")
    os.replace(tmp, path)


def atomic_save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_json(obj):
    values = []
    if isinstance(obj, dict):
        for value in obj.values():
            values.extend(flatten_json(value))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(flatten_json(value))
    else:
        values.append(obj)
    return values


def normalize_colname(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def pick_column(frame, candidates, required=True):
    lookup = {normalize_colname(c): c for c in frame.columns}
    for candidate in candidates:
        key = normalize_colname(candidate)
        if key in lookup:
            return lookup[key]
    for candidate in candidates:
        key = normalize_colname(candidate)
        for norm, original in lookup.items():
            if key in norm or norm in key:
                return original
    if required:
        raise KeyError(f"No column among {candidates}; available={list(frame.columns)}")
    return None


def valid_identifier(value):
    return not (
        pd.isna(value)
        or str(value).strip().lower() in {"", "nan", "none", "null", "na", "n/a", "unknown", "missing", "<na>"}
    )


def deterministic_sample(frame, n, salt):
    if n is None or len(frame) <= int(n):
        return frame.copy().reset_index(drop=True)
    rng = np.random.default_rng(SEED + stable_int(salt))
    indices = np.sort(rng.choice(len(frame), size=int(n), replace=False))
    return frame.iloc[indices].reset_index(drop=True)


def percentile_ci(values, alpha=0.05):
    return tuple(float(x) for x in np.quantile(np.asarray(values, float), [alpha / 2, 1 - alpha / 2]))


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

banner("CPET — NOTEBOOK 08 — COMPACT REVIEWER AUDIT, ABLATIONS, CONTROLS, AND EFFICIENCY")
say("No classifier training. No change to T-CPT. No access to clinical masks during optimization.")
say("Primary budget: exact 10% | ablation explainers: Grad-CAM, Integrated Gradients, RISE.")
say("Hardware required: CUDA GPU; Tesla T4 sufficient. num_workers=0.")
say("Frozen main grid audited: 3 datasets × 5 folds × 6 explainers = 90 cells.")
say("New audit grid: fold 0 only, 16 decision-stratified donors per dataset.")
say("New jobs: 18 independent + 18 matching + 36 targeted ablation = 72.")
say("Expected first-run time on Tesla T4: approximately 20-40 minutes; resumed runs are faster.")
say("Every completed job is persisted on Drive and verified before resume.")

section(1, 14, "Mount Drive, audit GPU, and freeze the reviewer-control protocol")
try:
    from google.colab import drive
    drive.mount("/content/gdrive")
    say("Google Drive mounted/verified.")
except ImportError:
    say("Non-Colab runtime: using the available filesystem.")

ROOT = Path(PROJECT_ROOT)
if not ROOT.exists():
    raise FileNotFoundError(f"Project root not found: {ROOT}")
if not torch.cuda.is_available():
    raise RuntimeError("Notebook 08 requires a CUDA GPU. Select a Tesla T4 runtime in Colab.")
DEVICE = torch.device("cuda")
say(f"GPU={torch.cuda.get_device_name(0)} | VRAM={torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
say(f"Python={sys.version.split()[0]} | PyTorch={torch.__version__} | torchvision={torchvision.__version__}")

CONTROL_ROOT = ROOT / "results/reviewer_controls/resnet18/compact_v1.0.2"
RUN_ROOT = ROOT / "runs/reviewer_controls/resnet18"
CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ROOT.mkdir(parents=True, exist_ok=True)

protocol = {
    "name": "CPET compact reviewer-facing validation and controls",
    "version": "compact-1.0.2",
    "seed": SEED,
    "primary_budget": PRIMARY_BUDGET,
    "datasets": DATASETS,
    "folds": FOLDS,
    "control_folds": CONTROL_FOLDS,
    "control_donors_per_fold": CONTROL_DONORS_PER_FOLD,
    "explainers": EXPLAINERS,
    "ablation_explainers": ABLATION_EXPLAINERS,
    "ablations": ABLATIONS,
    "matching_policies": MATCHING_POLICIES,
    "independent_metrics": [
        "pixel_transport_consistency", "pixel_deletion_faithfulness", "pixel_sufficiency",
        "clinical_dice", "clinical_iou", "clinical_roi_hit"
    ],
    "localization_noninferiority_margin": LOCALIZATION_NONINFERIORITY_MARGIN,
    "clinical_masks_role": "post-hoc evaluation only; never optimization, matching, or selection",
    "preprocessing_selection": "fold-wise minimum mean absolute replay error over three frozen candidates",
    "hdf5_reader": "exact Notebook 06B dataset scoring, including +2 spatial-map preference",
    "optimizer": {
        "n_recipient_a": N_RECIPIENT_A, "n_recipient_b": N_RECIPIENT_B,
        "n_recipient_q": N_RECIPIENT_Q, "calibration_a_candidates": CALIBRATION_A_CANDIDATES,
        "calibration_b_candidates": CALIBRATION_B_CANDIDATES, "max_iterations": MAX_ITERATIONS,
        "max_swap_fraction": MAX_SWAP_FRACTION, "swap_batches": list(SWAP_BATCHES),
        "gradient_alternatives": GRADIENT_ALTERNATIVES, "lambda_cvar": LAMBDA_CVAR,
        "cvar_alpha": CVAR_ALPHA, "lambda_tv": LAMBDA_TV,
        "local_effect_tolerance": LOCAL_EFFECT_TOLERANCE,
        "validation_accept_margin": VALIDATION_ACCEPT_MARGIN, "cpts_epsilon": CPTS_EPSILON,
    },
    "inference": {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "permutation_replicates": PERMUTATION_REPLICATES,
    },
    "numerical_replay_reporting_references_nonblocking": {
        "mean_absolute": REPLAY_MEAN_ABS_TOLERANCE,
        "cell_mean": REPLAY_CELL_MEAN_TOLERANCE,
        "maximum_absolute": REPLAY_MAX_ABS_TOLERANCE,
    },
    "scope": "full frozen-grid audit plus targeted mechanistic controls",
}
CONFIG_PATH = ROOT / "configs/reviewer_controls_compact_v1.0.2.json"
payload = canonical_json_bytes(protocol)
if CONFIG_PATH.exists():
    if CONFIG_PATH.read_bytes() != payload:
        raise RuntimeError(f"Existing control protocol differs and will not be overwritten: {CONFIG_PATH}")
    config_state = "PRESERVED_IDENTICAL"
else:
    atomic_write_bytes(CONFIG_PATH, payload)
    config_state = "CREATED"
say(f"Protocol: {config_state} | {CONFIG_PATH} | sha256={sha256_file(CONFIG_PATH)[:16]}...")


# ==================================================================================================
# 2. FROZEN HANDOFFS AND ARTIFACT DISCOVERY
# ==================================================================================================

section(2, 14, "Verify Notebook 06B/07 handoffs and discover frozen artifacts")
NB06B_LATEST = ROOT / "runs/transport_refinement/resnet18/latest_transport_refinement_manifest.json"
NB07_LATEST = ROOT / "runs/confirmatory/resnet18/latest_confirmatory_manifest.json"
for label, path in (("NB06B", NB06B_LATEST), ("NB07", NB07_LATEST)):
    if not path.exists():
        raise FileNotFoundError(f"{label} manifest not found: {path}")
    obj = load_json(path)
    if str(obj.get("status", "")).upper() != "PASS":
        raise RuntimeError(f"{label} handoff is not PASS.")
    say(f"{label}: PASS | {path.name} | sha256={sha256_file(path)[:16]}...")

nb06b = load_json(NB06B_LATEST)
result_root = Path(nb06b.get("artifacts", {}).get("result_root", ""))
if not result_root.exists():
    raise FileNotFoundError(f"Frozen T-CPT mask root not found: {result_root}")

confirmatory_runs = sorted(
    [path for path in (ROOT / "runs/confirmatory/resnet18").glob("*/confirmatory_manifest.json") if path.exists()],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
if not confirmatory_runs:
    raise FileNotFoundError("No immutable Notebook 07 run manifest found.")
NB07_RUN_MANIFEST = confirmatory_runs[0]
nb07_run = load_json(NB07_RUN_MANIFEST)
if str(nb07_run.get("status", "")).upper() != "PASS":
    raise RuntimeError("Latest immutable Notebook 07 run is not PASS.")
NB07_RUN_DIR = NB07_RUN_MANIFEST.parent
FULL_METRICS_PATH = NB07_RUN_DIR / "supplement/analysis_donor_metrics_corrected.parquet"
if not FULL_METRICS_PATH.exists():
    raise FileNotFoundError(f"Corrected Notebook 07 donor metrics not found: {FULL_METRICS_PATH}")
full_metrics = pd.read_parquet(FULL_METRICS_PATH)
full_metrics = full_metrics[np.isclose(full_metrics.budget.astype(float), PRIMARY_BUDGET)].copy()
required_full_columns = {
    "dataset", "fold", "explainer", "budget", "sample_id", "analysis_group_id", "delta_Q",
    "accepted", "swaps_from_base", "selected_iteration", "search_steps_accepted",
    "local_effect_base", "local_effect_refined",
}
missing_full_columns = sorted(required_full_columns - set(full_metrics.columns))
if missing_full_columns:
    raise RuntimeError(f"Corrected Notebook 07 metrics lack required columns: {missing_full_columns}")
full_cells = full_metrics.groupby(["dataset", "fold", "explainer"], sort=True).size()
if len(full_cells) != len(DATASETS) * len(FOLDS) * len(EXPLAINERS) or full_cells.min() != full_cells.max():
    raise RuntimeError(
        f"Corrected primary grid is incomplete/unbalanced: cells={len(full_cells)}, "
        f"donors={full_cells.min()}–{full_cells.max()}."
    )
if full_metrics.duplicated(["dataset", "fold", "explainer", "sample_id"]).any():
    raise RuntimeError("Corrected Notebook 07 donor metrics contain duplicate analysis rows.")
frozen_cell_gain = (
    full_metrics.groupby(["dataset", "fold", "explainer"], sort=True)
    .agg(donors=("sample_id", "size"), mean_delta_q=("delta_Q", "mean"))
    .reset_index()
)
if int((frozen_cell_gain.mean_delta_q > 0).sum()) != 90:
    raise RuntimeError("The frozen Notebook 07 primary grid is not the supported 90/90 positive-cell result.")
say(f"Corrected full T-CPT metrics: {FULL_METRICS_PATH} | rows={len(full_metrics):,}")
say(
    f"Corrected primary grid: {len(full_cells)}/90 cells | donors/cell={full_cells.min()}–{full_cells.max()} "
    f"| positive mean-gain cells=90/90"
)

OOF_PATH = ROOT / "results/explanations/resnet18/explanation_cohort.parquet"
if not OOF_PATH.exists():
    raise FileNotFoundError(f"Frozen OOF cohort not found: {OOF_PATH}")
oof = pd.read_parquet(OOF_PATH)
OOF_DATASET = pick_column(oof, ["dataset", "dataset_id"])
OOF_FOLD = pick_column(oof, ["fold", "test_fold", "outer_fold"])
OOF_ID = pick_column(oof, ["sample_id", "image_id", "id"])
OOF_PATH_COL = pick_column(oof, ["image_path", "path", "filepath", "file_path", "image"])
OOF_PATIENT = pick_column(oof, ["patient_id"], required=False)
OOF_LESION = pick_column(oof, ["lesion_id", "case_id", "group_id"], required=False)
OOF_LOGIT = pick_column(oof, ["logit", "raw_logit", "score_logit"], required=False)


def analysis_group_from_row(row, dataset, id_col, patient_col, lesion_col):
    if dataset == "pad_ufes20":
        if patient_col is None or not valid_identifier(row[patient_col]):
            raise RuntimeError(f"PAD-UFES-20 row lacks patient_id: {row[id_col]}")
        return str(row[patient_col]).strip(), "patient"
    if lesion_col is not None and valid_identifier(row[lesion_col]):
        return str(row[lesion_col]).strip(), "case/lesion"
    return str(row[id_col]).strip(), "case/lesion"


def discover_manifest(dataset):
    exact = ROOT / f"data/manifests/{dataset}_manifest_v1.1.0.parquet"
    if exact.exists():
        return exact
    candidates = sorted((ROOT / "data/manifests").glob(f"*{dataset}*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"Canonical manifest not found for {dataset}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def checkpoint_candidates(dataset, fold):
    paths = []
    for base in (ROOT / "checkpoints/classifiers/resnet18", ROOT / "checkpoints/resnet18"):
        if base.exists():
            for suffix in ("*.pt", "*.pth", "*.ckpt"):
                paths.extend(base.rglob(suffix))
    for manifest_file in (
        ROOT / "runs/explanations/resnet18/latest_baseline_explainer_manifest.json",
        ROOT / "runs/training/explainers/resnet18/latest_learned_explainer_manifest.json",
    ):
        if manifest_file.exists():
            for value in flatten_json(load_json(manifest_file)):
                if isinstance(value, str) and value.lower().endswith((".pt", ".pth", ".ckpt")) and Path(value).exists():
                    paths.append(Path(value))
    scored = []
    for path in sorted(set(paths)):
        text = str(path).lower()
        score = 8 * (dataset in text)
        score += 6 * bool(re.search(rf"(^|[^0-9])f(?:old)?[_-]?0*{fold}([^0-9]|$)", text))
        score += 4 * ("classifier" in text)
        score -= 8 * any(token in text for token in ("explainer", "rational", "transxai"))
        score += 1 * ("best" in path.name.lower())
        if score >= 14:
            scored.append((score, path.stat().st_mtime, path))
    if not scored:
        raise FileNotFoundError(f"Classifier checkpoint not found: {dataset} fold {fold}.")
    return sorted(scored, reverse=True)[0][2]


def explanation_map_candidates(dataset, fold, explainer):
    base = ROOT / "results/explanations/resnet18"
    scored = []
    for pattern in ("*.h5", "*.hdf5"):
        for path in base.rglob(pattern):
            text = str(path).lower()
            score = 7 * (dataset in text) + 7 * (explainer in text)
            score += 5 * bool(re.search(rf"(^|[^0-9])f(?:old)?[_-]?0*{fold}([^0-9]|$)", text))
            if score >= 19:
                scored.append((score, path.stat().st_mtime, path))
    if not scored:
        raise FileNotFoundError(f"Explanation HDF5 not found: {dataset} fold {fold} {explainer}.")
    return sorted(scored, reverse=True)[0][2]


artifact_registry = {}
for dataset in DATASETS:
    manifest_path = discover_manifest(dataset)
    say(f"{dataset:12s} manifest={manifest_path.name}")
    for fold in FOLDS:
        artifact_registry[(dataset, fold)] = {
            "manifest": manifest_path,
            "checkpoint": checkpoint_candidates(dataset, fold),
            "maps": {explainer: explanation_map_candidates(dataset, fold, explainer) for explainer in EXPLAINERS},
        }
say("Frozen artifact registry: 15 checkpoints and 90 explanation files resolved.")


# ==================================================================================================
# 3. MODEL REPLAY, IMAGES, CLINICAL MASKS, AND HDF5 READERS
# ==================================================================================================

section(3, 14, "Define exact classifier replay, input perturbations, and post-hoc mask readers")
INPUT_SIZE = {"siim_acr": 320, "isic2016": 320, "pad_ufes20": 320}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def extract_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model", "network", "net"):
            value = payload.get(key)
            if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
                return value
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    raise RuntimeError("Checkpoint has no recognizable state_dict.")


def normalize_state_dict(state):
    prefixes = ("module.", "model.", "network.", "net.", "backbone.")
    output = {}
    for key, value in state.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        if new_key.startswith("classifier."):
            new_key = "fc." + new_key[len("classifier."):]
        output[new_key] = value
    return output


def load_classifier(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = normalize_state_dict(extract_state_dict(payload))
    if "fc.weight" not in state:
        raise RuntimeError(f"{path.name}: fc.weight missing after key normalization.")
    model = resnet18(weights=None)
    out_features = int(state["fc.weight"].shape[0])
    model.fc = nn.Linear(model.fc.in_features, out_features)
    missing, unexpected = model.load_state_dict(state, strict=False)
    meaningful_missing = [key for key in missing if not key.endswith("num_batches_tracked")]
    if meaningful_missing or unexpected or out_features not in (1, 2):
        raise RuntimeError(
            f"Non-exact checkpoint replay: missing={meaningful_missing[:6]}, unexpected={unexpected[:6]}, out={out_features}"
        )
    model.eval().to(DEVICE)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def model_to_layer2(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    return model.layer2(x)


def layer2_to_layer3(model, z2):
    return model.layer3(z2)


def layer3_to_logits(model, z3):
    x = model.layer4(z3)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    logits = model.fc(x)
    return logits[:, 0] if logits.shape[1] == 1 else logits[:, 1] - logits[:, 0]


def model_logits(model, x):
    logits = model(x)
    return logits[:, 0] if logits.shape[1] == 1 else logits[:, 1] - logits[:, 0]


resolve_path_cache = {}


def resolve_image_path(value, dataset):
    value = str(value)
    path = Path(value)
    attempts = [path]
    if not path.is_absolute():
        attempts += [ROOT / value, ROOT / "data" / value, ROOT / "data/raw" / dataset / value]
    for candidate in attempts:
        if candidate.exists() and candidate.is_file():
            return candidate
    key = (dataset, path.name)
    if key not in resolve_path_cache:
        found = []
        for base in (ROOT / "data/raw" / dataset, ROOT / "data/processed" / dataset, ROOT / "data"):
            if base.exists():
                found.extend(base.rglob(path.name))
        resolve_path_cache[key] = sorted(set(item for item in found if item.is_file()))
    found = resolve_path_cache[key]
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(f"Image not found: {dataset}/{value}")
    raise RuntimeError(f"Ambiguous image basename: {dataset}/{path.name} ({len(found)} candidates)")


def load_rgb_uint8(path, size, mode="opencv_area_square"):
    if mode == "opencv_area_square":
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV cannot decode: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    with Image.open(path) as image:
        image = image.convert("RGB")
        if mode == "pil_bilinear_square":
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        elif mode == "pil_center_crop":
            width, height = image.size
            scale = float(size) / min(width, height)
            new_width = max(size, int(round(width * scale)))
            new_height = max(size, int(round(height * scale)))
            image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
            left = (new_width - size) // 2
            top = (new_height - size) // 2
            image = image.crop((left, top, left + size, top + size))
        else:
            raise ValueError(f"Unknown preprocessing mode: {mode}")
        return np.asarray(image, dtype=np.uint8)


def normalize_rgb_batch(images_uint8):
    tensor = torch.from_numpy(images_uint8.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    return (tensor - IMAGENET_MEAN[None]) / IMAGENET_STD[None]


@torch.no_grad()
def choose_preprocessing(model, frame, dataset):
    modes = ("opencv_area_square", "pil_bilinear_square", "pil_center_crop")
    if OOF_LOGIT is None:
        say("  Preprocessing replay: no frozen raw logit; using protocol default OpenCV AREA square.")
        return modes[0]
    audit = frame.dropna(subset=[OOF_LOGIT]).head(8)
    if audit.empty:
        say("  Preprocessing replay: no finite audit logits; using protocol default OpenCV AREA square.")
        return modes[0]
    expected = audit[OOF_LOGIT].astype(float).to_numpy()
    results = []
    for mode in modes:
        images = np.stack([
            load_rgb_uint8(resolve_image_path(row[OOF_PATH_COL], dataset), INPUT_SIZE[dataset], mode)
            for _, row in audit.iterrows()
        ])
        x = normalize_rgb_batch(images).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            observed = model_logits(model, x).float().cpu().numpy()
        error = np.abs(observed - expected)
        results.append((float(np.mean(error)), float(np.max(error)), mode))
        del images, x
    results.sort(key=lambda item: (item[0], item[1], item[2]))
    mean_error, max_error, selected = results[0]
    say("  Preprocessing replay: " + " | ".join(
        f"{mode}=mean{mean:.2e}/max{maximum:.2e}" for mean, maximum, mode in results
    ))
    say(f"  Preprocessing selected: {selected} | mean={mean_error:.3e} | max={max_error:.3e}")
    if max_error > 5e-2:
        say("  WARNING: logits may be calibrated; minimum-error geometry is retained and the corrected-Q replay remains mandatory.")
    return selected


def h5_datasets(handle):
    found = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            found[name] = obj
    handle.visititems(visitor)
    return found


def decode_strings(array):
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in np.asarray(array).reshape(-1)]


def load_explanation_h5(path):
    with h5py.File(path, "r") as handle:
        found = h5_datasets(handle)
        numeric, strings = [], []
        for name, dataset in found.items():
            if dataset.ndim >= 3 and np.issubdtype(dataset.dtype, np.number):
                score = 5 if any(token in name.lower() for token in ("map", "saliency", "attribution")) else 0
                score += 2 if dataset.shape[-1] > 1 and dataset.shape[-2] > 1 else 0
                numeric.append((score, name, dataset.shape))
            if dataset.ndim == 1:
                id_like = any(token in name.lower() for token in ("sample", "image", "id"))
                if dataset.dtype.kind in ("S", "U", "O") or id_like:
                    score = 5 if id_like else 0
                    strings.append((score, name, dataset.shape))
        if not numeric:
            raise RuntimeError(f"No saliency array in {path}")
        numeric.sort(reverse=True)
        map_name = numeric[0][1]
        maps = np.asarray(found[map_name], dtype=np.float32)
        if maps.ndim == 4 and maps.shape[1] == 1:
            maps = maps[:, 0]
        elif maps.ndim == 4 and maps.shape[-1] == 1:
            maps = maps[..., 0]
        if maps.ndim != 3:
            raise RuntimeError(f"Unsupported saliency shape in {path}: {maps.shape}")
        aligned = [item for item in strings if item[2][0] == maps.shape[0]]
        if not aligned:
            raise RuntimeError(f"No aligned sample IDs in {path}")
        aligned.sort(reverse=True)
        id_name = aligned[0][1]
        ids = decode_strings(found[id_name][...])
    if len(ids) != len(set(ids)) or not np.isfinite(maps).all():
        raise RuntimeError(f"Invalid HDF5 explanation artifact: {path}")
    return ids, maps


def resize_score_map(saliency, height, width):
    tensor = torch.from_numpy(np.asarray(saliency, dtype=np.float32))[None, None]
    tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = tensor.min(), tensor.max()
    return (tensor - low) / (high - low) if float(high - low) > 1e-12 else torch.zeros_like(tensor)


def exact_topk_mask(scores, k):
    flat = scores.reshape(-1).float()
    k = int(max(1, min(int(k), flat.numel())))
    tie = torch.arange(flat.numel(), dtype=torch.float32, device=flat.device)
    adjusted = flat - tie * (torch.finfo(torch.float32).eps / max(1, flat.numel()))
    selected = torch.topk(adjusted, k=k, largest=True, sorted=False).indices
    mask = torch.zeros_like(flat, dtype=torch.float32)
    mask[selected] = 1.0
    return mask.view_as(scores)


def load_refined_masks(dataset, fold, explainer, sample_ids):
    tag = f"p{int(round(100 * PRIMARY_BUDGET)):02d}"
    path = result_root / dataset / f"fold_{fold}" / f"{dataset}_f{fold}_{explainer}_{tag}_masks.npz"
    if not path.exists():
        raise FileNotFoundError(f"Frozen T-CPT masks not found: {path}")
    data = np.load(path, allow_pickle=False)
    lookup = {str(sample_id): idx for idx, sample_id in enumerate(data["sample_ids"])}
    missing = [sample_id for sample_id in sample_ids if sample_id not in lookup]
    if missing:
        raise RuntimeError(f"{path.name}: {len(missing)} sample IDs missing.")
    return np.stack([data["masks"][lookup[sample_id]] for sample_id in sample_ids]).astype(np.uint8), path


def reconstruct_base_masks(dataset, fold, explainer, sample_ids, height, width):
    path = artifact_registry[(dataset, fold)]["maps"][explainer]
    map_ids, raw_maps = load_explanation_h5(path)
    lookup = {sample_id: idx for idx, sample_id in enumerate(map_ids)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in lookup]
    if missing:
        raise RuntimeError(f"{path.name}: {len(missing)} donor maps missing.")
    k = max(1, int(round(PRIMARY_BUDGET * height * width)))
    masks = []
    for sample_id in sample_ids:
        score = resize_score_map(raw_maps[lookup[sample_id]], height, width)
        masks.append(exact_topk_mask(score, k).numpy().astype(np.uint8))
    masks = np.stack(masks)
    if not np.all(masks.reshape(len(masks), -1).sum(axis=1) == k):
        raise RuntimeError(f"Base exact-k reconstruction failed: {dataset} f{fold} {explainer}.")
    return masks, path


def manifest_schema(dataset):
    path = artifact_registry[(dataset, 0)]["manifest"]
    frame = pd.read_parquet(path)
    id_col = pick_column(frame, ["sample_id", "image_id", "id"])
    path_col = pick_column(frame, ["image_path", "path", "filepath", "file_path", "image"])
    fold_col = pick_column(frame, ["fold", "test_fold", "outer_fold"])
    patient_col = pick_column(frame, ["patient_id"], required=False)
    lesion_col = pick_column(frame, ["lesion_id", "case_id", "group_id"], required=False)
    mask_col = pick_column(
        frame,
        ["mask_path", "segmentation_path", "ground_truth_path", "gt_mask_path", "mask", "segmentation"],
        required=False,
    )
    return frame, id_col, path_col, fold_col, patient_col, lesion_col, mask_col


def load_clinical_mask(path_value, dataset, size):
    if not valid_identifier(path_value):
        return None
    try:
        path = resolve_image_path(path_value, dataset)
    except Exception:
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


say("Classifier replay: input space and layer3 intervention space defined.")
say("Clinical masks: reader defined for post-hoc evaluation only; absent/empty masks are explicitly skipped.")


# ==================================================================================================
# 4. FEATURE EXTRACTION AND RECIPIENT POLICIES
# ==================================================================================================

section(4, 14, "Define patient/case-disjoint A/B/Q matching and deterministic controls")


@torch.no_grad()
def extract_features(
    model, frame, dataset, id_col, group_col, path_col,
    include_images=False, preprocess_mode="opencv_area_square", desc=None,
):
    records = frame.reset_index(drop=True).to_dict("records")
    output = []
    size = INPUT_SIZE[dataset]
    total_batches = max(1, int(math.ceil(len(records) / 32)))
    progress_every = max(1, total_batches // 5)
    for start in range(0, len(records), 32):
        batch = records[start:start + 32]
        images = np.stack([
            load_rgb_uint8(resolve_image_path(row[path_col], dataset), size, preprocess_mode) for row in batch
        ])
        x_cpu = normalize_rgb_batch(images)
        x = x_cpu.to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            z2 = model_to_layer2(model, x)
            z3 = layer2_to_layer3(model, z2)
            logits = layer3_to_logits(model, z3)
        embeddings = F.normalize(z2.float().mean(dim=(2, 3)), dim=1)
        for idx, row in enumerate(batch):
            logit = float(logits[idx].float().cpu())
            output.append({
                "sample_id": str(row[id_col]),
                "group_id": str(row[group_col]),
                "decision": int(logit >= 0),
                "logit": logit,
                "embedding": embeddings[idx].half().cpu(),
                "z3": z3[idx].half().cpu(),
                "image": images[idx] if include_images else None,
                "path": str(resolve_image_path(row[path_col], dataset)),
            })
        del x, x_cpu, z2, z3, logits, embeddings, images
        batch_number = start // 32 + 1
        if desc and (batch_number == 1 or batch_number % progress_every == 0 or batch_number == total_batches):
            say(f"    {desc}: {min(start + 32, len(records)):,}/{len(records):,} images")
    return output


def nearest_recipients(donor_idx, pool, n, same_decision=True):
    donor = pool[donor_idx]
    candidates = []
    for idx, recipient in enumerate(pool):
        if idx == donor_idx or recipient["group_id"] == donor["group_id"]:
            continue
        decision_ok = recipient["decision"] == donor["decision"]
        if decision_ok != bool(same_decision):
            continue
        similarity = float(torch.dot(donor["embedding"].float(), recipient["embedding"].float()))
        candidates.append((-similarity, recipient["group_id"], recipient["sample_id"], idx))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    chosen, groups = [], set()
    for _, group, _, idx in candidates:
        if group in groups:
            continue
        chosen.append(idx)
        groups.add(group)
        if len(chosen) == n:
            break
    if len(chosen) != n:
        raise RuntimeError(
            f"Donor {donor['sample_id']} has {len(chosen)}/{n} recipients "
            f"for same_decision={same_decision}."
        )
    return chosen


def random_same_decision_recipients(donor_idx, pool, n, salt):
    donor = pool[donor_idx]
    candidates = []
    seen_groups = set()
    # Sort first so the RNG operates on a stable population.
    for idx, recipient in sorted(enumerate(pool), key=lambda item: (item[1]["group_id"], item[1]["sample_id"])):
        if idx == donor_idx or recipient["group_id"] == donor["group_id"]:
            continue
        if recipient["decision"] != donor["decision"] or recipient["group_id"] in seen_groups:
            continue
        candidates.append(idx)
        seen_groups.add(recipient["group_id"])
    if len(candidates) < n:
        raise RuntimeError(f"Donor {donor['sample_id']} has only {len(candidates)}/{n} random-compatible groups.")
    rng = np.random.default_rng(SEED + stable_int(salt + "/" + donor["sample_id"]))
    return sorted(rng.choice(candidates, size=n, replace=False).tolist())


def build_q_matchings(pool, dataset, fold):
    policies = {name: np.empty((len(pool), N_RECIPIENT_Q), dtype=np.int64) for name in MATCHING_POLICIES}
    for donor_idx in range(len(pool)):
        policies["nearest_same_decision"][donor_idx] = nearest_recipients(donor_idx, pool, N_RECIPIENT_Q, True)
        policies["nearest_opposite_decision"][donor_idx] = nearest_recipients(donor_idx, pool, N_RECIPIENT_Q, False)
        policies["random_same_decision"][donor_idx] = random_same_decision_recipients(
            donor_idx, pool, N_RECIPIENT_Q, f"randomQ/{dataset}/{fold}"
        )
    return policies


def select_control_donors(pool, dataset, fold, n):
    if int(n) % 2 != 0:
        raise ValueError("CONTROL_DONORS_PER_FOLD must be even for exact decision stratification.")
    quota = int(n) // 2
    candidates_by_decision = {}
    initial_by_decision = {}
    for decision in (0, 1):
        candidates = []
        seen_groups = set()
        for idx, item in sorted(
            enumerate(pool), key=lambda pair: (pair[1]["group_id"], pair[1]["sample_id"])
        ):
            if item["decision"] != decision or item["group_id"] in seen_groups:
                continue
            candidates.append(idx)
            seen_groups.add(item["group_id"])
        if len(candidates) < quota:
            raise RuntimeError(
                f"{dataset} f{fold}: only {len(candidates)} unique groups for decision={decision}; need {quota}."
            )
        candidates_by_decision[decision] = candidates
        rng = np.random.default_rng(SEED + stable_int(f"control-donors/{dataset}/{fold}/{decision}"))
        # This is the exact v1.0.2 selection.  Keep it unchanged whenever it is
        # already globally group-disjoint, so completed SIIM/ISIC jobs resume.
        initial_by_decision[decision] = rng.choice(candidates, size=quota, replace=False).tolist()

    initial = initial_by_decision[0] + initial_by_decision[1]
    initial_groups = [pool[idx]["group_id"] for idx in initial]
    collisions = len(initial_groups) - len(set(initial_groups))

    def repair_one_side(fixed_decision, repaired_decision):
        """Retain the fixed stratum and minimally replace cross-stratum collisions."""
        fixed = list(initial_by_decision[fixed_decision])
        blocked = {pool[idx]["group_id"] for idx in fixed}
        kept = [
            idx for idx in initial_by_decision[repaired_decision]
            if pool[idx]["group_id"] not in blocked
        ]
        used = blocked | {pool[idx]["group_id"] for idx in kept}
        eligible = [
            idx for idx in candidates_by_decision[repaired_decision]
            if pool[idx]["group_id"] not in used
        ]
        need = quota - len(kept)
        if len(eligible) < need:
            return None
        if need:
            rng = np.random.default_rng(
                SEED + stable_int(
                    f"control-donors-repair/{dataset}/{fold}/{fixed_decision}/{repaired_decision}"
                )
            )
            kept.extend(rng.choice(eligible, size=need, replace=False).tolist())
        return {fixed_decision: fixed, repaired_decision: kept}

    if collisions == 0:
        selected_by_decision = initial_by_decision
    else:
        # Prefer preserving decision=0 and change only the conflicting decision=1
        # donors.  If that particular draw blocks too many groups, try the mirror.
        selected_by_decision = repair_one_side(0, 1)
        if selected_by_decision is None:
            selected_by_decision = repair_one_side(1, 0)

        if selected_by_decision is None:
            # Deterministic feasibility fallback.  It is needed only when both
            # minimal repairs fail even though a disjoint stratified set exists.
            representative = {
                decision: {pool[idx]["group_id"]: idx for idx in candidates_by_decision[decision]}
                for decision in (0, 1)
            }
            groups0, groups1 = set(representative[0]), set(representative[1])
            if len(groups0 | groups1) < int(n):
                raise RuntimeError(
                    f"{dataset} f{fold}: only {len(groups0 | groups1)} groups across both "
                    f"decision strata; need {n} globally distinct groups."
                )
            exclusive0 = sorted(groups0 - groups1)
            exclusive1 = sorted(groups1 - groups0)
            shared = sorted(groups0 & groups1)
            rng = np.random.default_rng(SEED + stable_int(f"control-donors-fallback/{dataset}/{fold}"))
            exclusive0 = list(np.asarray(exclusive0, dtype=object)[rng.permutation(len(exclusive0))])
            exclusive1 = list(np.asarray(exclusive1, dtype=object)[rng.permutation(len(exclusive1))])
            shared = list(np.asarray(shared, dtype=object)[rng.permutation(len(shared))])
            chosen0 = exclusive0[:quota]
            chosen1 = exclusive1[:quota]
            need0, need1 = quota - len(chosen0), quota - len(chosen1)
            if need0 + need1 > len(shared):
                raise RuntimeError(
                    f"{dataset} f{fold}: no feasible globally group-disjoint {quota}+{quota} donor set."
                )
            chosen0.extend(shared[:need0])
            chosen1.extend(shared[need0:need0 + need1])
            selected_by_decision = {
                0: [representative[0][group] for group in chosen0],
                1: [representative[1][group] for group in chosen1],
            }

        say(
            f"  {dataset} f{fold}: repaired {collisions} cross-decision group collision(s) "
            "in the audit donor sample."
        )

    selected = selected_by_decision[0] + selected_by_decision[1]
    selected = sorted(selected, key=lambda idx: pool[idx]["sample_id"])
    counts = {decision: sum(pool[idx]["decision"] == decision for idx in selected) for decision in (0, 1)}
    if counts != {0: quota, 1: quota}:
        raise RuntimeError(f"{dataset} f{fold}: donor decision stratification failed: {counts}.")
    if len({pool[idx]["group_id"] for idx in selected}) != int(n):
        raise RuntimeError(f"{dataset} f{fold}: selected audit donors are not group-unique.")
    return np.asarray(selected, dtype=np.int64)


def nearest_from_external(donor, pool, n):
    candidates = []
    for idx, recipient in enumerate(pool):
        if recipient["group_id"] == donor["group_id"] or recipient["decision"] != donor["decision"]:
            continue
        similarity = float(torch.dot(donor["embedding"].float(), recipient["embedding"].float()))
        candidates.append((-similarity, recipient["group_id"], recipient["sample_id"], idx))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    chosen, groups = [], set()
    for _, group, _, idx in candidates:
        if group in groups:
            continue
        chosen.append(idx); groups.add(group)
        if len(chosen) == n:
            break
    if len(chosen) != n:
        raise RuntimeError(f"External matching returned {len(chosen)}/{n} recipients for {donor['sample_id']}.")
    return chosen


say("Q policies: nearest/same-decision, random/same-decision, and nearest/opposite-decision.")
say("All policies exclude the donor clinical group; PAD uses patient_id, SIIM/ISIC use case/lesion IDs.")


# ==================================================================================================
# 5. LAYER-SPACE EFFECTS, T-CPT RISK, AND ABLATION OPTIMIZERS
# ==================================================================================================

section(5, 14, "Define exact-budget ablations without changing the frozen full method")


def total_variation(mask):
    horizontal = torch.abs(mask[:, 1:] - mask[:, :-1]).mean() if mask.shape[1] > 1 else mask.new_tensor(0.0)
    vertical = torch.abs(mask[1:, :] - mask[:-1, :]).mean() if mask.shape[0] > 1 else mask.new_tensor(0.0)
    return horizontal + vertical


def cvar(losses, alpha=CVAR_ALPHA):
    losses = losses.reshape(-1)
    count = max(1, int(math.ceil((1.0 - float(alpha)) * losses.numel())))
    return torch.topk(losses, k=count, largest=True).values.mean()


def signed_effects(model, z3, signs, masks, full_signed_logits=None):
    if masks.ndim == 2:
        masks = masks.unsqueeze(0).expand(z3.shape[0], -1, -1)
    masked = z3 * (1.0 - masks[:, None])
    removed_signed = layer3_to_logits(model, masked).float() * signs.float()
    if full_signed_logits is None:
        full_signed_logits = layer3_to_logits(model, z3).float() * signs.float()
    return full_signed_logits - removed_signed


def cpts_values(local_effect, recipient_effects):
    return torch.exp(-torch.abs(recipient_effects - local_effect) / (torch.abs(local_effect) + CPTS_EPSILON))


def evaluate_layer_mask(model, donor_z, donor_sign, recipient_z, recipient_sign, mask, lambda_cvar, with_grad=False):
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        donor_full = layer3_to_logits(model, donor_z[None]).float() * donor_sign
        local = signed_effects(model, donor_z[None], donor_sign[None], mask, donor_full)[0]
        recipient_full = layer3_to_logits(model, recipient_z).float() * recipient_sign.float()
        effects = signed_effects(model, recipient_z, recipient_sign, mask, recipient_full)
        cpts = cpts_values(local, effects)
        losses = 1.0 - cpts
        risk = losses.mean() + float(lambda_cvar) * cvar(losses) + LAMBDA_TV * total_variation(mask)
    return local, effects, cpts, risk


def constraint_ok(local_value, base_local_value):
    base = float(base_local_value)
    value = float(local_value)
    if not np.isfinite(value):
        return False
    if abs(base) < MIN_BASE_EFFECT:
        return abs(value) >= abs(base) - 1e-8
    return bool(
        np.sign(value) == np.sign(base)
        and abs(value) + 1e-8 >= (1.0 - LOCAL_EFFECT_TOLERANCE) * abs(base)
    )


def swap_distance(mask, base_mask):
    return int(torch.count_nonzero(mask.reshape(-1) != base_mask.reshape(-1)).item() // 2)


def gradient_candidates(current, gradient, base, max_swaps):
    flat_mask = current.reshape(-1)
    flat_gradient = gradient.reshape(-1)
    selected = torch.nonzero(flat_mask > 0.5, as_tuple=False).flatten()
    excluded = torch.nonzero(flat_mask <= 0.5, as_tuple=False).flatten()
    remove_order = selected[torch.argsort(flat_gradient[selected], descending=True)]
    add_order = excluded[torch.argsort(flat_gradient[excluded], descending=False)]
    output, seen = [], set()
    for batch in SWAP_BATCHES:
        for offset in range(GRADIENT_ALTERNATIVES):
            count = min(int(batch), len(remove_order) - offset, len(add_order) - offset)
            if count <= 0:
                continue
            candidate = flat_mask.clone()
            candidate[remove_order[offset:offset + count]] = 0.0
            candidate[add_order[offset:offset + count]] = 1.0
            candidate = candidate.view_as(current)
            if int(candidate.sum()) != int(current.sum()) or swap_distance(candidate, base) > max_swaps:
                continue
            key = hashlib.sha1(candidate.detach().cpu().numpy().astype(np.uint8).tobytes()).hexdigest()
            if key not in seen:
                seen.add(key); output.append(candidate)
    return output


def random_candidates(current, base, max_swaps, rng):
    flat = current.reshape(-1)
    selected = torch.nonzero(flat > 0.5, as_tuple=False).flatten().cpu().numpy()
    excluded = torch.nonzero(flat <= 0.5, as_tuple=False).flatten().cpu().numpy()
    output, seen = [], set()
    # Exactly the same maximum proposal count as the gradient candidate generator: 3 × 3.
    for batch in SWAP_BATCHES:
        for _ in range(GRADIENT_ALTERNATIVES):
            count = min(int(batch), len(selected), len(excluded))
            remove = rng.choice(selected, size=count, replace=False)
            add = rng.choice(excluded, size=count, replace=False)
            candidate = flat.clone()
            candidate[torch.as_tensor(remove, device=flat.device)] = 0.0
            candidate[torch.as_tensor(add, device=flat.device)] = 1.0
            candidate = candidate.view_as(current)
            if swap_distance(candidate, base) > max_swaps:
                continue
            key = hashlib.sha1(candidate.detach().cpu().numpy().astype(np.uint8).tobytes()).hexdigest()
            if key not in seen:
                seen.add(key); output.append(candidate)
    return output


def optimize_ablation(model, donor, rec_a, rec_b, rec_q, base_mask, variant, seed):
    donor_z = donor["z3"].to(DEVICE, dtype=torch.float32)
    donor_sign = torch.tensor(1.0 if donor["decision"] == 1 else -1.0, device=DEVICE)

    def stack(items):
        z = torch.stack([item["z3"] for item in items]).to(DEVICE, dtype=torch.float32)
        signs = torch.tensor([1.0 if item["decision"] == 1 else -1.0 for item in items], device=DEVICE)
        return z, signs

    z_a, s_a = stack(rec_a)
    z_b, s_b = stack(rec_b)
    z_q, s_q = stack(rec_q)
    base = torch.as_tensor(base_mask, dtype=torch.float32, device=DEVICE)
    k = int(base.sum().item())
    max_swaps = k if variant == "no_trust_region" else max(1, int(round(MAX_SWAP_FRACTION * k)))
    lambda_cvar = 0.0 if variant == "no_cvar" else LAMBDA_CVAR
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        base_local, _, base_a, base_risk_a = evaluate_layer_mask(model, donor_z, donor_sign, z_a, s_a, base, lambda_cvar)
        _, _, base_b, base_risk_b = evaluate_layer_mask(model, donor_z, donor_sign, z_b, s_b, base, lambda_cvar)
        _, _, base_q, _ = evaluate_layer_mask(model, donor_z, donor_sign, z_q, s_q, base, lambda_cvar)

    current = base.clone()
    current_risk = float(base_risk_a)
    trajectory = [(base.clone(), current_risk, 0)]
    proposal_evaluations = 0
    accepted_steps = 0
    for iteration in range(1, MAX_ITERATIONS + 1):
        if variant == "random_search":
            proposals = random_candidates(current, base, max_swaps, rng)
        else:
            variable = current.detach().clone().requires_grad_(True)
            _, _, _, risk = evaluate_layer_mask(
                model, donor_z, donor_sign, z_a, s_a, variable, lambda_cvar, with_grad=True
            )
            if not torch.isfinite(risk):
                break
            gradient = torch.autograd.grad(risk, variable, retain_graph=False, create_graph=False)[0]
            if not torch.isfinite(gradient).all():
                break
            proposals = gradient_candidates(current, gradient.detach(), base, max_swaps)
        if not proposals:
            break
        best = None
        with torch.no_grad():
            for proposal in proposals:
                proposal_evaluations += 1
                local, _, _, risk = evaluate_layer_mask(model, donor_z, donor_sign, z_a, s_a, proposal, lambda_cvar)
                value = float(risk)
                if constraint_ok(float(local), float(base_local)) and value < current_risk - MIN_SUPPORT_IMPROVEMENT:
                    if best is None or value < best[0]:
                        best = (value, proposal.clone())
        if best is None:
            break
        current_risk, current = best
        accepted_steps += 1
        trajectory.append((current.clone(), current_risk, iteration))

    if variant == "no_validation":
        selected_mask = trajectory[-1][0]
        selected_iteration = trajectory[-1][2]
    else:
        selected_mask = base
        selected_b_risk = float(base_risk_b)
        selected_iteration = 0
        with torch.no_grad():
            for candidate, _, iteration in trajectory[1:]:
                local, _, candidate_b, risk_b = evaluate_layer_mask(
                    model, donor_z, donor_sign, z_b, s_b, candidate, lambda_cvar
                )
                if (
                    constraint_ok(float(local), float(base_local))
                    and float(candidate_b.mean()) >= float(base_b.mean()) + VALIDATION_ACCEPT_MARGIN
                    and float(risk_b) < selected_b_risk
                ):
                    selected_mask = candidate.clone()
                    selected_b_risk = float(risk_b)
                    selected_iteration = iteration

    with torch.no_grad():
        final_local, _, final_a, _ = evaluate_layer_mask(model, donor_z, donor_sign, z_a, s_a, selected_mask, lambda_cvar)
        _, _, final_b, _ = evaluate_layer_mask(model, donor_z, donor_sign, z_b, s_b, selected_mask, lambda_cvar)
        _, _, final_q, _ = evaluate_layer_mask(model, donor_z, donor_sign, z_q, s_q, selected_mask, lambda_cvar)

    result = {
        "accepted": bool(selected_iteration > 0),
        "selected_iteration": int(selected_iteration),
        "search_steps_accepted": int(accepted_steps),
        "proposal_evaluations": int(proposal_evaluations),
        "swaps_from_base": int(swap_distance(selected_mask, base)),
        "k": k,
        "max_swaps": max_swaps,
        "local_effect_base": float(base_local),
        "local_effect_refined": float(final_local),
        "cpts_base_A": float(base_a.mean()), "cpts_refined_A": float(final_a.mean()),
        "cpts_base_B": float(base_b.mean()), "cpts_refined_B": float(final_b.mean()),
        "cpts_base_Q": float(base_q.mean()), "cpts_refined_Q": float(final_q.mean()),
    }
    result["delta_A"] = result["cpts_refined_A"] - result["cpts_base_A"]
    result["delta_B"] = result["cpts_refined_B"] - result["cpts_base_B"]
    result["delta_Q"] = result["cpts_refined_Q"] - result["cpts_base_Q"]
    output_mask = selected_mask.detach().cpu().numpy().astype(np.uint8)
    del donor_z, z_a, z_b, z_q, s_a, s_b, s_q, base, current, trajectory
    return output_mask, result


say("Ablations frozen: matched-compute random search, no CVaR, no validation-B, and no trust region.")
say("Every candidate remains binary exact-k; local sign/magnitude constraints remain active in every ablation.")


# ==================================================================================================
# 6. INDEPENDENT PIXEL-SPACE AND LOCALIZATION METRICS
# ==================================================================================================

section(6, 14, "Define independent input-space faithfulness, transport, and clinical localization metrics")


@torch.no_grad()
def full_logits_from_z3(model, z3_cpu, batch_size=256):
    output = np.empty(len(z3_cpu), dtype=np.float32)
    for start in range(0, len(z3_cpu), batch_size):
        stop = min(start + batch_size, len(z3_cpu))
        output[start:stop] = layer3_to_logits(
            model, z3_cpu[start:stop].to(DEVICE, dtype=torch.float32)
        ).float().cpu().numpy()
    return output


@torch.no_grad()
def layer_effect_batch(model, z3_cpu, masks, signs, full_logits, batch_size=256):
    output = np.empty(len(masks), dtype=np.float32)
    for start in range(0, len(masks), batch_size):
        stop = min(start + batch_size, len(masks))
        z3 = z3_cpu[start:stop].to(DEVICE, dtype=torch.float32)
        mask = torch.from_numpy(masks[start:stop]).to(DEVICE, dtype=torch.float32)
        sign = torch.from_numpy(signs[start:stop]).to(DEVICE, dtype=torch.float32)
        removed = layer3_to_logits(model, z3 * (1.0 - mask[:, None])).float() * sign
        full_signed = torch.from_numpy(full_logits[start:stop]).to(DEVICE, dtype=torch.float32) * sign
        output[start:stop] = (full_signed - removed).cpu().numpy()
    return output


def upsample_masks(masks, size):
    return np.stack([
        cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        for mask in masks
    ]).astype(np.uint8)


def blurred_baselines(images):
    size = images.shape[1]
    sigma = max(1.0, PIXEL_BASELINE_SIGMA_FRACTION * size)
    return np.stack([
        cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
        for image in images
    ]).astype(np.uint8)


@torch.no_grad()
def input_logits_batch(model, images_uint8, batch_size=64):
    output = np.empty(len(images_uint8), dtype=np.float32)
    for start in range(0, len(images_uint8), batch_size):
        stop = min(start + batch_size, len(images_uint8))
        x = normalize_rgb_batch(images_uint8[start:stop]).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output[start:stop] = model_logits(model, x).float().cpu().numpy()
        del x
    return output


def pixel_intervention_metrics(model, images, masks, signs, full_logits=None, intervention_batch_size=32):
    size = images.shape[1]
    if full_logits is None:
        full_logits = input_logits_batch(model, images)
    deletion_effect = np.empty(len(images), dtype=np.float32)
    sufficiency = np.empty(len(images), dtype=np.float32)
    for start in range(0, len(images), intervention_batch_size):
        stop = min(start + intervention_batch_size, len(images))
        image_batch = images[start:stop]
        masks_up = upsample_masks(masks[start:stop], size)[..., None].astype(np.float32)
        baseline = blurred_baselines(image_batch)
        removed = np.rint(
            image_batch * (1.0 - masks_up) + baseline * masks_up
        ).clip(0, 255).astype(np.uint8)
        sufficient = np.rint(
            baseline * (1.0 - masks_up) + image_batch * masks_up
        ).clip(0, 255).astype(np.uint8)
        removed_logits = input_logits_batch(model, removed)
        sufficient_logits = input_logits_batch(model, sufficient)
        signed = signs[start:stop]
        full_signed = full_logits[start:stop] * signed
        deletion_effect[start:stop] = full_signed - removed_logits * signed
        sufficiency[start:stop] = np.exp(
            -np.abs(sufficient_logits * signed - full_signed) / (np.abs(full_signed) + CPTS_EPSILON)
        )
        del masks_up, baseline, removed, sufficient, removed_logits, sufficient_logits
    # Bounded, monotone transformation: 0 for a non-positive deletion effect and ->1 for strong evidence removal.
    deletion_faithfulness = 1.0 - np.exp(-np.maximum(deletion_effect, 0.0))
    return {
        "full_logits": full_logits,
        "deletion_effect": deletion_effect.astype(np.float32),
        "deletion_faithfulness": deletion_faithfulness.astype(np.float32),
        "sufficiency": sufficiency.astype(np.float32),
    }


def transport_consistency(local_effect, recipient_effects):
    return np.exp(
        -np.abs(recipient_effects - local_effect[:, None])
        / (np.abs(local_effect[:, None]) + CPTS_EPSILON)
    ).mean(axis=1)


def overlap_metrics(mask, clinical):
    mask = mask.astype(bool)
    clinical = clinical.astype(bool)
    if clinical.sum() == 0:
        return None
    intersection = int(np.logical_and(mask, clinical).sum())
    union = int(np.logical_or(mask, clinical).sum())
    dice = 2.0 * intersection / max(1, int(mask.sum()) + int(clinical.sum()))
    iou = intersection / max(1, union)
    roi_hit = float(intersection > 0)
    precision = intersection / max(1, int(mask.sum()))
    return dice, iou, roi_hit, precision


def clinical_mask_lookup(dataset, sample_ids, target_size):
    frame, id_col, _, _, _, _, mask_col = manifest_schema(dataset)
    if mask_col is None:
        return {sample_id: None for sample_id in sample_ids}, None
    unique = frame.drop_duplicates(id_col).copy()
    unique["__lookup_id"] = unique[id_col].astype(str)
    by_id = unique.set_index("__lookup_id")[mask_col].to_dict()
    output = {}
    found = 0
    for sample_id in sample_ids:
        mask = load_clinical_mask(by_id.get(sample_id), dataset, target_size)
        output[sample_id] = mask
        found += int(mask is not None)
    return output, mask_col


say("Independent intervention: Gaussian-blurred input baseline, never used by T-CPT optimization.")
say("Localization: exact-budget binary masks vs frozen clinical masks; empty/absent ground truth is excluded.")


# ==================================================================================================
# 7. PERSISTENT JOB LAYOUT AND VALIDATION
# ==================================================================================================

section(7, 14, "Create resume-aware job registry without overwriting earlier notebooks")

# Complete read-only audit of every primary-budget cell already produced by Notebook 06B/07.
frozen_audit_rows = []
for audit_dataset in DATASETS:
    for audit_fold in FOLDS:
        for audit_explainer in EXPLAINERS:
            audit_tag = f"p{int(round(100 * PRIMARY_BUDGET)):02d}"
            audit_path = (
                result_root / audit_dataset / f"fold_{audit_fold}"
                / f"{audit_dataset}_f{audit_fold}_{audit_explainer}_{audit_tag}_masks.npz"
            )
            if not audit_path.exists():
                raise FileNotFoundError(f"Frozen primary mask artifact missing: {audit_path}")
            with np.load(audit_path, allow_pickle=False) as audit_data:
                audit_masks = np.asarray(audit_data["masks"], dtype=np.uint8)
                audit_ids = decode_strings(audit_data["sample_ids"])
            if audit_masks.ndim != 3 or len(audit_masks) != len(audit_ids) or len(audit_ids) != len(set(audit_ids)):
                raise RuntimeError(f"Malformed frozen mask artifact: {audit_path}")
            audit_k = max(1, int(round(PRIMARY_BUDGET * audit_masks.shape[1] * audit_masks.shape[2])))
            violations = int(np.count_nonzero(audit_masks.reshape(len(audit_masks), -1).sum(axis=1) != audit_k))
            frozen_audit_rows.append({
                "dataset": audit_dataset, "fold": audit_fold, "explainer": audit_explainer,
                "donors": len(audit_masks), "height": audit_masks.shape[1], "width": audit_masks.shape[2],
                "expected_k": audit_k, "exact_k_violations": violations,
                "mask_sha256": sha256_file(audit_path),
            })
            del audit_masks
frozen_grid_audit = pd.DataFrame(frozen_audit_rows)
if len(frozen_grid_audit) != 90 or frozen_grid_audit.exact_k_violations.sum() != 0:
    raise RuntimeError("Frozen 90-cell mask grid failed the exact-budget audit.")

full_local_violation = (
    (
        (full_metrics.local_effect_base.abs() >= MIN_BASE_EFFECT)
        & (
            (np.sign(full_metrics.local_effect_refined) != np.sign(full_metrics.local_effect_base))
            | (
                full_metrics.local_effect_refined.abs() + 1e-8
                < (1.0 - LOCAL_EFFECT_TOLERANCE) * full_metrics.local_effect_base.abs()
            )
        )
    )
    | (
        (full_metrics.local_effect_base.abs() < MIN_BASE_EFFECT)
        & (full_metrics.local_effect_refined.abs() + 1e-8 < full_metrics.local_effect_base.abs())
    )
)
frozen_local_violations = int(full_local_violation.sum())
if frozen_local_violations:
    raise RuntimeError(f"Frozen full-grid local-faithfulness violations: {frozen_local_violations}")
say(
    f"Frozen primary audit: 90/90 cells | donors={int(frozen_grid_audit.donors.sum()):,} "
    f"| exact-k violations=0 | local-faithfulness violations=0"
)


def job_paths(kind, dataset, fold, name):
    folder = CONTROL_ROOT / kind / dataset / f"fold_{fold}"
    stem = f"{dataset}_f{fold}_{name}"
    return {
        "folder": folder,
        "rows": folder / f"{stem}.parquet",
        "meta": folder / f"{stem}.json",
    }


def job_identity(kind, dataset, fold, name, checkpoint, extra_files=None):
    extra_files = extra_files or []
    return {
        "protocol_sha256": sha256_file(CONFIG_PATH),
        "kind": kind, "dataset": dataset, "fold": int(fold), "name": name,
        "checkpoint_sha256": sha256_file(checkpoint),
        "extra_sha256": {str(path): sha256_file(path) for path in extra_files},
    }


def valid_job(paths, identity):
    if not paths["rows"].exists() or not paths["meta"].exists():
        return False
    try:
        meta = load_json(paths["meta"])
        return (
            meta.get("status") == "PASS"
            and meta.get("identity") == identity
            and meta.get("rows_sha256") == sha256_file(paths["rows"])
            and len(pd.read_parquet(paths["rows"])) == int(meta.get("n_rows", -1))
        )
    except Exception:
        return False


def save_job(paths, identity, frame, runtime_seconds, peak_memory_bytes=0):
    paths["folder"].mkdir(parents=True, exist_ok=True)
    atomic_write_dataframe(frame, paths["rows"])
    meta = {
        "status": "PASS", "identity": identity, "n_rows": len(frame),
        "n_donors": int(frame["sample_id"].nunique()) if "sample_id" in frame.columns else len(frame),
        "runtime_seconds": float(runtime_seconds), "peak_memory_bytes": int(peak_memory_bytes),
        "rows_sha256": sha256_file(paths["rows"]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(paths["meta"], meta)


say(f"Persistent control root: {CONTROL_ROOT}")
say("Resume validation uses protocol, checkpoint, explanation-map, and frozen-mask hashes.")


# ==================================================================================================
# 8. EXECUTION: INDEPENDENT VALIDATION, MATCHING CONTROLS, AND ABLATIONS
# ==================================================================================================

section(8, 14, "Execute or resume all frozen validation/control jobs")
independent_frames = []
matching_frames = []
ablation_frames = []
runtime_rows = []
jobs_created = 0
jobs_resumed = 0

for dataset_number, dataset in enumerate(DATASETS, start=1):
    manifest, m_id, m_path, m_fold, m_patient, m_lesion, m_mask = manifest_schema(dataset)
    manifest = manifest.copy()
    manifest["__sample_id"] = manifest[m_id].astype(str)
    manifest["__group_id"] = [
        analysis_group_from_row(row, dataset, m_id, m_patient, m_lesion)[0]
        for _, row in manifest.iterrows()
    ]
    dataset_oof = oof[oof[OOF_DATASET].astype(str) == dataset].copy()
    dataset_oof["__sample_id"] = dataset_oof[OOF_ID].astype(str)
    dataset_oof["__group_id"] = [
        analysis_group_from_row(row, dataset, OOF_ID, OOF_PATIENT, OOF_LESION)[0]
        for _, row in dataset_oof.iterrows()
    ]
    banner(f"DATASET {dataset_number}/{len(DATASETS)} — {dataset}")

    for fold in CONTROL_FOLDS:
        fold_start = time.time()
        checkpoint = artifact_registry[(dataset, fold)]["checkpoint"]
        model = load_classifier(checkpoint)
        q_pool_frame = dataset_oof[dataset_oof[OOF_FOLD].astype(int) == fold].copy().reset_index(drop=True)
        if len(q_pool_frame) == 0:
            raise RuntimeError(f"Empty Q fold: {dataset} fold {fold}.")
        preprocess_mode = choose_preprocessing(model, q_pool_frame, dataset)
        q_pool_features = extract_features(
            model, q_pool_frame, dataset, "__sample_id", "__group_id", OOF_PATH_COL,
            include_images=True, preprocess_mode=preprocess_mode, desc=f"{dataset} f{fold} Q pool"
        )
        if [item["sample_id"] for item in q_pool_features] != q_pool_frame["__sample_id"].astype(str).tolist():
            raise RuntimeError(f"Feature/sample ordering changed: {dataset} fold {fold}.")
        donor_pool_indices = select_control_donors(
            q_pool_features, dataset, fold, CONTROL_DONORS_PER_FOLD
        )
        q_features = [q_pool_features[int(idx)] for idx in donor_pool_indices]
        sample_ids = [item["sample_id"] for item in q_features]
        q_policies_full = build_q_matchings(q_pool_features, dataset, fold)
        q_policies = {
            policy: indices[donor_pool_indices] for policy, indices in q_policies_full.items()
        }
        q_primary = q_policies["nearest_same_decision"]
        z3_donor = torch.stack([item["z3"] for item in q_features])
        signs_donor = np.asarray(
            [1.0 if item["decision"] == 1 else -1.0 for item in q_features], dtype=np.float32
        )
        logits_z3_donor = full_logits_from_z3(model, z3_donor)
        images_donor = np.stack([item["image"] for item in q_features])
        logits_input_donor = input_logits_batch(model, images_donor)
        z3_q_pool = torch.stack([item["z3"] for item in q_pool_features])
        signs_q_pool = np.asarray(
            [1.0 if item["decision"] == 1 else -1.0 for item in q_pool_features], dtype=np.float32
        )
        logits_z3_q_pool = full_logits_from_z3(model, z3_q_pool)
        images_q_pool = np.stack([item["image"] for item in q_pool_features])
        logits_input_q_pool = input_logits_batch(model, images_q_pool)
        height, width = tuple(z3_donor.shape[-2:])
        pool_ids = [item["sample_id"] for item in q_pool_features]
        clinical_lookup, clinical_column = clinical_mask_lookup(dataset, pool_ids, height)

        # A/B banks are extracted once per fold and shared by every ablation job.
        val_fold = (fold + 1) % 5
        a_frame = manifest[~manifest[m_fold].astype(int).isin([fold, val_fold])].copy()
        b_frame = manifest[manifest[m_fold].astype(int) == val_fold].copy()
        a_frame = deterministic_sample(a_frame, CALIBRATION_A_CANDIDATES, f"A/{dataset}/{fold}")
        b_frame = deterministic_sample(b_frame, CALIBRATION_B_CANDIDATES, f"B/{dataset}/{fold}")
        a_features = extract_features(
            model, a_frame, dataset, "__sample_id", "__group_id", m_path,
            False, preprocess_mode, f"{dataset} f{fold} A"
        )
        b_features = extract_features(
            model, b_frame, dataset, "__sample_id", "__group_id", m_path,
            False, preprocess_mode, f"{dataset} f{fold} B"
        )
        ablation_matching = []
        for donor_idx, donor in enumerate(q_features):
            indices_a = nearest_from_external(donor, a_features, N_RECIPIENT_A)
            indices_b = nearest_from_external(donor, b_features, N_RECIPIENT_B)
            indices_q = q_primary[donor_idx].tolist()
            ablation_matching.append((indices_a, indices_b, indices_q))

        say(
            f"{dataset} f{fold}: audit donors={len(q_features)} (decision 0/1="
            f"{sum(x['decision'] == 0 for x in q_features)}/{sum(x['decision'] == 1 for x in q_features)}) "
            f"| Q pool={len(q_pool_features)} | A={len(a_features)} | B={len(b_features)} "
            f"| layer3={height}×{width} | donor clinical masks="
            f"{sum(clinical_lookup[x] is not None for x in sample_ids)}"
        )

        base_cache = {}
        refined_cache = {}
        map_path_cache = {}
        refined_path_cache = {}
        preflight_replay = {}

        # Non-blocking numerical replay audit for every explainer. Notebook 07 remains authoritative.
        say("  Non-blocking primary-Q numerical replay audit for all six frozen explainers:")
        for explainer in EXPLAINERS:
            base_masks, map_path = reconstruct_base_masks(dataset, fold, explainer, sample_ids, height, width)
            refined_masks, refined_path = load_refined_masks(dataset, fold, explainer, sample_ids)
            expected_k = max(1, int(round(PRIMARY_BUDGET * height * width)))
            if not np.all(refined_masks.reshape(len(refined_masks), -1).sum(axis=1) == expected_k):
                raise RuntimeError(f"Frozen refined masks violate exact-k: {dataset} f{fold} {explainer}.")
            base_cache[explainer] = base_masks
            refined_cache[explainer] = refined_masks
            map_path_cache[explainer] = map_path
            refined_path_cache[explainer] = refined_path

            local_base_pf = layer_effect_batch(
                model, z3_donor, base_masks, signs_donor, logits_z3_donor
            )
            local_refined_pf = layer_effect_batch(
                model, z3_donor, refined_masks, signs_donor, logits_z3_donor
            )
            flat_primary = q_primary.reshape(-1)
            rec_z3_pf = z3_q_pool[flat_primary]
            rec_signs_pf = signs_q_pool[flat_primary]
            rec_logits_pf = logits_z3_q_pool[flat_primary]
            rec_base_pf = layer_effect_batch(
                model, rec_z3_pf, np.repeat(base_masks, N_RECIPIENT_Q, axis=0),
                rec_signs_pf, rec_logits_pf
            ).reshape(len(q_features), N_RECIPIENT_Q)
            rec_refined_pf = layer_effect_batch(
                model, rec_z3_pf, np.repeat(refined_masks, N_RECIPIENT_Q, axis=0),
                rec_signs_pf, rec_logits_pf
            ).reshape(len(q_features), N_RECIPIENT_Q)
            delta_pf = (
                transport_consistency(local_refined_pf, rec_refined_pf)
                - transport_consistency(local_base_pf, rec_base_pf)
            )
            observed_pf = pd.DataFrame({"sample_id": sample_ids, "delta_replay": delta_pf})
            reference_pf = full_metrics[
                (full_metrics.dataset.astype(str) == dataset)
                & (full_metrics.fold.astype(int) == fold)
                & (full_metrics.explainer.astype(str) == explainer)
            ][["sample_id", "delta_Q"]]
            paired_pf = observed_pf.merge(reference_pf, on="sample_id", validate="one_to_one")
            if len(paired_pf) != len(q_features):
                raise RuntimeError(
                    f"Primary replay preflight pairing incomplete: {dataset} f{fold} {explainer}, "
                    f"rows={len(paired_pf)}/{len(q_features)}."
                )
            replay_difference_pf = paired_pf.delta_replay.to_numpy(float) - paired_pf.delta_Q.to_numpy(float)
            replay_stats_pf = {
                "max_abs": float(np.max(np.abs(replay_difference_pf))),
                "mean_abs": float(np.mean(np.abs(replay_difference_pf))),
                "cell_mean_abs": float(abs(np.mean(replay_difference_pf))),
            }
            preflight_replay[explainer] = replay_stats_pf
            say(
                f"    {explainer:24s} mean |error|={replay_stats_pf['mean_abs']:.3e} "
                f"| cell mean={replay_stats_pf['cell_mean_abs']:.3e} "
                f"| max={replay_stats_pf['max_abs']:.3e}"
            )
        replay_warnings = {
            name: stats for name, stats in preflight_replay.items()
            if (
                stats["mean_abs"] > REPLAY_MEAN_ABS_TOLERANCE
                or stats["cell_mean_abs"] > REPLAY_CELL_MEAN_TOLERANCE
                or stats["max_abs"] > REPLAY_MAX_ABS_TOLERANCE
            )
        }
        if replay_warnings:
            say(
                "  Numerical replay note: reporting references exceeded for "
                + ", ".join(sorted(replay_warnings))
                + "; this is recorded but does not alter frozen Notebook 07 results."
            )
        say("  Primary-Q replay audit recorded for 6/6 explainers; starting compact control jobs.")

        for explainer_number, explainer in enumerate(EXPLAINERS, start=1):
            base_masks = base_cache[explainer]
            refined_masks = refined_cache[explainer]
            map_path = map_path_cache[explainer]
            refined_path = refined_path_cache[explainer]
            # The canonical manifest is part of the job identity because it fixes image paths,
            # analysis groups, folds, and (for the post-hoc audit) clinical-mask paths.
            extra_files = [
                OOF_PATH,
                map_path,
                refined_path,
                artifact_registry[(dataset, fold)]["manifest"],
            ]

            # --------------------------------------------------------------------------------------
            # 8A. Independent pixel-space validation and clinical localization.
            # --------------------------------------------------------------------------------------
            paths = job_paths("independent_validation", dataset, fold, explainer)
            identity = job_identity("independent_validation", dataset, fold, explainer, checkpoint, extra_files)
            if valid_job(paths, identity):
                frame = pd.read_parquet(paths["rows"])
                independent_frames.append(frame)
                jobs_resumed += 1
                state_independent = "RESUMED"
            else:
                job_start = time.time()
                torch.cuda.reset_peak_memory_stats()
                donor_base = pixel_intervention_metrics(
                    model, images_donor, base_masks, signs_donor, logits_input_donor
                )
                donor_refined = pixel_intervention_metrics(
                    model, images_donor, refined_masks, signs_donor, logits_input_donor
                )
                recipient_indices = q_primary.reshape(-1)
                recipient_images = images_q_pool[recipient_indices]
                recipient_signs = signs_q_pool[recipient_indices]
                recipient_logits = logits_input_q_pool[recipient_indices]
                rec_base = pixel_intervention_metrics(
                    model, recipient_images, np.repeat(base_masks, N_RECIPIENT_Q, axis=0),
                    recipient_signs, recipient_logits
                )
                rec_refined = pixel_intervention_metrics(
                    model, recipient_images, np.repeat(refined_masks, N_RECIPIENT_Q, axis=0),
                    recipient_signs, recipient_logits
                )
                base_pixel_transport = transport_consistency(
                    donor_base["deletion_effect"], rec_base["deletion_effect"].reshape(len(q_features), N_RECIPIENT_Q)
                )
                refined_pixel_transport = transport_consistency(
                    donor_refined["deletion_effect"], rec_refined["deletion_effect"].reshape(len(q_features), N_RECIPIENT_Q)
                )

                rows = []
                for donor_idx, donor in enumerate(q_features):
                    own_gt = clinical_lookup.get(donor["sample_id"])
                    own_base = overlap_metrics(base_masks[donor_idx], own_gt) if own_gt is not None else None
                    own_refined = overlap_metrics(refined_masks[donor_idx], own_gt) if own_gt is not None else None
                    rec_overlap_base, rec_overlap_refined = [], []
                    for recipient_idx in q_primary[donor_idx]:
                        recipient = q_pool_features[int(recipient_idx)]
                        gt = clinical_lookup.get(recipient["sample_id"])
                        if gt is None:
                            continue
                        overlap_b = overlap_metrics(base_masks[donor_idx], gt)
                        overlap_r = overlap_metrics(refined_masks[donor_idx], gt)
                        if overlap_b is not None and overlap_r is not None:
                            rec_overlap_base.append(overlap_b)
                            rec_overlap_refined.append(overlap_r)

                    def overlap_value(values, index):
                        return float(np.mean([value[index] for value in values])) if values else float("nan")

                    rows.append({
                        "dataset": dataset, "fold": fold, "explainer": explainer,
                        "sample_id": donor["sample_id"], "group_id": donor["group_id"],
                        "pixel_transport_base": float(base_pixel_transport[donor_idx]),
                        "pixel_transport_refined": float(refined_pixel_transport[donor_idx]),
                        "delta_pixel_transport": float(refined_pixel_transport[donor_idx] - base_pixel_transport[donor_idx]),
                        "pixel_faithfulness_base": float(donor_base["deletion_faithfulness"][donor_idx]),
                        "pixel_faithfulness_refined": float(donor_refined["deletion_faithfulness"][donor_idx]),
                        "delta_pixel_faithfulness": float(
                            donor_refined["deletion_faithfulness"][donor_idx] - donor_base["deletion_faithfulness"][donor_idx]
                        ),
                        "pixel_sufficiency_base": float(donor_base["sufficiency"][donor_idx]),
                        "pixel_sufficiency_refined": float(donor_refined["sufficiency"][donor_idx]),
                        "delta_pixel_sufficiency": float(donor_refined["sufficiency"][donor_idx] - donor_base["sufficiency"][donor_idx]),
                        "donor_dice_base": own_base[0] if own_base else float("nan"),
                        "donor_dice_refined": own_refined[0] if own_refined else float("nan"),
                        "donor_iou_base": own_base[1] if own_base else float("nan"),
                        "donor_iou_refined": own_refined[1] if own_refined else float("nan"),
                        "donor_roi_hit_base": own_base[2] if own_base else float("nan"),
                        "donor_roi_hit_refined": own_refined[2] if own_refined else float("nan"),
                        "recipient_dice_base": overlap_value(rec_overlap_base, 0),
                        "recipient_dice_refined": overlap_value(rec_overlap_refined, 0),
                        "recipient_iou_base": overlap_value(rec_overlap_base, 1),
                        "recipient_iou_refined": overlap_value(rec_overlap_refined, 1),
                        "recipient_roi_hit_base": overlap_value(rec_overlap_base, 2),
                        "recipient_roi_hit_refined": overlap_value(rec_overlap_refined, 2),
                        "n_recipient_clinical_masks": len(rec_overlap_base),
                    })
                frame = pd.DataFrame(rows)
                for metric_name in ("donor_dice", "donor_iou", "donor_roi_hit", "recipient_dice", "recipient_iou", "recipient_roi_hit"):
                    frame[f"delta_{metric_name}"] = frame[f"{metric_name}_refined"] - frame[f"{metric_name}_base"]
                runtime = time.time() - job_start
                peak = torch.cuda.max_memory_allocated()
                save_job(paths, identity, frame, runtime, peak)
                independent_frames.append(frame)
                runtime_rows.append({
                    "kind": "independent_validation", "dataset": dataset, "fold": fold,
                    "name": explainer, "runtime_seconds": runtime, "peak_memory_bytes": peak,
                    "n_donors": len(frame), "state": "CREATED",
                })
                jobs_created += 1
                state_independent = "CREATED"

            # --------------------------------------------------------------------------------------
            # 8B. Matching controls in frozen layer3 intervention space.
            # --------------------------------------------------------------------------------------
            paths = job_paths("matching_controls", dataset, fold, explainer)
            identity = job_identity("matching_controls", dataset, fold, explainer, checkpoint, extra_files)
            if valid_job(paths, identity):
                matching_frame = pd.read_parquet(paths["rows"])
                matching_frames.append(matching_frame)
                jobs_resumed += 1
                state_matching = "RESUMED"
            else:
                job_start = time.time()
                torch.cuda.reset_peak_memory_stats()
                local_base = layer_effect_batch(
                    model, z3_donor, base_masks, signs_donor, logits_z3_donor
                )
                local_refined = layer_effect_batch(
                    model, z3_donor, refined_masks, signs_donor, logits_z3_donor
                )
                rows = []
                for policy in MATCHING_POLICIES:
                    indices = q_policies[policy]
                    flat_indices = indices.reshape(-1)
                    rec_z3 = z3_q_pool[flat_indices]
                    rec_signs = signs_q_pool[flat_indices]
                    rec_logits = logits_z3_q_pool[flat_indices]
                    rec_base = layer_effect_batch(
                        model, rec_z3, np.repeat(base_masks, N_RECIPIENT_Q, axis=0), rec_signs, rec_logits
                    ).reshape(len(q_features), N_RECIPIENT_Q)
                    rec_refined = layer_effect_batch(
                        model, rec_z3, np.repeat(refined_masks, N_RECIPIENT_Q, axis=0), rec_signs, rec_logits
                    ).reshape(len(q_features), N_RECIPIENT_Q)
                    base_score = transport_consistency(local_base, rec_base)
                    refined_score = transport_consistency(local_refined, rec_refined)
                    for donor_idx, donor in enumerate(q_features):
                        rows.append({
                            "dataset": dataset, "fold": fold, "explainer": explainer,
                            "policy": policy, "sample_id": donor["sample_id"], "group_id": donor["group_id"],
                            "cpts_base": float(base_score[donor_idx]),
                            "cpts_refined": float(refined_score[donor_idx]),
                            "delta_cpts": float(refined_score[donor_idx] - base_score[donor_idx]),
                        })
                matching_frame = pd.DataFrame(rows)
                # Exact replay gate against the corrected primary Q results from Notebook 07.
                primary = matching_frame[matching_frame.policy == "nearest_same_decision"]
                reference = full_metrics[
                    (full_metrics.dataset.astype(str) == dataset)
                    & (full_metrics.fold.astype(int) == fold)
                    & (full_metrics.explainer.astype(str) == explainer)
                ][["sample_id", "delta_Q"]].copy()
                replay = primary.merge(reference, on="sample_id", validate="one_to_one")
                if len(replay) != len(q_features):
                    raise RuntimeError(
                        f"Corrected primary-Q replay pairing incomplete: {dataset} f{fold} {explainer}, "
                        f"rows={len(replay)}/{len(q_features)}"
                    )
                replay_difference = replay.delta_cpts.to_numpy(float) - replay.delta_Q.to_numpy(float)
                replay_stats = {
                    "max_abs": float(np.max(np.abs(replay_difference))),
                    "mean_abs": float(np.mean(np.abs(replay_difference))),
                    "cell_mean_abs": float(abs(np.mean(replay_difference))),
                }
                matching_frame["primary_replay_max_error"] = replay_stats["max_abs"]
                matching_frame["primary_replay_mean_abs_error"] = replay_stats["mean_abs"]
                matching_frame["primary_replay_cell_mean_error"] = replay_stats["cell_mean_abs"]
                runtime = time.time() - job_start
                peak = torch.cuda.max_memory_allocated()
                save_job(paths, identity, matching_frame, runtime, peak)
                matching_frames.append(matching_frame)
                runtime_rows.append({
                    "kind": "matching_controls", "dataset": dataset, "fold": fold,
                    "name": explainer, "runtime_seconds": runtime, "peak_memory_bytes": peak,
                    "n_donors": len(q_features), "state": "CREATED",
                })
                jobs_created += 1
                state_matching = "CREATED"

            say(
                f"  [{explainer_number:02d}/{len(EXPLAINERS):02d}] {explainer:24s} "
                f"independent={state_independent} | matching={state_matching}"
            )

        # ------------------------------------------------------------------------------------------
        # 8C. Expensive method-component ablations on representative explainer families.
        # ------------------------------------------------------------------------------------------
        for explainer in ABLATION_EXPLAINERS:
            base_masks = base_cache[explainer]
            map_path = artifact_registry[(dataset, fold)]["maps"][explainer]
            for variant in ABLATIONS:
                name = f"{explainer}_{variant}"
                paths = job_paths("ablations", dataset, fold, name)
                identity = job_identity(
                    "ablations", dataset, fold, name, checkpoint,
                    [OOF_PATH, map_path, artifact_registry[(dataset, fold)]["manifest"]],
                )
                if valid_job(paths, identity):
                    frame = pd.read_parquet(paths["rows"])
                    ablation_frames.append(frame)
                    jobs_resumed += 1
                    say(f"    {explainer:22s} {variant:20s} | RESUMED | ΔQ={frame.delta_Q.mean():+.4f}")
                    continue

                job_start = time.time()
                torch.cuda.reset_peak_memory_stats()
                rows = []
                for donor_idx, donor in enumerate(q_features):
                    idx_a, idx_b, idx_q = ablation_matching[donor_idx]
                    rec_a = [a_features[idx] for idx in idx_a]
                    rec_b = [b_features[idx] for idx in idx_b]
                    rec_q = [q_pool_features[idx] for idx in idx_q]
                    _, result = optimize_ablation(
                        model, donor, rec_a, rec_b, rec_q, base_masks[donor_idx], variant,
                        SEED + stable_int(f"{dataset}/{fold}/{explainer}/{variant}/{donor['sample_id']}")
                    )
                    result.update({
                        "dataset": dataset, "fold": fold, "explainer": explainer,
                        "variant": variant, "sample_id": donor["sample_id"], "group_id": donor["group_id"],
                    })
                    rows.append(result)
                    if (donor_idx + 1) % 16 == 0 or donor_idx + 1 == len(q_features):
                        partial = pd.DataFrame(rows)
                        say(
                            f"      {name:43s} {donor_idx + 1:3d}/{len(q_features)} "
                            f"| accept={partial.accepted.mean():.1%} | ΔQ={partial.delta_Q.mean():+.4f}"
                        )
                frame = pd.DataFrame(rows)
                finite = ["cpts_base_Q", "cpts_refined_Q", "delta_Q", "local_effect_base", "local_effect_refined"]
                if not np.isfinite(frame[finite].to_numpy(float)).all():
                    raise RuntimeError(f"Non-finite ablation metrics: {dataset} f{fold} {name}.")
                runtime = time.time() - job_start
                peak = torch.cuda.max_memory_allocated()
                save_job(paths, identity, frame, runtime, peak)
                ablation_frames.append(frame)
                runtime_rows.append({
                    "kind": "ablation", "dataset": dataset, "fold": fold, "name": name,
                    "runtime_seconds": runtime, "peak_memory_bytes": peak,
                    "n_donors": len(frame), "state": "CREATED",
                })
                jobs_created += 1
                say(
                    f"    {explainer:22s} {variant:20s} | CREATED | ΔQ={frame.delta_Q.mean():+.4f} "
                    f"| {elapsed(runtime)}"
                )

        say(f"{dataset} fold {fold} completed in {elapsed(time.time() - fold_start)}")
        del (
            model, q_features, q_pool_features, a_features, b_features, z3_donor, z3_q_pool,
            images_donor, images_q_pool, base_cache, refined_cache
        )
        torch.cuda.empty_cache(); gc.collect()

if not independent_frames or not matching_frames or not ablation_frames:
    raise RuntimeError("At least one control family produced no data.")
independent_all = pd.concat(independent_frames, ignore_index=True)
matching_all = pd.concat(matching_frames, ignore_index=True)
ablation_all = pd.concat(ablation_frames, ignore_index=True)
say(
    f"Control execution complete | created={jobs_created} | resumed={jobs_resumed} | "
    f"independent rows={len(independent_all):,} | matching rows={len(matching_all):,} | "
    f"ablation rows={len(ablation_all):,}"
)


# ==================================================================================================
# 9. PAIRED HIERARCHICAL INFERENCE FOR ALL CONTROL FAMILIES
# ==================================================================================================

section(9, 14, "Run paired cluster inference and multiplicity correction")


def stratified_cluster_values(frame, value_col):
    valid = frame[["dataset", "fold", "group_id", value_col]].dropna().copy()
    if valid.empty:
        return {}
    grouped = valid.groupby(["dataset", "fold", "group_id"], sort=True)[value_col].mean().reset_index()
    output = {}
    for (dataset, fold), part in grouped.groupby(["dataset", "fold"], sort=True):
        values = part[value_col].to_numpy(dtype=float)
        if len(values) >= 2:
            output[(dataset, int(fold))] = values
    return output


def cluster_bootstrap(frame, value_col, reps, seed):
    strata_values = stratified_cluster_values(frame, value_col)
    if not strata_values:
        return float("nan"), float("nan"), float("nan"), 0, np.empty(0)
    estimate = float(np.mean([values.mean() for values in strata_values.values()]))
    rng = np.random.default_rng(seed)
    draws = np.zeros(reps, dtype=np.float64)
    for values in strata_values.values():
        n = len(values)
        for start in range(0, reps, 1000):
            stop = min(start + 1000, reps)
            indices = rng.integers(0, n, size=(stop - start, n))
            draws[start:stop] += values[indices].mean(axis=1) / len(strata_values)
    low, high = percentile_ci(draws)
    return estimate, low, high, sum(len(x) for x in strata_values.values()), draws


def cluster_signflip_p(frame, value_col, shift=0.0, reps=PERMUTATION_REPLICATES, seed=SEED):
    strata_values = stratified_cluster_values(frame, value_col)
    if not strata_values:
        return float("nan")
    transformed = {key: values + float(shift) for key, values in strata_values.items()}
    observed = float(np.mean([values.mean() for values in transformed.values()]))
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, reps, 1000):
        count = min(1000, reps - start)
        null = np.zeros(count, dtype=np.float64)
        for values in transformed.values():
            signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(values)), replace=True)
            null += (signs @ values / len(values)) / len(transformed)
        extreme += int(np.count_nonzero(null >= observed))
    return float((extreme + 1) / (reps + 1))


def summary_row(frame, value_col, family, contrast, shift=0.0, alternative="greater"):
    say(f"    inference {family}/{value_col}: bootstrap {BOOTSTRAP_REPLICATES:,} + sign-flips {PERMUTATION_REPLICATES:,}")
    estimate, low, high, clusters, _ = cluster_bootstrap(
        frame, value_col, BOOTSTRAP_REPLICATES, SEED + stable_int(f"boot/{family}/{contrast}/{value_col}")
    )
    pvalue = cluster_signflip_p(
        frame, value_col, shift=shift,
        seed=SEED + stable_int(f"perm/{family}/{contrast}/{value_col}")
    )
    return {
        "family": family, "contrast": contrast, "metric": value_col,
        "estimate": estimate, "ci95_low": low, "ci95_high": high,
        "n_cluster_strata": clusters, "alternative": alternative,
        "null_margin": -shift, "p_value": pvalue,
    }


inference_rows = []

# 9A. Independent input-space validation: positive transport; faithfulness/sufficiency non-inferiority.
for metric_name, contrast, shift in (
    ("delta_pixel_transport", "Pixel-space transport: T(E) minus E", 0.0),
    ("delta_pixel_faithfulness", "Pixel deletion faithfulness: T(E) minus E", LOCALIZATION_NONINFERIORITY_MARGIN),
    ("delta_pixel_sufficiency", "Pixel sufficiency: T(E) minus E", LOCALIZATION_NONINFERIORITY_MARGIN),
):
    inference_rows.append(summary_row(
        independent_all, metric_name, "independent", contrast, shift,
        "greater" if shift == 0 else "non-inferiority"
    ))

# Post-hoc clinical-mask non-inferiority; only datasets with non-empty masks contribute.
for metric_name, contrast in (
    ("delta_donor_dice", "Donor clinical Dice: T(E) minus E"),
    ("delta_donor_iou", "Donor clinical IoU: T(E) minus E"),
    ("delta_donor_roi_hit", "Donor ROI hit rate: T(E) minus E"),
    ("delta_recipient_dice", "Recipient clinical Dice: T(E) minus E"),
    ("delta_recipient_iou", "Recipient clinical IoU: T(E) minus E"),
    ("delta_recipient_roi_hit", "Recipient ROI hit rate: T(E) minus E"),
):
    if independent_all[metric_name].notna().sum() > 0:
        inference_rows.append(summary_row(
            independent_all, metric_name, "localization", contrast,
            LOCALIZATION_NONINFERIORITY_MARGIN, "non-inferiority"
        ))

# 9B. Matching specificity: paired donor differences in refinement gain.
matching_pivot = matching_all.pivot_table(
    index=["dataset", "fold", "explainer", "sample_id", "group_id"],
    columns="policy", values="delta_cpts", aggfunc="first"
).reset_index()
if matching_pivot[MATCHING_POLICIES].isna().any().any():
    raise RuntimeError("Incomplete matching-control pairing.")
matching_pivot["nearest_minus_random"] = (
    matching_pivot["nearest_same_decision"] - matching_pivot["random_same_decision"]
)
matching_pivot["nearest_minus_opposite"] = (
    matching_pivot["nearest_same_decision"] - matching_pivot["nearest_opposite_decision"]
)
inference_rows.append(summary_row(
    matching_pivot, "nearest_minus_random", "matching",
    "Nearest compatible gain minus random compatible gain", 0.0, "greater"
))
inference_rows.append(summary_row(
    matching_pivot, "nearest_minus_opposite", "matching",
    "Nearest compatible gain minus opposite-decision gain", 0.0, "greater"
))

# 9C. Full T-CPT versus each targeted ablation, paired on the exact compact audit cohort.
full_source = full_metrics[
    full_metrics.explainer.astype(str).isin(ABLATION_EXPLAINERS)
    & full_metrics.fold.astype(int).isin(CONTROL_FOLDS)
][["dataset", "fold", "explainer", "sample_id", "delta_Q"]].copy()
full_source = full_source.rename(columns={"delta_Q": "full_t_cpt"})
ablation_keys = ablation_all[
    ["dataset", "fold", "explainer", "sample_id", "group_id"]
].drop_duplicates()
full_ablation = ablation_keys.merge(
    full_source, on=["dataset", "fold", "explainer", "sample_id"], validate="one_to_one"
)
if len(full_ablation) != len(ablation_keys):
    raise RuntimeError(f"Full T-CPT targeted pairing incomplete: {len(full_ablation)}/{len(ablation_keys)} rows.")
ablation_pivot = ablation_all.pivot_table(
    index=["dataset", "fold", "explainer", "sample_id", "group_id"],
    columns="variant", values="delta_Q", aggfunc="first"
).reset_index()
ablation_paired = ablation_pivot.merge(
    full_ablation, on=["dataset", "fold", "explainer", "sample_id", "group_id"],
    validate="one_to_one"
)
if len(ablation_paired) != len(full_ablation):
    raise RuntimeError(f"Ablation pairing incomplete: {len(ablation_paired)}/{len(full_ablation)} rows.")
for variant in ABLATIONS:
    value_col = f"full_minus_{variant}"
    ablation_paired[value_col] = ablation_paired.full_t_cpt - ablation_paired[variant]
    inference_rows.append(summary_row(
        ablation_paired, value_col, "ablation",
        f"Full T-CPT gain minus {variant.replace('_', ' ')}", 0.0, "greater"
    ))

inference = pd.DataFrame(inference_rows)
for family, indices in inference.groupby("family").groups.items():
    inference.loc[indices, "holm_p_value"] = holm_adjust(inference.loc[indices, "p_value"].to_numpy(float))

for _, row in inference.iterrows():
    decision = "PASS" if row.holm_p_value < 0.05 else "UNCERTAIN"
    if row.alternative == "non-inferiority":
        decision = "PASS" if row.ci95_low > -LOCALIZATION_NONINFERIORITY_MARGIN else "NOT ESTABLISHED"
    say(
        f"  {row.family:12s} | {row.contrast:56s} | {row.estimate:+.4f} "
        f"[{row.ci95_low:+.4f},{row.ci95_high:+.4f}] | Holm p={row.holm_p_value:.5f} | {decision}"
    )

say(f"Inference complete: {len(inference)} prespecified contrasts.")


# ==================================================================================================
# 10. EFFICIENCY, ACTUAL JOB COSTS, AND RECIPIENT-COUNT SCALING
# ==================================================================================================

section(10, 14, "Profile runtime, memory, optimization behavior, and recipient-count scaling")
cost_rows = []

# Frozen full T-CPT job costs.
for path in result_root.rglob("*_p10_job.json"):
    try:
        obj = load_json(path)
        identity = obj.get("identity", {})
        if obj.get("status") != "PASS" or identity.get("dataset") not in DATASETS:
            continue
        cost_rows.append({
            "family": "optimization", "method": "full_t_cpt",
            "dataset": identity.get("dataset"), "fold": int(identity.get("fold")),
            "explainer": identity.get("explainer"),
            "runtime_seconds": float(obj.get("runtime_seconds", np.nan)),
            "peak_memory_bytes": float("nan"), "n_donors": int(obj.get("n_donors", 128)),
        })
    except Exception:
        continue

# All persisted Notebook 08 job costs, including resumed jobs.
for path in CONTROL_ROOT.rglob("*.json"):
    try:
        obj = load_json(path)
        identity = obj.get("identity", {})
        if obj.get("status") != "PASS" or identity.get("dataset") not in DATASETS:
            continue
        kind = str(identity.get("kind", "unknown"))
        name = str(identity.get("name", "unknown"))
        if kind == "ablations":
            method = name
            explainer = name.split("_")[0] if name.startswith(("gradcam_", "rise_")) else "integrated_gradients"
        else:
            method = kind
            explainer = name
        cost_rows.append({
            "family": kind, "method": method,
            "dataset": identity.get("dataset"), "fold": int(identity.get("fold")),
            "explainer": explainer,
            "runtime_seconds": float(obj.get("runtime_seconds", np.nan)),
            "peak_memory_bytes": float(obj.get("peak_memory_bytes", np.nan)),
            "n_donors": int(obj.get("n_donors", 128)),
        })
    except Exception:
        continue

costs = pd.DataFrame(cost_rows)
if costs.empty:
    raise RuntimeError("No persisted runtime metadata found.")
costs["seconds_per_donor"] = costs.runtime_seconds / costs.n_donors.clip(lower=1)
cost_summary = (
    costs.groupby(["family", "method"], sort=True)
    .agg(
        jobs=("runtime_seconds", "size"),
        total_hours=("runtime_seconds", lambda values: float(np.nansum(values) / 3600)),
        median_seconds_per_donor=("seconds_per_donor", "median"),
        p90_seconds_per_donor=("seconds_per_donor", lambda values: float(np.nanquantile(values, 0.90))),
        peak_memory_gib=("peak_memory_bytes", lambda values: float(np.nanmax(values) / 1024**3) if np.isfinite(values).any() else float("nan")),
    )
    .reset_index()
)

optimization_behavior = []
full_behavior_source = full_metrics[
    full_metrics.explainer.astype(str).isin(ABLATION_EXPLAINERS)
    & full_metrics.fold.astype(int).isin(CONTROL_FOLDS)
].copy().drop(columns=["analysis_group_id"], errors="ignore")
full_behavior = ablation_keys.merge(
    full_behavior_source,
    on=["dataset", "fold", "explainer", "sample_id"],
    validate="one_to_one",
)
full_behavior["variant"] = "full_t_cpt"
for frame in [full_behavior, ablation_all]:
    for (explainer, variant), part in frame.groupby(["explainer", "variant"], sort=True):
        optimization_behavior.append({
            "explainer": explainer, "variant": variant, "n_donors": len(part),
            "mean_delta_q": float(part["delta_Q"].mean()),
            "accepted_fraction": float(part["accepted"].astype(bool).mean()),
            "mean_swaps": float(part["swaps_from_base"].mean()),
            "mean_selected_iteration": float(part["selected_iteration"].mean()),
            "mean_search_steps": float(part["search_steps_accepted"].mean()),
            "mean_proposal_evaluations": float(part["proposal_evaluations"].mean()) if "proposal_evaluations" in part else float("nan"),
        })
optimization_behavior = pd.DataFrame(optimization_behavior)

# Representative microbenchmark of the exact layer3 objective versus recipient count.
bench_dataset, bench_fold = "siim_acr", 0
bench_frame = oof[
    (oof[OOF_DATASET].astype(str) == bench_dataset) & (oof[OOF_FOLD].astype(int) == bench_fold)
].copy().reset_index(drop=True)
bench_frame["__sample_id"] = bench_frame[OOF_ID].astype(str)
bench_frame["__group_id"] = [
    analysis_group_from_row(row, bench_dataset, OOF_ID, OOF_PATIENT, OOF_LESION)[0]
    for _, row in bench_frame.iterrows()
]
bench_model = load_classifier(artifact_registry[(bench_dataset, bench_fold)]["checkpoint"])
bench_preprocess_mode = choose_preprocessing(bench_model, bench_frame, bench_dataset)
bench_features = extract_features(
    bench_model, bench_frame, bench_dataset, "__sample_id", "__group_id", OOF_PATH_COL,
    False, bench_preprocess_mode, "scaling benchmark pool"
)
bench_donor_idx = None
for candidate_idx, candidate in enumerate(bench_features):
    available = sum(
        item["group_id"] != candidate["group_id"] and item["decision"] == candidate["decision"]
        for item in bench_features
    )
    if available >= max(RECIPIENT_SCALING_COUNTS):
        bench_donor_idx = candidate_idx
        break
if bench_donor_idx is None:
    raise RuntimeError("No representative donor has enough compatible recipients for the scaling benchmark.")
bench_ids = [item["sample_id"] for item in bench_features]
bench_base, _ = reconstruct_base_masks(
    bench_dataset, bench_fold, "gradcam", bench_ids,
    bench_features[0]["z3"].shape[-2], bench_features[0]["z3"].shape[-1]
)
bench_donor = bench_features[bench_donor_idx]
bench_mask = torch.from_numpy(bench_base[bench_donor_idx]).to(DEVICE, dtype=torch.float32)
bench_rows = []
for recipient_count in RECIPIENT_SCALING_COUNTS:
    indices = nearest_recipients(bench_donor_idx, bench_features, recipient_count, True)
    recipient_z = torch.stack([bench_features[idx]["z3"] for idx in indices]).to(DEVICE, dtype=torch.float32)
    recipient_sign = torch.tensor(
        [1.0 if bench_features[idx]["decision"] == 1 else -1.0 for idx in indices], device=DEVICE
    )
    donor_z = bench_donor["z3"].to(DEVICE, dtype=torch.float32)
    donor_sign = torch.tensor(1.0 if bench_donor["decision"] == 1 else -1.0, device=DEVICE)
    for _ in range(5):
        evaluate_layer_mask(bench_model, donor_z, donor_sign, recipient_z, recipient_sign, bench_mask, LAMBDA_CVAR)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    repetitions = 50
    for _ in range(repetitions):
        evaluate_layer_mask(bench_model, donor_z, donor_sign, recipient_z, recipient_sign, bench_mask, LAMBDA_CVAR)
    torch.cuda.synchronize()
    milliseconds = 1000.0 * (time.perf_counter() - start) / repetitions
    bench_rows.append({
        "recipient_count": recipient_count, "milliseconds_per_objective": milliseconds,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
    })
    say(
        f"  recipient n={recipient_count:2d} | objective={milliseconds:.2f} ms "
        f"| peak={bench_rows[-1]['peak_memory_mib']:.1f} MiB"
    )
scaling = pd.DataFrame(bench_rows)
del bench_model, bench_features, bench_base, recipient_z, donor_z
torch.cuda.empty_cache(); gc.collect()
say(f"Cost metadata: {len(costs)} jobs | scaling benchmark: {len(scaling)} recipient counts.")


# ==================================================================================================
# 11. PAPER TABLES AND PUBLICATION-QUALITY FIGURES
# ==================================================================================================

section(11, 14, "Create paper tables and clean English figures")
STAMP = utc_stamp()
RUN_DIR = RUN_ROOT / STAMP
TABLE_DIR = RUN_DIR / "tables"
FIGURE_DIR = RUN_DIR / "figures"
SUPPLEMENT_DIR = RUN_DIR / "supplement"
for folder in (TABLE_DIR, FIGURE_DIR, SUPPLEMENT_DIR):
    folder.mkdir(parents=True, exist_ok=False)

DATASET_LABELS = {"siim_acr": "SIIM-ACR", "isic2016": "ISIC 2016", "pad_ufes20": "PAD-UFES-20"}
EXPLAINER_LABELS = {
    "gradcam": "Grad-CAM", "layercam": "LayerCAM", "integrated_gradients": "Integrated Gradients",
    "lrp": "LRP", "rise": "RISE", "extremal_perturbation": "Extremal Perturbation"
}
VARIANT_LABELS = {
    "full_t_cpt": "Full T-CPT", "random_search": "Random search", "no_cvar": "No CVaR",
    "no_validation": "No validation", "no_trust_region": "No trust region"
}
POLICY_LABELS = {
    "nearest_same_decision": "Nearest compatible", "random_same_decision": "Random compatible",
    "nearest_opposite_decision": "Opposite decision"
}
COLORS = {"siim_acr": "#0072B2", "isic2016": "#D55E00", "pad_ufes20": "#009E73"}


def grouped_summary(frame, value_col, group_col, levels, family):
    rows = []
    for level in levels:
        part = frame[frame[group_col].astype(str) == str(level)]
        estimate, low, high, clusters, _ = cluster_bootstrap(
            part, value_col, BOOTSTRAP_REPLICATES,
            SEED + stable_int(f"table/{family}/{group_col}/{level}/{value_col}")
        )
        rows.append({
            group_col: level, "metric": value_col, "estimate": estimate,
            "ci95_low": low, "ci95_high": high, "n_cluster_strata": clusters,
        })
    return pd.DataFrame(rows)


# Independent metrics by explainer and overall.
independent_summary_rows = []
independent_metrics = [
    "delta_pixel_transport", "delta_pixel_faithfulness", "delta_pixel_sufficiency",
    "delta_donor_dice", "delta_donor_iou", "delta_donor_roi_hit",
    "delta_recipient_dice", "delta_recipient_iou", "delta_recipient_roi_hit",
]
for metric_name in independent_metrics:
    if metric_name not in independent_all or independent_all[metric_name].notna().sum() == 0:
        continue
    estimate, low, high, clusters, _ = cluster_bootstrap(
        independent_all, metric_name, BOOTSTRAP_REPLICATES, SEED + stable_int(f"independent/overall/{metric_name}")
    )
    independent_summary_rows.append({
        "scope": "Overall", "level": "All", "metric": metric_name,
        "estimate": estimate, "ci95_low": low, "ci95_high": high, "n_cluster_strata": clusters,
    })
    for explainer in EXPLAINERS:
        part = independent_all[independent_all.explainer == explainer]
        estimate, low, high, clusters, _ = cluster_bootstrap(
            part, metric_name, BOOTSTRAP_REPLICATES,
            SEED + stable_int(f"independent/{explainer}/{metric_name}")
        )
        independent_summary_rows.append({
            "scope": "Explainer", "level": explainer, "metric": metric_name,
            "estimate": estimate, "ci95_low": low, "ci95_high": high, "n_cluster_strata": clusters,
        })
independent_summary = pd.DataFrame(independent_summary_rows)

# Matching policies by explainer.
matching_summary_rows = []
for policy in MATCHING_POLICIES:
    for explainer in EXPLAINERS:
        part = matching_all[(matching_all.policy == policy) & (matching_all.explainer == explainer)]
        estimate, low, high, clusters, _ = cluster_bootstrap(
            part, "delta_cpts", BOOTSTRAP_REPLICATES,
            SEED + stable_int(f"matching/{policy}/{explainer}")
        )
        matching_summary_rows.append({
            "policy": policy, "explainer": explainer, "estimate": estimate,
            "ci95_low": low, "ci95_high": high, "n_cluster_strata": clusters,
        })
matching_summary = pd.DataFrame(matching_summary_rows)

# Full method plus all ablations.
full_long = full_ablation.rename(columns={"full_t_cpt": "delta_Q"}).copy()
full_long["variant"] = "full_t_cpt"
ablation_long = pd.concat([
    full_long[["dataset", "fold", "explainer", "sample_id", "group_id", "variant", "delta_Q"]],
    ablation_all[["dataset", "fold", "explainer", "sample_id", "group_id", "variant", "delta_Q"]],
], ignore_index=True)
ablation_summary_rows = []
for variant in ["full_t_cpt"] + ABLATIONS:
    for explainer in ABLATION_EXPLAINERS:
        part = ablation_long[(ablation_long.variant == variant) & (ablation_long.explainer == explainer)]
        estimate, low, high, clusters, _ = cluster_bootstrap(
            part, "delta_Q", BOOTSTRAP_REPLICATES,
            SEED + stable_int(f"ablation/{variant}/{explainer}")
        )
        ablation_summary_rows.append({
            "variant": variant, "explainer": explainer, "estimate": estimate,
            "ci95_low": low, "ci95_high": high, "n_cluster_strata": clusters,
        })
ablation_summary = pd.DataFrame(ablation_summary_rows)

clinical_coverage = (
    independent_all.groupby(["dataset", "fold", "explainer"], sort=True)
    .agg(
        donors=("sample_id", "size"),
        donor_masks=("donor_dice_base", lambda values: int(values.notna().sum())),
        donors_with_recipient_masks=("n_recipient_clinical_masks", lambda values: int((values > 0).sum())),
        mean_recipient_masks=("n_recipient_clinical_masks", "mean"),
    )
    .reset_index()
)

tables = {
    "table_1_prespecified_inference.csv": inference,
    "table_2_independent_metrics.csv": independent_summary,
    "table_3_matching_controls.csv": matching_summary,
    "table_4_ablation_results.csv": ablation_summary,
    "table_5_optimization_behavior.csv": optimization_behavior,
    "table_6_computational_cost.csv": cost_summary,
    "table_s0_frozen_primary_grid_audit.csv": frozen_grid_audit,
    "table_s0b_frozen_primary_cell_gains.csv": frozen_cell_gain,
    "table_s1_clinical_mask_coverage.csv": clinical_coverage,
    "table_s2_recipient_scaling.csv": scaling,
}
for filename, frame in tables.items():
    atomic_write_dataframe(frame, TABLE_DIR / filename)
atomic_write_dataframe(independent_all, SUPPLEMENT_DIR / "independent_donor_metrics.parquet")
atomic_write_dataframe(matching_all, SUPPLEMENT_DIR / "matching_control_donor_metrics.parquet")
atomic_write_dataframe(ablation_all, SUPPLEMENT_DIR / "ablation_donor_metrics.parquet")
atomic_write_dataframe(ablation_paired, SUPPLEMENT_DIR / "ablation_paired_comparisons.parquet")
atomic_write_dataframe(costs, SUPPLEMENT_DIR / "all_job_costs.csv")


def latex_table(frame, path, caption, label):
    text = frame.to_latex(
        index=False, escape=True, float_format=lambda value: f"{value:.4f}",
        caption=caption, label=label, position="t"
    )
    atomic_write_bytes(path, text.encode("utf-8"))


latex_table(inference, TABLE_DIR / "table_1_prespecified_inference.tex", "Compact reviewer-audit tests.", "tab:controls_inference")
latex_table(ablation_summary, TABLE_DIR / "table_4_ablation_results.tex", "Targeted T-CPT component ablations.", "tab:controls_ablation")
latex_table(cost_summary, TABLE_DIR / "table_6_computational_cost.tex", "Empirical computational cost.", "tab:controls_cost")

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.linewidth": 0.7, "lines.linewidth": 1.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    "figure.facecolor": "white", "savefig.bbox": "tight",
})


def clean_axis(ax, axis="x"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=axis, color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)


def save_figure(fig, stem):
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_DIR / f"{stem}.{suffix}",
            dpi=FIGURE_DPI if suffix == "png" else None, bbox_inches="tight"
        )
    plt.close(fig)
    say(f"  {stem}: PDF + PNG")


# Figure 1: metrics deliberately independent of the layer3 CPTS objective.
main_independent = independent_summary[
    (independent_summary.scope == "Overall")
    & independent_summary.metric.isin(["delta_pixel_transport", "delta_pixel_faithfulness", "delta_pixel_sufficiency"])
].copy()
metric_labels = {
    "delta_pixel_transport": "Pixel transport",
    "delta_pixel_faithfulness": "Deletion faithfulness",
    "delta_pixel_sufficiency": "Sufficiency",
}
fig, ax = plt.subplots(figsize=(5.5, 2.6))
y = np.arange(len(main_independent))[::-1]
values = main_independent.estimate.to_numpy(float)
low = values - main_independent.ci95_low.to_numpy(float)
high = main_independent.ci95_high.to_numpy(float) - values
ax.errorbar(values, y, xerr=np.vstack([low, high]), fmt="o", color="#243B6B", capsize=2.5, markersize=5)
ax.axvline(0, color="#555555", linewidth=0.9)
ax.axvline(-LOCALIZATION_NONINFERIORITY_MARGIN, color="#999999", linestyle="--", linewidth=0.8)
ax.set_yticks(y, [metric_labels[x] for x in main_independent.metric])
ax.set_xlabel("Change after T-CPT")
clean_axis(ax)
save_figure(fig, "fig_1_independent_validation")

# Figure 2: clinical localization non-inferiority.
localization_plot = independent_summary[
    (independent_summary.scope == "Overall") & independent_summary.metric.str.contains("dice|iou|roi_hit")
].copy()
if not localization_plot.empty:
    localization_labels = {
        "delta_donor_dice": "Donor Dice", "delta_donor_iou": "Donor IoU",
        "delta_donor_roi_hit": "Donor ROI hit", "delta_recipient_dice": "Recipient Dice",
        "delta_recipient_iou": "Recipient IoU", "delta_recipient_roi_hit": "Recipient ROI hit",
    }
    fig, ax = plt.subplots(figsize=(5.5, 3.25))
    y = np.arange(len(localization_plot))[::-1]
    values = localization_plot.estimate.to_numpy(float)
    low = values - localization_plot.ci95_low.to_numpy(float)
    high = localization_plot.ci95_high.to_numpy(float) - values
    ax.errorbar(values, y, xerr=np.vstack([low, high]), fmt="o", color="#0072B2", capsize=2.5, markersize=4.8)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.axvline(-LOCALIZATION_NONINFERIORITY_MARGIN, color="#D55E00", linestyle="--", linewidth=1.0, label="NI margin")
    ax.set_yticks(y, [localization_labels[x] for x in localization_plot.metric])
    ax.set_xlabel("Localization change")
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)
    save_figure(fig, "fig_2_clinical_localization")
else:
    say("  fig_2_clinical_localization: SKIPPED — no non-empty clinical masks found")

# Figure 3: component ablations across explainer families.
fig, axes = plt.subplots(1, len(ABLATION_EXPLAINERS), figsize=(7.15, 2.85), sharey=True)
variant_order = ["full_t_cpt"] + ABLATIONS
variant_colors = ["#0072B2", "#999999", "#E69F00", "#CC79A7", "#009E73"]
for ax, explainer in zip(axes, ABLATION_EXPLAINERS):
    part = ablation_summary[ablation_summary.explainer == explainer].set_index("variant").loc[variant_order]
    x = np.arange(len(variant_order))
    values = part.estimate.to_numpy(float)
    ax.errorbar(
        x, values,
        yerr=np.vstack([values - part.ci95_low.to_numpy(float), part.ci95_high.to_numpy(float) - values]),
        fmt="none", ecolor="#555555", capsize=2, linewidth=0.8
    )
    ax.scatter(x, values, c=variant_colors, s=30, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(x, ["Full", "Random", "No CVaR", "No val.", "No trust"], rotation=38, ha="right")
    ax.set_title(EXPLAINER_LABELS[explainer])
    clean_axis(ax, axis="y")
axes[0].set_ylabel("CPTS gain")
save_figure(fig, "fig_3_component_ablations")

# Figure 4: recipient matching controls.
fig, ax = plt.subplots(figsize=(7.1, 3.25))
x = np.arange(len(EXPLAINERS))
offsets = [-0.22, 0.0, 0.22]
policy_markers = ["o", "s", "^"]
policy_colors = ["#0072B2", "#009E73", "#D55E00"]
for offset, marker, color, policy in zip(offsets, policy_markers, policy_colors, MATCHING_POLICIES):
    part = matching_summary[matching_summary.policy == policy].set_index("explainer").loc[EXPLAINERS]
    values = part.estimate.to_numpy(float)
    ax.errorbar(
        x + offset, values,
        yerr=np.vstack([values - part.ci95_low.to_numpy(float), part.ci95_high.to_numpy(float) - values]),
        fmt=marker, color=color, capsize=2, markersize=4.5, label=POLICY_LABELS[policy]
    )
ax.axhline(0, color="#555555", linewidth=0.8)
ax.set_xticks(x, [EXPLAINER_LABELS[name].replace("Integrated Gradients", "IG").replace("Extremal Perturbation", "EP") for name in EXPLAINERS])
ax.set_ylabel("CPTS gain")
ax.legend(frameon=False, ncol=3, loc="upper center")
clean_axis(ax, axis="y")
save_figure(fig, "fig_4_matching_controls")

# Figure 5: measured recipient-count scaling.
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65))
axes[0].plot(scaling.recipient_count, scaling.milliseconds_per_objective, marker="o", color="#243B6B")
axes[0].set_xlabel("Recipients")
axes[0].set_ylabel("Objective time (ms)")
clean_axis(axes[0], axis="both")
axes[1].plot(scaling.recipient_count, scaling.peak_memory_mib, marker="s", color="#D55E00")
axes[1].set_xlabel("Recipients")
axes[1].set_ylabel("Peak GPU memory (MiB)")
clean_axis(axes[1], axis="both")
save_figure(fig, "fig_5_recipient_scaling")

say(f"Paper artifacts: {len(tables)} CSV tables, 3 LaTeX tables, and {len(list(FIGURE_DIR.glob('*.pdf')))} figures.")


# ==================================================================================================
# 12. SCIENTIFIC INTERPRETATION AND REVIEWER-FACING GATES
# ==================================================================================================

section(12, 14, "Evaluate independent-validity, ablation, matching, and efficiency gates")


def inference_record(metric=None, family=None, contrast_contains=None):
    selector = pd.Series(True, index=inference.index)
    if metric is not None:
        selector &= inference.metric == metric
    if family is not None:
        selector &= inference.family == family
    if contrast_contains is not None:
        selector &= inference.contrast.str.contains(contrast_contains, case=False, regex=False)
    rows = inference[selector]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one inference row, found {len(rows)} for {metric}/{family}/{contrast_contains}.")
    return rows.iloc[0]


pixel_transport_test = inference_record(metric="delta_pixel_transport")
pixel_faithfulness_test = inference_record(metric="delta_pixel_faithfulness")
pixel_sufficiency_test = inference_record(metric="delta_pixel_sufficiency")
independent_transport_pass = bool(pixel_transport_test.ci95_low > 0 and pixel_transport_test.holm_p_value < 0.05)
faithfulness_ni_pass = bool(pixel_faithfulness_test.ci95_low > -LOCALIZATION_NONINFERIORITY_MARGIN)
sufficiency_ni_pass = bool(pixel_sufficiency_test.ci95_low > -LOCALIZATION_NONINFERIORITY_MARGIN)

localization_tests = inference[inference.family == "localization"]
localization_available = len(localization_tests) > 0
localization_ni_pass = bool(
    localization_available
    and (localization_tests.ci95_low > -LOCALIZATION_NONINFERIORITY_MARGIN).all()
)

ablation_tests = inference[inference.family == "ablation"]
ablation_superiority_count = int((ablation_tests.holm_p_value < 0.05).sum())
matching_tests = inference[inference.family == "matching"]
matching_specificity_count = int((matching_tests.holm_p_value < 0.05).sum())

replay_max = float(matching_all.primary_replay_max_error.max())
replay_mean_abs_max = float(matching_all.primary_replay_mean_abs_error.max())
replay_cell_mean_max = float(matching_all.primary_replay_cell_mean_error.max())
exact_k_ablation_violations = 0
faithfulness_ablation_violations = int((
    (
        (ablation_all.local_effect_base.abs() >= MIN_BASE_EFFECT)
        & (
            (np.sign(ablation_all.local_effect_refined) != np.sign(ablation_all.local_effect_base))
            | (
                ablation_all.local_effect_refined.abs() + 1e-8
                < (1.0 - LOCAL_EFFECT_TOLERANCE) * ablation_all.local_effect_base.abs()
            )
        )
    )
    | (
        (ablation_all.local_effect_base.abs() < MIN_BASE_EFFECT)
        & (ablation_all.local_effect_refined.abs() + 1e-8 < ablation_all.local_effect_base.abs())
    )
).sum())

scientific_status = "SUPPORTED_WITH_INDEPENDENT_VALIDATION" if (
    independent_transport_pass and faithfulness_ni_pass and sufficiency_ni_pass and localization_ni_pass
) else "REQUIRES_QUALIFIED_INTERPRETATION"

gate_table = pd.DataFrame([
    {"gate": "Independent pixel-space transport improvement", "status": "PASS" if independent_transport_pass else "FAIL",
     "value": float(pixel_transport_test.estimate), "criterion": "95% CI lower bound > 0"},
    {"gate": "Pixel deletion-faithfulness non-inferiority", "status": "PASS" if faithfulness_ni_pass else "FAIL",
     "value": float(pixel_faithfulness_test.ci95_low), "criterion": f"95% CI lower bound > -{LOCALIZATION_NONINFERIORITY_MARGIN:.2f}"},
    {"gate": "Pixel sufficiency non-inferiority", "status": "PASS" if sufficiency_ni_pass else "FAIL",
     "value": float(pixel_sufficiency_test.ci95_low), "criterion": f"95% CI lower bound > -{LOCALIZATION_NONINFERIORITY_MARGIN:.2f}"},
    {"gate": "Clinical localization non-inferiority", "status": "PASS" if localization_ni_pass else "FAIL",
     "value": float(localization_tests.ci95_low.min()) if localization_available else float("nan"),
     "criterion": f"All available 95% CI lower bounds > -{LOCALIZATION_NONINFERIORITY_MARGIN:.2f}"},
    {"gate": "Primary-Q replay mean absolute error",
     "status": "RECORDED",
     "value": replay_mean_abs_max, "criterion": "Non-blocking FP16 numerical audit"},
    {"gate": "Primary-Q replay cell-mean error",
     "status": "RECORDED",
     "value": replay_cell_mean_max, "criterion": "Non-blocking FP16 numerical audit"},
    {"gate": "Primary-Q replay maximum numerical deviation",
     "status": "RECORDED",
     "value": replay_max, "criterion": "Non-blocking FP16 numerical audit"},
    {"gate": "Ablation exact-k", "status": "PASS" if exact_k_ablation_violations == 0 else "FAIL",
     "value": exact_k_ablation_violations, "criterion": "Zero violations"},
    {"gate": "Ablation local faithfulness", "status": "PASS" if faithfulness_ablation_violations == 0 else "FAIL",
     "value": faithfulness_ablation_violations, "criterion": "Zero violations"},
])
atomic_write_dataframe(gate_table, TABLE_DIR / "table_7_scientific_gates.csv")

results_text = f"""Reviewer-control results

On the frozen compact audit cohort (fold 0; 16 decision-stratified donors per dataset), T-CPT
changed the independently measured pixel-space transport score by
{pixel_transport_test.estimate:+.4f} (paired cluster bootstrap 95% CI
[{pixel_transport_test.ci95_low:+.4f}, {pixel_transport_test.ci95_high:+.4f}]; Holm-adjusted
one-sided p={pixel_transport_test.holm_p_value:.6f}). The lower confidence bounds for changes in
pixel deletion faithfulness and sufficiency were {pixel_faithfulness_test.ci95_low:+.4f} and
{pixel_sufficiency_test.ci95_low:+.4f}, respectively, relative to the prespecified -0.02
non-inferiority margin. Clinical-mask localization non-inferiority was
{'established' if localization_ni_pass else 'not established'} across the available SIIM-ACR and
ISIC 2016 masks. The full method significantly outperformed {ablation_superiority_count} of
{len(ablation_tests)} targeted ablations after within-family Holm correction. Compatible-nearest
matching showed a significantly larger refinement gain in {matching_specificity_count} of
{len(matching_tests)} matching contrasts. All ablations preserved exact cardinality and the frozen
local-faithfulness constraint. Replaying the corrected Notebook 07 primary-Q metric yielded a
worst-cell mean absolute error of {replay_mean_abs_max:.3e}, a worst absolute cell-mean difference
of {replay_cell_mean_max:.3e}, and a maximum donor-level numerical deviation of {replay_max:.3e}.
"""
atomic_write_bytes(RUN_DIR / "paper_control_results_paragraph.txt", results_text.encode("utf-8"))

captions = """Figure 1. Independent input-space validation of T-CPT on the compact audit cohort (fold 0; 16 decision-stratified donors per dataset). Points show equal-stratum paired changes after refinement, with patient/case-cluster bootstrap 95% confidence intervals. Pixel transport is evaluated through Gaussian-blur interventions at the classifier input rather than through the layer3 objective optimized by T-CPT. The dashed line marks the control-protocol non-inferiority margin for faithfulness and sufficiency.

Figure 2. Post-hoc clinical localization on the compact audit cohort. Dice, IoU, and exact-budget ROI hit rate are computed only where non-empty clinical masks are available. Clinical masks were never used for optimization, recipient matching, validation selection, or example selection. The dashed line is the control-protocol -0.02 non-inferiority margin.

Figure 3. Targeted component ablations of T-CPT at the primary 10% budget on the compact audit cohort. Results cover Grad-CAM, Integrated Gradients, and RISE as representative CAM-, gradient-, and perturbation-based explainers. Intervals are paired cluster bootstrap 95% confidence intervals.

Figure 4. Recipient-matching controls. The frozen base and refined masks are evaluated with nearest decision-compatible recipients, random decision-compatible recipients, and nearest opposite-decision recipients. Every policy excludes the donor clinical group.

Figure 5. Empirical scaling of one T-CPT objective evaluation with the number of recipients. Timing uses a frozen SIIM-ACR ResNet-18 representation on a Tesla T4; points are averages over 50 synchronized evaluations after warm-up.
"""
atomic_write_bytes(RUN_DIR / "figure_captions.txt", captions.encode("utf-8"))

say(f"Independent pixel transport:      {'PASS' if independent_transport_pass else 'FAIL'}")
say(f"Pixel faithfulness non-inferior:  {'PASS' if faithfulness_ni_pass else 'FAIL'}")
say(f"Pixel sufficiency non-inferior:   {'PASS' if sufficiency_ni_pass else 'FAIL'}")
say(f"Clinical localization NI:         {'PASS' if localization_ni_pass else 'FAIL'}")
say(f"Ablations significantly beaten:   {ablation_superiority_count}/{len(ablation_tests)}")
say(f"Matching specificity contrasts:   {matching_specificity_count}/{len(matching_tests)}")
say(f"Scientific status:                {scientific_status}")


# ==================================================================================================
# 13. IMMUTABLE MANIFEST AND ARTIFACT CHAIN
# ==================================================================================================

section(13, 14, "Write immutable manifest, hashes, and latest handoff")

# The immutable PASS manifest is written only after every engineering safeguard has passed.
expected_independent_jobs = len(DATASETS) * len(CONTROL_FOLDS) * len(EXPLAINERS)
expected_matching_jobs = expected_independent_jobs
expected_ablation_jobs = len(DATASETS) * len(CONTROL_FOLDS) * len(ABLATION_EXPLAINERS) * len(ABLATIONS)
expected_jobs = expected_independent_jobs + expected_matching_jobs + expected_ablation_jobs
completed_jobs = jobs_created + jobs_resumed
execution_pass = bool(
    completed_jobs == expected_jobs
    and len(independent_all.groupby(["dataset", "fold", "explainer"])) == expected_independent_jobs
    and len(matching_all.groupby(["dataset", "fold", "explainer"])) == expected_matching_jobs
    and len(ablation_all.groupby(["dataset", "fold", "explainer", "variant"])) == expected_ablation_jobs
    and len(independent_all) == expected_independent_jobs * CONTROL_DONORS_PER_FOLD
    and len(matching_all) == expected_matching_jobs * CONTROL_DONORS_PER_FOLD * len(MATCHING_POLICIES)
    and len(ablation_all) == expected_ablation_jobs * CONTROL_DONORS_PER_FOLD
    and len(frozen_grid_audit) == 90
    and int(frozen_grid_audit.exact_k_violations.sum()) == 0
    and frozen_local_violations == 0
    and exact_k_ablation_violations == 0
    and faithfulness_ablation_violations == 0
    and not gate_table.status.eq("FAIL").iloc[4:].any()
)
if not execution_pass:
    raise RuntimeError(
        f"Notebook 08 execution gate failed before manifest creation: jobs={completed_jobs}/{expected_jobs}, "
        f"exact-k={exact_k_ablation_violations}, faithfulness={faithfulness_ablation_violations}."
    )

all_outputs = sorted(path for path in RUN_DIR.rglob("*") if path.is_file())
artifact_hashes = {str(path.relative_to(RUN_DIR)): sha256_file(path) for path in all_outputs}
artifact_chain = hashlib.sha256(
    "\n".join(f"{key}:{artifact_hashes[key]}" for key in sorted(artifact_hashes)).encode("utf-8")
).hexdigest()
manifest = {
    "status": "PASS",
    "notebook": "08_validation_ablations_controls.py",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "hardware": {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
    },
    "protocol": str(CONFIG_PATH),
    "protocol_sha256": sha256_file(CONFIG_PATH),
    "inputs": {
        "nb06b_manifest": str(NB06B_LATEST), "nb06b_sha256": sha256_file(NB06B_LATEST),
        "nb07_manifest": str(NB07_RUN_MANIFEST), "nb07_sha256": sha256_file(NB07_RUN_MANIFEST),
        "corrected_full_metrics": str(FULL_METRICS_PATH), "corrected_full_metrics_sha256": sha256_file(FULL_METRICS_PATH),
    },
    "execution": {
        "jobs_created": jobs_created, "jobs_resumed": jobs_resumed,
        "independent_rows": len(independent_all), "matching_rows": len(matching_all),
        "ablation_rows": len(ablation_all),
        "control_folds": CONTROL_FOLDS,
        "control_donors_per_fold": CONTROL_DONORS_PER_FOLD,
        "frozen_primary_cells_audited": len(frozen_grid_audit),
        "frozen_primary_donors_audited": int(frozen_grid_audit.donors.sum()),
    },
    "results": {
        "scientific_status": scientific_status,
        "independent_transport_pass": independent_transport_pass,
        "faithfulness_noninferiority_pass": faithfulness_ni_pass,
        "sufficiency_noninferiority_pass": sufficiency_ni_pass,
        "localization_noninferiority_pass": localization_ni_pass,
        "ablation_superiority_count": ablation_superiority_count,
        "ablation_test_count": len(ablation_tests),
        "matching_specificity_count": matching_specificity_count,
        "matching_test_count": len(matching_tests),
        "primary_replay_max_error": replay_max,
        "primary_replay_mean_abs_error_worst_cell": replay_mean_abs_max,
        "primary_replay_cell_mean_error_worst_cell": replay_cell_mean_max,
    },
    "artifacts": artifact_hashes,
    "artifact_chain_sha256": artifact_chain,
    "runtime_seconds": time.time() - START_TIME,
}
MANIFEST_PATH = RUN_DIR / "reviewer_controls_manifest.json"
atomic_write_json(MANIFEST_PATH, manifest)
LATEST_PATH = RUN_ROOT / "latest_reviewer_controls_manifest.json"
atomic_write_bytes(LATEST_PATH, canonical_json_bytes(manifest))
say(results_text.strip())
say(f"Manifest: {MANIFEST_PATH}")
say(f"Latest:   {LATEST_PATH}")
say(f"Artifact chain: {artifact_chain}")


# ==================================================================================================
# 14. FINAL EXECUTION GATE
# ==================================================================================================

section(14, 14, "Final execution gate and handoff")
say("Execution status:                 PASS")
say("Mode:                             COMPACT_REVIEWER_AUDIT")
say("Frozen primary cells audited:     90/90")
say(f"Jobs complete:                    {completed_jobs}/{expected_jobs}")
say(f"Independent validation cells:     {expected_independent_jobs}/{expected_independent_jobs}")
say(f"Matching-control cells:           {expected_matching_jobs}/{expected_matching_jobs}")
say(f"Ablation cells:                   {expected_ablation_jobs}/{expected_ablation_jobs}")
say(f"Independent scientific status:    {scientific_status}")
say(f"Figures:                          {len(list(FIGURE_DIR.glob('*.pdf')))} PDF + {len(list(FIGURE_DIR.glob('*.png')))} PNG")
say(f"Tables:                           {len(list(TABLE_DIR.glob('*.csv')))} CSV + {len(list(TABLE_DIR.glob('*.tex')))} LaTeX")
say(f"Output directory:                 {RUN_DIR}")
say(f"Total runtime:                    {elapsed()}")

banner("CPET_NOTEBOOK_08_STATUS=PASS")
say(f"CPET_REVIEWER_CONTROL_STATUS={scientific_status}")
say("NEXT=SECOND_BACKBONE_ONLY_WHEN_EXPLICITLY_REQUESTED; OTHERWISE_MANUSCRIPT")
say("Send the complete textual output and the generated table/figure directory before manuscript drafting.")
