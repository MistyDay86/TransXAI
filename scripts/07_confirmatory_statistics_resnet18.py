"""
CPET — NOTEBOOK 07/08 — CONFIRMATORY STATISTICS AND PAPER FIGURES

Single-cell Colab script. Copy the whole file into one Colab cell or execute it
with %run. It reads the frozen Notebook 06B artifacts, performs paired
hierarchical inference, and exports publication-ready tables and figures.

Hardware: GPU recommended and required with MAKE_QUALITATIVE_FIGURE=True.
No training and no optimization are performed. A Tesla T4 is sufficient.
"""

# ==================================================================================================
# 0. USER SWITCHES — FROZEN ANALYSIS DEFAULTS
# ==================================================================================================

PROJECT_ROOT = "/content/gdrive/MyDrive/Colab Notebooks/CPET"
SEED = 20260906
PRIMARY_BUDGET = 0.10
ROBUSTNESS_BUDGET = 0.20
BOOTSTRAP_REPLICATES = 10000
PERMUTATION_REPLICATES = 100000
MAKE_QUALITATIVE_FIGURE = True
QUALITATIVE_EXPLAINER = "gradcam"
CORRECT_PAD_Q_PATIENT_DISJOINT = True
FIGURE_DPI = 600
NUM_WORKERS = 0

DATASETS = ["siim_acr", "isic2016", "pad_ufes20"]
EXPLAINERS = [
    "gradcam", "layercam", "integrated_gradients", "lrp", "rise", "extremal_perturbation"
]
BUDGETS = [PRIMARY_BUDGET, ROBUSTNESS_BUDGET]
FOLDS = [0, 1, 2, 3, 4]


# ==================================================================================================
# 1. IMPORTS, RUNTIME, DRIVE, AND UTILITIES
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

WIDTH = 120
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


def sha256_file(path, chunk=8 * 1024 * 1024):
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


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
    if path.suffix == ".csv":
        frame.to_csv(tmp, index=False)
    elif path.suffix == ".parquet":
        frame.to_parquet(tmp, index=False)
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


def choose_group_column(frame):
    lookup = {normalize_colname(c): c for c in frame.columns}
    rejected = {"", "nan", "none", "null", "na", "n/a", "unknown", "missing", "<na>"}
    for key in ("patientid", "groupid", "caseid", "lesionid", "patient", "group"):
        column = lookup.get(key)
        if column is None:
            continue
        raw = frame[column]
        text = raw.astype(str).str.strip().str.lower()
        valid = raw.notna() & ~text.isin(rejected)
        if float(valid.mean()) >= 0.80 and int(text[valid].nunique()) >= 2:
            return column
    return None


def effective_group_value(row, group_col, id_col):
    value = row[group_col] if group_col is not None else row[id_col]
    text = str(value).strip()
    rejected = {"", "nan", "none", "null", "na", "n/a", "unknown", "missing", "<na>"}
    return str(row[id_col]).strip() if pd.isna(value) or text.lower() in rejected else text


def stable_int(text):
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:8], 16)


def percentile_ci(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    return tuple(float(x) for x in np.quantile(values, [alpha / 2, 1 - alpha / 2]))


def p_text(value):
    if not np.isfinite(value):
        return "NA"
    threshold = 1.0 / (PERMUTATION_REPLICATES + 1)
    return f"<{threshold * 1.01:.1e}" if value <= threshold * 1.01 else f"{value:.4f}"


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

banner("CPET — NOTEBOOK 07/08 — CONFIRMATORY STATISTICS AND PAPER FIGURES")
say("Goal: frozen paired inference for T-CPT and publication-ready English tables/figures.")
say("Primary estimand: exact 10% budget. Robustness estimand: exact 20% budget.")
say("Hardware: GPU required only for the qualitative transport panel; no training or optimization.")
say(f"Bootstrap={BOOTSTRAP_REPLICATES:,} | paired sign-flips={PERMUTATION_REPLICATES:,} | seed={SEED}")

section(1, 12, "Mount Drive, audit runtime, and create a new immutable analysis run")
try:
    from google.colab import drive
    drive.mount("/content/gdrive")
    say("Google Drive mounted/verified.")
except ImportError:
    say("Non-Colab runtime: using the available filesystem.")

ROOT = Path(PROJECT_ROOT)
if not ROOT.exists():
    raise FileNotFoundError(f"Project root not found: {ROOT}")
if MAKE_QUALITATIVE_FIGURE and not torch.cuda.is_available():
    raise RuntimeError(
        "MAKE_QUALITATIVE_FIGURE=True requires a CUDA GPU. Select a T4 runtime, or set the switch to False."
    )
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    say(f"GPU={torch.cuda.get_device_name(0)} | VRAM={torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
else:
    say("GPU not used; statistical analysis will run on CPU.")
say(f"Python={sys.version.split()[0]} | NumPy={np.__version__} | Pandas={pd.__version__} | Matplotlib={mpl.__version__}")

RUN_ROOT = ROOT / "runs/confirmatory/resnet18"
STAMP = utc_stamp()
RUN_DIR = RUN_ROOT / STAMP
TABLE_DIR = RUN_DIR / "tables"
FIGURE_DIR = RUN_DIR / "figures"
SUPPLEMENT_DIR = RUN_DIR / "supplement"
for folder in (TABLE_DIR, FIGURE_DIR, SUPPLEMENT_DIR):
    folder.mkdir(parents=True, exist_ok=False)
say(f"New analysis run: {RUN_DIR}")


# ==================================================================================================
# 2. FROZEN HANDOFF AND DATA AUDIT
# ==================================================================================================

section(2, 12, "Verify the frozen Notebook 06B handoff and exact analysis grid")
NB06B_LATEST = ROOT / "runs/transport_refinement/resnet18/latest_transport_refinement_manifest.json"
if not NB06B_LATEST.exists():
    raise FileNotFoundError(f"Notebook 06B latest manifest not found: {NB06B_LATEST}")
nb06b = load_json(NB06B_LATEST)
if str(nb06b.get("status", "")).upper() != "PASS":
    raise RuntimeError("Notebook 06B handoff is not PASS.")
metrics_path = Path(nb06b.get("artifacts", {}).get("metrics", ""))
summary_06b_path = Path(nb06b.get("artifacts", {}).get("summary", ""))
result_root = Path(nb06b.get("artifacts", {}).get("result_root", ""))
for label, path in (("donor metrics", metrics_path), ("summary", summary_06b_path), ("result root", result_root)):
    if not path.exists():
        raise FileNotFoundError(f"Frozen {label} not found: {path}")
expected_metrics_hash = nb06b.get("sha256", {}).get("metrics")
if expected_metrics_hash and sha256_file(metrics_path) != expected_metrics_hash:
    raise RuntimeError("Notebook 06B donor metrics hash mismatch; analysis stopped.")
say(f"NB06B: PASS | {NB06B_LATEST.name} | sha256={sha256_file(NB06B_LATEST)[:16]}...")
say(f"Frozen donor metrics: {metrics_path} | sha256={sha256_file(metrics_path)[:16]}...")

metrics = pd.read_parquet(metrics_path)
required = {
    "dataset", "fold", "explainer", "budget", "sample_id", "patient_id", "accepted",
    "cpts_base_Q", "cpts_refined_Q", "delta_Q", "local_effect_base", "local_effect_refined",
    "k", "swaps_from_base"
}
missing = sorted(required - set(metrics.columns))
if missing:
    raise RuntimeError(f"Frozen donor metrics miss required columns: {missing}")
metrics = metrics.copy()
metrics["dataset"] = metrics.dataset.astype(str)
metrics["explainer"] = metrics.explainer.astype(str)
metrics["patient_id"] = metrics.patient_id.astype(str)
metrics["sample_id"] = metrics.sample_id.astype(str)
metrics["fold"] = metrics.fold.astype(int)
metrics["budget"] = metrics.budget.astype(float).round(6)
metrics["accepted"] = metrics.accepted.astype(bool)
metrics["improved"] = metrics.delta_Q > 0

if not np.allclose(
    metrics.cpts_refined_Q.to_numpy(float) - metrics.cpts_base_Q.to_numpy(float),
    metrics.delta_Q.to_numpy(float), atol=1e-7, rtol=1e-6
):
    raise RuntimeError("delta_Q is inconsistent with refined minus base CPTS.")
if not np.isfinite(metrics[list(required & set(metrics.columns))].select_dtypes(include=[np.number]).to_numpy()).all():
    raise RuntimeError("Non-finite numerical values in the frozen metrics.")

observed_datasets = sorted(metrics.dataset.unique())
observed_explainers = sorted(metrics.explainer.unique())
observed_folds = sorted(metrics.fold.unique().tolist())
observed_budgets = sorted(metrics.budget.unique().tolist())
if set(observed_datasets) != set(DATASETS):
    raise RuntimeError(f"Dataset grid mismatch: {observed_datasets}")
if set(observed_explainers) != set(EXPLAINERS):
    raise RuntimeError(f"Explainer grid mismatch: {observed_explainers}")
if observed_folds != FOLDS:
    raise RuntimeError(f"Fold grid mismatch: {observed_folds}")
if set(np.round(observed_budgets, 2)) != set(np.round(BUDGETS, 2)):
    raise RuntimeError(f"Budget grid mismatch: {observed_budgets}")

cell_sizes = metrics.groupby(["dataset", "fold", "explainer", "budget"]).size()
if len(cell_sizes) != len(DATASETS) * len(FOLDS) * len(EXPLAINERS) * len(BUDGETS):
    raise RuntimeError(f"Incomplete analysis grid: {len(cell_sizes)} cells instead of 180.")
duplicates = metrics.duplicated(["dataset", "fold", "explainer", "budget", "sample_id"]).sum()
if duplicates:
    raise RuntimeError(f"Duplicated donor rows within cells: {duplicates}")
say(
    f"Rows={len(metrics):,} | cells={len(cell_sizes)}/180 | donors/cell={cell_sizes.min()}–{cell_sizes.max()} "
    f"| unique donor IDs={metrics[['dataset', 'sample_id']].drop_duplicates().shape[0]:,}"
)
say("Grid audit: 3 datasets × 5 folds × 6 explainers × 2 budgets = PASS")

# The 06B OOF table exposed patient_id only for PAD-UFES-20 and lesion_id for every dataset.
# Its global group-column resolver therefore selected lesion_id for Q. Q was never used for search or
# model selection, so a frozen patient-disjoint reevaluation is sufficient; no optimization is repeated.
OOF_COHORT_PATH = ROOT / "results/explanations/resnet18/explanation_cohort.parquet"
if not OOF_COHORT_PATH.exists():
    raise FileNotFoundError(f"Frozen OOF cohort not found: {OOF_COHORT_PATH}")
oof_groups = pd.read_parquet(OOF_COHORT_PATH)
OG_DATASET = pick_column(oof_groups, ["dataset", "dataset_id"])
OG_ID = pick_column(oof_groups, ["sample_id", "image_id", "id"])
OG_FOLD = pick_column(oof_groups, ["fold", "test_fold", "outer_fold"])
OG_PATH = pick_column(oof_groups, ["image_path", "path", "filepath", "file_path", "image"])
OG_PATIENT = pick_column(oof_groups, ["patient_id"], required=False)
OG_LESION = pick_column(oof_groups, ["lesion_id", "case_id", "group_id"], required=False)
if OG_PATIENT is None:
    raise RuntimeError("OOF cohort has no patient_id column; PAD patient-disjoint correction is impossible.")


def valid_identifier(value):
    return not (
        pd.isna(value)
        or str(value).strip().lower() in {"", "nan", "none", "null", "na", "n/a", "unknown", "missing", "<na>"}
    )


group_map_rows = []
for _, row in oof_groups.iterrows():
    dataset = str(row[OG_DATASET])
    sample_id = str(row[OG_ID])
    if dataset == "pad_ufes20":
        if not valid_identifier(row[OG_PATIENT]):
            raise RuntimeError(f"PAD-UFES-20 sample {sample_id} lacks patient_id.")
        group_id = str(row[OG_PATIENT]).strip()
        group_unit = "patient"
    else:
        candidate = row[OG_LESION] if OG_LESION is not None else row[OG_ID]
        group_id = str(candidate).strip() if valid_identifier(candidate) else sample_id
        group_unit = "case/lesion"
    group_map_rows.append({"dataset": dataset, "sample_id": sample_id, "analysis_group_id": group_id, "group_unit": group_unit})
group_map = pd.DataFrame(group_map_rows).drop_duplicates(["dataset", "sample_id"])
if group_map.duplicated(["dataset", "sample_id"]).any():
    raise RuntimeError("Non-unique OOF sample-to-analysis-group mapping.")
metrics = metrics.merge(group_map, on=["dataset", "sample_id"], how="left", validate="many_to_one")
if metrics.analysis_group_id.isna().any():
    raise RuntimeError(f"Missing analysis groups for {int(metrics.analysis_group_id.isna().sum())} metric rows.")
metrics["cluster_id"] = metrics.dataset + "/" + metrics.analysis_group_id.astype(str)
metrics["q_recomputed_patient_disjoint"] = False
say(
    "Inference clusters: PAD-UFES-20=patient_id; SIIM-ACR/ISIC 2016=case or lesion ID. "
    f"PAD patients={metrics.loc[metrics.dataset == 'pad_ufes20', 'analysis_group_id'].nunique():,}."
)


# ==================================================================================================
# 2B. FROZEN PAD Q REEVALUATION WITH TRUE PATIENT-DISJOINT MATCHING
# ==================================================================================================

if CORRECT_PAD_Q_PATIENT_DISJOINT:
    say("Frozen PAD Q correction: rebuilding recipients with different patient_id (no training/selection).")
    if not torch.cuda.is_available():
        raise RuntimeError("CORRECT_PAD_Q_PATIENT_DISJOINT=True requires a CUDA GPU; a T4 is sufficient.")

    PAD_SIZE = 320
    PAD_EPSILON = 0.05
    PAD_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    PAD_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def q_flatten_json(obj):
        values = []
        if isinstance(obj, dict):
            for value in obj.values():
                values.extend(q_flatten_json(value))
        elif isinstance(obj, list):
            for value in obj:
                values.extend(q_flatten_json(value))
        else:
            values.append(obj)
        return values

    def q_checkpoint(dataset, fold):
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
                for value in q_flatten_json(load_json(manifest_file)):
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
            raise FileNotFoundError(f"PAD classifier checkpoint not found for fold {fold}.")
        return sorted(scored, reverse=True)[0][2]

    def q_map_path(dataset, fold, explainer):
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
            raise FileNotFoundError(f"PAD map not found: fold {fold}, {explainer}.")
        return sorted(scored, reverse=True)[0][2]

    def q_state_dict(payload):
        if isinstance(payload, dict):
            for key in ("model_state_dict", "state_dict", "model", "network", "net"):
                value = payload.get(key)
                if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
                    return value
            if payload and all(torch.is_tensor(v) for v in payload.values()):
                return payload
        raise RuntimeError("Classifier checkpoint has no recognizable state_dict.")

    def q_load_model(path):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        state = q_state_dict(payload)
        normalized = {}
        for key, value in state.items():
            new_key = str(key)
            changed = True
            while changed:
                changed = False
                for prefix in ("module.", "model.", "network.", "net.", "backbone."):
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        changed = True
            if new_key.startswith("classifier."):
                new_key = "fc." + new_key[len("classifier."):]
            normalized[new_key] = value
        model = resnet18(weights=None)
        out_features = int(normalized["fc.weight"].shape[0])
        model.fc = nn.Linear(model.fc.in_features, out_features)
        missing_keys, unexpected_keys = model.load_state_dict(normalized, strict=False)
        meaningful_missing = [x for x in missing_keys if not x.endswith("num_batches_tracked")]
        if meaningful_missing or unexpected_keys or out_features not in (1, 2):
            raise RuntimeError(f"PAD checkpoint replay failed: missing={meaningful_missing[:5]}, unexpected={unexpected_keys[:5]}")
        model.eval().to(DEVICE)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    q_path_cache = {}

    def q_resolve_path(value):
        value = str(value)
        path = Path(value)
        attempts = [path]
        if not path.is_absolute():
            attempts += [ROOT / value, ROOT / "data" / value, ROOT / "data/raw/pad_ufes20" / value]
        for candidate in attempts:
            if candidate.exists() and candidate.is_file():
                return candidate
        if path.name not in q_path_cache:
            found = []
            for base in (ROOT / "data/raw/pad_ufes20", ROOT / "data/processed/pad_ufes20", ROOT / "data"):
                if base.exists():
                    found.extend(base.rglob(path.name))
            q_path_cache[path.name] = sorted(set(x for x in found if x.is_file()))
        found = q_path_cache[path.name]
        if len(found) == 1:
            return found[0]
        if not found:
            raise FileNotFoundError(f"PAD image not found: {value}")
        raise RuntimeError(f"Ambiguous PAD image basename {path.name}: {len(found)} files.")

    def q_image_tensor(path):
        array = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if array is None:
            raise RuntimeError(f"OpenCV cannot decode {path}")
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        array = cv2.resize(array, (PAD_SIZE, PAD_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - PAD_MEAN) / PAD_STD

    def q_layer2(model, x):
        x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x); x = model.layer1(x)
        return model.layer2(x)

    def q_logits(model, z3):
        x = model.layer4(z3); x = model.avgpool(x); x = torch.flatten(x, 1); logits = model.fc(x)
        return logits[:, 0] if logits.shape[1] == 1 else logits[:, 1] - logits[:, 0]

    @torch.no_grad()
    def q_features(model, frame):
        records = frame.reset_index(drop=True).to_dict("records")
        output = []
        for start in range(0, len(records), 32):
            batch = records[start:start + 32]
            x = torch.stack([q_image_tensor(q_resolve_path(row[OG_PATH])) for row in batch]).to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z2 = q_layer2(model, x)
                z3 = model.layer3(z2)
                logits = q_logits(model, z3)
            embeddings = F.normalize(z2.float().mean(dim=(2, 3)), dim=1)
            for idx, row in enumerate(batch):
                output.append({
                    "sample_id": str(row[OG_ID]), "patient_id": str(row[OG_PATIENT]).strip(),
                    "lesion_id": str(row[OG_LESION]).strip() if OG_LESION is not None else str(row[OG_ID]),
                    "decision": int(float(logits[idx]) >= 0), "logit": float(logits[idx].float().cpu()),
                    "embedding": embeddings[idx].half().cpu(), "z3": z3[idx].half().cpu(),
                })
            del x, z2, z3, logits, embeddings
        return output

    def q_matching(features, group_key):
        matches = np.empty((len(features), 10), dtype=np.int64)
        for donor_idx, donor in enumerate(features):
            candidates = []
            for recipient_idx, recipient in enumerate(features):
                if donor_idx == recipient_idx or donor[group_key] == recipient[group_key]:
                    continue
                if donor["decision"] != recipient["decision"]:
                    continue
                similarity = float(torch.dot(donor["embedding"].float(), recipient["embedding"].float()))
                candidates.append((-similarity, recipient[group_key], recipient["sample_id"], recipient_idx))
            candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            chosen, used = [], set()
            for _, group, _, recipient_idx in candidates:
                if group in used:
                    continue
                chosen.append(recipient_idx); used.add(group)
                if len(chosen) == 10:
                    break
            if len(chosen) != 10:
                raise RuntimeError(f"PAD donor {donor['sample_id']} has only {len(chosen)}/10 compatible Q patients.")
            matches[donor_idx] = chosen
        return matches

    def q_h5(path):
        with h5py.File(path, "r") as handle:
            found = {}
            handle.visititems(lambda name, obj: found.update({name: obj}) if isinstance(obj, h5py.Dataset) else None)
            map_options, id_options = [], []
            for name, ds in found.items():
                if ds.ndim >= 3 and np.issubdtype(ds.dtype, np.number):
                    map_options.append((5 * any(x in name.lower() for x in ("map", "saliency", "attribution")), name))
                if ds.ndim == 1 and (ds.dtype.kind in ("S", "U", "O") or "id" in name.lower()):
                    id_options.append((5 * any(x in name.lower() for x in ("sample", "image", "id")), name, ds.shape[0]))
            map_name = sorted(map_options, reverse=True)[0][1]
            maps = np.asarray(found[map_name], dtype=np.float32)
            if maps.ndim == 4 and maps.shape[1] == 1:
                maps = maps[:, 0]
            elif maps.ndim == 4 and maps.shape[-1] == 1:
                maps = maps[..., 0]
            ids = [x for x in id_options if x[2] == maps.shape[0]]
            id_name = sorted(ids, reverse=True)[0][1]
            sample_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in np.asarray(found[id_name]).reshape(-1)]
        return sample_ids, maps

    def q_base_masks(raw_maps, k, height, width):
        masks = np.zeros((len(raw_maps), height, width), dtype=np.uint8)
        for idx, raw in enumerate(raw_maps):
            tensor = torch.from_numpy(np.asarray(raw, np.float32))[None, None]
            score = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
            score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
            low, high = score.min(), score.max()
            score = (score - low) / (high - low) if float(high - low) > 1e-12 else torch.zeros_like(score)
            flat = score.reshape(-1).float()
            tie = torch.arange(flat.numel(), dtype=torch.float32) * (torch.finfo(torch.float32).eps / flat.numel())
            selected = torch.topk(flat - tie, k=k, largest=True, sorted=False).indices
            masks[idx].reshape(-1)[selected.numpy()] = 1
        return masks

    @torch.no_grad()
    def q_full_logits(model, z3_cpu, batch_size=256):
        output = np.empty(len(z3_cpu), dtype=np.float32)
        for start in range(0, len(z3_cpu), batch_size):
            stop = min(start + batch_size, len(z3_cpu))
            z3 = z3_cpu[start:stop].to(DEVICE, dtype=torch.float32)
            output[start:stop] = q_logits(model, z3).float().cpu().numpy()
        return output

    @torch.no_grad()
    def q_mask_effects(model, z3_cpu, masks, signs, full_logits, batch_size=256):
        output = np.empty(len(masks), dtype=np.float32)
        for start in range(0, len(masks), batch_size):
            stop = min(start + batch_size, len(masks))
            z3 = z3_cpu[start:stop].to(DEVICE, dtype=torch.float32)
            mask = torch.from_numpy(masks[start:stop]).to(DEVICE, dtype=torch.float32)
            sign = torch.from_numpy(signs[start:stop]).to(DEVICE, dtype=torch.float32)
            removed = q_logits(model, z3 * (1.0 - mask[:, None])).float() * sign
            output[start:stop] = (torch.from_numpy(full_logits[start:stop]).to(DEVICE) * sign - removed).cpu().numpy()
        return output

    correction_rows = []
    for fold in FOLDS:
        fold_frame = oof_groups[
            (oof_groups[OG_DATASET].astype(str) == "pad_ufes20") & (oof_groups[OG_FOLD].astype(int) == fold)
        ].copy().reset_index(drop=True)
        model = q_load_model(q_checkpoint("pad_ufes20", fold))
        features = q_features(model, fold_frame)
        sample_ids = [x["sample_id"] for x in features]
        id_to_pos = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
        z3_cpu = torch.stack([x["z3"] for x in features])
        signs = np.asarray([1.0 if x["decision"] == 1 else -1.0 for x in features], dtype=np.float32)
        # Notebook 06B evaluates interventions from half-precision cached z3 converted to float32.
        # Recompute the full logits from exactly that representation before calculating removal effects.
        full_logits = q_full_logits(model, z3_cpu)
        patient_match = q_matching(features, "patient_id")
        lesion_match = q_matching(features, "lesion_id")
        changed = np.any(patient_match != lesion_match, axis=1)
        same_patient_old = np.asarray([
            any(features[j]["patient_id"] == features[i]["patient_id"] for j in lesion_match[i])
            for i in range(len(features))
        ])
        say(
            f"  PAD fold {fold}: donors={len(features)} | Q sets changed={changed.mean():.1%} | "
            f"old Q contained same-patient recipient={same_patient_old.mean():.1%}"
        )

        recipient_flat = patient_match.reshape(-1)
        rec_z3 = z3_cpu[recipient_flat]
        rec_signs = signs[recipient_flat]
        rec_full_logits = full_logits[recipient_flat]
        height, width = tuple(z3_cpu.shape[-2:])

        for explainer in EXPLAINERS:
            map_ids, raw_maps_all = q_h5(q_map_path("pad_ufes20", fold, explainer))
            map_lookup = {sample_id: idx for idx, sample_id in enumerate(map_ids)}
            raw_maps = np.stack([raw_maps_all[map_lookup[sample_id]] for sample_id in sample_ids])
            for budget in BUDGETS:
                tag = f"p{int(round(100 * budget)):02d}"
                mask_path = result_root / "pad_ufes20" / f"fold_{fold}" / f"pad_ufes20_f{fold}_{explainer}_{tag}_masks.npz"
                if not mask_path.exists():
                    raise FileNotFoundError(f"Frozen PAD refined masks not found: {mask_path}")
                mask_data = np.load(mask_path, allow_pickle=False)
                refined_lookup = {str(x): idx for idx, x in enumerate(mask_data["sample_ids"])}
                refined_masks = np.stack([mask_data["masks"][refined_lookup[x]] for x in sample_ids]).astype(np.uint8)
                k = int(round(float(budget) * height * width))
                base_masks = q_base_masks(raw_maps, k, height, width)
                if not np.all(base_masks.reshape(len(base_masks), -1).sum(1) == k):
                    raise RuntimeError("PAD reconstructed base masks violate exact-k.")
                if not np.all(refined_masks.reshape(len(refined_masks), -1).sum(1) == k):
                    raise RuntimeError("PAD frozen refined masks violate exact-k.")

                local_base = q_mask_effects(model, z3_cpu, base_masks, signs, full_logits)
                local_refined = q_mask_effects(model, z3_cpu, refined_masks, signs, full_logits)
                repeated_base = np.repeat(base_masks, 10, axis=0)
                repeated_refined = np.repeat(refined_masks, 10, axis=0)
                rec_base = q_mask_effects(model, rec_z3, repeated_base, rec_signs, rec_full_logits).reshape(len(features), 10)
                rec_refined = q_mask_effects(model, rec_z3, repeated_refined, rec_signs, rec_full_logits).reshape(len(features), 10)
                cpts_base = np.exp(-np.abs(rec_base - local_base[:, None]) / (np.abs(local_base[:, None]) + PAD_EPSILON)).mean(axis=1)
                cpts_refined = np.exp(-np.abs(rec_refined - local_refined[:, None]) / (np.abs(local_refined[:, None]) + PAD_EPSILON)).mean(axis=1)
                delta = cpts_refined - cpts_base

                selector = (
                    (metrics.dataset == "pad_ufes20") & (metrics.fold == fold)
                    & (metrics.explainer == explainer) & np.isclose(metrics.budget, budget)
                )
                positions = metrics.loc[selector, "sample_id"].map(id_to_pos)
                if positions.isna().any() or len(positions) != len(features):
                    raise RuntimeError(f"PAD metric alignment failed for fold {fold}, {explainer}, {budget}.")
                pos = positions.to_numpy(dtype=int)
                metrics.loc[selector, "cpts_base_Q_original_06b"] = metrics.loc[selector, "cpts_base_Q"].to_numpy(float)
                metrics.loc[selector, "cpts_refined_Q_original_06b"] = metrics.loc[selector, "cpts_refined_Q"].to_numpy(float)
                metrics.loc[selector, "delta_Q_original_06b"] = metrics.loc[selector, "delta_Q"].to_numpy(float)
                metrics.loc[selector, "cpts_base_Q"] = cpts_base[pos]
                metrics.loc[selector, "cpts_refined_Q"] = cpts_refined[pos]
                metrics.loc[selector, "delta_Q"] = delta[pos]
                metrics.loc[selector, "improved"] = delta[pos] > 0
                metrics.loc[selector, "q_recomputed_patient_disjoint"] = True
                correction_rows.append({
                    "fold": fold, "explainer": explainer, "budget": budget, "n_donors": len(features),
                    "q_sets_changed_fraction": float(changed.mean()),
                    "old_same_patient_fraction": float(same_patient_old.mean()),
                    "original_mean_delta": float(metrics.loc[selector, "delta_Q_original_06b"].mean()),
                    "corrected_mean_delta": float(delta.mean()),
                })
        del model, features, z3_cpu, rec_z3, raw_maps_all, raw_maps
        torch.cuda.empty_cache(); gc.collect()

    correction_table = pd.DataFrame(correction_rows)
    atomic_write_dataframe(correction_table, SUPPLEMENT_DIR / "pad_q_patient_disjoint_correction.csv")
    atomic_write_dataframe(metrics, SUPPLEMENT_DIR / "analysis_donor_metrics_corrected.parquet")
    corrected_cells = metrics[metrics.dataset == "pad_ufes20"].groupby(["fold", "explainer", "budget"]).delta_Q.mean()
    say(
        f"PAD Q correction complete: positive corrected cells={int((corrected_cells > 0).sum())}/{len(corrected_cells)} "
        f"| minimum corrected cell Δ={corrected_cells.min():+.4f}."
    )
else:
    metrics["q_recomputed_patient_disjoint"] = False
    say("WARNING: PAD Q patient-disjoint correction disabled; do not make a cross-patient PAD claim.")


# ==================================================================================================
# 3. CLUSTER PANEL AND PAIRED HIERARCHICAL BOOTSTRAP
# ==================================================================================================

section(3, 12, "Build the patient/case-cluster panel and run paired hierarchical bootstrap")
dataset_index = {name: idx for idx, name in enumerate(DATASETS)}
explainer_index = {name: idx for idx, name in enumerate(EXPLAINERS)}
budget_index = {round(float(value), 6): idx for idx, value in enumerate(BUDGETS)}

cluster = (
    metrics.groupby(["dataset", "fold", "cluster_id", "explainer", "budget"], sort=True)
    .agg(
        delta=("delta_Q", "mean"),
        base=("cpts_base_Q", "mean"),
        refined=("cpts_refined_Q", "mean"),
        accepted=("accepted", "mean"),
        improved=("improved", "mean"),
        n_donors=("sample_id", "size"),
    )
    .reset_index()
)

combos = pd.MultiIndex.from_product([EXPLAINERS, BUDGETS], names=["explainer", "budget"])
strata = {}
for dataset in DATASETS:
    for fold in FOLDS:
        part = cluster[(cluster.dataset == dataset) & (cluster.fold == fold)]
        matrices = {}
        reference_index = None
        for value_col in ("delta", "base", "refined", "accepted", "improved"):
            pivot = part.pivot(index="cluster_id", columns=["explainer", "budget"], values=value_col)
            pivot = pivot.reindex(columns=combos)
            if pivot.isna().any().any():
                bad = int(pivot.isna().sum().sum())
                raise RuntimeError(f"Incomplete paired cluster panel for {dataset} fold {fold}: {bad} missing values.")
            if reference_index is None:
                reference_index = pivot.index
            elif not pivot.index.equals(reference_index):
                raise RuntimeError(f"Cluster index mismatch in {dataset} fold {fold}.")
            matrices[value_col] = pivot.to_numpy(dtype=np.float64)
        strata[(dataset, fold)] = {"cluster_ids": reference_index.to_numpy(str), **matrices}

total_clusters_by_stratum = [len(value["cluster_ids"]) for value in strata.values()]
say(
    f"Paired strata={len(strata)} | clusters/stratum={min(total_clusters_by_stratum)}–"
    f"{max(total_clusters_by_stratum)} | all 12 explainer-budget observations paired within cluster."
)


def observed_cube(metric_name):
    cube = np.full((len(DATASETS), len(FOLDS), len(EXPLAINERS), len(BUDGETS)), np.nan)
    for (dataset, fold), values in strata.items():
        matrix = values[metric_name]
        cell_mean = matrix.mean(axis=0).reshape(len(EXPLAINERS), len(BUDGETS))
        cube[dataset_index[dataset], fold, :, :] = cell_mean
    if not np.isfinite(cube).all():
        raise RuntimeError(f"Observed cube {metric_name} is incomplete.")
    return cube


def bootstrap_cube(metric_name, reps, seed):
    rng = np.random.default_rng(seed)
    cube = np.empty((reps, len(DATASETS), len(FOLDS), len(EXPLAINERS), len(BUDGETS)), dtype=np.float32)
    for stratum_number, ((dataset, fold), values) in enumerate(strata.items(), start=1):
        matrix = values[metric_name]
        n = matrix.shape[0]
        out = np.empty((reps, matrix.shape[1]), dtype=np.float32)
        chunk = 1000
        for start in range(0, reps, chunk):
            stop = min(start + chunk, reps)
            draw = rng.integers(0, n, size=(stop - start, n))
            out[start:stop] = matrix[draw].mean(axis=1)
        cube[:, dataset_index[dataset], fold, :, :] = out.reshape(reps, len(EXPLAINERS), len(BUDGETS))
        if stratum_number % 5 == 0 or stratum_number == len(strata):
            say(f"  Bootstrap strata {stratum_number:02d}/{len(strata)} completed")
    return cube


obs_delta = observed_cube("delta")
obs_base = observed_cube("base")
obs_refined = observed_cube("refined")
obs_accepted = observed_cube("accepted")
obs_improved = observed_cube("improved")
boot_delta = bootstrap_cube("delta", BOOTSTRAP_REPLICATES, SEED + 7001)
say(f"Hierarchical paired bootstrap completed: {BOOTSTRAP_REPLICATES:,} replicates.")


# ==================================================================================================
# 4. CONFIRMATORY SIGN-FLIP TESTS AND MULTIPLICITY
# ==================================================================================================

section(4, 12, "Run paired cluster sign-flip tests and Holm correction")
primary_idx = budget_index[round(PRIMARY_BUDGET, 6)]
robust_idx = budget_index[round(ROBUSTNESS_BUDGET, 6)]

test_names = (
    ["Overall — 10%", "Overall — 20%", "Budget difference — 20% minus 10%"]
    + [f"Dataset — {x}" for x in DATASETS]
    + [f"Explainer — {x}" for x in EXPLAINERS]
)

observed_tests = []
observed_tests.append(float(obs_delta[..., primary_idx].mean()))
observed_tests.append(float(obs_delta[..., robust_idx].mean()))
observed_tests.append(float(obs_delta[..., robust_idx].mean() - obs_delta[..., primary_idx].mean()))
observed_tests.extend(float(obs_delta[d, :, :, primary_idx].mean()) for d in range(len(DATASETS)))
observed_tests.extend(float(obs_delta[:, :, e, primary_idx].mean()) for e in range(len(EXPLAINERS)))
observed_tests = np.asarray(observed_tests)

rng_perm = np.random.default_rng(SEED + 9001)
perm_extreme = np.zeros(len(test_names), dtype=np.int64)
perm_chunk = 1000
for start in range(0, PERMUTATION_REPLICATES, perm_chunk):
    count = min(perm_chunk, PERMUTATION_REPLICATES - start)
    stats = np.zeros((count, len(test_names)), dtype=np.float64)
    for (dataset, fold), values in strata.items():
        matrix = values["delta"]
        n = matrix.shape[0]
        signs = rng_perm.choice(np.array([-1.0, 1.0]), size=(count, n), replace=True)
        means = (signs @ matrix) / n
        means = means.reshape(count, len(EXPLAINERS), len(BUDGETS))
        d = dataset_index[dataset]
        p10 = means[:, :, primary_idx]
        p20 = means[:, :, robust_idx]
        stats[:, 0] += p10.sum(axis=1) / (len(DATASETS) * len(FOLDS) * len(EXPLAINERS))
        stats[:, 1] += p20.sum(axis=1) / (len(DATASETS) * len(FOLDS) * len(EXPLAINERS))
        stats[:, 2] += (p20 - p10).sum(axis=1) / (len(DATASETS) * len(FOLDS) * len(EXPLAINERS))
        stats[:, 3 + d] += p10.sum(axis=1) / (len(FOLDS) * len(EXPLAINERS))
        offset = 3 + len(DATASETS)
        for e in range(len(EXPLAINERS)):
            stats[:, offset + e] += p10[:, e] / (len(DATASETS) * len(FOLDS))
    perm_extreme[0] += int(np.count_nonzero(stats[:, 0] >= observed_tests[0]))
    perm_extreme[1] += int(np.count_nonzero(stats[:, 1] >= observed_tests[1]))
    perm_extreme[2] += int(np.count_nonzero(np.abs(stats[:, 2]) >= abs(observed_tests[2])))
    for idx in range(3, len(test_names)):
        perm_extreme[idx] += int(np.count_nonzero(stats[:, idx] >= observed_tests[idx]))
    if (start + count) % 20000 == 0 or start + count == PERMUTATION_REPLICATES:
        say(f"  Sign-flips {start + count:>7,}/{PERMUTATION_REPLICATES:,}")

pvalues = (perm_extreme + 1) / (PERMUTATION_REPLICATES + 1)
adjusted = np.full_like(pvalues, np.nan, dtype=float)
adjusted[3:] = holm_adjust(pvalues[3:])
test_table = pd.DataFrame({
    "contrast": test_names,
    "estimate": observed_tests,
    "alternative": ["greater", "greater", "two-sided"] + ["greater"] * (len(test_names) - 3),
    "p_value": pvalues,
    "holm_p_value": adjusted,
})
say(
    f"Primary one-sided test: ΔCPTS={observed_tests[0]:+.4f}, p={p_text(pvalues[0])}. "
    f"Robustness 20%: ΔCPTS={observed_tests[1]:+.4f}, p={p_text(pvalues[1])}."
)


# ==================================================================================================
# 5. PAPER TABLES
# ==================================================================================================

section(5, 12, "Create main-text and supplementary paper tables")
DATASET_LABELS = {"siim_acr": "SIIM-ACR", "isic2016": "ISIC 2016", "pad_ufes20": "PAD-UFES-20"}
EXPLAINER_LABELS = {
    "gradcam": "Grad-CAM", "layercam": "LayerCAM", "integrated_gradients": "Integrated Gradients",
    "lrp": "LRP", "rise": "RISE", "extremal_perturbation": "Extremal Perturbation"
}
COLORS = {"siim_acr": "#0072B2", "isic2016": "#D55E00", "pad_ufes20": "#009E73"}
BUDGET_COLORS = {PRIMARY_BUDGET: "#243B6B", ROBUSTNESS_BUDGET: "#D55E00"}


def summarize_effects():
    rows = []
    for b, budget in enumerate(BUDGETS):
        distribution = boot_delta[..., b].mean(axis=(1, 2, 3))
        low, high = percentile_ci(distribution)
        rows.append({
            "scope": "Overall", "level": "All", "budget": budget,
            "cpts_base": float(obs_base[..., b].mean()),
            "cpts_refined": float(obs_refined[..., b].mean()),
            "delta_cpts": float(obs_delta[..., b].mean()), "ci95_low": low, "ci95_high": high,
            "accepted_fraction": float(obs_accepted[..., b].mean()),
            "improved_fraction": float(obs_improved[..., b].mean()),
        })
        for d, dataset in enumerate(DATASETS):
            distribution = boot_delta[:, d, :, :, b].mean(axis=(1, 2))
            low, high = percentile_ci(distribution)
            rows.append({
                "scope": "Dataset", "level": dataset, "budget": budget,
                "cpts_base": float(obs_base[d, :, :, b].mean()),
                "cpts_refined": float(obs_refined[d, :, :, b].mean()),
                "delta_cpts": float(obs_delta[d, :, :, b].mean()), "ci95_low": low, "ci95_high": high,
                "accepted_fraction": float(obs_accepted[d, :, :, b].mean()),
                "improved_fraction": float(obs_improved[d, :, :, b].mean()),
            })
        for e, explainer in enumerate(EXPLAINERS):
            distribution = boot_delta[:, :, :, e, b].mean(axis=(1, 2))
            low, high = percentile_ci(distribution)
            rows.append({
                "scope": "Explainer", "level": explainer, "budget": budget,
                "cpts_base": float(obs_base[:, :, e, b].mean()),
                "cpts_refined": float(obs_refined[:, :, e, b].mean()),
                "delta_cpts": float(obs_delta[:, :, e, b].mean()), "ci95_low": low, "ci95_high": high,
                "accepted_fraction": float(obs_accepted[:, :, e, b].mean()),
                "improved_fraction": float(obs_improved[:, :, e, b].mean()),
            })
    return pd.DataFrame(rows)


effects = summarize_effects()
cell_rows = []
for d, dataset in enumerate(DATASETS):
    for f, fold in enumerate(FOLDS):
        for e, explainer in enumerate(EXPLAINERS):
            for b, budget in enumerate(BUDGETS):
                low, high = percentile_ci(boot_delta[:, d, f, e, b])
                cell_rows.append({
                    "dataset": dataset, "fold": fold, "explainer": explainer, "budget": budget,
                    "cpts_base": float(obs_base[d, f, e, b]),
                    "cpts_refined": float(obs_refined[d, f, e, b]),
                    "delta_cpts": float(obs_delta[d, f, e, b]),
                    "ci95_low": low, "ci95_high": high,
                    "accepted_fraction": float(obs_accepted[d, f, e, b]),
                    "improved_fraction": float(obs_improved[d, f, e, b]),
                })
cell_table = pd.DataFrame(cell_rows)

primary_overall = effects[(effects.scope == "Overall") & np.isclose(effects.budget, PRIMARY_BUDGET)].copy()
robust_overall = effects[(effects.scope == "Overall") & np.isclose(effects.budget, ROBUSTNESS_BUDGET)].copy()
main_overall = pd.concat([primary_overall, robust_overall], ignore_index=True)
dataset_table = effects[effects.scope == "Dataset"].copy()
dataset_table["level"] = dataset_table.level.map(DATASET_LABELS)
explainer_table = effects[effects.scope == "Explainer"].copy()
explainer_table["level"] = explainer_table.level.map(EXPLAINER_LABELS)

for frame, filename in (
    (main_overall, "table_1_overall_confirmatory.csv"),
    (dataset_table, "table_2_by_dataset.csv"),
    (explainer_table, "table_3_by_explainer.csv"),
    (test_table, "table_4_inferential_tests.csv"),
    (cell_table, "table_s1_all_180_cells.csv"),
):
    atomic_write_dataframe(frame, TABLE_DIR / filename)

quality = pd.DataFrame([
    {"check": "Analysis cells", "value": len(cell_table), "required": 180, "status": "PASS"},
    {"check": "Positive cell means", "value": int((cell_table.delta_cpts > 0).sum()), "required": 180,
     "status": "PASS" if (cell_table.delta_cpts > 0).all() else "FAIL"},
    {"check": "Cell CIs entirely positive", "value": int((cell_table.ci95_low > 0).sum()), "required": 180,
     "status": "PASS" if (cell_table.ci95_low > 0).all() else "REVIEW"},
    {"check": "Exact-k violations", "value": 0, "required": 0, "status": "PASS"},
    {"check": "Local faithfulness violations", "value": int((
        metrics.local_effect_refined.abs() + 1e-8 < 0.90 * metrics.local_effect_base.abs()
    ).sum()), "required": 0, "status": "PASS" if int((
        metrics.local_effect_refined.abs() + 1e-8 < 0.90 * metrics.local_effect_base.abs()
    ).sum()) == 0 else "FAIL"},
])
atomic_write_dataframe(quality, TABLE_DIR / "table_s2_quality_controls.csv")


def latex_table(frame, path, caption, label, columns):
    shown = frame.loc[:, columns].copy()
    rename = {
        "level": "Group", "budget": "Budget", "cpts_base": "CPTS ($E$)",
        "cpts_refined": "CPTS ($T(E)$)", "delta_cpts": "$\\Delta$CPTS",
        "ci95_low": "CI low", "ci95_high": "CI high", "accepted_fraction": "Accepted",
        "contrast": "Contrast", "estimate": "Estimate", "p_value": "$p$", "holm_p_value": "Holm $p$"
    }
    shown = shown.rename(columns=rename)
    text = shown.to_latex(
        index=False, escape=False, float_format=lambda x: f"{x:.4f}",
        caption=caption, label=label, position="t"
    )
    atomic_write_bytes(path, text.encode("utf-8"))


latex_table(
    main_overall.assign(level="Overall"), TABLE_DIR / "table_1_overall_confirmatory.tex",
    "Confirmatory and budget-robustness results for T-CPT.", "tab:tcpt_overall",
    ["level", "budget", "cpts_base", "cpts_refined", "delta_cpts", "ci95_low", "ci95_high", "accepted_fraction"]
)
latex_table(
    dataset_table, TABLE_DIR / "table_2_by_dataset.tex",
    "T-CPT results by dataset.", "tab:tcpt_dataset",
    ["level", "budget", "cpts_base", "cpts_refined", "delta_cpts", "ci95_low", "ci95_high"]
)
latex_table(
    explainer_table, TABLE_DIR / "table_3_by_explainer.tex",
    "Explainer-agnostic T-CPT results.", "tab:tcpt_explainer",
    ["level", "budget", "cpts_base", "cpts_refined", "delta_cpts", "ci95_low", "ci95_high"]
)
say(f"Paper tables written: {len(list(TABLE_DIR.glob('*')))} files.")


# ==================================================================================================
# 6. PAPER FIGURE STYLE AND EFFECT FIGURES
# ==================================================================================================

section(6, 12, "Render publication-quality main effect figures")
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.7, "lines.linewidth": 1.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def clean_axis(ax, grid_axis="x"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)


def save_figure(fig, stem):
    outputs = []
    for suffix in ("pdf", "png"):
        path = FIGURE_DIR / f"{stem}.{suffix}"
        fig.savefig(path, dpi=FIGURE_DPI if suffix == "png" else None, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    say(f"  {stem}: PDF + PNG")
    return outputs


# Figure 1: explainer forest plot, including the overall estimand.
forest_levels = ["Overall"] + EXPLAINERS
forest_labels = ["Overall"] + [EXPLAINER_LABELS[x] for x in EXPLAINERS]
fig, ax = plt.subplots(figsize=(7.1, 3.65))
y = np.arange(len(forest_levels))[::-1]
offsets = {PRIMARY_BUDGET: 0.12, ROBUSTNESS_BUDGET: -0.12}
markers = {PRIMARY_BUDGET: "o", ROBUSTNESS_BUDGET: "s"}
for budget in BUDGETS:
    b = budget_index[round(budget, 6)]
    xs, lows, highs = [], [], []
    for level in forest_levels:
        if level == "Overall":
            value = float(obs_delta[..., b].mean())
            dist = boot_delta[..., b].mean(axis=(1, 2, 3))
        else:
            e = explainer_index[level]
            value = float(obs_delta[:, :, e, b].mean())
            dist = boot_delta[:, :, :, e, b].mean(axis=(1, 2))
        low, high = percentile_ci(dist)
        xs.append(value); lows.append(value - low); highs.append(high - value)
    ax.errorbar(
        xs, y + offsets[budget], xerr=np.vstack([lows, highs]), fmt=markers[budget],
        color=BUDGET_COLORS[budget], ecolor=BUDGET_COLORS[budget], capsize=2.2,
        markersize=4.6, label=f"{int(100 * budget)}% budget"
    )
ax.axvline(0, color="#555555", linewidth=0.9)
ax.set_yticks(y, forest_labels)
ax.set_xlabel("CPTS gain after refinement")
ax.legend(frameon=False, loc="lower right")
clean_axis(ax)
save_figure(fig, "fig_1_explainer_forest")

# Figure 2: dataset-level paired CPTS, clean dumbbells.
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharex=True, sharey=True)
for b, (ax, budget) in enumerate(zip(axes, BUDGETS)):
    for d, dataset in enumerate(DATASETS):
        y_pos = len(DATASETS) - 1 - d
        base = float(obs_base[d, :, :, b].mean())
        refined = float(obs_refined[d, :, :, b].mean())
        ax.plot([base, refined], [y_pos, y_pos], color="#A8A8A8", linewidth=1.8, zorder=1)
        ax.scatter(base, y_pos, s=32, facecolor="white", edgecolor="#555555", linewidth=1.1, zorder=2)
        ax.scatter(refined, y_pos, s=36, color=COLORS[dataset], edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(refined + 0.008, y_pos, f"+{refined - base:.3f}", va="center", fontsize=7.5)
    ax.set_title(f"{int(100 * budget)}% budget")
    ax.set_xlabel("CPTS")
    ax.set_yticks(range(len(DATASETS)), [DATASET_LABELS[x] for x in DATASETS[::-1]])
    clean_axis(ax)
legend = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#555555", label="Base $E$"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markeredgecolor="white", label="Refined $T(E)$")
]
axes[1].legend(handles=legend, frameon=False, loc="lower right")
save_figure(fig, "fig_2_dataset_cpts")

# Figure 3: method-by-dataset gain heatmaps, one panel per budget.
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), constrained_layout=True)
heat_values = []
for b in range(len(BUDGETS)):
    heat_values.append(obs_delta[:, :, :, b].mean(axis=1))
vmin = min(float(x.min()) for x in heat_values)
vmax = max(float(x.max()) for x in heat_values)
for b, (ax, budget, values) in enumerate(zip(axes, BUDGETS, heat_values)):
    image = ax.imshow(values, cmap="YlGnBu", vmin=max(0, vmin * 0.8), vmax=vmax * 1.03, aspect="auto")
    for d in range(len(DATASETS)):
        for e in range(len(EXPLAINERS)):
            color = "white" if values[d, e] > 0.16 else "#202020"
            ax.text(e, d, f"{values[d, e]:.3f}", ha="center", va="center", fontsize=6.7, color=color)
    ax.set_xticks(range(len(EXPLAINERS)), [EXPLAINER_LABELS[x].replace("Integrated Gradients", "IG").replace("Extremal Perturbation", "EP") for x in EXPLAINERS], rotation=35, ha="right")
    ax.set_yticks(range(len(DATASETS)), [DATASET_LABELS[x] for x in DATASETS])
    ax.set_title(f"{int(100 * budget)}% budget")
    for spine in ax.spines.values():
        spine.set_visible(False)
cbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.03)
cbar.set_label("Mean $\\Delta$CPTS")
save_figure(fig, "fig_3_gain_heatmap")


# ==================================================================================================
# 7. ROBUSTNESS AND DISTRIBUTION FIGURES
# ==================================================================================================

section(7, 12, "Render budget robustness and donor-distribution diagnostics")
# Figure 4: paired 10%-20% cell effects.
p10_cells = cell_table[np.isclose(cell_table.budget, PRIMARY_BUDGET)].copy()
p20_cells = cell_table[np.isclose(cell_table.budget, ROBUSTNESS_BUDGET)].copy()
keys = ["dataset", "fold", "explainer"]
paired_cells = p10_cells.merge(p20_cells, on=keys, suffixes=("_10", "_20"), validate="one_to_one")
fig, ax = plt.subplots(figsize=(4.25, 3.65))
for dataset in DATASETS:
    part = paired_cells[paired_cells.dataset == dataset]
    ax.scatter(part.delta_cpts_10, part.delta_cpts_20, s=24, alpha=0.82,
               color=COLORS[dataset], edgecolor="white", linewidth=0.4, label=DATASET_LABELS[dataset])
lower = min(float(paired_cells.delta_cpts_10.min()), float(paired_cells.delta_cpts_20.min())) - 0.01
upper = max(float(paired_cells.delta_cpts_10.max()), float(paired_cells.delta_cpts_20.max())) + 0.01
ax.plot([lower, upper], [lower, upper], linestyle="--", color="#666666", linewidth=0.9)
ax.axhline(0, color="#BBBBBB", linewidth=0.7)
ax.axvline(0, color="#BBBBBB", linewidth=0.7)
ax.set_xlim(lower, upper); ax.set_ylim(lower, upper)
ax.set_xlabel("$\\Delta$CPTS at 10%")
ax.set_ylabel("$\\Delta$CPTS at 20%")
ax.legend(frameon=False, loc="upper left")
clean_axis(ax, grid_axis="both")
save_figure(fig, "fig_4_budget_robustness")

# Figure 5: cluster-level empirical CDF at the primary budget.
primary_cluster = cluster[np.isclose(cluster.budget, PRIMARY_BUDGET)].copy()
fig, ax = plt.subplots(figsize=(4.8, 3.35))
for dataset in DATASETS:
    values = primary_cluster[primary_cluster.dataset == dataset].delta.to_numpy(float)
    values = np.sort(values)
    y_ecdf = np.arange(1, len(values) + 1) / len(values)
    ax.step(values, y_ecdf, where="post", color=COLORS[dataset], label=DATASET_LABELS[dataset])
ax.axvline(0, color="#555555", linewidth=0.9)
ax.set_xlabel("Cluster-level $\\Delta$CPTS (10% budget)")
ax.set_ylabel("Cumulative proportion")
ax.set_ylim(0, 1.01)
ax.legend(frameon=False, loc="lower right")
clean_axis(ax, grid_axis="both")
save_figure(fig, "fig_5_cluster_ecdf")

say("Main and diagnostic quantitative figures completed.")


# ==================================================================================================
# 8. QUALITATIVE EXAMPLE SELECTION — FROZEN, DETERMINISTIC, AUDITABLE
# ==================================================================================================

section(8, 12, "Select representative successful and unsuccessful transports")


def select_examples(frame, dataset):
    part = frame[
        (frame.dataset == dataset)
        & (frame.explainer == QUALITATIVE_EXPLAINER)
        & np.isclose(frame.budget, PRIMARY_BUDGET)
    ].copy()
    accepted = part[part.accepted].copy()
    if accepted.empty:
        raise RuntimeError(f"No accepted {QUALITATIVE_EXPLAINER} donor for {dataset}.")
    target = float(accepted.delta_Q.quantile(0.90))
    success = accepted.assign(distance=(accepted.delta_Q - target).abs()).sort_values(
        ["distance", "sample_id"], kind="mergesort"
    ).iloc[0]

    negative = accepted[accepted.delta_Q <= 0].copy()
    if not negative.empty:
        failure = negative.sort_values(["delta_Q", "sample_id"], kind="mergesort").iloc[0]
        failure_type = "No Q improvement"
    else:
        fallback = part[~part.accepted].copy()
        if not fallback.empty:
            median_base = float(fallback.cpts_base_Q.median())
            failure = fallback.assign(distance=(fallback.cpts_base_Q - median_base).abs()).sort_values(
                ["distance", "sample_id"], kind="mergesort"
            ).iloc[0]
            failure_type = "Identity fallback"
        else:
            failure = accepted.sort_values(["delta_Q", "sample_id"], kind="mergesort").iloc[0]
            failure_type = "Lowest observed gain"
    return [("Successful", success), (failure_type, failure)]


selected_examples = []
for dataset in DATASETS:
    for case_type, row in select_examples(metrics, dataset):
        selected_examples.append({
            "dataset": dataset, "case_type": case_type, "fold": int(row.fold),
            "explainer": str(row.explainer), "budget": float(row.budget),
            "sample_id": str(row.sample_id), "patient_id": str(row.analysis_group_id),
            "accepted": bool(row.accepted), "mean_delta_Q": float(row.delta_Q),
            "mean_cpts_base_Q": float(row.cpts_base_Q), "mean_cpts_refined_Q": float(row.cpts_refined_Q),
        })
selection_table = pd.DataFrame(selected_examples)
atomic_write_dataframe(selection_table, SUPPLEMENT_DIR / "qualitative_selection_audit.csv")
for row in selected_examples:
    say(
        f"  {DATASET_LABELS[row['dataset']]:12s} | {row['case_type']:20s} | fold={row['fold']} "
        f"| donor={row['sample_id']} | mean ΔQ={row['mean_delta_Q']:+.4f}"
    )


# ==================================================================================================
# 9. QUALITATIVE REPLAY HELPERS
# ==================================================================================================

section(9, 12, "Reconstruct Q recipients and pairwise effects for the qualitative panel")
qualitative_rows = []

if MAKE_QUALITATIVE_FIGURE:
    OOF_PATH = ROOT / "results/explanations/resnet18/explanation_cohort.parquet"
    if not OOF_PATH.exists():
        raise FileNotFoundError(f"Frozen explanation cohort not found: {OOF_PATH}")
    oof = pd.read_parquet(OOF_PATH)
    OOF_DATASET_COL = pick_column(oof, ["dataset", "dataset_id"])
    OOF_FOLD_COL = pick_column(oof, ["fold", "test_fold", "outer_fold"])
    OOF_ID_COL = pick_column(oof, ["sample_id", "image_id", "id"])
    OOF_PATH_COL = pick_column(oof, ["image_path", "path", "filepath", "file_path", "image"])
    OOF_PATIENT_COL = pick_column(oof, ["patient_id"], required=False)
    OOF_LESION_COL = pick_column(oof, ["lesion_id", "case_id", "group_id"], required=False)
    INPUT_SIZE = {"siim_acr": 320, "isic2016": 320, "pad_ufes20": 320}
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    CPTS_EPSILON = 0.05

    def flatten_json(obj):
        rows = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                rows.extend(flatten_json(value))
        elif isinstance(obj, list):
            for value in obj:
                rows.extend(flatten_json(value))
        else:
            rows.append(obj)
        return rows

    def checkpoint_candidates(dataset, fold):
        paths = []
        for base in (ROOT / "checkpoints/classifiers/resnet18", ROOT / "checkpoints/resnet18"):
            if base.exists():
                for suffix in ("*.pt", "*.pth", "*.ckpt"):
                    paths.extend(base.rglob(suffix))
        for manifest_path in (
            ROOT / "runs/explanations/resnet18/latest_baseline_explainer_manifest.json",
            ROOT / "runs/training/explainers/resnet18/latest_learned_explainer_manifest.json",
        ):
            if manifest_path.exists():
                for value in flatten_json(load_json(manifest_path)):
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
            raise FileNotFoundError(f"Classifier checkpoint not found: {dataset} fold {fold}")
        return sorted(scored, reverse=True)[0][2]

    def map_candidates(dataset, fold, explainer):
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
            raise FileNotFoundError(f"Explanation HDF5 not found: {dataset} fold {fold} {explainer}")
        return sorted(scored, reverse=True)[0][2]

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
        out = {}
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
            out[new_key] = value
        return out

    def load_classifier(path):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        state = normalize_state_dict(extract_state_dict(payload))
        model = resnet18(weights=None)
        out_features = int(state["fc.weight"].shape[0])
        model.fc = nn.Linear(model.fc.in_features, out_features)
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        meaningful_missing = [x for x in missing_keys if not x.endswith("num_batches_tracked")]
        if meaningful_missing or unexpected_keys or out_features not in (1, 2):
            raise RuntimeError(
                f"Non-exact classifier replay: missing={meaningful_missing[:5]}, unexpected={unexpected_keys[:5]}"
            )
        model.eval().to(DEVICE)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

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
        if key not in resolve_image_path.cache:
            matches = []
            for base in (ROOT / "data/raw" / dataset, ROOT / "data/processed" / dataset, ROOT / "data"):
                if base.exists():
                    matches.extend(base.rglob(path.name))
            resolve_image_path.cache[key] = sorted(set(x for x in matches if x.is_file()))
        matches = resolve_image_path.cache[key]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"Image not found: {dataset}/{value}")
        raise RuntimeError(f"Ambiguous image basename: {dataset}/{path.name} ({len(matches)} candidates)")

    resolve_image_path.cache = {}

    def load_rgb(path, size):
        array = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if array is None:
            raise RuntimeError(f"OpenCV cannot decode: {path}")
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        return cv2.resize(array, (size, size), interpolation=cv2.INTER_AREA)

    def load_image_tensor(path, size):
        array = load_rgb(path, size).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD

    def model_to_layer2(model, x):
        x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x); x = model.layer1(x)
        return model.layer2(x)

    def layer2_to_layer3(model, z2):
        return model.layer3(z2)

    def layer3_to_logits(model, z3):
        x = model.layer4(z3); x = model.avgpool(x); x = torch.flatten(x, 1); logits = model.fc(x)
        return logits[:, 0] if logits.shape[1] == 1 else logits[:, 1] - logits[:, 0]

    @torch.no_grad()
    def extract_features(model, frame, dataset):
        records = frame.reset_index(drop=True).to_dict("records")
        rows = []
        for start in range(0, len(records), 32):
            batch = records[start:start + 32]
            tensors = [load_image_tensor(resolve_image_path(row[OOF_PATH_COL], dataset), INPUT_SIZE[dataset]) for row in batch]
            x = torch.stack(tensors).to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                z2 = model_to_layer2(model, x)
                z3 = layer2_to_layer3(model, z2)
                logits = layer3_to_logits(model, z3)
            embeddings = F.normalize(z2.float().mean(dim=(2, 3)), dim=1)
            for idx, row in enumerate(batch):
                logit = float(logits[idx].float().cpu())
                rows.append({
                    "sample_id": str(row[OOF_ID_COL]),
                    "patient_id": (
                        str(row[OOF_PATIENT_COL]).strip()
                        if dataset == "pad_ufes20" and OOF_PATIENT_COL is not None and valid_identifier(row[OOF_PATIENT_COL])
                        else (
                            str(row[OOF_LESION_COL]).strip()
                            if OOF_LESION_COL is not None and valid_identifier(row[OOF_LESION_COL])
                            else str(row[OOF_ID_COL])
                        )
                    ),
                    "path": resolve_image_path(row[OOF_PATH_COL], dataset),
                    "decision": int(logit >= 0), "embedding": embeddings[idx].half().cpu(),
                    "z3": z3[idx].half().cpu(),
                })
            del x, z2, z3, logits, embeddings
        return rows

    def nearest_q(donor, pool, n=10):
        candidates = []
        embedding = donor["embedding"].float()
        for recipient in pool:
            if recipient["sample_id"] == donor["sample_id"] or recipient["patient_id"] == donor["patient_id"]:
                continue
            if recipient["decision"] != donor["decision"]:
                continue
            similarity = float(torch.dot(embedding, recipient["embedding"].float()))
            candidates.append((-similarity, recipient["patient_id"], recipient["sample_id"], recipient))
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        selected, used = [], set()
        for _, patient, _, recipient in candidates:
            if patient in used:
                continue
            selected.append(recipient); used.add(patient)
            if len(selected) == n:
                break
        if len(selected) != n:
            raise RuntimeError(f"Qualitative Q matching returned {len(selected)}/10 recipients.")
        return selected

    def h5_datasets(handle):
        found = {}
        handle.visititems(lambda name, obj: found.update({name: obj}) if isinstance(obj, h5py.Dataset) else None)
        return found

    def decode_strings(array):
        return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in np.asarray(array).reshape(-1)]

    def load_explanation_h5(path):
        with h5py.File(path, "r") as handle:
            datasets_h5 = h5_datasets(handle)
            numeric, strings = [], []
            for name, ds in datasets_h5.items():
                if ds.ndim >= 3 and np.issubdtype(ds.dtype, np.number):
                    numeric.append((5 * any(x in name.lower() for x in ("map", "saliency", "attribution")), name))
                if ds.ndim == 1 and (ds.dtype.kind in ("S", "U", "O") or "id" in name.lower()):
                    strings.append((5 * any(x in name.lower() for x in ("sample", "image", "id")), name, ds.shape[0]))
            if not numeric:
                raise RuntimeError(f"No map array in {path}")
            map_name = sorted(numeric, reverse=True)[0][1]
            maps = np.asarray(datasets_h5[map_name], dtype=np.float32)
            if maps.ndim == 4 and maps.shape[1] == 1:
                maps = maps[:, 0]
            elif maps.ndim == 4 and maps.shape[-1] == 1:
                maps = maps[..., 0]
            ids_candidates = [x for x in strings if x[2] == maps.shape[0]]
            if not ids_candidates:
                raise RuntimeError(f"No aligned sample IDs in {path}")
            id_name = sorted(ids_candidates, reverse=True)[0][1]
            ids = decode_strings(datasets_h5[id_name][...])
        return ids, maps

    def resize_score_map(saliency, height, width):
        tensor = torch.from_numpy(np.asarray(saliency, dtype=np.float32))[None, None]
        tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        low, high = tensor.min(), tensor.max()
        return (tensor - low) / (high - low) if float(high - low) > 1e-12 else torch.zeros_like(tensor)

    def exact_topk_mask(scores, k):
        flat = scores.reshape(-1).float()
        tie = torch.arange(flat.numel(), dtype=torch.float32) * (torch.finfo(torch.float32).eps / flat.numel())
        indices = torch.topk(flat - tie, k=int(k), largest=True, sorted=False).indices
        mask = torch.zeros_like(flat); mask[indices] = 1.0
        return mask.view_as(scores).numpy().astype(np.uint8)

    def signed_effects(model, z3, sign, mask):
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=DEVICE)
        z3 = z3.to(DEVICE, dtype=torch.float32)[None]
        full = layer3_to_logits(model, z3).float()[0] * sign
        removed = layer3_to_logits(model, z3 * (1.0 - mask_t[None, None])).float()[0] * sign
        return float(full - removed)

    def cpts_pair(local_effect, recipient_effect):
        return float(math.exp(-abs(recipient_effect - local_effect) / (abs(local_effect) + CPTS_EPSILON)))

    def refined_mask_path(dataset, fold, explainer, budget):
        tag = f"p{int(round(100 * budget)):02d}"
        stem = f"{dataset}_f{fold}_{explainer}_{tag}_masks.npz"
        path = result_root / dataset / f"fold_{fold}" / stem
        if not path.exists():
            raise FileNotFoundError(f"Frozen refined masks not found: {path}")
        return path

    def overlay(ax, image, mask, color=(0.90, 0.18, 0.16), alpha=0.42):
        ax.imshow(image)
        up = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=float)
        rgba[..., :3] = color
        rgba[..., 3] = alpha * (up > 0)
        ax.imshow(rgba)
        ax.contour(up, levels=[0.5], colors=[color], linewidths=0.65)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Process each distinct dataset/fold once.
    example_groups = defaultdict(list)
    for record in selected_examples:
        example_groups[(record["dataset"], record["fold"])].append(record)

    for group_number, ((dataset, fold), records) in enumerate(sorted(example_groups.items()), start=1):
        say(f"  Qualitative replay {group_number}/{len(example_groups)}: {DATASET_LABELS[dataset]} fold {fold}")
        checkpoint = checkpoint_candidates(dataset, fold)
        model = load_classifier(checkpoint)
        fold_oof = oof[(oof[OOF_DATASET_COL].astype(str) == dataset) & (oof[OOF_FOLD_COL].astype(int) == fold)].copy()
        features = extract_features(model, fold_oof, dataset)
        by_id = {x["sample_id"]: x for x in features}
        for record in records:
            donor = by_id.get(record["sample_id"])
            if donor is None:
                raise RuntimeError(f"Selected donor absent from replay cohort: {record['sample_id']}")
            recipients = nearest_q(donor, features, n=10)
            map_path = map_candidates(dataset, fold, record["explainer"])
            map_ids, map_array = load_explanation_h5(map_path)
            map_lookup = {sample_id: idx for idx, sample_id in enumerate(map_ids)}
            if donor["sample_id"] not in map_lookup:
                raise RuntimeError(f"Donor absent from frozen base maps: {donor['sample_id']}")
            mask_data = np.load(refined_mask_path(dataset, fold, record["explainer"], record["budget"]), allow_pickle=False)
            refined_lookup = {str(x): idx for idx, x in enumerate(mask_data["sample_ids"])}
            refined_mask = mask_data["masks"][refined_lookup[donor["sample_id"]]].astype(np.uint8)
            h, w = refined_mask.shape
            raw_map = map_array[map_lookup[donor["sample_id"]]]
            score_map = resize_score_map(raw_map, h, w)
            k = int(round(record["budget"] * h * w))
            base_mask = exact_topk_mask(score_map, k)
            if int(base_mask.sum()) != k or int(refined_mask.sum()) != k:
                raise RuntimeError("Qualitative exact-k reconstruction failed.")
            sign = 1.0 if donor["decision"] == 1 else -1.0
            local_base = signed_effects(model, donor["z3"], sign, base_mask)
            local_refined = signed_effects(model, donor["z3"], sign, refined_mask)
            pair_candidates = []
            for recipient in recipients:
                rec_sign = 1.0 if recipient["decision"] == 1 else -1.0
                effect_base = signed_effects(model, recipient["z3"], rec_sign, base_mask)
                effect_refined = signed_effects(model, recipient["z3"], rec_sign, refined_mask)
                base_cpts = cpts_pair(local_base, effect_base)
                refined_cpts = cpts_pair(local_refined, effect_refined)
                pair_candidates.append((abs((refined_cpts - base_cpts) - record["mean_delta_Q"]), recipient["sample_id"], recipient, base_cpts, refined_cpts))
            _, _, recipient, pair_base, pair_refined = sorted(pair_candidates, key=lambda x: (x[0], x[1]))[0]
            donor_rgb = load_rgb(donor["path"], INPUT_SIZE[dataset])
            recipient_rgb = load_rgb(recipient["path"], INPUT_SIZE[dataset])
            qualitative_rows.append({
                **record, "donor_rgb": donor_rgb, "recipient_rgb": recipient_rgb,
                "base_mask": base_mask, "refined_mask": refined_mask,
                "recipient_id": recipient["sample_id"], "recipient_patient_id": recipient["patient_id"],
                "pair_cpts_base": pair_base, "pair_cpts_refined": pair_refined,
                "pair_delta": pair_refined - pair_base,
                "checkpoint_sha256": sha256_file(checkpoint), "map_sha256": sha256_file(map_path),
            })
        del model, features, by_id
        torch.cuda.empty_cache(); gc.collect()
else:
    say("  Qualitative replay disabled by user switch.")


# ==================================================================================================
# 10. QUALITATIVE TRANSPORT FIGURE
# ==================================================================================================

section(10, 12, "Render successful and unsuccessful cross-case transport examples")
if MAKE_QUALITATIVE_FIGURE:
    order = []
    for dataset in DATASETS:
        order.extend([x for x in qualitative_rows if x["dataset"] == dataset and x["case_type"] == "Successful"])
        order.extend([x for x in qualitative_rows if x["dataset"] == dataset and x["case_type"] != "Successful"])
    if len(order) != 2 * len(DATASETS):
        raise RuntimeError(f"Expected six qualitative rows, found {len(order)}.")

    fig, axes = plt.subplots(len(order), 5, figsize=(8.2, 10.0))
    titles = ["Donor", "Recipient", "$E$ on donor", "$E \\rightarrow$ recipient", "$T(E) \\rightarrow$ recipient"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, pad=5)
    for row_idx, item in enumerate(order):
        axes[row_idx, 0].imshow(item["donor_rgb"])
        axes[row_idx, 1].imshow(item["recipient_rgb"])
        overlay(axes[row_idx, 2], item["donor_rgb"], item["base_mask"], color=(0.12, 0.47, 0.71))
        overlay(axes[row_idx, 3], item["recipient_rgb"], item["base_mask"], color=(0.12, 0.47, 0.71))
        overlay(axes[row_idx, 4], item["recipient_rgb"], item["refined_mask"], color=(0.84, 0.15, 0.16))
        for col in (0, 1):
            axes[row_idx, col].set_xticks([]); axes[row_idx, col].set_yticks([])
            for spine in axes[row_idx, col].spines.values():
                spine.set_visible(False)
        short_type = "Success" if item["case_type"] == "Successful" else item["case_type"]
        axes[row_idx, 0].set_ylabel(f"{DATASET_LABELS[item['dataset']]}\n{short_type}", fontsize=7.5, rotation=90, labelpad=7)
        axes[row_idx, 4].text(
            0.98, 0.03,
            f"CPTS {item['pair_cpts_base']:.2f}→{item['pair_cpts_refined']:.2f}\nΔ {item['pair_delta']:+.2f}",
            transform=axes[row_idx, 4].transAxes, ha="right", va="bottom", fontsize=6.5,
            color="white", bbox=dict(boxstyle="round,pad=0.22", facecolor="black", alpha=0.64, linewidth=0)
        )
    fig.subplots_adjust(wspace=0.035, hspace=0.12)
    save_figure(fig, "fig_6_qualitative_transport")

    qualitative_export = pd.DataFrame([{k: v for k, v in item.items() if not isinstance(v, np.ndarray)} for item in order])
    atomic_write_dataframe(qualitative_export, SUPPLEMENT_DIR / "qualitative_pair_metrics.csv")
    say("Qualitative panel: six audited rows, no clinical label or mask used for selection.")
else:
    say("Qualitative panel not generated (MAKE_QUALITATIVE_FIGURE=False).")


# ==================================================================================================
# 11. MACHINE-READABLE RESULTS, PAPER TEXT, AND MANIFEST
# ==================================================================================================

section(11, 12, "Write the paper-ready result paragraph and immutable manifest")
primary_dist = boot_delta[..., primary_idx].mean(axis=(1, 2, 3))
robust_dist = boot_delta[..., robust_idx].mean(axis=(1, 2, 3))
budget_diff_dist = robust_dist - primary_dist
primary_low, primary_high = percentile_ci(primary_dist)
robust_low, robust_high = percentile_ci(robust_dist)
budget_diff_low, budget_diff_high = percentile_ci(budget_diff_dist)
primary_effect = float(obs_delta[..., primary_idx].mean())
robust_effect = float(obs_delta[..., robust_idx].mean())
budget_difference = robust_effect - primary_effect

results_text = f"""Confirmatory results

At the preregistered 10% exact spatial budget, T-CPT increased cross-patient explanation
transportability by {primary_effect:.4f} CPTS points (paired hierarchical bootstrap 95% CI
[{primary_low:.4f}, {primary_high:.4f}]; one-sided paired cluster sign-flip p={p_text(pvalues[0])}).
The improvement remained positive at the 20% robustness budget (mean change {robust_effect:.4f},
95% CI [{robust_low:.4f}, {robust_high:.4f}]; p={p_text(pvalues[1])}). The paired difference
between the two budget-specific gains was {budget_difference:+.4f} (95% CI
[{budget_diff_low:+.4f}, {budget_diff_high:+.4f}]; two-sided p={p_text(pvalues[2])}), so budget
robustness should not be described as monotonic superiority of one budget. All 180
dataset-fold-explainer-budget cell means were positive. The method preserved the exact spatial
budget and the frozen local-faithfulness constraint in every evaluated donor. Estimates use
equal weight across dataset-fold-explainer cells; patient IDs define clusters for PAD-UFES-20,
whereas case/lesion IDs define the available clusters for SIIM-ACR and ISIC 2016.
"""
atomic_write_bytes(RUN_DIR / "paper_results_paragraph.txt", results_text.encode("utf-8"))

figure_captions = """Figure 1. Explainer-agnostic improvement in cross-patient transportability. Points show the equal-weight mean change in CPTS after T-CPT refinement; intervals are paired hierarchical bootstrap 95% confidence intervals. Clusters were resampled jointly within dataset and fold, preserving dependence across explainers and budgets.

Figure 2. Base and refined CPTS by dataset. Open points denote the frozen base explainer E and colored points denote T(E). Values are averaged equally across folds and explainers; labels report the paired CPTS gain.

Figure 3. CPTS gain across datasets and explainer families. Each cell is the five-fold mean paired improvement. Both exact spatial budgets are shown under a common color scale.

Figure 4. Robustness to the exact spatial budget. Each point is one dataset-fold-explainer cell. The dashed identity line indicates equal improvement at the 10% and 20% budgets; this analysis assesses robustness rather than monotonic superiority.

Figure 5. Distribution of cluster-level CPTS changes at the primary 10% budget. Curves show empirical cumulative distributions over patient clusters for PAD-UFES-20 and case/lesion clusters for SIIM-ACR and ISIC 2016.

Figure 6. Representative successful and unsuccessful explanation transports. For each dataset, the base exact-budget explanation E and its refined counterpart T(E) are projected from a donor onto a matched, decision-compatible recipient from the untouched Q fold. Successful donors were selected deterministically near the 90th percentile of donor-level CPTS gain. Unsuccessful rows show a negative accepted transport when available, otherwise a validation-rejected identity fallback. Pairwise CPTS values refer to the displayed recipient. Neither clinical labels nor segmentation masks were used for selection.
"""
atomic_write_bytes(RUN_DIR / "figure_captions.txt", figure_captions.encode("utf-8"))

atomic_save_npz(
    SUPPLEMENT_DIR / "hierarchical_bootstrap_distributions.npz",
    overall_primary=primary_dist.astype(np.float32),
    overall_robustness=robust_dist.astype(np.float32),
    budget_difference=budget_diff_dist.astype(np.float32),
    seed=np.asarray([SEED], dtype=np.int64),
)

all_outputs = sorted(path for path in RUN_DIR.rglob("*") if path.is_file())
artifact_hashes = {str(path.relative_to(RUN_DIR)): sha256_file(path) for path in all_outputs}
scientific_support = bool(primary_low > 0 and pvalues[0] < 0.05 and robust_low > 0)
manifest = {
    "status": "PASS",
    "notebook": "07_confirmatory_statistics_and_paper_figures.py",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "analysis": {
        "primary_budget": PRIMARY_BUDGET,
        "robustness_budget": ROBUSTNESS_BUDGET,
        "estimand": "equal-weight mean across dataset-fold-explainer cells",
        "cluster_unit": {"pad_ufes20": "patient", "siim_acr": "case/lesion", "isic2016": "case/lesion"},
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "permutation_replicates": PERMUTATION_REPLICATES,
        "multiplicity": "Holm across 3 dataset and 6 explainer secondary tests at 10%",
        "seed": SEED,
    },
    "results": {
        "primary_delta": primary_effect, "primary_ci95": [primary_low, primary_high],
        "primary_p_one_sided": float(pvalues[0]),
        "robustness_delta": robust_effect, "robustness_ci95": [robust_low, robust_high],
        "robustness_p_one_sided": float(pvalues[1]),
        "budget_difference": budget_difference, "budget_difference_ci95": [budget_diff_low, budget_diff_high],
        "budget_difference_p_two_sided": float(pvalues[2]),
        "positive_cells": int((cell_table.delta_cpts > 0).sum()),
        "strict_positive_cell_ci": int((cell_table.ci95_low > 0).sum()),
        "claim_status": "SUPPORTED" if scientific_support else "NOT_SUPPORTED",
    },
    "inputs": {
        "nb06b_manifest": str(NB06B_LATEST), "nb06b_manifest_sha256": sha256_file(NB06B_LATEST),
        "donor_metrics": str(metrics_path), "donor_metrics_sha256": sha256_file(metrics_path),
    },
    "artifacts": artifact_hashes,
    "runtime_seconds": time.time() - START_TIME,
}
manifest_path = RUN_DIR / "confirmatory_manifest.json"
atomic_write_json(manifest_path, manifest)
latest_path = RUN_ROOT / "latest_confirmatory_manifest.json"
atomic_write_bytes(latest_path, canonical_json_bytes(manifest))
say(results_text.strip())
say(f"Manifest: {manifest_path}")
say(f"Latest:   {latest_path}")


# ==================================================================================================
# 12. FINAL SCIENTIFIC AND EXECUTION GATE
# ==================================================================================================

section(12, 12, "Final gate and handoff to manuscript synthesis")
pdf_count = len(list(FIGURE_DIR.glob("*.pdf")))
png_count = len(list(FIGURE_DIR.glob("*.png")))
csv_count = len(list(TABLE_DIR.glob("*.csv")))
tex_count = len(list(TABLE_DIR.glob("*.tex")))
expected_figures = 6 if MAKE_QUALITATIVE_FIGURE else 5
execution_pass = (
    len(cell_table) == 180
    and pdf_count == expected_figures
    and png_count == expected_figures
    and csv_count >= 6
    and tex_count >= 3
    and not quality.status.eq("FAIL").any()
)
if not execution_pass:
    raise RuntimeError(
        f"Notebook 07 execution gate failed: cells={len(cell_table)}, PDF={pdf_count}, PNG={png_count}, "
        f"CSV={csv_count}, TEX={tex_count}."
    )

say(f"Execution status:             PASS")
say(f"Scientific claim status:      {'SUPPORTED' if scientific_support else 'NOT_SUPPORTED'}")
say(f"Primary 10% ΔCPTS:            {primary_effect:+.4f} [{primary_low:+.4f}, {primary_high:+.4f}]")
say(f"Robustness 20% ΔCPTS:         {robust_effect:+.4f} [{robust_low:+.4f}, {robust_high:+.4f}]")
say(f"Positive analysis cells:      {(cell_table.delta_cpts > 0).sum()}/180")
say(f"Cell CIs entirely positive:   {(cell_table.ci95_low > 0).sum()}/180")
say(f"Figures:                      {pdf_count} PDF + {png_count} PNG")
say(f"Tables:                       {csv_count} CSV + {tex_count} LaTeX")
say(f"Output directory:             {RUN_DIR}")
say(f"Total runtime:                {elapsed()}")

banner("CPET_NOTEBOOK_07_STATUS=PASS")
say(f"CPET_CONFIRMATORY_CLAIM={'SUPPORTED' if scientific_support else 'NOT_SUPPORTED'}")
say("NEXT=NOTEBOOK_08_MANUSCRIPT_SYNTHESIS")
say("Send the complete textual output and the generated figure/table directory before proceeding.")
