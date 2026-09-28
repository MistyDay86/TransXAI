"""
CPET — NOTEBOOK 06B/08 — EXPLAINER-AGNOSTIC TRANSPORT REFINEMENT

Single-cell Colab script. Copy the whole file into one Colab cell or execute it
with `%run`. The only user-facing switches are in SECTION 0 below.

This notebook does not retrain or overwrite anything produced by Notebooks
01–06. It reads the frozen ResNet-18 checkpoints and post-hoc explanation maps,
then applies an exact-budget, trust-region refinement T_k(E).
"""

# ==================================================================================================
# 0. USER SWITCHES — SAFE PILOT DEFAULTS
# ==================================================================================================

N_FOLDS = 5                 # integer in {1,2,3,4,5}; 1 runs fold 0, 5 runs folds 0..4
EXPLICIT_FOLDS = None       # e.g. [2] or [0, 2, 4]; when not None, overrides N_FOLDS

# The pilot targets the already frozen primary estimand: exact area 10%, k=10 recipients.
# After a successful pilot, BUDGETS may be changed to [0.05, 0.10, 0.20].
BUDGETS = [0.10, 0.20]
DATASETS = ["siim_acr", "isic2016", "pad_ufes20"]
EXPLAINERS = [
    "gradcam",
    "layercam",
    "integrated_gradients",
    "lrp",
    "rise",
    "extremal_perturbation",
]

# None uses every frozen OOF donor in the selected fold. A positive integer is allowed only for
# engineering smoke tests and must not be reported as a scientific result.
MAX_DONORS_PER_DATASET = None

PROJECT_ROOT = "/content/gdrive/MyDrive/Colab Notebooks/CPET"
SEED = 20260906
NUM_WORKERS = 0

# Calibration/reference-bank sizes. Sampling is deterministic and never uses clinical labels/masks.
CALIBRATION_A_CANDIDATES = 384
CALIBRATION_B_CANDIDATES = 256
N_RECIPIENT_A = 12
N_RECIPIENT_B = 12
N_RECIPIENT_Q = 10

# T_k(E) optimization. Every evaluated candidate is binary and has exactly k active cells.
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

# Statistics are descriptive in this pilot; Notebook 07 will perform the frozen confirmatory tests.
BOOTSTRAP_REPLICATES = 2000


# ==================================================================================================
# 1. IMPORTS, DRIVE, REPRODUCIBILITY AND HARDWARE
# ==================================================================================================

import os
import re
import gc
import sys
import json
import math
import time
import shutil
import random
import hashlib
import platform
import warnings
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd

try:
    import h5py
    from PIL import Image
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from torchvision.models import resnet18
    from torchvision.transforms import functional as TF
except Exception as exc:
    raise RuntimeError(
        "Dipendenze mancanti. Eseguire prima il Notebook 01 nello stesso progetto. "
        f"Dettaglio: {type(exc).__name__}: {exc}"
    ) from exc

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


WIDTH = 118
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
    cache_key = (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
    if cache_key in sha256_file.cache:
        return sha256_file.cache[cache_key]
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    digest = h.hexdigest()
    sha256_file.cache[cache_key] = digest
    return digest


sha256_file.cache = {}


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
        raise ValueError(f"Formato tabellare non supportato: {path}")
    os.replace(tmp, path)


def stable_int(text):
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:8], 16)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if EXPLICIT_FOLDS is None:
    if not isinstance(N_FOLDS, int) or not 1 <= N_FOLDS <= 5:
        raise ValueError("N_FOLDS deve essere un intero compreso fra 1 e 5.")
    FOLDS_TO_RUN = list(range(N_FOLDS))
else:
    FOLDS_TO_RUN = sorted({int(x) for x in EXPLICIT_FOLDS})
    if not FOLDS_TO_RUN or any(x < 0 or x > 4 for x in FOLDS_TO_RUN):
        raise ValueError("EXPLICIT_FOLDS deve contenere uno o più fold in {0,1,2,3,4}.")

if any(float(p) <= 0 or float(p) >= 1 for p in BUDGETS):
    raise ValueError("Ogni budget deve essere strettamente compreso fra 0 e 1.")
if MAX_DONORS_PER_DATASET is not None and int(MAX_DONORS_PER_DATASET) < 8:
    raise ValueError("MAX_DONORS_PER_DATASET deve essere None oppure almeno 8.")

banner("CPET — NOTEBOOK 06B/08 — EXPLAINER-AGNOSTIC EXACT-BUDGET TRANSPORT REFINEMENT")
say("Obiettivo: applicare T_k(E) a explainer congelati e verificare CPTS[T_k(E)] > CPTS[E].")
say("Metodo: coordinate swap discreto, exact-k, trust region, mean+CVaR, validation fallback.")
say("Hardware richiesto: GPU. Tesla T4 sufficiente. num_workers=0.")
say(f"Fold selezionati: {FOLDS_TO_RUN} | budget: {BUDGETS} | seed algoritmica: {SEED}")
say("Nota: i fold NON sono seed indipendenti; usano i cinque classificatori OOF gia congelati.")
say("Notebook 01–06: sola lettura; nessun checkpoint, split o risultato precedente verra sovrascritto.")

section(1, 12, "Mount di Drive, versioni e GPU")
try:
    from google.colab import drive
    drive.mount("/content/gdrive")
    say("Google Drive montato/verificato.")
except ImportError:
    say("Ambiente non-Colab rilevato: uso il filesystem gia disponibile.")

ROOT = Path(PROJECT_ROOT)
if not ROOT.exists():
    raise FileNotFoundError(f"Project root non trovato: {ROOT}")

if not torch.cuda.is_available():
    raise RuntimeError("Questo notebook richiede una GPU CUDA. In Colab selezionare una runtime GPU.")
DEVICE = torch.device("cuda")
GPU_NAME = torch.cuda.get_device_name(0)
GPU_GIB = torch.cuda.get_device_properties(0).total_memory / 1024**3
seed_everything(SEED)
say(f"Python={sys.version.split()[0]} | PyTorch={torch.__version__} | torchvision={torchvision.__version__}")
say(f"GPU={GPU_NAME} | VRAM={GPU_GIB:.2f} GiB | CUDA={torch.version.cuda}")


# ==================================================================================================
# 2. READ-ONLY HANDOFF AUDIT AND PROTOCOL FREEZE
# ==================================================================================================


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_json(obj, prefix=""):
    rows = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_json(value, name))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            rows.extend(flatten_json(value, f"{prefix}[{idx}]"))
    else:
        rows.append((prefix, obj))
    return rows


def manifest_has_pass(obj):
    values = [str(v).upper() for _, v in flatten_json(obj) if isinstance(v, (str, int, float, bool))]
    return any(v == "PASS" or v.endswith("_STATUS=PASS") for v in values)


section(2, 12, "Gate crittografico e read-only dei Notebook 04, 05 e 06")
handoff_paths = {
    "NB04": ROOT / "runs/explanations/resnet18/latest_baseline_explainer_manifest.json",
    "NB05": ROOT / "runs/training/explainers/resnet18/latest_learned_explainer_manifest.json",
    "NB06": ROOT / "runs/metrics/resnet18/latest_cpet_metrics_manifest.json",
}
handoffs = {}
for label, path in handoff_paths.items():
    if not path.exists():
        raise FileNotFoundError(f"{label}: manifest latest non trovato: {path}")
    obj = load_json(path)
    if not manifest_has_pass(obj):
        raise RuntimeError(f"{label}: il manifest non contiene un gate PASS verificabile: {path}")
    handoffs[label] = obj
    say(f"{label}: PASS | {path.name} | sha256={sha256_file(path)[:16]}...")

protocol = {
    "name": "CPET explainer-agnostic exact-budget transport refinement",
    "short_name": "T-CPT",
    "implementation": "1.0.1",
    "parent_protocol": "1.1.0",
    "base_explainers": EXPLAINERS,
    "confirmatory_datasets": DATASETS,
    "primary_budget": 0.10,
    "primary_recipient_k": 10,
    "operator": "exact-cardinality projected coordinate swap with validation identity fallback",
    "objective": {
        "pair_loss": "1-exp(-abs(delta_recipient-delta_donor)/(abs(delta_donor)+epsilon))",
        "mean_weight": 1.0,
        "cvar_weight": LAMBDA_CVAR,
        "cvar_alpha": CVAR_ALPHA,
        "tv_weight": LAMBDA_TV,
    },
    "constraints": {
        "exact_k": True,
        "max_swap_fraction": MAX_SWAP_FRACTION,
        "local_effect_relative_tolerance": LOCAL_EFFECT_TOLERANCE,
        "preserve_local_effect_sign": True,
        "clinical_masks_used": False,
        "ground_truth_labels_used": False,
    },
    "selection": {
        "support_A": "outer-train patients",
        "validation_B": "outer-validation patients",
        "query_Q": "outer-test OOF patients",
        "accept_margin": VALIDATION_ACCEPT_MARGIN,
        "fallback": "identity/base explanation",
    },
    "seed": SEED,
}
CONFIG_PATH = ROOT / "configs/transport_refinement_t_cpt_v1.0.1.json"
config_payload = canonical_json_bytes(protocol)
if CONFIG_PATH.exists():
    old_payload = CONFIG_PATH.read_bytes()
    if old_payload != config_payload:
        raise RuntimeError(
            f"Config scientifica gia esistente ma differente: {CONFIG_PATH}. "
            "Non viene sovrascritta: incrementare esplicitamente la versione del protocollo."
        )
    config_state = "PRESERVED_IDENTICAL"
else:
    atomic_write_bytes(CONFIG_PATH, config_payload)
    config_state = "CREATED"
say(f"Protocollo T-CPT: {config_state} | {CONFIG_PATH} | sha256={sha256_file(CONFIG_PATH)[:16]}...")


# ==================================================================================================
# 3. ARTIFACT DISCOVERY — FAIL LOUDLY ON AMBIGUITY
# ==================================================================================================


def strings_from_json(obj):
    return [str(v) for _, v in flatten_json(obj) if isinstance(v, str)]


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
        raise KeyError(f"Nessuna colonna fra {candidates}; disponibili={list(frame.columns)}")
    return None


def choose_group_column(frame):
    """Prefer a real patient id; reject empty/placeholder columns and fall back to case/group id."""
    lookup = {normalize_colname(c): c for c in frame.columns}
    ordered = ["patientid", "groupid", "caseid", "lesionid", "patient", "group"]
    rejected = {"", "nan", "none", "null", "na", "n/a", "unknown", "missing", "<na>"}
    audits = []
    for key in ordered:
        column = lookup.get(key)
        if column is None:
            continue
        raw = frame[column]
        text = raw.astype(str).str.strip().str.lower()
        valid = raw.notna() & ~text.isin(rejected)
        coverage = float(valid.mean()) if len(frame) else 0.0
        unique = int(text[valid].nunique())
        audits.append((column, coverage, unique))
        if coverage >= 0.80 and unique >= 2:
            return column, audits
    return None, audits


def effective_group_value(row, group_col, id_col):
    value = row[group_col] if group_col is not None else row[id_col]
    text = str(value).strip()
    rejected = {"", "nan", "none", "null", "na", "n/a", "unknown", "missing", "<na>"}
    if pd.isna(value) or text.lower() in rejected:
        text = str(row[id_col]).strip()
    return text


def parquet_columns(path):
    try:
        import pyarrow.parquet as pq
        return pq.read_schema(path).names
    except Exception:
        return list(pd.read_parquet(path).columns)


def discover_oof_cohort():
    candidates = []
    for value in strings_from_json(handoffs["NB04"]):
        if value.lower().endswith(".parquet"):
            p = Path(value)
            if p.exists():
                candidates.append(p)
    search_roots = [ROOT / "data/manifests", ROOT / "results/explanations", ROOT / "runs/explanations"]
    for base in search_roots:
        if base.exists():
            candidates.extend(base.rglob("*.parquet"))
    scored = []
    for path in sorted(set(candidates)):
        try:
            cols = [normalize_colname(c) for c in parquet_columns(path)]
            score = 0
            score += 5 if any("dataset" == c for c in cols) else 0
            score += 5 if any("fold" == c for c in cols) else 0
            score += 5 if any("sampleid" == c for c in cols) else 0
            score += 3 if any("donor" in c for c in cols) else 0
            score += 2 if "oof" in path.name.lower() else 0
            score += 2 if "cohort" in path.name.lower() else 0
            if score >= 15:
                scored.append((score, path))
        except Exception:
            continue
    if not scored:
        raise FileNotFoundError("Impossibile individuare la coorte OOF del Notebook 04.")
    scored.sort(key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
    top_score = scored[0][0]
    top = [p for s, p in scored if s == top_score]
    # Prefer an explicitly named donor/cohort artifact; report the deterministic choice.
    top.sort(key=lambda p: ("cohort" in p.name.lower(), "donor" in p.name.lower(), p.stat().st_mtime), reverse=True)
    return top[0]


def discover_dataset_manifest(dataset):
    exact = ROOT / f"data/manifests/{dataset}_manifest_v1.1.0.parquet"
    if exact.exists():
        return exact
    candidates = sorted((ROOT / "data/manifests").glob(f"{dataset}*manifest*.parquet"))
    if not candidates:
        candidates = sorted((ROOT / "data/manifests").glob(f"*{dataset}*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"Manifest canonico non trovato per {dataset}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


section(3, 12, "Individuazione coorte OOF, manifest clinici, checkpoint e mappe")
OOF_COHORT_PATH = discover_oof_cohort()
oof = pd.read_parquet(OOF_COHORT_PATH)
OOF_DATASET_COL = pick_column(oof, ["dataset", "dataset_id"])
OOF_FOLD_COL = pick_column(oof, ["fold", "test_fold", "outer_fold"])
OOF_ID_COL = pick_column(oof, ["sample_id", "image_id", "id"])
OOF_PATIENT_COL, OOF_GROUP_AUDIT = choose_group_column(oof)
if OOF_PATIENT_COL is None:
    OOF_PATIENT_COL = OOF_ID_COL
OOF_PATH_COL = pick_column(oof, ["image_path", "path", "filepath", "file_path", "image"])
OOF_LOGIT_COL = pick_column(
    oof,
    ["logit_raw", "raw_logit", "logit", "decision_logit", "model_logit"],
    required=False,
)
say(f"Coorte OOF: {OOF_COHORT_PATH} | righe={len(oof):,}")
say(f"Schema: dataset={OOF_DATASET_COL}, fold={OOF_FOLD_COL}, id={OOF_ID_COL}, patient={OOF_PATIENT_COL}")
say("Group-column audit: " + ", ".join(f"{c}:coverage={v:.1%},unique={u}" for c, v, u in OOF_GROUP_AUDIT))
say(f"Replay reference logit: {OOF_LOGIT_COL if OOF_LOGIT_COL else 'non disponibile; audit strutturale'}")


def checkpoint_candidates(dataset, fold):
    bases = [ROOT / "checkpoints/classifiers/resnet18", ROOT / "checkpoints/resnet18"]
    paths = []
    for base in bases:
        if base.exists():
            for suffix in ("*.pt", "*.pth", "*.ckpt"):
                paths.extend(base.rglob(suffix))
    # Add only existing checkpoint paths explicitly recorded by NB04/NB05.
    for obj in (handoffs["NB04"], handoffs["NB05"]):
        for value in strings_from_json(obj):
            if value.lower().endswith((".pt", ".pth", ".ckpt")) and Path(value).exists():
                paths.append(Path(value))
    unique = sorted(set(paths))
    scored = []
    for path in unique:
        text = str(path).lower()
        score = 0
        score += 8 if dataset.lower() in text else 0
        score += 6 if re.search(rf"(^|[^0-9])f(?:old)?[_-]?0*{fold}([^0-9]|$)", text) else 0
        score += 4 if "classifier" in text else 0
        score -= 8 if "explainer" in text or "rational" in text or "transxai" in text else 0
        score += 1 if "best" in path.name.lower() else 0
        if score >= 14:
            scored.append((score, path))
    if not scored:
        raise FileNotFoundError(f"Checkpoint classificatore non trovato: {dataset} fold {fold}")
    scored.sort(key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
    return scored[0][1]


def map_candidates(dataset, fold, explainer):
    base = ROOT / "results/explanations/resnet18"
    if not base.exists():
        raise FileNotFoundError(f"Directory mappe NB04 non trovata: {base}")
    paths = []
    for pattern in ("*.h5", "*.hdf5"):
        paths.extend(base.rglob(pattern))
    scored = []
    for path in sorted(set(paths)):
        text = str(path).lower()
        score = 0
        score += 7 if dataset.lower() in text else 0
        score += 7 if explainer.lower() in text else 0
        score += 5 if re.search(rf"(^|[^0-9])f(?:old)?[_-]?0*{fold}([^0-9]|$)", text) else 0
        if score >= 19:
            scored.append((score, path))
    if not scored:
        raise FileNotFoundError(f"Mappa HDF5 non trovata: {dataset} f{fold} {explainer}")
    scored.sort(key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
    return scored[0][1]


artifact_registry = {}
for dataset in DATASETS:
    manifest_path = discover_dataset_manifest(dataset)
    say(f"{dataset:12s} manifest={manifest_path.name}")
    for fold in FOLDS_TO_RUN:
        ckpt = checkpoint_candidates(dataset, fold)
        maps = {method: map_candidates(dataset, fold, method) for method in EXPLAINERS}
        artifact_registry[(dataset, fold)] = {"manifest": manifest_path, "checkpoint": ckpt, "maps": maps}
        say(f"  fold {fold}: checkpoint={ckpt.name} | mappe={len(maps)}/{len(EXPLAINERS)}")


# ==================================================================================================
# 4. MODEL, IMAGE PIPELINE, FEATURE REPLAY
# ==================================================================================================


INPUT_SIZE = {"bus_uclm": 256, "siim_acr": 320, "isic2016": 320, "pad_ufes20": 320}
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
    raise RuntimeError("Checkpoint privo di uno state_dict riconoscibile.")


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
    if "fc.weight" not in state:
        raise RuntimeError(f"{path.name}: fc.weight assente dopo normalizzazione delle chiavi.")
    out_features = int(state["fc.weight"].shape[0])
    model.fc = nn.Linear(model.fc.in_features, out_features)
    missing, unexpected = model.load_state_dict(state, strict=False)
    meaningful_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if meaningful_missing or unexpected:
        raise RuntimeError(
            f"Caricamento checkpoint non esatto: missing={meaningful_missing[:8]}, unexpected={unexpected[:8]}"
        )
    if out_features not in (1, 2):
        raise RuntimeError(f"Output classificatore inatteso: {out_features}")
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
    # Deterministic basename fallback, cached per dataset; ambiguity is an error.
    basename = path.name
    cache_key = (dataset, basename)
    if cache_key not in resolve_image_path.cache:
        roots = [ROOT / "data/raw" / dataset, ROOT / "data/processed" / dataset, ROOT / "data"]
        matches = []
        for base in roots:
            if base.exists():
                matches.extend(base.rglob(basename))
        matches = sorted(set(p for p in matches if p.is_file()))
        resolve_image_path.cache[cache_key] = matches
    matches = resolve_image_path.cache[cache_key]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Immagine non trovata per {dataset}: {value}")
    raise RuntimeError(f"Path ambiguo per {dataset}/{basename}: {len(matches)} candidati.")


resolve_image_path.cache = {}


def load_image_tensor(path, size, mode="opencv_area_square"):
    if mode == "opencv_area_square":
        array = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if array is None:
            raise RuntimeError(f"OpenCV non decodifica: {path}")
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        array = cv2.resize(array, (size, size), interpolation=cv2.INTER_AREA)
        array = array.astype(np.float32) / 255.0
    else:
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
                raise ValueError(f"Modalita preprocessing ignota: {mode}")
            array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


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
    if logits.shape[1] == 1:
        return logits[:, 0]
    return logits[:, 1] - logits[:, 0]


@torch.no_grad()
def extract_features(model, frame, dataset, id_col, patient_col, path_col, desc, preprocess_mode):
    size = INPUT_SIZE[dataset]
    rows = []
    batch_size = 32
    records = frame.reset_index(drop=True).to_dict("records")
    for start in tqdm(range(0, len(records), batch_size), desc=desc, leave=False):
        batch = records[start:start + batch_size]
        tensors = []
        for row in batch:
            path = resolve_image_path(row[path_col], dataset)
            tensors.append(load_image_tensor(path, size, preprocess_mode))
        x = torch.stack(tensors).to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            z2 = model_to_layer2(model, x)
            z3 = layer2_to_layer3(model, z2)
            logits = layer3_to_logits(model, z3)
        embeddings = F.normalize(z2.float().mean(dim=(2, 3)), dim=1)
        for idx, row in enumerate(batch):
            logit = float(logits[idx].float().cpu())
            rows.append({
                "sample_id": str(row[id_col]),
                "patient_id": effective_group_value(row, patient_col, id_col),
                "logit": logit,
                "decision": int(logit >= 0),
                "embedding": embeddings[idx].half().cpu(),
                "z3": z3[idx].half().cpu(),
            })
        del x, z2, z3, logits, embeddings
    return rows


@torch.no_grad()
def choose_preprocessing(model, frame, dataset):
    """Select geometry/decoder by replaying frozen raw OOF logits when available."""
    modes = ("opencv_area_square", "pil_bilinear_square", "pil_center_crop")
    if OOF_LOGIT_COL is None:
        say("  Preprocessing replay: raw logit OOF non disponibile; uso OpenCV AREA square (protocol default).")
        return modes[0]
    audit = frame.dropna(subset=[OOF_LOGIT_COL]).head(8)
    if audit.empty:
        say("  Preprocessing replay: logit OOF vuoti; uso OpenCV AREA square (protocol default).")
        return modes[0]
    expected = audit[OOF_LOGIT_COL].astype(float).to_numpy()
    results = []
    for mode in modes:
        tensors = [
            load_image_tensor(resolve_image_path(row[OOF_PATH_COL], dataset), INPUT_SIZE[dataset], mode)
            for _, row in audit.iterrows()
        ]
        x = torch.stack(tensors).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            z2 = model_to_layer2(model, x)
            logits = layer3_to_logits(model, layer2_to_layer3(model, z2)).float().cpu().numpy()
        # A two-logit checkpoint is already converted to the binary margin by layer3_to_logits.
        error = np.abs(logits - expected)
        results.append((float(np.mean(error)), float(np.max(error)), mode))
        del x, z2
    results.sort()
    mean_error, max_error, selected = results[0]
    say(
        "  Preprocessing replay: "
        + " | ".join(f"{mode}=mean{mean:.2e}/max{maximum:.2e}" for mean, maximum, mode in results)
    )
    if max_error > 5e-2:
        selected = "opencv_area_square"
        say(
            "  WARNING: nessuna variante replica i logit entro 5e-2; il campo potrebbe essere calibrato. "
            "Non uso il ranking dell'errore calibrato per cambiare geometria. "
            f"Scelta protocollare: {selected}."
        )
    else:
        say(f"  Preprocessing selezionato: {selected} | replay max={max_error:.3e}")
    return selected


section(4, 12, "Definizione replay ResNet-18 e preprocessing identico ai classificatori congelati")
say("Replay: input -> layer2 per matching; layer3 per intervento; layer4+fc frozen per gli effetti.")
say("Clinical mask e ground-truth label non vengono lette dal metodo.")


# ==================================================================================================
# 5. HDF5 MAP READER WITH SCHEMA AUDIT
# ==================================================================================================


def h5_datasets(handle):
    found = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            found[name] = obj
    handle.visititems(visitor)
    return found


def decode_strings(array):
    out = []
    for item in np.asarray(array).reshape(-1):
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def load_explanation_h5(path):
    with h5py.File(path, "r") as handle:
        datasets = h5_datasets(handle)
        numeric = []
        strings = []
        for name, ds in datasets.items():
            if ds.ndim >= 3 and np.issubdtype(ds.dtype, np.number):
                score = 5 if any(token in name.lower() for token in ("map", "saliency", "attribution")) else 0
                score += 2 if ds.shape[-1] > 1 and ds.shape[-2] > 1 else 0
                numeric.append((score, name, ds.shape))
            if ds.ndim == 1:
                id_like = any(token in name.lower() for token in ("sample", "image", "id"))
                if ds.dtype.kind in ("S", "U", "O") or id_like:
                    score = 5 if id_like else 0
                    strings.append((score, name, ds.shape))
        if not numeric:
            raise RuntimeError(f"{path.name}: nessun dataset numerico 3D/4D per le mappe.")
        numeric.sort(reverse=True)
        map_name = numeric[0][1]
        maps = np.asarray(datasets[map_name], dtype=np.float32)
        if maps.ndim == 4 and maps.shape[1] == 1:
            maps = maps[:, 0]
        elif maps.ndim == 4 and maps.shape[-1] == 1:
            maps = maps[..., 0]
        if maps.ndim != 3:
            raise RuntimeError(f"{path.name}: shape mappe non supportata {maps.shape}")
        matching_strings = [x for x in strings if x[2][0] == maps.shape[0]]
        if not matching_strings:
            raise RuntimeError(f"{path.name}: sample_id non individuabili per {maps.shape[0]} mappe.")
        matching_strings.sort(reverse=True)
        id_name = matching_strings[0][1]
        ids = decode_strings(datasets[id_name][...])
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{path.name}: sample_id duplicati nell'HDF5.")
    if not np.isfinite(maps).all():
        raise RuntimeError(f"{path.name}: mappe non finite.")
    return ids, maps, map_name, id_name


def resize_map(saliency, height, width):
    tensor = torch.from_numpy(np.asarray(saliency, dtype=np.float32))[None, None]
    tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    lo = tensor.min()
    hi = tensor.max()
    if float(hi - lo) > 1e-12:
        tensor = (tensor - lo) / (hi - lo)
    else:
        tensor = torch.zeros_like(tensor)
    return tensor


section(5, 12, "Audit reader HDF5 e comparabilita delle mappe")
first_key = next(iter(artifact_registry))
first_path = artifact_registry[first_key]["maps"][EXPLAINERS[0]]
ids_smoke, maps_smoke, map_ds, id_ds = load_explanation_h5(first_path)
say(f"HDF5 smoke: {first_path.name} | maps={maps_smoke.shape} ({map_ds}) | ids={len(ids_smoke)} ({id_ds})")
del ids_smoke, maps_smoke


# ==================================================================================================
# 6. CPTS LOSS, EXACT-K FEASIBLE SET AND DISCRETE OPTIMIZER
# ==================================================================================================


def exact_topk_mask(scores, k):
    flat = scores.reshape(-1)
    k = int(max(1, min(k, flat.numel())))
    # Stable infinitesimal tie-break: lower flat index wins, deterministic across devices.
    tie = torch.arange(flat.numel(), device=flat.device, dtype=torch.float32)
    adjusted = flat.float() - tie * (torch.finfo(torch.float32).eps / max(1, flat.numel()))
    indices = torch.topk(adjusted, k=k, largest=True, sorted=False).indices
    mask = torch.zeros_like(flat, dtype=torch.float32)
    mask[indices] = 1.0
    return mask.view_as(scores)


def total_variation(mask):
    horizontal = torch.abs(mask[:, 1:] - mask[:, :-1]).mean() if mask.shape[1] > 1 else mask.new_tensor(0.0)
    vertical = torch.abs(mask[1:, :] - mask[:-1, :]).mean() if mask.shape[0] > 1 else mask.new_tensor(0.0)
    return horizontal + vertical


def signed_effects(model, z3, signs, masks, full_signed_logits=None):
    """Removal effect after zeroing selected layer3 positions. Shapes: z3 B,C,H,W; masks B/H/W or H/W."""
    if masks.ndim == 2:
        masks = masks.unsqueeze(0).expand(z3.shape[0], -1, -1)
    if masks.shape[0] != z3.shape[0]:
        raise ValueError("Batch incompatibile fra feature e maschere.")
    masked_z3 = z3 * (1.0 - masks[:, None, :, :])
    removed_signed = layer3_to_logits(model, masked_z3).float() * signs.float()
    if full_signed_logits is None:
        full_signed_logits = layer3_to_logits(model, z3).float() * signs.float()
    return full_signed_logits - removed_signed


def cpts_values(local_effect, recipient_effects):
    scale = torch.abs(local_effect) + float(CPTS_EPSILON)
    relative_gap = torch.abs(recipient_effects - local_effect) / scale
    return torch.exp(-relative_gap)


def cvar(losses, alpha=CVAR_ALPHA):
    losses = losses.reshape(-1)
    count = max(1, int(math.ceil((1.0 - float(alpha)) * losses.numel())))
    return torch.topk(losses, k=count, largest=True).values.mean()


def evaluate_mask(model, donor_z, donor_sign, recipient_z, recipient_sign, mask, with_grad=False):
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        donor_full = layer3_to_logits(model, donor_z[None]).float() * donor_sign
        local = signed_effects(model, donor_z[None], donor_sign[None], mask, donor_full)[0]
        if recipient_z.shape[0] == 0:
            empty = mask.new_empty((0,))
            return local, empty, empty, mask.new_tensor(float("nan"))
        recipient_full = layer3_to_logits(model, recipient_z).float() * recipient_sign.float()
        transported = signed_effects(model, recipient_z, recipient_sign, mask, recipient_full)
        cpts = cpts_values(local, transported)
        losses = 1.0 - cpts
        risk = losses.mean() + LAMBDA_CVAR * cvar(losses) + LAMBDA_TV * total_variation(mask)
    return local, transported, cpts, risk


def constraint_ok(local_value, base_local_value):
    base = float(base_local_value)
    value = float(local_value)
    if not np.isfinite(value):
        return False
    if abs(base) < MIN_BASE_EFFECT:
        return abs(value) >= abs(base) - 1e-8
    same_sign = np.sign(value) == np.sign(base)
    enough_magnitude = abs(value) + 1e-8 >= (1.0 - LOCAL_EFFECT_TOLERANCE) * abs(base)
    return bool(same_sign and enough_magnitude)


def swap_distance(mask, base_mask):
    return int(torch.count_nonzero(mask.reshape(-1) != base_mask.reshape(-1)).item() // 2)


def candidate_swaps(current, gradient, base, max_swaps):
    flat_m = current.reshape(-1)
    flat_g = gradient.reshape(-1)
    selected = torch.nonzero(flat_m > 0.5, as_tuple=False).flatten()
    excluded = torch.nonzero(flat_m <= 0.5, as_tuple=False).flatten()
    remove_order = selected[torch.argsort(flat_g[selected], descending=True)]
    add_order = excluded[torch.argsort(flat_g[excluded], descending=False)]
    candidates = []
    seen = set()
    for batch in SWAP_BATCHES:
        for offset in range(GRADIENT_ALTERNATIVES):
            n = min(int(batch), len(remove_order) - offset, len(add_order) - offset)
            if n <= 0:
                continue
            candidate = flat_m.clone()
            remove = remove_order[offset:offset + n]
            add = add_order[offset:offset + n]
            candidate[remove] = 0.0
            candidate[add] = 1.0
            candidate = candidate.view_as(current)
            if int(candidate.sum().item()) != int(current.sum().item()):
                continue
            if swap_distance(candidate, base) > max_swaps:
                continue
            key = hashlib.sha1(candidate.detach().cpu().numpy().astype(np.uint8).tobytes()).hexdigest()
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def optimize_one(model, donor, rec_a, rec_b, rec_q, base_mask):
    donor_z = donor["z3"].to(DEVICE, dtype=torch.float32)
    donor_sign = torch.tensor(1.0 if donor["decision"] == 1 else -1.0, device=DEVICE)

    def stack_recipients(items):
        if not items:
            h, w = donor_z.shape[-2:]
            return (
                torch.empty((0, donor_z.shape[0], h, w), device=DEVICE),
                torch.empty((0,), device=DEVICE),
            )
        z = torch.stack([r["z3"] for r in items]).to(DEVICE, dtype=torch.float32)
        signs = torch.tensor([1.0 if r["decision"] == 1 else -1.0 for r in items], device=DEVICE)
        return z, signs

    z_a, s_a = stack_recipients(rec_a)
    z_b, s_b = stack_recipients(rec_b)
    z_q, s_q = stack_recipients(rec_q)
    base = base_mask.to(DEVICE, dtype=torch.float32)
    k = int(base.sum().item())
    max_swaps = max(1, int(round(MAX_SWAP_FRACTION * k)))

    with torch.no_grad():
        base_local, _, base_cpts_a, base_risk_a = evaluate_mask(model, donor_z, donor_sign, z_a, s_a, base)
        _, _, base_cpts_b, base_risk_b = evaluate_mask(model, donor_z, donor_sign, z_b, s_b, base)
        _, _, base_cpts_q, _ = evaluate_mask(model, donor_z, donor_sign, z_q, s_q, base)

    current = base.clone()
    current_risk = float(base_risk_a)
    trajectory = [(base.clone(), current_risk, 0)]
    accepted_steps = 0

    for iteration in range(1, MAX_ITERATIONS + 1):
        variable = current.detach().clone().requires_grad_(True)
        local, _, _, risk = evaluate_mask(model, donor_z, donor_sign, z_a, s_a, variable, with_grad=True)
        if not torch.isfinite(risk):
            break
        gradient = torch.autograd.grad(risk, variable, retain_graph=False, create_graph=False)[0]
        if not torch.isfinite(gradient).all():
            break
        proposals = candidate_swaps(current, gradient.detach(), base, max_swaps)
        if not proposals:
            break
        best = None
        with torch.no_grad():
            for proposal in proposals:
                local_p, _, _, risk_p = evaluate_mask(model, donor_z, donor_sign, z_a, s_a, proposal)
                value = float(risk_p)
                if not constraint_ok(float(local_p), float(base_local)):
                    continue
                if value < current_risk - MIN_SUPPORT_IMPROVEMENT and (best is None or value < best[0]):
                    best = (value, proposal.clone())
        if best is None:
            break
        current_risk, current = best
        accepted_steps += 1
        trajectory.append((current.clone(), current_risk, iteration))

    # B is never used in the coordinate search. It selects one point from the finite A trajectory.
    selected_mask = base
    selected_b_risk = float(base_risk_b)
    selected_iteration = 0
    selected_b_cpts = float(base_cpts_b.mean()) if base_cpts_b.numel() else float("nan")
    with torch.no_grad():
        for mask, _, iteration in trajectory[1:]:
            local_b, _, cpts_b, risk_b = evaluate_mask(model, donor_z, donor_sign, z_b, s_b, mask)
            if not constraint_ok(float(local_b), float(base_local)):
                continue
            mean_b = float(cpts_b.mean()) if cpts_b.numel() else float("nan")
            if (
                np.isfinite(mean_b)
                and mean_b >= float(base_cpts_b.mean()) + VALIDATION_ACCEPT_MARGIN
                and float(risk_b) < selected_b_risk
            ):
                selected_mask = mask.clone()
                selected_b_risk = float(risk_b)
                selected_b_cpts = mean_b
                selected_iteration = iteration

        final_local, _, final_cpts_a, _ = evaluate_mask(model, donor_z, donor_sign, z_a, s_a, selected_mask)
        _, _, final_cpts_b, _ = evaluate_mask(model, donor_z, donor_sign, z_b, s_b, selected_mask)
        _, _, final_cpts_q, _ = evaluate_mask(model, donor_z, donor_sign, z_q, s_q, selected_mask)

    def safe_mean(tensor):
        return float(tensor.mean()) if tensor.numel() else float("nan")

    result = {
        "accepted": bool(selected_iteration > 0),
        "selected_iteration": int(selected_iteration),
        "search_steps_accepted": int(accepted_steps),
        "swaps_from_base": int(swap_distance(selected_mask, base)),
        "k": k,
        "max_swaps": max_swaps,
        "local_effect_base": float(base_local),
        "local_effect_refined": float(final_local),
        "cpts_base_A": safe_mean(base_cpts_a),
        "cpts_refined_A": safe_mean(final_cpts_a),
        "cpts_base_B": safe_mean(base_cpts_b),
        "cpts_refined_B": safe_mean(final_cpts_b),
        "cpts_base_Q": safe_mean(base_cpts_q),
        "cpts_refined_Q": safe_mean(final_cpts_q),
        "n_A": int(len(rec_a)),
        "n_B": int(len(rec_b)),
        "n_Q": int(len(rec_q)),
    }
    result["delta_A"] = result["cpts_refined_A"] - result["cpts_base_A"]
    result["delta_B"] = result["cpts_refined_B"] - result["cpts_base_B"]
    result["delta_Q"] = result["cpts_refined_Q"] - result["cpts_base_Q"]
    output_mask = selected_mask.detach().cpu().numpy().astype(np.uint8)
    del donor_z, z_a, z_b, z_q, s_a, s_b, s_q, base, current, trajectory
    return output_mask, result


section(6, 12, "Definizione T_k(E): objective mean+CVaR, exact-k, trust region e identity fallback")
say("Ottimizzazione: solo recipient A. Selezione: solo recipient B. Valutazione: solo recipient Q.")
say("Forward e valutazione usano sempre mask binarie exact-k: nessun soft/hard surrogate gap.")


# ==================================================================================================
# 7. PATIENT-DISJOINT REFERENCE BANKS AND MATCHING
# ==================================================================================================


def deterministic_sample(frame, n, salt):
    if len(frame) <= n:
        return frame.copy().reset_index(drop=True)
    rng = np.random.default_rng(SEED + stable_int(salt))
    indices = np.sort(rng.choice(len(frame), size=n, replace=False))
    return frame.iloc[indices].reset_index(drop=True)


def nearest_compatible(donor, pool, n, exclude_patient=True):
    candidates = []
    d_emb = donor["embedding"].float()
    for recipient in pool:
        if recipient["sample_id"] == donor["sample_id"]:
            continue
        if exclude_patient and recipient["patient_id"] == donor["patient_id"]:
            continue
        if recipient["decision"] != donor["decision"]:
            continue
        similarity = float(torch.dot(d_emb, recipient["embedding"].float()))
        candidates.append((-similarity, recipient["patient_id"], recipient["sample_id"], recipient))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    selected = []
    used_patients = set()
    for _, patient_id, _, recipient in candidates:
        if patient_id in used_patients:
            continue
        selected.append(recipient)
        used_patients.add(patient_id)
        if len(selected) >= n:
            break
    return selected


def standardize_frame(frame, dataset, is_oof=False):
    if is_oof:
        patient_col, _ = choose_group_column(frame)
        if patient_col is None:
            patient_col = OOF_ID_COL
        return frame.copy(), OOF_ID_COL, patient_col, OOF_PATH_COL, OOF_FOLD_COL
    id_col = pick_column(frame, ["sample_id", "image_id", "id"])
    patient_col, _ = choose_group_column(frame)
    if patient_col is None:
        patient_col = id_col
    path_col = pick_column(frame, ["image_path", "path", "filepath", "file_path", "image"])
    fold_col = pick_column(frame, ["fold", "test_fold", "outer_fold"])
    return frame.copy(), id_col, patient_col, path_col, fold_col


section(7, 12, "Costruzione patient-disjoint delle coorti A/B/Q")
say("Per il fold f: Q=fold f; B=fold (f+1) mod 5; A=gli altri tre fold.")
say("Matching: stessa decisione del classificatore, paziente diverso, nearest layer2 embedding.")


# ==================================================================================================
# 8. RESUME-AWARE EXECUTION
# ==================================================================================================


RESULT_ROOT = ROOT / "results/transport_refinement/resnet18/t_cpt_v1.0.1"
RUN_ROOT = ROOT / "runs/transport_refinement/resnet18"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ROOT.mkdir(parents=True, exist_ok=True)


def budget_tag(p):
    return f"p{int(round(100 * float(p))):02d}"


def job_paths(dataset, fold, explainer, budget):
    folder = RESULT_ROOT / dataset / f"fold_{fold}"
    stem = f"{dataset}_f{fold}_{explainer}_{budget_tag(budget)}"
    return {
        "folder": folder,
        "masks": folder / f"{stem}_masks.npz",
        "rows": folder / f"{stem}_donor_metrics.parquet",
        "meta": folder / f"{stem}_job.json",
    }


def job_identity(dataset, fold, explainer, budget, checkpoint, map_path, donor_ids):
    return {
        "implementation": protocol["implementation"],
        "dataset": dataset,
        "fold": int(fold),
        "explainer": explainer,
        "budget": float(budget),
        "seed": SEED,
        "checkpoint_sha256": sha256_file(checkpoint),
        "base_map_sha256": sha256_file(map_path),
        "donor_ids_sha256": hashlib.sha256("\n".join(donor_ids).encode("utf-8")).hexdigest(),
        "algorithm_config_sha256": sha256_file(CONFIG_PATH),
    }


def valid_completed_job(paths, identity):
    if not all(paths[key].exists() for key in ("masks", "rows", "meta")):
        return False
    try:
        meta = load_json(paths["meta"])
        if meta.get("status") != "PASS" or meta.get("identity") != identity:
            return False
        if meta.get("sha256", {}).get("masks") != sha256_file(paths["masks"]):
            return False
        if meta.get("sha256", {}).get("rows") != sha256_file(paths["rows"]):
            return False
        data = np.load(paths["masks"], allow_pickle=False)
        rows = pd.read_parquet(paths["rows"])
        return data["masks"].shape[0] == len(rows) == len(data["sample_ids"])
    except Exception:
        return False


def save_npz_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


section(8, 12, "Esecuzione T_k(E) con resume per dataset × fold × explainer × budget")
all_metric_frames = []
jobs_created = 0
jobs_resumed = 0
job_errors = []
runtime_samples = []

for dataset_index, dataset in enumerate(DATASETS, start=1):
    dataset_oof = oof[oof[OOF_DATASET_COL].astype(str) == dataset].copy()
    if dataset_oof.empty:
        raise RuntimeError(f"La coorte OOF non contiene il dataset {dataset}.")
    dataset_manifest = pd.read_parquet(artifact_registry[(dataset, FOLDS_TO_RUN[0])]["manifest"])
    manifest_frame, m_id, m_patient, m_path, m_fold = standardize_frame(dataset_manifest, dataset)

    banner(f"DATASET {dataset_index}/{len(DATASETS)} — {dataset}")
    for fold in FOLDS_TO_RUN:
        fold_start = time.time()
        fold_oof = dataset_oof[dataset_oof[OOF_FOLD_COL].astype(int) == fold].copy().reset_index(drop=True)
        if MAX_DONORS_PER_DATASET is not None:
            fold_oof = deterministic_sample(fold_oof, int(MAX_DONORS_PER_DATASET), f"donor/{dataset}/{fold}")
        if fold_oof.empty:
            raise RuntimeError(f"Nessun donor OOF per {dataset} fold {fold}.")
        _, _, q_patient_col, _, _ = standardize_frame(fold_oof, dataset, is_oof=True)

        val_fold = (fold + 1) % 5
        pool_a_frame = manifest_frame[
            (~manifest_frame[m_fold].astype(int).isin([fold, val_fold]))
        ].copy()
        pool_b_frame = manifest_frame[manifest_frame[m_fold].astype(int) == val_fold].copy()
        pool_a_frame = deterministic_sample(pool_a_frame, CALIBRATION_A_CANDIDATES, f"A/{dataset}/{fold}")
        pool_b_frame = deterministic_sample(pool_b_frame, CALIBRATION_B_CANDIDATES, f"B/{dataset}/{fold}")

        checkpoint = artifact_registry[(dataset, fold)]["checkpoint"]
        model = load_classifier(checkpoint)
        preprocess_mode = choose_preprocessing(model, fold_oof, dataset)
        say(
            f"\n{dataset} f{fold}: donor-Q={len(fold_oof)} | candidati-A={len(pool_a_frame)} "
            f"| candidati-B={len(pool_b_frame)} | checkpoint={checkpoint.name}"
        )
        say(f"  Leakage groups: Q={q_patient_col} | A/B={m_patient}")
        donor_features = extract_features(
            model, fold_oof, dataset, OOF_ID_COL, q_patient_col, OOF_PATH_COL,
            desc=f"{dataset} f{fold} donor-Q", preprocess_mode=preprocess_mode
        )
        pool_a = extract_features(
            model, pool_a_frame, dataset, m_id, m_patient, m_path,
            desc=f"{dataset} f{fold} pool-A", preprocess_mode=preprocess_mode
        )
        pool_b = extract_features(
            model, pool_b_frame, dataset, m_id, m_patient, m_path,
            desc=f"{dataset} f{fold} pool-B", preprocess_mode=preprocess_mode
        )
        donor_by_id = {row["sample_id"]: row for row in donor_features}

        # Query recipients come only from the untouched outer fold and are independent of A/B selection.
        matching = {}
        insufficient = defaultdict(int)
        for donor in donor_features:
            rec_a = nearest_compatible(donor, pool_a, N_RECIPIENT_A)
            rec_b = nearest_compatible(donor, pool_b, N_RECIPIENT_B)
            rec_q = nearest_compatible(donor, donor_features, N_RECIPIENT_Q)
            matching[donor["sample_id"]] = (rec_a, rec_b, rec_q)
            if len(rec_a) < N_RECIPIENT_A:
                insufficient["A"] += 1
            if len(rec_b) < N_RECIPIENT_B:
                insufficient["B"] += 1
            if len(rec_q) < N_RECIPIENT_Q:
                insufficient["Q"] += 1
        say(
            f"  Matching pronto | donor insufficient A/B/Q="
            f"{insufficient['A']}/{insufficient['B']}/{insufficient['Q']}"
        )
        counts_a = np.asarray([len(value[0]) for value in matching.values()], dtype=int)
        counts_b = np.asarray([len(value[1]) for value in matching.values()], dtype=int)
        counts_q = np.asarray([len(value[2]) for value in matching.values()], dtype=int)
        say(
            "  Recipient count min/median/max | "
            f"A={counts_a.min()}/{np.median(counts_a):.0f}/{counts_a.max()} | "
            f"B={counts_b.min()}/{np.median(counts_b):.0f}/{counts_b.max()} | "
            f"Q={counts_q.min()}/{np.median(counts_q):.0f}/{counts_q.max()}"
        )
        if (
            counts_a.min() < N_RECIPIENT_A
            or counts_b.min() < N_RECIPIENT_B
            or counts_q.min() < N_RECIPIENT_Q
        ):
            raise RuntimeError(
                f"{dataset} f{fold}: gate matching fallito. Nessun job viene prodotto: "
                f"richiesti A/B/Q={N_RECIPIENT_A}/{N_RECIPIENT_B}/{N_RECIPIENT_Q}, "
                f"minimi={counts_a.min()}/{counts_b.min()}/{counts_q.min()}."
            )

        for method_index, explainer in enumerate(EXPLAINERS, start=1):
            map_path = artifact_registry[(dataset, fold)]["maps"][explainer]
            map_ids, base_maps, _, _ = load_explanation_h5(map_path)
            map_lookup = {sample_id: idx for idx, sample_id in enumerate(map_ids)}
            missing_ids = [sample_id for sample_id in donor_by_id if sample_id not in map_lookup]
            if missing_ids:
                raise RuntimeError(
                    f"{dataset} f{fold} {explainer}: {len(missing_ids)} donor assenti dalle mappe, "
                    f"esempio={missing_ids[:3]}"
                )

            for budget in BUDGETS:
                paths = job_paths(dataset, fold, explainer, budget)
                donor_ids = [row["sample_id"] for row in donor_features]
                identity = job_identity(dataset, fold, explainer, budget, checkpoint, map_path, donor_ids)
                if valid_completed_job(paths, identity):
                    frame = pd.read_parquet(paths["rows"])
                    all_metric_frames.append(frame)
                    jobs_resumed += 1
                    say(
                        f"  [{method_index:02d}/{len(EXPLAINERS):02d}] {explainer:24s} "
                        f"{budget_tag(budget)} | RESUMED | n={len(frame)} | ΔQ={frame.delta_Q.mean():+.4f}"
                    )
                    continue

                job_start = time.time()
                job_rows = []
                refined_masks = []
                for donor_idx, donor in enumerate(donor_features):
                    sample_id = donor["sample_id"]
                    raw_map = base_maps[map_lookup[sample_id]]
                    h, w = tuple(donor["z3"].shape[-2:])
                    score_map = resize_map(raw_map, h, w)
                    k = max(1, int(round(float(budget) * h * w)))
                    base_mask = exact_topk_mask(score_map, k)
                    rec_a, rec_b, rec_q = matching[sample_id]
                    output_mask, metrics = optimize_one(model, donor, rec_a, rec_b, rec_q, base_mask)
                    metrics.update({
                        "dataset": dataset,
                        "fold": int(fold),
                        "explainer": explainer,
                        "budget": float(budget),
                        "sample_id": sample_id,
                        "patient_id": donor["patient_id"],
                        "decision": int(donor["decision"]),
                    })
                    job_rows.append(metrics)
                    refined_masks.append(output_mask)
                    if (donor_idx + 1) % 16 == 0 or donor_idx + 1 == len(donor_features):
                        partial = pd.DataFrame(job_rows)
                        say(
                            f"      {dataset} f{fold} {explainer} {budget_tag(budget)} "
                            f"{donor_idx + 1:3d}/{len(donor_features)} | "
                            f"accept={partial.accepted.mean():.1%} | ΔA={partial.delta_A.mean():+.4f} "
                            f"ΔB={partial.delta_B.mean():+.4f} | ΔQ={partial.delta_Q.mean():+.4f}"
                        )

                frame = pd.DataFrame(job_rows)
                finite_columns = [
                    "cpts_base_A", "cpts_refined_A", "cpts_base_B", "cpts_refined_B",
                    "cpts_base_Q", "cpts_refined_Q", "delta_A", "delta_B", "delta_Q",
                    "local_effect_base", "local_effect_refined",
                ]
                if not np.isfinite(frame[finite_columns].to_numpy(dtype=float)).all():
                    bad = int((~np.isfinite(frame[finite_columns].to_numpy(dtype=float))).sum())
                    raise RuntimeError(
                        f"{dataset} f{fold} {explainer}: {bad} valori non finiti. "
                        "Il job non viene salvato e non puo superare il gate."
                    )
                masks_array = np.stack(refined_masks).astype(np.uint8)
                observed_k = masks_array.reshape(len(masks_array), -1).sum(axis=1)
                expected_k_array = frame["k"].to_numpy(dtype=int)
                if not np.array_equal(observed_k.astype(int), expected_k_array):
                    bad = int(np.count_nonzero(observed_k.astype(int) != expected_k_array))
                    raise RuntimeError(f"{dataset} f{fold} {explainer}: exact-k violato per {bad} donor.")
                paths["folder"].mkdir(parents=True, exist_ok=True)
                save_npz_atomic(
                    paths["masks"],
                    masks=masks_array,
                    sample_ids=np.asarray(donor_ids, dtype="U"),
                    budget=np.asarray([float(budget)], dtype=np.float32),
                )
                atomic_write_dataframe(frame, paths["rows"])
                meta = {
                    "status": "PASS",
                    "identity": identity,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "n_donors": len(frame),
                    "accepted_fraction": float(frame.accepted.mean()),
                    "mean_delta_Q": float(frame.delta_Q.mean()),
                    "runtime_seconds": time.time() - job_start,
                    "sha256": {
                        "masks": sha256_file(paths["masks"]),
                        "rows": sha256_file(paths["rows"]),
                    },
                }
                atomic_write_json(paths["meta"], meta)
                all_metric_frames.append(frame)
                jobs_created += 1
                job_seconds = time.time() - job_start
                runtime_samples.append(job_seconds)
                say(
                    f"  [{method_index:02d}/{len(EXPLAINERS):02d}] {explainer:24s} {budget_tag(budget)} "
                    f"| CREATED | accept={frame.accepted.mean():.1%} | ΔQ={frame.delta_Q.mean():+.4f} "
                    f"| {elapsed(job_seconds)}"
                )
                if len(runtime_samples) == 1:
                    remaining = (
                        len(DATASETS) * len(FOLDS_TO_RUN) * len(EXPLAINERS) * len(BUDGETS)
                        - jobs_resumed - jobs_created
                    )
                    say(f"      Prima stima ETA grezza: <= {elapsed(np.mean(runtime_samples) * max(0, remaining))}")

        say(f"{dataset} f{fold} completato in {elapsed(time.time() - fold_start)}")
        del model, donor_features, pool_a, pool_b, donor_by_id, matching, base_maps
        torch.cuda.empty_cache()
        gc.collect()


# ==================================================================================================
# 9. PAIRED PATIENT-CLUSTER BOOTSTRAP
# ==================================================================================================


def patient_cluster_bootstrap(frame, value_col="delta_Q", reps=BOOTSTRAP_REPLICATES, seed=SEED):
    valid = frame[["patient_id", value_col]].dropna().copy()
    if valid.empty:
        return float("nan"), float("nan"), float("nan"), 0
    patient_means = valid.groupby("patient_id", sort=True)[value_col].mean().to_numpy(dtype=float)
    estimate = float(patient_means.mean())
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=float)
    for idx in range(reps):
        draw = rng.integers(0, len(patient_means), size=len(patient_means))
        values[idx] = patient_means[draw].mean()
    low, high = np.quantile(values, [0.025, 0.975])
    return estimate, float(low), float(high), len(patient_means)


section(9, 12, "Bootstrap paired clusterizzato per paziente sul query set Q")
if not all_metric_frames:
    raise RuntimeError("Nessuna metrica prodotta o ripresa.")
metrics_all = pd.concat(all_metric_frames, ignore_index=True)
summary_rows = []
group_cols = ["dataset", "fold", "explainer", "budget"]
for keys, frame in metrics_all.groupby(group_cols, sort=True):
    estimate, low, high, n_patients = patient_cluster_bootstrap(
        frame, seed=SEED + stable_int("/".join(map(str, keys)))
    )
    summary_rows.append({
        "dataset": keys[0],
        "fold": int(keys[1]),
        "explainer": keys[2],
        "budget": float(keys[3]),
        "n_donors": len(frame),
        "n_patients": n_patients,
        "accepted_fraction": float(frame.accepted.mean()),
        "mean_cpts_base_Q": float(frame.cpts_base_Q.mean()),
        "mean_cpts_refined_Q": float(frame.cpts_refined_Q.mean()),
        "mean_delta_Q": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "fraction_improved_Q": float((frame.delta_Q > 0).mean()),
        "mean_swaps": float(frame.swaps_from_base.mean()),
        "local_effect_ratio": float(
            (frame.local_effect_refined.abs() / frame.local_effect_base.abs().clip(lower=MIN_BASE_EFFECT)).mean()
        ),
    })
summary = pd.DataFrame(summary_rows)

for _, row in summary.iterrows():
    signal = "POSITIVE" if row.ci95_low > 0 else ("NEGATIVE" if row.ci95_high < 0 else "UNCERTAIN")
    say(
        f"{row.dataset:12s} f{int(row.fold)} {row.explainer:24s} p={row.budget:.2f} | "
        f"CPTS {row.mean_cpts_base_Q:.4f}->{row.mean_cpts_refined_Q:.4f} | "
        f"Δ={row.mean_delta_Q:+.4f} [{row.ci95_low:+.4f},{row.ci95_high:+.4f}] | "
        f"accept={row.accepted_fraction:.1%} | {signal}"
    )


# ==================================================================================================
# 10. POOLED PILOT SIGNAL AND SCIENTIFIC GUARDS
# ==================================================================================================


section(10, 12, "Riepilogo del segnale del claim senza riscrittura post-hoc delle soglie")
pooled_est, pooled_low, pooled_high, pooled_patients = patient_cluster_bootstrap(
    metrics_all.assign(patient_id=metrics_all.dataset.astype(str) + "/" + metrics_all.patient_id.astype(str)),
    seed=SEED + 991,
)
positive_cells = int((summary.mean_delta_Q > 0).sum())
strict_positive_cells = int((summary.ci95_low > 0).sum())
total_cells = len(summary)
faithfulness_violations = int((
    metrics_all.local_effect_refined.abs() + 1e-8
    < (1.0 - LOCAL_EFFECT_TOLERANCE) * metrics_all.local_effect_base.abs()
).sum())
exact_k_violations = 0

if pooled_low > 0:
    claim_signal = "POSITIVE_PILOT"
elif pooled_high < 0:
    claim_signal = "NEGATIVE_PILOT"
else:
    claim_signal = "INCONCLUSIVE_PILOT"

say(f"Pooled donor/patient paired ΔCPTS-Q={pooled_est:+.4f} | CI95% [{pooled_low:+.4f},{pooled_high:+.4f}]")
say(f"Celle con ΔQ>0: {positive_cells}/{total_cells} | CI interamente >0: {strict_positive_cells}/{total_cells}")
say(f"Identity fallback: {(~metrics_all.accepted.astype(bool)).mean():.1%} dei donor")
say(f"Violazioni faithfulness: {faithfulness_violations} | violazioni exact-k: {exact_k_violations}")
say(f"Segnale scientifico pilot: {claim_signal}")
say("Un pilot su un fold misura il segnale e il funzionamento; non costituisce ancora evidenza confirmatoria.")


# ==================================================================================================
# 11. SAVE TABLES, MANIFEST AND ARTIFACT CHAIN
# ==================================================================================================


section(11, 12, "Salvataggio atomico di risultati, summary e manifest")
stamp = utc_stamp()
run_dir = RUN_ROOT / stamp
run_dir.mkdir(parents=True, exist_ok=False)
metrics_path = run_dir / "t_cpt_donor_metrics.parquet"
summary_path = run_dir / "t_cpt_summary.csv"
run_config_path = run_dir / "run_selection.json"
manifest_path = run_dir / "transport_refinement_manifest.json"
latest_path = RUN_ROOT / "latest_transport_refinement_manifest.json"

atomic_write_dataframe(metrics_all, metrics_path)
atomic_write_dataframe(summary, summary_path)
run_selection = {
    "folds": FOLDS_TO_RUN,
    "n_folds": len(FOLDS_TO_RUN),
    "datasets": DATASETS,
    "explainers": EXPLAINERS,
    "budgets": [float(x) for x in BUDGETS],
    "max_donors_per_dataset": MAX_DONORS_PER_DATASET,
    "is_full_confirmatory_grid": bool(
        set(FOLDS_TO_RUN) == set(range(5))
        and set(round(float(x), 2) for x in BUDGETS) == {0.05, 0.10, 0.20}
        and MAX_DONORS_PER_DATASET is None
    ),
}
atomic_write_json(run_config_path, run_selection)

artifact_hashes = {
    "protocol": sha256_file(CONFIG_PATH),
    "metrics": sha256_file(metrics_path),
    "summary": sha256_file(summary_path),
    "run_selection": sha256_file(run_config_path),
    "nb04": sha256_file(handoff_paths["NB04"]),
    "nb05": sha256_file(handoff_paths["NB05"]),
    "nb06": sha256_file(handoff_paths["NB06"]),
}
chain = hashlib.sha256("\n".join(f"{k}:{artifact_hashes[k]}" for k in sorted(artifact_hashes)).encode()).hexdigest()
manifest = {
    "status": "PASS",
    "notebook": "06B_transport_refinement_pilot.py",
    "method": "T-CPT",
    "implementation": protocol["implementation"],
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "hardware": {"gpu": GPU_NAME, "vram_gib": GPU_GIB},
    "run_selection": run_selection,
    "jobs": {
        "expected": len(DATASETS) * len(FOLDS_TO_RUN) * len(EXPLAINERS) * len(BUDGETS),
        "created": jobs_created,
        "resumed": jobs_resumed,
        "errors": job_errors,
    },
    "results": {
        "claim_signal": claim_signal,
        "pooled_delta_Q": pooled_est,
        "pooled_ci95": [pooled_low, pooled_high],
        "positive_cells": positive_cells,
        "strict_positive_cells": strict_positive_cells,
        "total_cells": total_cells,
        "faithfulness_violations": faithfulness_violations,
        "exact_k_violations": exact_k_violations,
    },
    "artifacts": {
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "protocol": str(CONFIG_PATH),
        "result_root": str(RESULT_ROOT),
    },
    "sha256": artifact_hashes,
    "artifact_chain_sha256": chain,
    "runtime_seconds": time.time() - START_TIME,
}
atomic_write_json(manifest_path, manifest)
atomic_write_bytes(latest_path, canonical_json_bytes(manifest))
say(f"Metriche donor: {metrics_path}")
say(f"Summary:         {summary_path}")
say(f"Manifest:        {manifest_path}")
say(f"Latest:          {latest_path}")
say(f"Artifact chain:  {chain}")


# ==================================================================================================
# 12. FINAL GATE
# ==================================================================================================


section(12, 12, "Gate esecutivo e istruzioni")
expected_jobs = len(DATASETS) * len(FOLDS_TO_RUN) * len(EXPLAINERS) * len(BUDGETS)
completed_jobs = jobs_created + jobs_resumed
execution_pass = (
    completed_jobs == expected_jobs
    and not job_errors
    and faithfulness_violations == 0
    and exact_k_violations == 0
)
if not execution_pass:
    raise RuntimeError(
        f"Gate esecutivo fallito: completed={completed_jobs}/{expected_jobs}, errors={len(job_errors)}, "
        f"faithfulness={faithfulness_violations}, exact_k={exact_k_violations}"
    )

say(f"Stato esecuzione:          PASS")
say(f"Fold completati:           {len(FOLDS_TO_RUN)}/5 -> {FOLDS_TO_RUN}")
say(f"Job completi:              {completed_jobs}/{expected_jobs}")
say(f"Creati / ripresi:          {jobs_created} / {jobs_resumed}")
say(f"Donor-metric rows:         {len(metrics_all):,}")
say(f"Segnale pilot:             {claim_signal}")
say(f"Tempo totale:              {elapsed()}")
if len(FOLDS_TO_RUN) == 1:
    say("Prossima decisione: inviare l'intero output. NON impostare ancora N_FOLDS=5 prima dell'analisi del pilot.")
else:
    say("Prossima decisione: integrare T-CPT nel Notebook 07 per l'inferenza gerarchica confirmatoria.")

banner("CPET_NOTEBOOK_06B_STATUS=PASS")
say(f"CPET_T_CPT_PILOT_SIGNAL={claim_signal}")
say("Inviare in chat l'intero output testuale di questa cella prima di procedere.")
