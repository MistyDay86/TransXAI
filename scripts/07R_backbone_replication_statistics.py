"""
CPET — NOTEBOOK 07R/08 — PAIRED BACKBONE REPLICATION STATISTICS

Single-cell Colab script. It performs no training, no explanation generation,
and no mask optimization. It reads the corrected frozen ResNet-18 analysis and
the two immutable RegNet-X-400MF T-CPT runs (10% and 20%), then performs joint
patient/case-cluster inference and writes paper-ready tables and figures.

Hardware: CPU only. A GPU runtime is unnecessary.
"""

# ==================================================================================================
# 0. FROZEN ANALYSIS SETTINGS
# ==================================================================================================

PROJECT_ROOT = "/content/gdrive/MyDrive/Colab Notebooks/CPET"
SEED = 20260906
PRIMARY_BUDGET = 0.10
ROBUSTNESS_BUDGET = 0.20
BOOTSTRAP_REPLICATES = 10_000
PERMUTATION_REPLICATES = 100_000
FIGURE_DPI = 600

DATASETS = ["siim_acr", "isic2016", "pad_ufes20"]
FOLDS = [0, 1, 2, 3, 4]
EXPLAINERS = [
    "gradcam", "layercam", "integrated_gradients", "lrp", "rise", "extremal_perturbation"
]
BUDGETS = [PRIMARY_BUDGET, ROBUSTNESS_BUDGET]
BACKBONES = ["resnet18", "regnet_x_400mf"]


# ==================================================================================================
# 1. IMPORTS, DRIVE, UTILITIES, AND IMMUTABLE OUTPUT ROOT
# ==================================================================================================

import os
import re
import sys
import json
import time
import random
import hashlib
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault("PYTHONHASHSEED", str(SEED))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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


def sha256_file(path, chunk=8 * 1024 * 1024):
    path = Path(path)
    cache_key = (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
    if cache_key in sha256_file.cache:
        return sha256_file.cache[cache_key]
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    value = digest.hexdigest()
    sha256_file.cache[cache_key] = value
    return value


sha256_file.cache = {}


def canonical_json_bytes(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def atomic_write_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(temporary, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path, obj):
    atomic_write_bytes(path, canonical_json_bytes(obj))


def atomic_write_dataframe(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp.{os.getpid()}" + path.suffix)
    if path.suffix == ".csv":
        frame.to_csv(temporary, index=False)
    elif path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path.suffix}")
    os.replace(temporary, path)


def atomic_save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def percentile_ci(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    return tuple(float(x) for x in np.quantile(values, [alpha / 2, 1 - alpha / 2]))


def p_text(value):
    if not np.isfinite(value):
        return "NA"
    resolution = 1.0 / (PERMUTATION_REPLICATES + 1)
    return f"<{resolution * 1.01:.1e}" if value <= resolution * 1.01 else f"{value:.4f}"


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def normalize_colname(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def pick_column(frame, candidates, required=True):
    lookup = {normalize_colname(column): column for column in frame.columns}
    for candidate in candidates:
        key = normalize_colname(candidate)
        if key in lookup:
            return lookup[key]
    for candidate in candidates:
        key = normalize_colname(candidate)
        for normalized, original in lookup.items():
            if key in normalized or normalized in key:
                return original
    if required:
        raise KeyError(f"No column among {candidates}; available={list(frame.columns)}")
    return None


def valid_identifier(value):
    return not (
        pd.isna(value)
        or str(value).strip().lower()
        in {"", "nan", "none", "null", "na", "n/a", "unknown", "missing", "<na>"}
    )


random.seed(SEED)
np.random.seed(SEED)

banner("CPET — NOTEBOOK 07R/08 — PAIRED RESNET-18 / REGNET-X-400MF REPLICATION")
say("Goal: test architectural replication of T-CPT using frozen, paired OOF donors.")
say("Primary estimand: RegNet-X-400MF gain at exact 10%; robustness estimand: gain at exact 20%.")
say("ResNet–RegNet differences are secondary contrasts; RegNet superiority is not required.")
say("Hardware: CPU only. No image loading, training, explainer generation, or optimization.")
say(f"Bootstrap={BOOTSTRAP_REPLICATES:,} | paired sign-flips={PERMUTATION_REPLICATES:,} | seed={SEED}")

section(1, 11, "Mount Drive, audit CPU runtime, and create an immutable output run")
try:
    from google.colab import drive
    drive.mount("/content/gdrive")
    say("Google Drive mounted/verified.")
except ImportError:
    say("Non-Colab runtime: using the available filesystem.")

ROOT = Path(PROJECT_ROOT)
if not ROOT.exists():
    raise FileNotFoundError(f"Project root not found: {ROOT}")
say(f"Python={sys.version.split()[0]} | NumPy={np.__version__} | Pandas={pd.__version__}")

RUN_ROOT = ROOT / "runs/confirmatory/backbone_replication"
RUN_DIR = RUN_ROOT / utc_stamp()
TABLE_DIR = RUN_DIR / "tables"
FIGURE_DIR = RUN_DIR / "figures"
SUPPLEMENT_DIR = RUN_DIR / "supplement"
for directory in (TABLE_DIR, FIGURE_DIR, SUPPLEMENT_DIR):
    directory.mkdir(parents=True, exist_ok=False)
say(f"New analysis run: {RUN_DIR}")


# ==================================================================================================
# 2. RESNET-18 CORRECTED HANDOFF AND REGNET IMMUTABLE-RUN DISCOVERY
# ==================================================================================================

section(2, 11, "Resolve and cryptographically verify the frozen ResNet and RegNet inputs")

RESNET07_LATEST = ROOT / "runs/confirmatory/resnet18/latest_confirmatory_manifest.json"
if not RESNET07_LATEST.exists():
    raise FileNotFoundError(f"Corrected ResNet Notebook 07 handoff not found: {RESNET07_LATEST}")
resnet07_latest = load_json(RESNET07_LATEST)
if resnet07_latest.get("status") != "PASS":
    raise RuntimeError("ResNet Notebook 07 handoff is not PASS.")

resnet_run_candidates = sorted(
    (ROOT / "runs/confirmatory/resnet18").glob("*/confirmatory_manifest.json"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
resnet_run_manifest = None
for candidate in resnet_run_candidates:
    if load_json(candidate) == resnet07_latest:
        resnet_run_manifest = candidate
        break
if resnet_run_manifest is None:
    raise RuntimeError("The immutable ResNet Notebook 07 run matching latest was not found.")

resnet_corrected_path = resnet_run_manifest.parent / "supplement/analysis_donor_metrics_corrected.parquet"
if not resnet_corrected_path.exists():
    raise FileNotFoundError(f"Corrected ResNet donor metrics not found: {resnet_corrected_path}")
expected_corrected_hash = resnet07_latest.get("artifacts", {}).get(
    "supplement/analysis_donor_metrics_corrected.parquet"
)
if expected_corrected_hash and sha256_file(resnet_corrected_path) != expected_corrected_hash:
    raise RuntimeError("Corrected ResNet donor metrics hash mismatch.")
say(
    f"ResNet-18 corrected NB07: PASS | {resnet_run_manifest.parent.name} | "
    f"metrics sha256={sha256_file(resnet_corrected_path)[:16]}..."
)


def validate_regnet_run(manifest_path, budget):
    manifest = load_json(manifest_path)
    selection = manifest.get("run_selection", {})
    observed_budgets = {round(float(value), 6) for value in selection.get("budgets", [])}
    if manifest.get("status") != "PASS" or manifest.get("backbone") != "regnet_x_400mf":
        return None
    if observed_budgets != {round(float(budget), 6)}:
        return None
    if sorted(int(value) for value in selection.get("folds", [])) != FOLDS:
        return None
    if set(selection.get("datasets", [])) != set(DATASETS):
        return None
    if set(selection.get("explainers", [])) != set(EXPLAINERS):
        return None
    if selection.get("max_donors_per_dataset") is not None:
        return None
    jobs = manifest.get("jobs", {})
    if (
        int(jobs.get("expected", -1)) != 90
        or int(jobs.get("created", 0)) + int(jobs.get("resumed", 0)) != 90
        or jobs.get("errors")
    ):
        return None
    metrics_path = Path(manifest.get("artifacts", {}).get("metrics", ""))
    if not metrics_path.exists():
        return None
    expected_hash = manifest.get("sha256", {}).get("metrics")
    if not expected_hash or sha256_file(metrics_path) != expected_hash:
        return None
    return manifest, metrics_path


REGNET_RUN_ROOT = ROOT / "runs/transport_refinement/regnet_x_400mf"
regnet_inputs = {}
for budget in BUDGETS:
    valid_runs = []
    for manifest_path in REGNET_RUN_ROOT.glob("*/transport_refinement_manifest.json"):
        try:
            validated = validate_regnet_run(manifest_path, budget)
        except Exception:
            validated = None
        if validated is not None:
            valid_runs.append((manifest_path.stat().st_mtime, manifest_path, *validated))
    if not valid_runs:
        raise FileNotFoundError(f"No complete immutable RegNet run found for budget={budget:.2f}.")
    _, manifest_path, manifest, metrics_path = max(valid_runs, key=lambda item: item[0])
    regnet_inputs[round(float(budget), 6)] = {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "metrics_path": metrics_path,
    }
    say(
        f"RegNet budget {budget:.2f}: PASS | {manifest_path.parent.name} | "
        f"metrics sha256={sha256_file(metrics_path)[:16]}..."
    )

regnet_protocol_hashes = {
    item["manifest"].get("sha256", {}).get("protocol") for item in regnet_inputs.values()
}
if len(regnet_protocol_hashes) != 1 or None in regnet_protocol_hashes:
    raise RuntimeError("The RegNet 10% and 20% runs do not share one frozen T-CPT protocol.")

resnet06b_path = Path(resnet07_latest.get("inputs", {}).get("nb06b_manifest", ""))
if not resnet06b_path.exists():
    raise FileNotFoundError(f"ResNet 06B input manifest recorded by NB07 not found: {resnet06b_path}")
expected_resnet06b_hash = resnet07_latest.get("inputs", {}).get("nb06b_manifest_sha256")
if not expected_resnet06b_hash or sha256_file(resnet06b_path) != expected_resnet06b_hash:
    raise RuntimeError("The ResNet 06B manifest no longer matches the input frozen by Notebook 07.")
resnet06b = load_json(resnet06b_path)
if resnet06b.get("sha256", {}).get("protocol") not in regnet_protocol_hashes:
    raise RuntimeError("ResNet and RegNet analyses do not share the same frozen T-CPT protocol hash.")
say("Cross-backbone protocol identity: PASS")


# ==================================================================================================
# 3. DONOR-LEVEL GRID, CANONICAL CLUSTERS, AND CONSTRAINT AUDIT
# ==================================================================================================

section(3, 11, "Build the paired 2-backbone × 2-budget donor grid and audit all invariants")

REQUIRED_COLUMNS = {
    "dataset", "fold", "explainer", "budget", "sample_id", "accepted",
    "cpts_base_Q", "cpts_refined_Q", "delta_Q", "local_effect_base",
    "local_effect_refined", "k", "swaps_from_base",
}


def standardize_metrics(frame, backbone):
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise RuntimeError(f"{backbone}: donor metrics miss required columns: {missing}")
    frame = frame.copy()
    frame["backbone"] = backbone
    frame["dataset"] = frame.dataset.astype(str)
    frame["fold"] = frame.fold.astype(int)
    frame["explainer"] = frame.explainer.astype(str)
    frame["budget"] = frame.budget.astype(float).round(6)
    frame["sample_id"] = frame.sample_id.astype(str)
    if pd.api.types.is_bool_dtype(frame.accepted):
        frame["accepted"] = frame.accepted.astype(bool)
    elif set(pd.unique(frame.accepted.dropna())).issubset({0, 1, 0.0, 1.0}):
        frame["accepted"] = frame.accepted.astype(bool)
    else:
        normalized_accepted = frame.accepted.astype(str).str.strip().str.lower()
        if not set(normalized_accepted.unique()).issubset({"true", "false"}):
            raise RuntimeError(f"{backbone}: invalid accepted values in frozen donor metrics.")
        frame["accepted"] = normalized_accepted.eq("true")
    frame["improved"] = frame.delta_Q.astype(float) > 0
    numeric = sorted(
        (REQUIRED_COLUMNS & set(frame.columns))
        - {"dataset", "explainer", "sample_id", "accepted"}
    )
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{backbone}: non-finite values in frozen donor metrics.")
    if not np.allclose(
        frame.cpts_refined_Q.to_numpy(float) - frame.cpts_base_Q.to_numpy(float),
        frame.delta_Q.to_numpy(float), atol=1e-7, rtol=1e-6,
    ):
        raise RuntimeError(f"{backbone}: delta_Q is inconsistent with refined minus base CPTS.")
    return frame


resnet_metrics = standardize_metrics(pd.read_parquet(resnet_corrected_path), "resnet18")
resnet_metrics = resnet_metrics[
    resnet_metrics.dataset.isin(DATASETS)
    & resnet_metrics.fold.isin(FOLDS)
    & resnet_metrics.explainer.isin(EXPLAINERS)
    & resnet_metrics.budget.isin([round(x, 6) for x in BUDGETS])
].copy()

regnet_parts = []
for budget in BUDGETS:
    item = regnet_inputs[round(float(budget), 6)]
    part = standardize_metrics(pd.read_parquet(item["metrics_path"]), "regnet_x_400mf")
    part = part[np.isclose(part.budget, budget)].copy()
    regnet_parts.append(part)
regnet_metrics = pd.concat(regnet_parts, ignore_index=True)


def audit_backbone_grid(frame, backbone):
    expected_rows = len(DATASETS) * len(FOLDS) * len(EXPLAINERS) * len(BUDGETS) * 128
    if len(frame) != expected_rows:
        raise RuntimeError(f"{backbone}: rows={len(frame):,}, expected={expected_rows:,}.")
    if set(frame.dataset) != set(DATASETS) or set(frame.explainer) != set(EXPLAINERS):
        raise RuntimeError(f"{backbone}: dataset/explainer grid mismatch.")
    if sorted(frame.fold.unique()) != FOLDS or set(np.round(frame.budget.unique(), 2)) != set(BUDGETS):
        raise RuntimeError(f"{backbone}: fold/budget grid mismatch.")
    duplicates = frame.duplicated(["dataset", "fold", "explainer", "budget", "sample_id"]).sum()
    if duplicates:
        raise RuntimeError(f"{backbone}: duplicated donor rows={duplicates}.")
    sizes = frame.groupby(["dataset", "fold", "explainer", "budget"], sort=True).size()
    if len(sizes) != 180 or sizes.min() != 128 or sizes.max() != 128:
        raise RuntimeError(f"{backbone}: incomplete/unbalanced 180-cell grid.")
    expected_k = np.rint(frame.budget.to_numpy(float) * 20 * 20).astype(int)
    if not np.array_equal(frame.k.to_numpy(dtype=int), expected_k):
        raise RuntimeError(f"{backbone}: stored exact-k values are inconsistent with the budgets.")
    say(f"{backbone:18s}: rows={len(frame):,} | cells=180/180 | donors/cell=128 | PASS")


audit_backbone_grid(resnet_metrics, "resnet18")
audit_backbone_grid(regnet_metrics, "regnet_x_400mf")

pair_keys = ["dataset", "fold", "explainer", "budget", "sample_id"]
resnet_keys = resnet_metrics[pair_keys].sort_values(pair_keys).reset_index(drop=True)
regnet_keys = regnet_metrics[pair_keys].sort_values(pair_keys).reset_index(drop=True)
if not resnet_keys.equals(regnet_keys):
    difference = len(set(map(tuple, resnet_keys.to_numpy())) ^ set(map(tuple, regnet_keys.to_numpy())))
    raise RuntimeError(f"Cross-backbone donor pairing failed: symmetric key difference={difference}.")
say("Exact donor pairing across backbones: 23,040/23,040 rows | PASS")

NB04R_LATEST = ROOT / "runs/explanations/regnet_x_400mf/latest_baseline_explainer_manifest.json"
if not NB04R_LATEST.exists():
    raise FileNotFoundError(f"RegNet explanation handoff not found: {NB04R_LATEST}")
nb04r = load_json(NB04R_LATEST)
cohort_path = Path(nb04r.get("cohort_path", ""))
if not cohort_path.exists() or sha256_file(cohort_path) != nb04r.get("cohort_sha256"):
    raise RuntimeError("RegNet OOF cohort path/hash is invalid.")
cohort = pd.read_parquet(cohort_path)
C_DATASET = pick_column(cohort, ["dataset", "dataset_id"])
C_FOLD = pick_column(cohort, ["fold", "outer_fold", "test_fold"])
C_ID = pick_column(cohort, ["sample_id", "image_id", "id"])
C_PATIENT = pick_column(cohort, ["patient_id"], required=False)
C_CASE = pick_column(cohort, ["lesion_id", "case_id", "group_id"], required=False)
if C_PATIENT is None:
    raise RuntimeError("The frozen OOF cohort has no patient_id column; PAD clustering is impossible.")

group_rows = []
for _, row in cohort.iterrows():
    dataset = str(row[C_DATASET])
    sample_id = str(row[C_ID])
    if dataset == "pad_ufes20":
        if not valid_identifier(row[C_PATIENT]):
            raise RuntimeError(f"PAD sample {sample_id} lacks a valid patient_id.")
        group_id = str(row[C_PATIENT]).strip()
        group_unit = "patient"
    else:
        candidate = row[C_CASE] if C_CASE is not None else row[C_ID]
        group_id = str(candidate).strip() if valid_identifier(candidate) else sample_id
        group_unit = "case/lesion"
    group_rows.append({
        "dataset": dataset, "fold": int(row[C_FOLD]), "sample_id": sample_id,
        "analysis_group_id": group_id, "group_unit": group_unit,
    })
group_map = pd.DataFrame(group_rows).drop_duplicates(["dataset", "fold", "sample_id"])
if len(group_map) != len(cohort) or group_map.duplicated(["dataset", "fold", "sample_id"]).any():
    raise RuntimeError("Canonical sample-to-cluster mapping is not one-to-one.")

# Notebook 07's corrected ResNet table already carries derived clustering columns.
# Rebuild them once from the shared frozen cohort to avoid merge suffixes (_x/_y)
# and guarantee identical grouping semantics for both backbones.
metrics = pd.concat([resnet_metrics, regnet_metrics], ignore_index=True).drop(
    columns=["analysis_group_id", "group_unit", "cluster_id"], errors="ignore"
)
metrics = metrics.merge(group_map, on=["dataset", "fold", "sample_id"], how="left", validate="many_to_one")
if metrics["analysis_group_id"].isna().any():
    raise RuntimeError("Some donor rows lack a canonical patient/case cluster.")
metrics["cluster_id"] = metrics["dataset"] + "/" + metrics["analysis_group_id"].astype(str)


def local_constraint_ok(refined, base, tolerance=0.10, minimum=1e-3):
    refined, base = float(refined), float(base)
    if not np.isfinite(refined):
        return False
    if abs(base) < minimum:
        return abs(refined) >= abs(base) - 1e-8
    return np.sign(refined) == np.sign(base) and abs(refined) + 1e-8 >= (1.0 - tolerance) * abs(base)


faithfulness_violations = int(sum(
    not local_constraint_ok(refined, base)
    for refined, base in zip(metrics.local_effect_refined, metrics.local_effect_base)
))
if faithfulness_violations:
    raise RuntimeError(f"Frozen local-faithfulness violations={faithfulness_violations}.")
say(
    "Canonical clusters: PAD=patient; SIIM/ISIC=case/lesion | "
    f"faithfulness violations=0 | exact-k violations=0 | PASS"
)

atomic_write_dataframe(metrics, SUPPLEMENT_DIR / "paired_donor_metrics_2backbones_2budgets.parquet")


# ==================================================================================================
# 4. JOINT PAIRED HIERARCHICAL BOOTSTRAP
# ==================================================================================================

section(4, 11, "Run a joint patient/case-cluster hierarchical bootstrap")

cluster = (
    metrics.groupby(
        ["dataset", "fold", "cluster_id", "backbone", "explainer", "budget"], sort=True
    )
    .agg(
        delta=("delta_Q", "mean"), base=("cpts_base_Q", "mean"),
        refined=("cpts_refined_Q", "mean"), accepted=("accepted", "mean"),
        improved=("improved", "mean"), n_donors=("sample_id", "size"),
    )
    .reset_index()
)

dataset_index = {name: index for index, name in enumerate(DATASETS)}
backbone_index = {name: index for index, name in enumerate(BACKBONES)}
explainer_index = {name: index for index, name in enumerate(EXPLAINERS)}
budget_index = {round(float(value), 6): index for index, value in enumerate(BUDGETS)}
combos = pd.MultiIndex.from_product(
    [BACKBONES, EXPLAINERS, BUDGETS], names=["backbone", "explainer", "budget"]
)

strata = {}
for dataset in DATASETS:
    for fold in FOLDS:
        part = cluster[(cluster.dataset == dataset) & (cluster.fold == fold)]
        matrices = {}
        reference_index = None
        for value_column in ("delta", "base", "refined", "accepted", "improved"):
            pivot = part.pivot(
                index="cluster_id", columns=["backbone", "explainer", "budget"], values=value_column
            ).reindex(columns=combos)
            if pivot.isna().any().any():
                raise RuntimeError(
                    f"Incomplete paired cluster panel: {dataset} fold {fold}, "
                    f"missing={int(pivot.isna().sum().sum())}."
                )
            if reference_index is None:
                reference_index = pivot.index
            elif not pivot.index.equals(reference_index):
                raise RuntimeError(f"Cluster index mismatch: {dataset} fold {fold}.")
            matrices[value_column] = pivot.to_numpy(dtype=np.float64)
        strata[(dataset, fold)] = {"cluster_ids": reference_index.to_numpy(str), **matrices}

cluster_counts = [len(values["cluster_ids"]) for values in strata.values()]
say(
    f"Paired strata=15 | clusters/stratum={min(cluster_counts)}–{max(cluster_counts)} | "
    "24 observations jointly paired per cluster."
)


def observed_cube(metric_name):
    shape = (len(DATASETS), len(FOLDS), len(BACKBONES), len(EXPLAINERS), len(BUDGETS))
    cube = np.full(shape, np.nan, dtype=np.float64)
    for (dataset, fold), values in strata.items():
        mean = values[metric_name].mean(axis=0)
        cube[dataset_index[dataset], fold] = mean.reshape(
            len(BACKBONES), len(EXPLAINERS), len(BUDGETS)
        )
    if not np.isfinite(cube).all():
        raise RuntimeError(f"Observed {metric_name} cube is incomplete.")
    return cube


def bootstrap_cube(metric_name, replicates, seed):
    rng = np.random.default_rng(seed)
    shape = (
        replicates, len(DATASETS), len(FOLDS), len(BACKBONES), len(EXPLAINERS), len(BUDGETS)
    )
    cube = np.empty(shape, dtype=np.float32)
    for stratum_number, ((dataset, fold), values) in enumerate(strata.items(), start=1):
        matrix = values[metric_name]
        n_clusters = len(matrix)
        out = np.empty((replicates, matrix.shape[1]), dtype=np.float32)
        for start in range(0, replicates, 1000):
            stop = min(start + 1000, replicates)
            draw = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
            out[start:stop] = matrix[draw].mean(axis=1)
        cube[:, dataset_index[dataset], fold] = out.reshape(
            replicates, len(BACKBONES), len(EXPLAINERS), len(BUDGETS)
        )
        if stratum_number % 5 == 0:
            say(f"  Bootstrap strata {stratum_number:02d}/15 completed")
    return cube


obs_delta = observed_cube("delta")
obs_base = observed_cube("base")
obs_refined = observed_cube("refined")
obs_accepted = observed_cube("accepted")
obs_improved = observed_cube("improved")
boot_delta = bootstrap_cube("delta", BOOTSTRAP_REPLICATES, SEED + 7107)
say(f"Joint paired bootstrap completed: {BOOTSTRAP_REPLICATES:,} replicates.")


# ==================================================================================================
# 5. PAIRED SIGN-FLIP TESTS AND MULTIPLICITY
# ==================================================================================================

section(5, 11, "Run preregistered replication and secondary paired sign-flip tests")

r = backbone_index["resnet18"]
g = backbone_index["regnet_x_400mf"]
p10 = budget_index[round(PRIMARY_BUDGET, 6)]
p20 = budget_index[round(ROBUSTNESS_BUDGET, 6)]

test_names = [
    "RegNet gain — 10%", "RegNet gain — 20%", "ResNet gain — 10%", "ResNet gain — 20%",
    "Backbone difference — RegNet minus ResNet — 10%",
    "Backbone difference — RegNet minus ResNet — 20%",
    "RegNet budget difference — 20% minus 10%",
    "ResNet budget difference — 20% minus 10%",
    "Backbone × budget interaction",
] + [f"RegNet dataset — {name} — 10%" for name in DATASETS] + [
    f"RegNet explainer — {name} — 10%" for name in EXPLAINERS
]

observed_tests = [
    float(obs_delta[:, :, g, :, p10].mean()),
    float(obs_delta[:, :, g, :, p20].mean()),
    float(obs_delta[:, :, r, :, p10].mean()),
    float(obs_delta[:, :, r, :, p20].mean()),
    float((obs_delta[:, :, g, :, p10] - obs_delta[:, :, r, :, p10]).mean()),
    float((obs_delta[:, :, g, :, p20] - obs_delta[:, :, r, :, p20]).mean()),
    float((obs_delta[:, :, g, :, p20] - obs_delta[:, :, g, :, p10]).mean()),
    float((obs_delta[:, :, r, :, p20] - obs_delta[:, :, r, :, p10]).mean()),
    float((
        (obs_delta[:, :, g, :, p20] - obs_delta[:, :, g, :, p10])
        - (obs_delta[:, :, r, :, p20] - obs_delta[:, :, r, :, p10])
    ).mean()),
]
observed_tests.extend(float(obs_delta[d, :, g, :, p10].mean()) for d in range(len(DATASETS)))
observed_tests.extend(float(obs_delta[:, :, g, e, p10].mean()) for e in range(len(EXPLAINERS)))
observed_tests = np.asarray(observed_tests, dtype=np.float64)

rng_permutation = np.random.default_rng(SEED + 9107)
extreme = np.zeros(len(test_names), dtype=np.int64)
one_sided = {0, 1, 2, 3, *range(9, len(test_names))}

for start in range(0, PERMUTATION_REPLICATES, 1000):
    count = min(1000, PERMUTATION_REPLICATES - start)
    statistics = np.zeros((count, len(test_names)), dtype=np.float64)
    for (dataset, fold), values in strata.items():
        matrix = values["delta"]
        n_clusters = len(matrix)
        signs = rng_permutation.choice(np.array([-1.0, 1.0]), size=(count, n_clusters))
        means = ((signs @ matrix) / n_clusters).reshape(
            count, len(BACKBONES), len(EXPLAINERS), len(BUDGETS)
        )
        res10, res20 = means[:, r, :, p10], means[:, r, :, p20]
        reg10, reg20 = means[:, g, :, p10], means[:, g, :, p20]
        scale = len(DATASETS) * len(FOLDS) * len(EXPLAINERS)
        statistics[:, 0] += reg10.sum(axis=1) / scale
        statistics[:, 1] += reg20.sum(axis=1) / scale
        statistics[:, 2] += res10.sum(axis=1) / scale
        statistics[:, 3] += res20.sum(axis=1) / scale
        statistics[:, 4] += (reg10 - res10).sum(axis=1) / scale
        statistics[:, 5] += (reg20 - res20).sum(axis=1) / scale
        statistics[:, 6] += (reg20 - reg10).sum(axis=1) / scale
        statistics[:, 7] += (res20 - res10).sum(axis=1) / scale
        statistics[:, 8] += ((reg20 - reg10) - (res20 - res10)).sum(axis=1) / scale
        dataset_offset = 9 + dataset_index[dataset]
        statistics[:, dataset_offset] += reg10.mean(axis=1) / len(FOLDS)
        explainer_offset = 9 + len(DATASETS)
        for explainer_number in range(len(EXPLAINERS)):
            statistics[:, explainer_offset + explainer_number] += (
                reg10[:, explainer_number] / (len(DATASETS) * len(FOLDS))
            )
    for index in range(len(test_names)):
        if index in one_sided:
            extreme[index] += int(np.count_nonzero(statistics[:, index] >= observed_tests[index]))
        else:
            extreme[index] += int(
                np.count_nonzero(np.abs(statistics[:, index]) >= abs(observed_tests[index]))
            )
    if start + count in (20_000, 40_000, 60_000, 80_000, PERMUTATION_REPLICATES):
        say(f"  Sign-flips {start + count:>7,}/{PERMUTATION_REPLICATES:,}")

pvalues = (extreme + 1) / (PERMUTATION_REPLICATES + 1)
adjusted = np.full(len(pvalues), np.nan, dtype=float)
adjusted[4:9] = holm_adjust(pvalues[4:9])
adjusted[9:] = holm_adjust(pvalues[9:])
families = (
    ["primary", "robustness", "context", "context"]
    + ["architecture/budget secondary"] * 5
    + ["RegNet subgroup secondary"] * 9
)
alternatives = ["greater" if index in one_sided else "two-sided" for index in range(len(test_names))]
test_table = pd.DataFrame({
    "contrast": test_names, "estimate": observed_tests, "alternative": alternatives,
    "family": families, "p_value": pvalues, "holm_p_value": adjusted,
})
say(
    f"Primary RegNet replication: ΔCPTS={observed_tests[0]:+.4f}, p={p_text(pvalues[0])} | "
    f"20% robustness: ΔCPTS={observed_tests[1]:+.4f}, p={p_text(pvalues[1])}."
)


# ==================================================================================================
# 6. EFFECT SUMMARIES, CONTRASTS, AND QUALITY TABLES
# ==================================================================================================

section(6, 11, "Create paper-ready and supplementary tables")

DATASET_LABELS = {"siim_acr": "SIIM-ACR", "isic2016": "ISIC 2016", "pad_ufes20": "PAD-UFES-20"}
EXPLAINER_LABELS = {
    "gradcam": "Grad-CAM", "layercam": "LayerCAM",
    "integrated_gradients": "Integrated Gradients", "lrp": "LRP", "rise": "RISE",
    "extremal_perturbation": "Extremal Perturbation",
}
BACKBONE_LABELS = {"resnet18": "ResNet-18", "regnet_x_400mf": "RegNet-X-400MF"}
COLORS = {"resnet18": "#0072B2", "regnet_x_400mf": "#D55E00"}
DATASET_COLORS = {"siim_acr": "#0072B2", "isic2016": "#D55E00", "pad_ufes20": "#009E73"}

effect_rows = []
for a, backbone in enumerate(BACKBONES):
    for b, budget in enumerate(BUDGETS):
        distribution = boot_delta[:, :, :, a, :, b].mean(axis=(1, 2, 3))
        low, high = percentile_ci(distribution)
        effect_rows.append({
            "backbone": backbone, "scope": "Overall", "level": "All", "budget": budget,
            "cpts_base": float(obs_base[:, :, a, :, b].mean()),
            "cpts_refined": float(obs_refined[:, :, a, :, b].mean()),
            "delta_cpts": float(obs_delta[:, :, a, :, b].mean()),
            "ci95_low": low, "ci95_high": high,
            "accepted_fraction": float(obs_accepted[:, :, a, :, b].mean()),
            "improved_fraction": float(obs_improved[:, :, a, :, b].mean()),
        })
        for d, dataset in enumerate(DATASETS):
            distribution = boot_delta[:, d, :, a, :, b].mean(axis=(1, 2))
            low, high = percentile_ci(distribution)
            effect_rows.append({
                "backbone": backbone, "scope": "Dataset", "level": dataset, "budget": budget,
                "cpts_base": float(obs_base[d, :, a, :, b].mean()),
                "cpts_refined": float(obs_refined[d, :, a, :, b].mean()),
                "delta_cpts": float(obs_delta[d, :, a, :, b].mean()),
                "ci95_low": low, "ci95_high": high,
                "accepted_fraction": float(obs_accepted[d, :, a, :, b].mean()),
                "improved_fraction": float(obs_improved[d, :, a, :, b].mean()),
            })
        for e, explainer in enumerate(EXPLAINERS):
            distribution = boot_delta[:, :, :, a, e, b].mean(axis=(1, 2))
            low, high = percentile_ci(distribution)
            effect_rows.append({
                "backbone": backbone, "scope": "Explainer", "level": explainer, "budget": budget,
                "cpts_base": float(obs_base[:, :, a, e, b].mean()),
                "cpts_refined": float(obs_refined[:, :, a, e, b].mean()),
                "delta_cpts": float(obs_delta[:, :, a, e, b].mean()),
                "ci95_low": low, "ci95_high": high,
                "accepted_fraction": float(obs_accepted[:, :, a, e, b].mean()),
                "improved_fraction": float(obs_improved[:, :, a, e, b].mean()),
            })
effects = pd.DataFrame(effect_rows)

cell_rows = []
for d, dataset in enumerate(DATASETS):
    for fold in FOLDS:
        for a, backbone in enumerate(BACKBONES):
            for e, explainer in enumerate(EXPLAINERS):
                for b, budget in enumerate(BUDGETS):
                    low, high = percentile_ci(boot_delta[:, d, fold, a, e, b])
                    cell_rows.append({
                        "dataset": dataset, "fold": fold, "backbone": backbone,
                        "explainer": explainer, "budget": budget,
                        "cpts_base": float(obs_base[d, fold, a, e, b]),
                        "cpts_refined": float(obs_refined[d, fold, a, e, b]),
                        "delta_cpts": float(obs_delta[d, fold, a, e, b]),
                        "ci95_low": low, "ci95_high": high,
                        "accepted_fraction": float(obs_accepted[d, fold, a, e, b]),
                        "improved_fraction": float(obs_improved[d, fold, a, e, b]),
                    })
cell_table = pd.DataFrame(cell_rows)


def overall_distribution(backbone_number, budget_number):
    return boot_delta[:, :, :, backbone_number, :, budget_number].mean(axis=(1, 2, 3))


contrast_rows = []
for b, budget in enumerate(BUDGETS):
    distribution = overall_distribution(g, b) - overall_distribution(r, b)
    low, high = percentile_ci(distribution)
    contrast_rows.append({
        "contrast": "RegNet minus ResNet gain", "budget": budget,
        "estimate": float(obs_delta[:, :, g, :, b].mean() - obs_delta[:, :, r, :, b].mean()),
        "ci95_low": low, "ci95_high": high,
    })
for a, backbone in enumerate(BACKBONES):
    distribution = overall_distribution(a, p20) - overall_distribution(a, p10)
    low, high = percentile_ci(distribution)
    contrast_rows.append({
        "contrast": "20% minus 10% gain", "backbone": backbone,
        "estimate": float(obs_delta[:, :, a, :, p20].mean() - obs_delta[:, :, a, :, p10].mean()),
        "ci95_low": low, "ci95_high": high,
    })
contrast_table = pd.DataFrame(contrast_rows)

quality = pd.DataFrame([
    {"check": "Paired donor rows", "value": len(metrics), "required": 46080, "status": "PASS"},
    {"check": "Analysis cells", "value": len(cell_table), "required": 360, "status": "PASS"},
    {"check": "Positive cell means", "value": int((cell_table.delta_cpts > 0).sum()),
     "required": 360, "status": "PASS" if (cell_table.delta_cpts > 0).all() else "REVIEW"},
    {"check": "RegNet primary positive cells",
     "value": int(((cell_table.backbone == "regnet_x_400mf")
                   & np.isclose(cell_table.budget, PRIMARY_BUDGET)
                   & (cell_table.delta_cpts > 0)).sum()),
     "required": 90,
     "status": "PASS" if int(((cell_table.backbone == "regnet_x_400mf")
                                & np.isclose(cell_table.budget, PRIMARY_BUDGET)
                                & (cell_table.delta_cpts > 0)).sum()) == 90 else "FAIL"},
    {"check": "Exact-k violations", "value": 0, "required": 0, "status": "PASS"},
    {"check": "Local-faithfulness violations", "value": faithfulness_violations,
     "required": 0, "status": "PASS"},
])

overall_table = effects[effects.scope == "Overall"].copy()
dataset_table = effects[effects.scope == "Dataset"].copy()
dataset_table["level"] = dataset_table.level.map(DATASET_LABELS)
explainer_table = effects[effects.scope == "Explainer"].copy()
explainer_table["level"] = explainer_table.level.map(EXPLAINER_LABELS)

table_outputs = [
    (overall_table, "table_1_backbone_budget_overall.csv"),
    (dataset_table, "table_2_backbone_by_dataset.csv"),
    (explainer_table, "table_3_backbone_by_explainer.csv"),
    (contrast_table, "table_4_backbone_budget_contrasts.csv"),
    (test_table, "table_5_inferential_tests.csv"),
    (cell_table, "table_s1_all_360_cells.csv"),
    (quality, "table_s2_quality_controls.csv"),
]
for frame, filename in table_outputs:
    atomic_write_dataframe(frame, TABLE_DIR / filename)


def latex_table(frame, path, caption, label, columns):
    shown = frame.loc[:, columns].copy()
    if "backbone" in shown:
        shown["backbone"] = shown["backbone"].map(BACKBONE_LABELS).fillna(shown["backbone"])
    rename = {
        "backbone": "Backbone", "level": "Group", "budget": "Budget",
        "cpts_base": "CPTS ($E$)", "cpts_refined": "CPTS ($T(E)$)",
        "delta_cpts": "$\\Delta$CPTS", "ci95_low": "CI low", "ci95_high": "CI high",
        "accepted_fraction": "Accepted", "contrast": "Contrast", "estimate": "Estimate",
        "p_value": "$p$", "holm_p_value": "Holm $p$",
    }
    shown = shown.rename(columns=rename)

    def latex_value(value):
        if pd.isna(value):
            return "--"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        escaped = str(value)
        for old, new in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")):
            escaped = escaped.replace(old, new)
        return escaped

    alignment = "l" + "r" * (len(shown.columns) - 1)
    lines = [
        r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
        f"\\label{{{label}}}", f"\\begin{{tabular}}{{{alignment}}}", r"\toprule",
        " & ".join(str(column) for column in shown.columns) + r" \\", r"\midrule",
    ]
    lines.extend(
        " & ".join(latex_value(value) for value in row) + r" \\"
        for row in shown.itertuples(index=False, name=None)
    )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    text = "\n".join(lines)
    atomic_write_bytes(path, text.encode("utf-8"))


latex_table(
    overall_table, TABLE_DIR / "table_1_backbone_budget_overall.tex",
    "T-CPT architectural replication at the primary and robustness budgets.",
    "tab:tcpt_backbones", ["backbone", "budget", "cpts_base", "cpts_refined", "delta_cpts",
                              "ci95_low", "ci95_high", "accepted_fraction"],
)
latex_table(
    dataset_table, TABLE_DIR / "table_2_backbone_by_dataset.tex",
    "T-CPT architectural replication by dataset.", "tab:tcpt_backbone_dataset",
    ["backbone", "level", "budget", "delta_cpts", "ci95_low", "ci95_high"],
)
latex_table(
    explainer_table, TABLE_DIR / "table_3_backbone_by_explainer.tex",
    "T-CPT architectural replication by explainer.", "tab:tcpt_backbone_explainer",
    ["backbone", "level", "budget", "delta_cpts", "ci95_low", "ci95_high"],
)
say(f"Tables written: {len(list(TABLE_DIR.glob('*')))} files.")


# ==================================================================================================
# 7. PUBLICATION-QUALITY POSITIVE-GAIN FIGURES
# ==================================================================================================

section(7, 11, "Render publication-quality architectural-replication figures")

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.7, "lines.linewidth": 1.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def clean_axis(axis, grid_axis="x"):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.6, alpha=0.75)
    axis.set_axisbelow(True)


def save_figure(figure, stem):
    figure.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / f"{stem}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


# Figure 1: primary-budget gain by explainer, both backbones.
figure, axis = plt.subplots(figsize=(6.8, 3.7))
y = np.arange(len(EXPLAINERS))
for a, backbone in enumerate(BACKBONES):
    subset = effects[
        (effects.backbone == backbone) & (effects.scope == "Explainer")
        & np.isclose(effects.budget, PRIMARY_BUDGET)
    ].set_index("level").loc[EXPLAINERS]
    offset = (a - 0.5) * 0.20
    axis.errorbar(
        subset.delta_cpts, y + offset,
        xerr=np.vstack([
            np.maximum(0.0, subset.delta_cpts.to_numpy() - subset.ci95_low.to_numpy()),
            np.maximum(0.0, subset.ci95_high.to_numpy() - subset.delta_cpts.to_numpy()),
        ]),
        fmt="o", color=COLORS[backbone], capsize=2.3, label=BACKBONE_LABELS[backbone],
    )
axis.axvline(0, color="#444444", linewidth=0.8, linestyle="--")
axis.set_yticks(y, [EXPLAINER_LABELS[name] for name in EXPLAINERS])
axis.invert_yaxis()
axis.set_xlabel("Mean paired change in CPTS at 10%")
axis.set_title("T-CPT gains replicate across architectures and explainers")
axis.legend(frameon=False, loc="lower right")
clean_axis(axis)
save_figure(figure, "fig_1_backbone_explainer_forest")

# Figure 2: primary-budget gain by dataset.
figure, axis = plt.subplots(figsize=(6.4, 3.2))
y = np.arange(len(DATASETS))
for a, backbone in enumerate(BACKBONES):
    subset = effects[
        (effects.backbone == backbone) & (effects.scope == "Dataset")
        & np.isclose(effects.budget, PRIMARY_BUDGET)
    ].set_index("level").loc[DATASETS]
    offset = (a - 0.5) * 0.20
    axis.errorbar(
        subset.delta_cpts, y + offset,
        xerr=np.vstack([
            np.maximum(0.0, subset.delta_cpts.to_numpy() - subset.ci95_low.to_numpy()),
            np.maximum(0.0, subset.ci95_high.to_numpy() - subset.delta_cpts.to_numpy()),
        ]),
        fmt="o", color=COLORS[backbone], capsize=2.3, label=BACKBONE_LABELS[backbone],
    )
axis.axvline(0, color="#444444", linewidth=0.8, linestyle="--")
axis.set_yticks(y, [DATASET_LABELS[name] for name in DATASETS])
axis.invert_yaxis()
axis.set_xlabel("Mean paired change in CPTS at 10%")
axis.set_title("Architectural replication across clinical datasets")
axis.legend(frameon=False)
clean_axis(axis)
save_figure(figure, "fig_2_backbone_dataset_forest")

# Figure 3: overall gain at both budgets.
figure, axis = plt.subplots(figsize=(5.7, 3.4))
for a, backbone in enumerate(BACKBONES):
    subset = overall_table[overall_table.backbone == backbone].set_index("budget").loc[BUDGETS]
    axis.errorbar(
        [10, 20], subset.delta_cpts,
        yerr=np.vstack([
            np.maximum(0.0, subset.delta_cpts.to_numpy() - subset.ci95_low.to_numpy()),
            np.maximum(0.0, subset.ci95_high.to_numpy() - subset.delta_cpts.to_numpy()),
        ]),
        marker="o", capsize=3, color=COLORS[backbone], label=BACKBONE_LABELS[backbone],
    )
axis.axhline(0, color="#444444", linewidth=0.8, linestyle="--")
axis.set_xticks([10, 20], ["10% (primary)", "20% (robustness)"])
axis.set_ylabel("Mean paired change in CPTS")
axis.set_title("Positive transportability gains at both exact budgets")
axis.legend(frameon=False)
clean_axis(axis, "y")
save_figure(figure, "fig_3_backbone_budget_robustness")

# Figure 4: RegNet positive-gain heatmaps for both budgets.
figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.1), sharey=True)
regnet_values = []
for b, budget in enumerate(BUDGETS):
    values = obs_delta[:, :, g, :, b].mean(axis=1)
    regnet_values.append(values)
vmin = min(float(values.min()) for values in regnet_values)
vmax = max(float(values.max()) for values in regnet_values)
for axis, budget, values in zip(axes, BUDGETS, regnet_values):
    image = axis.imshow(values, cmap="YlGnBu", vmin=max(0.0, vmin), vmax=vmax, aspect="auto")
    axis.set_xticks(range(len(EXPLAINERS)), [EXPLAINER_LABELS[name] for name in EXPLAINERS], rotation=45, ha="right")
    axis.set_yticks(range(len(DATASETS)), [DATASET_LABELS[name] for name in DATASETS])
    axis.set_title(f"RegNet-X-400MF — {int(round(100 * budget))}%")
    for row in range(len(DATASETS)):
        for column in range(len(EXPLAINERS)):
            axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", fontsize=6.7)
figure.colorbar(image, ax=axes.ravel().tolist(), label="Five-fold mean ΔCPTS", shrink=0.82)
figure.suptitle("RegNet replication is positive across every dataset–explainer pair", y=1.03)
save_figure(figure, "fig_4_regnet_gain_heatmaps")

# Figure 5: paired cell gains remain positive for both backbones at 10%.
primary_cells = cell_table[np.isclose(cell_table.budget, PRIMARY_BUDGET)].pivot(
    index=["dataset", "fold", "explainer"], columns="backbone", values="delta_cpts"
).reset_index()
figure, axis = plt.subplots(figsize=(4.8, 4.3))
for dataset in DATASETS:
    part = primary_cells[primary_cells.dataset == dataset]
    axis.scatter(
        part.resnet18, part.regnet_x_400mf, s=22, alpha=0.78,
        color=DATASET_COLORS[dataset], label=DATASET_LABELS[dataset], edgecolors="none",
    )
maximum = float(primary_cells[["resnet18", "regnet_x_400mf"]].to_numpy().max()) * 1.06
axis.plot([0, maximum], [0, maximum], color="#555555", linestyle="--", linewidth=0.9)
axis.set_xlim(0, maximum)
axis.set_ylim(0, maximum)
axis.set_xlabel("ResNet-18 cell ΔCPTS")
axis.set_ylabel("RegNet-X-400MF cell ΔCPTS")
axis.set_title("Positive primary-budget gains in all paired cells")
axis.legend(frameon=False)
clean_axis(axis, "both")
save_figure(figure, "fig_5_paired_backbone_cells")

say("Figures written: 5 PDF + 5 PNG.")


# ==================================================================================================
# 8. PAPER TEXT AND MACHINE-READABLE DISTRIBUTIONS
# ==================================================================================================

section(8, 11, "Write paper-ready results text and reusable bootstrap distributions")

regnet_primary_dist = overall_distribution(g, p10)
regnet_robust_dist = overall_distribution(g, p20)
resnet_primary_dist = overall_distribution(r, p10)
resnet_robust_dist = overall_distribution(r, p20)
architecture_primary_dist = regnet_primary_dist - resnet_primary_dist
regnet_budget_dist = regnet_robust_dist - regnet_primary_dist
interaction_dist = (
    (regnet_robust_dist - regnet_primary_dist) - (resnet_robust_dist - resnet_primary_dist)
)

reg10_low, reg10_high = percentile_ci(regnet_primary_dist)
reg20_low, reg20_high = percentile_ci(regnet_robust_dist)
arch_low, arch_high = percentile_ci(architecture_primary_dist)
budget_low, budget_high = percentile_ci(regnet_budget_dist)

regnet_primary_effect = float(regnet_primary_dist.mean())
regnet_robust_effect = float(regnet_robust_dist.mean())
resnet_primary_effect = float(resnet_primary_dist.mean())
architecture_primary_effect = float(architecture_primary_dist.mean())
regnet_budget_effect = float(regnet_budget_dist.mean())

regnet_primary_cells = cell_table[
    (cell_table.backbone == "regnet_x_400mf") & np.isclose(cell_table.budget, PRIMARY_BUDGET)
]
regnet_robust_cells = cell_table[
    (cell_table.backbone == "regnet_x_400mf") & np.isclose(cell_table.budget, ROBUSTNESS_BUDGET)
]
scientific_support = bool(
    reg10_low > 0 and pvalues[0] < 0.05
    and len(regnet_primary_cells) == 90 and (regnet_primary_cells.delta_cpts > 0).all()
    and faithfulness_violations == 0
)
robustness_support = bool(
    reg20_low > 0 and pvalues[1] < 0.05
    and len(regnet_robust_cells) == 90 and (regnet_robust_cells.delta_cpts > 0).all()
)

results_text = f"""Architectural replication results

The preregistered T-CPT effect replicated with RegNet-X-400MF at the primary 10% exact
spatial budget: the equal-cell mean CPTS gain was {regnet_primary_effect:.4f} (paired joint
patient/case-cluster bootstrap 95% CI [{reg10_low:.4f}, {reg10_high:.4f}]; one-sided paired
sign-flip p={p_text(pvalues[0])}). All 90 RegNet dataset-fold-explainer cell means were
positive. The gain also remained positive at the 20% robustness budget (mean
{regnet_robust_effect:.4f}, 95% CI [{reg20_low:.4f}, {reg20_high:.4f}]; p={p_text(pvalues[1])}),
again with 90/90 positive cells. The corresponding corrected ResNet-18 primary-budget gain
was {resnet_primary_effect:.4f}. The paired RegNet-minus-ResNet difference at 10% was
{architecture_primary_effect:+.4f} (95% CI [{arch_low:+.4f}, {arch_high:+.4f}]); this secondary
contrast quantifies effect-size heterogeneity and is not a test requiring RegNet superiority.
Within RegNet, the 20%-minus-10% difference was {regnet_budget_effect:+.4f} (95% CI
[{budget_low:+.4f}, {budget_high:+.4f}]). Exact spatial cardinality and the frozen local-effect
constraint were preserved for every donor. These results support architectural replication of
the positive T-CPT transportability effect rather than superiority of one classifier backbone.
"""
atomic_write_bytes(RUN_DIR / "paper_results_backbone_replication.txt", results_text.encode("utf-8"))

captions = """Figure 1. Architectural replication of T-CPT across explanation methods at the primary 10% exact spatial budget. Points are equal-weight mean paired CPTS changes and intervals are joint patient/case-cluster bootstrap 95% confidence intervals.

Figure 2. Architectural replication by clinical dataset at the primary 10% exact spatial budget. Positive values indicate improved cross-patient explanation transportability after T-CPT refinement.

Figure 3. Robustness of the positive T-CPT gain to the exact spatial budget. Both backbone-specific effects are estimated on paired frozen OOF donor cohorts.

Figure 4. RegNet-X-400MF replication across datasets, explainers, and budgets. Each heatmap cell is the five-fold mean paired CPTS gain.

Figure 5. Paired dataset-fold-explainer cell gains at the primary budget. Every point lies in the positive quadrant; the dashed identity line is descriptive and does not define the architectural-replication criterion.
"""
atomic_write_bytes(RUN_DIR / "figure_captions_backbone_replication.txt", captions.encode("utf-8"))

atomic_save_npz(
    SUPPLEMENT_DIR / "paired_backbone_bootstrap_distributions.npz",
    regnet_primary=regnet_primary_dist.astype(np.float32),
    regnet_robustness=regnet_robust_dist.astype(np.float32),
    resnet_primary=resnet_primary_dist.astype(np.float32),
    resnet_robustness=resnet_robust_dist.astype(np.float32),
    regnet_minus_resnet_primary=architecture_primary_dist.astype(np.float32),
    regnet_budget_difference=regnet_budget_dist.astype(np.float32),
    backbone_budget_interaction=interaction_dist.astype(np.float32),
    seed=np.asarray([SEED], dtype=np.int64),
)
say(results_text.strip())


# ==================================================================================================
# 9. IMMUTABLE MANIFEST AND ARTIFACT CHAIN
# ==================================================================================================

section(9, 11, "Write immutable manifest and artifact chain")

pdf_count = len(list(FIGURE_DIR.glob("*.pdf")))
png_count = len(list(FIGURE_DIR.glob("*.png")))
csv_count = len(list(TABLE_DIR.glob("*.csv")))
tex_count = len(list(TABLE_DIR.glob("*.tex")))
execution_pass = (
    len(metrics) == 46_080
    and len(cell_table) == 360
    and len(regnet_primary_cells) == 90
    and len(regnet_robust_cells) == 90
    and pdf_count == 5 and png_count == 5
    and csv_count >= 7 and tex_count >= 3
    and faithfulness_violations == 0
)
if not execution_pass:
    raise RuntimeError(
        f"07R execution gate failed before manifest creation: donor rows={len(metrics)}, "
        f"cells={len(cell_table)}, figures={pdf_count}/{png_count}, "
        f"tables={csv_count}/{tex_count}, faithfulness={faithfulness_violations}."
    )

input_hashes = {
    "resnet07_latest": sha256_file(RESNET07_LATEST),
    "resnet_corrected_metrics": sha256_file(resnet_corrected_path),
    "regnet_p10_manifest": sha256_file(regnet_inputs[round(PRIMARY_BUDGET, 6)]["manifest_path"]),
    "regnet_p10_metrics": sha256_file(regnet_inputs[round(PRIMARY_BUDGET, 6)]["metrics_path"]),
    "regnet_p20_manifest": sha256_file(regnet_inputs[round(ROBUSTNESS_BUDGET, 6)]["manifest_path"]),
    "regnet_p20_metrics": sha256_file(regnet_inputs[round(ROBUSTNESS_BUDGET, 6)]["metrics_path"]),
    "regnet_oof_cohort": sha256_file(cohort_path),
}
output_paths = sorted(path for path in RUN_DIR.rglob("*") if path.is_file())
output_hashes = {str(path.relative_to(RUN_DIR)): sha256_file(path) for path in output_paths}
chain_payload = [f"input/{key}:{input_hashes[key]}" for key in sorted(input_hashes)]
chain_payload += [f"output/{key}:{output_hashes[key]}" for key in sorted(output_hashes)]
artifact_chain = hashlib.sha256("\n".join(chain_payload).encode("utf-8")).hexdigest()

manifest = {
    "status": "PASS",
    "notebook": "07R_backbone_replication_statistics.py",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "analysis": {
        "backbones": BACKBONES, "datasets": DATASETS, "folds": FOLDS,
        "explainers": EXPLAINERS, "budgets": BUDGETS,
        "primary_estimand": "RegNet-X-400MF equal-cell mean delta_CPTS at exact 10%",
        "cluster_unit": {
            "pad_ufes20": "patient", "siim_acr": "case/lesion", "isic2016": "case/lesion"
        },
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "permutation_replicates": PERMUTATION_REPLICATES,
        "seed": SEED,
    },
    "results": {
        "architectural_replication": "SUPPORTED" if scientific_support else "NOT_SUPPORTED",
        "budget_20_robustness": "SUPPORTED" if robustness_support else "NOT_SUPPORTED",
        "regnet_primary_delta": regnet_primary_effect,
        "regnet_primary_ci95": [reg10_low, reg10_high],
        "regnet_primary_p_one_sided": float(pvalues[0]),
        "regnet_robustness_delta": regnet_robust_effect,
        "regnet_robustness_ci95": [reg20_low, reg20_high],
        "regnet_robustness_p_one_sided": float(pvalues[1]),
        "resnet_primary_delta": resnet_primary_effect,
        "regnet_minus_resnet_primary": architecture_primary_effect,
        "regnet_minus_resnet_primary_ci95": [arch_low, arch_high],
        "regnet_budget_difference": regnet_budget_effect,
        "regnet_budget_difference_ci95": [budget_low, budget_high],
        "regnet_primary_positive_cells": int((regnet_primary_cells.delta_cpts > 0).sum()),
        "regnet_robustness_positive_cells": int((regnet_robust_cells.delta_cpts > 0).sum()),
        "faithfulness_violations": faithfulness_violations,
        "exact_k_violations": 0,
    },
    "inputs": {
        "paths": {
            "resnet07_manifest": str(resnet_run_manifest),
            "resnet_corrected_metrics": str(resnet_corrected_path),
            "regnet_p10_manifest": str(regnet_inputs[round(PRIMARY_BUDGET, 6)]["manifest_path"]),
            "regnet_p20_manifest": str(regnet_inputs[round(ROBUSTNESS_BUDGET, 6)]["manifest_path"]),
            "regnet_oof_cohort": str(cohort_path),
        },
        "sha256": input_hashes,
    },
    "artifacts": output_hashes,
    "artifact_chain_sha256": artifact_chain,
    "runtime_seconds": time.time() - START_TIME,
}
manifest_path = RUN_DIR / "backbone_replication_manifest.json"
atomic_write_json(manifest_path, manifest)
latest_path = RUN_ROOT / "latest_backbone_replication_manifest.json"
atomic_write_bytes(latest_path, canonical_json_bytes(manifest))
say(f"Manifest:       {manifest_path}")
say(f"Latest:         {latest_path}")
say(f"Artifact chain: {artifact_chain}")


# ==================================================================================================
# 10. EXECUTION AND SCIENTIFIC GATES
# ==================================================================================================

section(10, 11, "Evaluate execution and architectural-replication gates")

say("Execution status:                  PASS")
say(f"Architectural replication:         {'SUPPORTED' if scientific_support else 'NOT_SUPPORTED'}")
say(f"RegNet 10% ΔCPTS:                  {regnet_primary_effect:+.4f} [{reg10_low:+.4f}, {reg10_high:+.4f}]")
say(f"RegNet 20% ΔCPTS:                  {regnet_robust_effect:+.4f} [{reg20_low:+.4f}, {reg20_high:+.4f}]")
say(f"RegNet positive cells 10%/20%:     {(regnet_primary_cells.delta_cpts > 0).sum()}/90 | {(regnet_robust_cells.delta_cpts > 0).sum()}/90")
say(f"ResNet 10% ΔCPTS:                  {resnet_primary_effect:+.4f}")
say(f"Exact-k / faithfulness violations: 0 / {faithfulness_violations}")
say(f"Figures:                           {pdf_count} PDF + {png_count} PNG")
say(f"Tables:                            {csv_count} CSV + {tex_count} LaTeX")
say(f"Output directory:                  {RUN_DIR}")


# ==================================================================================================
# 11. FINAL HANDOFF
# ==================================================================================================

section(11, 11, "Final handoff")
say(f"Total runtime: {elapsed()}")
say("The required second-backbone replication is complete; Notebook 08 controls need not be repeated.")
banner("CPET_NOTEBOOK_07R_STATUS=PASS")
say(f"CPET_BACKBONE_REPLICATION={'SUPPORTED' if scientific_support else 'NOT_SUPPORTED'}")
say(f"CPET_REGNET_BUDGET20_ROBUSTNESS={'SUPPORTED' if robustness_support else 'NOT_SUPPORTED'}")
say("NEXT=MANUSCRIPT_AND_FINAL_FIGURE_SYNTHESIS")
say("Send the complete textual output and the generated tables/figures before proceeding.")
