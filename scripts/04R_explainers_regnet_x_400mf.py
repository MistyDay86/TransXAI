# CPET — NOTEBOOK 04R/08 — EXPLAINER POST-HOC OOF PER REGNET-X-400MF
# Unica cella Python autocontenuta per Google Colab.
# Hardware: GPU richiesta; Tesla T4 sufficiente. num_workers=0.

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


# ======================================================================================
# CONFIGURAZIONE SCIENTIFICA — CONGELATA PRIMA DELLE METRICHE XAI
# ======================================================================================
PROJECT_ROOT = Path('/content/gdrive/MyDrive/Colab Notebooks/CPET')
PROTOCOL_VERSION = '1.1.0'
BASE_SEED = 20260906
NOTEBOOK_ID = '04R_baseline_explainers_regnet_x_400mf'
BACKBONE = 'regnet_x_400mf'
NUM_WORKERS = 0
DATASETS = ['siim_acr', 'isic2016', 'pad_ufes20']
FOLDS = [0, 1, 2, 3, 4]
EXPLAINERS = ['gradcam', 'layercam', 'integrated_gradients', 'lrp', 'rise', 'extremal_perturbation']

# Riutilizziamo esattamente la coorte XAI stratificata congelata con ResNet-18.
# Non selezioniamo nuovi donor in base alla backbone o alle spiegazioni.
MAX_DONORS_PER_FOLD = 128
MAX_IMAGES_PER_EFFECTIVE_GROUP = 4
IG_STEPS = 32
RISE_MASKS = 512
RISE_GRID = 8
RISE_P_KEEP = 0.5
EP_STEPS = 40
EP_GRID = 16
EP_AREA = 0.10
EP_AREA_WEIGHT = 12.0
EP_TV_WEIGHT = 0.15
EXPLAIN_BATCH = 8
CAM_BATCH = 16
RISE_FORWARD_BATCH = 64
# Riusa direttamente la cache del Notebook 03R se il runtime Colab e' ancora vivo.
# In un nuovo runtime copia in parallelo soltanto i 1,920 donor, non l'intero dataset.
LOCAL_STAGE_ROOT = Path('/content/cpet_stage')
STAGE_COPY_WORKERS = 8
PRELOAD_DECODE_WORKERS = 8
RESUME = True

DATASET_SETTINGS = {
    'siim_acr':   {'input_size': 320, 'modality': 'xray',        'role': 'primary'},
    'isic2016':   {'input_size': 320, 'modality': 'dermoscopy',  'role': 'masked_replication'},
    'pad_ufes20': {'input_size': 320, 'modality': 'clinical',    'role': 'external_patient_subgroup'},
}

started = time.time()
LINE = '=' * 112


def say(message=''):
    print(message, flush=True)


def step(i, n, title):
    say(f'\n[{i}/{n}] {title}')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def human_seconds(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def stable_hash(obj):
    payload = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def sha256_file(path, chunk=4 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    os.replace(tmp, path)


def atomic_parquet(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def deterministic_rank(value, seed):
    return hashlib.sha256(f'{seed}|{value}'.encode('utf-8')).hexdigest()


say(LINE)
say('CPET — NOTEBOOK 04R/08 — EXPLAINER POST-HOC OOF PER REGNET-X-400MF')
say(LINE)
say('Obiettivo: produrre mappe OOF comparabili per sei explainer post-hoc, senza addestrare classificatori.')
say('Hardware richiesto: GPU. Tesla T4 sufficiente. CPU non consigliata.')
say('Target XAI: decisione OOF del classificatore (segno del logit coerente con prediction), non ground truth.')
say('Output: un HDF5 atomico per dataset × fold × explainer, con resume per artefatto completo.')
say('I/O veloce: cache 03R riutilizzata, staging parallelo e preprocessing in RAM una sola volta per fold.')
say('Dataset: esclusivamente i tre dataset confirmatori; BUS-UCLM non viene elaborato.')


# ======================================================================================
# 1. MOUNT, DIPENDENZE E GPU
# ======================================================================================
step(1, 10, 'Mount di Drive, dipendenze e audit GPU')
try:
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=False)
except Exception as exc:
    raise RuntimeError(f'Notebook destinato a Google Colab con Drive: {exc}')

if not PROJECT_ROOT.is_dir():
    raise RuntimeError(f'Project root non trovata: {PROJECT_ROOT}')

required = {
    'numpy': 'numpy', 'pandas': 'pandas', 'cv2': 'opencv-python-headless',
    'torch': 'torch', 'torchvision': 'torchvision', 'captum': 'captum',
    'h5py': 'h5py', 'tqdm': 'tqdm', 'sklearn': 'scikit-learn',
}
missing = []
for module, package in required.items():
    try:
        importlib.import_module(module)
    except Exception:
        missing.append(package)
if missing:
    say('Installazione dipendenze mancanti: ' + ', '.join(sorted(set(missing))))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *sorted(set(missing))], check=True)
else:
    say('Tutte le dipendenze richieste sono gia disponibili.')

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients, LRP
from torch.utils.data import DataLoader, Dataset
from torchvision.models import regnet_x_400mf
from tqdm.auto import tqdm

if not torch.cuda.is_available():
    raise RuntimeError('GPU non disponibile: Runtime > Cambia tipo di runtime > T4 GPU.')
device = torch.device('cuda')
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
seed_everything(BASE_SEED)
probe = torch.randn(256, 256, device=device)
probe = probe @ probe.T
torch.cuda.synchronize()
if not torch.isfinite(probe).all():
    raise RuntimeError('Smoke test GPU non finito.')
del probe
torch.cuda.empty_cache()
say(f'Python={sys.version.split()[0]} | PyTorch={torch.__version__} | CUDA={torch.version.cuda}')
say(f'GPU={torch.cuda.get_device_name(0)} | VRAM={torch.cuda.get_device_properties(0).total_memory/2**30:.2f} GiB')


# ======================================================================================
# 2. HANDOFF 03R E QUALITA DEI CLASSIFICATORI
# ======================================================================================
step(2, 10, 'Verifica del manifest 03R e dei 15 checkpoint OOF RegNet')
classifier_manifest_path = (
    PROJECT_ROOT / 'runs/training/classifiers' / BACKBONE / 'latest_classifier_manifest.json'
)
if not classifier_manifest_path.exists():
    raise RuntimeError(f'Manifest 03R mancante: {classifier_manifest_path}')
classifier_manifest = load_json(classifier_manifest_path)
training_backbone = classifier_manifest.get('training_config', {}).get('backbone')
if (
    classifier_manifest.get('status') != 'PASS'
    or not bool(classifier_manifest.get('full_grid_complete'))
    or training_backbone != BACKBONE
    or classifier_manifest.get('protocol_version') != PROTOCOL_VERSION
):
    raise RuntimeError(
        'Handoff 03R non valido: '
        f'status={classifier_manifest.get("status")}, '
        f'full_grid={classifier_manifest.get("full_grid_complete")}, '
        f'backbone={training_backbone}, protocol={classifier_manifest.get("protocol_version")}.'
    )
classifier_chain = classifier_manifest.get('artifact_chain_sha256')
if not isinstance(classifier_chain, str) or len(classifier_chain) != 64:
    raise RuntimeError('Artifact chain 03R mancante o non valida.')
say(f'Handoff 03R: PASS | full grid=15/15 | chain={classifier_chain[:16]}...')

checkpoint_root = PROJECT_ROOT / 'checkpoints/classifiers' / BACKBONE
classifier_results_root = PROJECT_ROOT / 'results/classifiers' / BACKBONE
fold_performance = {}
checkpoint_hashes = {}
for dataset in DATASETS:
    rows = []
    for fold in FOLDS:
        done_path = checkpoint_root / dataset / f'fold_{fold}' / 'COMPLETE.json'
        best_path = checkpoint_root / dataset / f'fold_{fold}' / 'best.pt'
        pred_path = classifier_results_root / dataset / f'fold_{fold}' / 'test_predictions.parquet'
        missing_paths = [str(p) for p in [done_path, best_path, pred_path] if not p.exists()]
        if missing_paths:
            raise RuntimeError('Artefatti Notebook 03 mancanti: ' + ' | '.join(missing_paths))
        done = load_json(done_path)
        if (
            done.get('dataset') != dataset
            or int(done.get('fold', -1)) != fold
            or done.get('backbone') != BACKBONE
            or done.get('protocol_version') != PROTOCOL_VERSION
        ):
            raise RuntimeError(f'Identita checkpoint incoerente: {done_path}')
        if done.get('best_checkpoint_sha256') != sha256_file(best_path):
            raise RuntimeError(f'Hash checkpoint incoerente: {best_path}')
        rows.append(float(done['test_metrics']['auroc']))
        checkpoint_hashes[f'{dataset}/fold_{fold}'] = sha256_file(best_path)
    fold_performance[dataset] = {'mean_auroc': float(np.mean(rows)), 'min_fold_auroc': float(np.min(rows))}
    role = DATASET_SETTINGS[dataset]['role']
    status = 'PASS'
    if np.mean(rows) < .70 or np.min(rows) < .60:
        raise RuntimeError(f'{dataset}: classificatore non supera il gate confirmatorio.')
    say(f'{dataset:12s} | role={role:26s} | AUROC={np.mean(rows):.4f} | min={np.min(rows):.4f} | {status}')
say(f'Checkpoint RegNet verificati e firmati: {len(checkpoint_hashes)}/15')


# ======================================================================================
# 3. FREEZE DEL PROTOCOLLO DEL NOTEBOOK 04
# ======================================================================================
step(3, 10, 'Freeze del protocollo degli explainer e dei vincoli di comparabilita')
reference_cohort_path = PROJECT_ROOT / 'results/explanations/resnet18/explanation_cohort.parquet'
if not reference_cohort_path.exists():
    raise RuntimeError(
        f'Coorte donor ResNet-18 mancante: {reference_cohort_path}. '
        'La replica RegNet richiede gli stessi sample_id per un confronto appaiato.'
    )
reference_cohort_sha256 = sha256_file(reference_cohort_path)
explanation_config = {
    'notebook_id': NOTEBOOK_ID,
    'protocol_version': PROTOCOL_VERSION,
    'backbone': BACKBONE,
    'datasets': DATASETS,
    'folds': FOLDS,
    'explainers': EXPLAINERS,
    'target': 'OOF predicted decision; signed binary logit (+logit for predicted positive, -logit for predicted negative)',
    'cohort': {
        'selection': 'exact sample_id replay of the frozen ResNet-18 donor cohort; RegNet OOF predictions and target signs',
        'reference_path': str(reference_cohort_path),
        'reference_sha256': reference_cohort_sha256,
        'max_donors_per_fold': MAX_DONORS_PER_FOLD,
        'max_images_per_effective_group': MAX_IMAGES_PER_EFFECTIVE_GROUP,
    },
    'map_semantics': 'positive evidence for the explained decision',
    'normalization': 'ReLU then per-map maximum scaling to [0,1]; degenerate maps retained and flagged',
    'storage': 'float16 continuous saliency in HDF5; thresholding deferred',
    'area_budgets_deferred': [0.05, 0.10, 0.20],
    # Ultimo stadio convoluzionale, analogo a ResNet-18 layer4[-1].
    # block2/block3 restano riservati a matching/intervento nel futuro 06R.
    'gradcam_layer': 'regnet_x_400mf.trunk_output.block4',
    'layercam_layer': 'regnet_x_400mf.trunk_output.block4',
    'integrated_gradients': {'steps': IG_STEPS, 'method': 'gausslegendre', 'baseline': 'Gaussian-blurred input'},
    'lrp': {'implementation': 'Captum LRP', 'channel_reduction': 'signed sum then positive evidence'},
    'rise': {'masks': RISE_MASKS, 'grid': RISE_GRID, 'p_keep': RISE_P_KEEP, 'baseline': 'Gaussian-blurred input'},
    'extremal_perturbation': {
        'steps': EP_STEPS, 'grid': EP_GRID, 'optimization_area': EP_AREA,
        'area_weight': EP_AREA_WEIGHT, 'tv_weight': EP_TV_WEIGHT,
        'baseline': 'Gaussian-blurred input',
    },
    'classifier_artifact_chain_sha256': classifier_chain,
    'io': {
        'stage_root': str(LOCAL_STAGE_ROOT),
        'stage_copy_workers': STAGE_COPY_WORKERS,
        'preload_decode_workers': PRELOAD_DECODE_WORKERS,
        'preprocess_once_per_fold_in_ram': True,
    },
    'seed': BASE_SEED,
    'num_workers': NUM_WORKERS,
    'resume': RESUME,
}
config_hash = stable_hash(explanation_config)
config_path = PROJECT_ROOT / f'configs/baseline_explainers_{BACKBONE}_v{PROTOCOL_VERSION}.json'
if config_path.exists():
    existing = load_json(config_path)
    if stable_hash(existing) != config_hash:
        raise RuntimeError(f'{config_path} esiste con configurazione diversa: non mescolo artefatti XAI incompatibili.')
    config_state = 'PRESERVED_IDENTICAL'
else:
    atomic_json(config_path, explanation_config)
    config_state = 'CREATED'
say(f'Config: {config_path} | {config_state} | sha256={sha256_file(config_path)}')
say('Clinical masks NON usate per generare o scegliere le spiegazioni; serviranno soltanto alla valutazione successiva.')
say('Le mappe restano continue: i budget equal-area 5/10/20% non vengono ancora applicati.')


# ======================================================================================
# 4. REPLAY APPAIATO DELLA COORTE OOF RESNET-18
# ======================================================================================
step(4, 10, 'Replay degli stessi donor ResNet-18 con predizioni OOF RegNet')
explanation_root = PROJECT_ROOT / 'results/explanations' / BACKBONE
run_root = PROJECT_ROOT / 'runs/explanations' / BACKBONE
explanation_root.mkdir(parents=True, exist_ok=True)
run_root.mkdir(parents=True, exist_ok=True)
cohort_path = explanation_root / 'explanation_cohort.parquet'
cohort_identity_path = explanation_root / 'explanation_cohort_identity.json'


def replay_reference_fold(frame, reference, dataset, fold):
    frame = frame.copy().reset_index(drop=True)
    required_cols = {'sample_id', 'label', 'prediction', 'logit', 'probability_calibrated', 'group_id_effective', 'image_path'}
    missing_cols = required_cols - set(frame.columns)
    if missing_cols:
        raise RuntimeError(f'{dataset} fold {fold}: colonne OOF mancanti {sorted(missing_cols)}')
    frame['_sample_key'] = frame.sample_id.astype(str)
    if frame._sample_key.duplicated().any():
        raise RuntimeError(f'{dataset} fold {fold}: sample_id duplicati nelle OOF.')
    reference_keys = reference.sample_id.astype(str).tolist()
    if len(reference_keys) != MAX_DONORS_PER_FOLD or len(set(reference_keys)) != len(reference_keys):
        raise RuntimeError(
            f'{dataset} fold {fold}: coorte ResNet di riferimento inattesa '
            f'(n={len(reference_keys)}, unici={len(set(reference_keys))}, atteso={MAX_DONORS_PER_FOLD}).'
        )
    indexed = frame.set_index('_sample_key', drop=False)
    missing = sorted(set(reference_keys) - set(indexed.index))
    if missing:
        raise RuntimeError(f'{dataset} fold {fold}: {len(missing)} donor ResNet assenti dalle OOF RegNet.')
    chosen = indexed.loc[reference_keys].copy().reset_index(drop=True)
    chosen['dataset'] = dataset
    chosen['fold'] = fold
    chosen['xai_role'] = 'confirmatory'
    chosen['explained_class'] = chosen['prediction'].astype(int)
    chosen['target_sign'] = np.where(chosen['explained_class'].to_numpy() == 1, 1.0, -1.0)
    return chosen.drop(columns=['_sample_key'], errors='ignore')


oof_hashes = {}
cohort_parts = []
datasets_for_xai = list(DATASETS)
reference_cohort = pd.read_parquet(reference_cohort_path)
reference_required = {'dataset', 'fold', 'sample_id'}
if reference_required - set(reference_cohort.columns):
    raise RuntimeError(f'Coorte ResNet priva delle colonne {sorted(reference_required-set(reference_cohort.columns))}.')
for dataset in datasets_for_xai:
    oof_path = classifier_results_root / f'{dataset}_oof_predictions.parquet'
    if not oof_path.exists():
        raise RuntimeError(f'OOF aggregate mancanti: {oof_path}')
    oof_hashes[dataset] = sha256_file(oof_path)
    oof = pd.read_parquet(oof_path)
    for fold in FOLDS:
        fold_frame = oof[oof.fold.astype(int) == fold].copy()
        reference_fold = reference_cohort[
            (reference_cohort.dataset.astype(str) == dataset)
            & (reference_cohort.fold.astype(int) == fold)
        ].copy()
        selected = replay_reference_fold(fold_frame, reference_fold, dataset, fold)
        cohort_parts.append(selected)
        say(
            f'{dataset:12s} fold={fold} | OOF={len(fold_frame):4d} -> donor={len(selected):3d} | '
            f'pos={int(selected.label.sum()):3d} | pred+={int(selected.prediction.sum()):3d} | '
            f'gruppi={selected.group_id_effective.nunique():3d}'
        )
cohort_candidate = pd.concat(cohort_parts, ignore_index=True)
cohort_candidate = cohort_candidate.sort_values(['dataset', 'fold', 'sample_id']).reset_index(drop=True)
cohort_identity = {
    'config_hash': config_hash,
    'oof_hashes': oof_hashes,
    'reference_cohort_sha256': reference_cohort_sha256,
    'rows': int(len(cohort_candidate)),
    'sample_chain': stable_hash(cohort_candidate[['dataset', 'fold', 'sample_id']].astype(str).to_dict('records')),
}
if cohort_path.exists() or cohort_identity_path.exists():
    if not (cohort_path.exists() and cohort_identity_path.exists()):
        raise RuntimeError('Coorte XAI parzialmente presente: non sovrascrivo uno stato ambiguo.')
    old_identity = load_json(cohort_identity_path)
    if old_identity != cohort_identity:
        raise RuntimeError('La coorte XAI esistente non coincide con la configurazione corrente.')
    cohort = pd.read_parquet(cohort_path)
    if stable_hash(cohort[['dataset', 'fold', 'sample_id']].astype(str).to_dict('records')) != cohort_identity['sample_chain']:
        raise RuntimeError('Firma della coorte XAI esistente non valida.')
    cohort_state = 'RESUME_IDENTICAL'
else:
    atomic_parquet(cohort_path, cohort_candidate)
    atomic_json(cohort_identity_path, cohort_identity)
    cohort = cohort_candidate
    cohort_state = 'CREATED'
say(f'Coorte XAI: {cohort_state} | donor={len(cohort):,} | sha256={sha256_file(cohort_path)}')


# ======================================================================================
# 5. STAGING LOCALE E PREPROCESSING IDENTICO AL NOTEBOOK 03
# ======================================================================================
step(5, 10, 'Cache locale parallela e preprocessing in RAM una sola volta per fold')
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
cv2.setNumThreads(1)


def resize_pad(image, size):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def manifest_positions(dataset):
    manifest_path = PROJECT_ROOT / f'data/manifests/{dataset}_manifest_v{PROTOCOL_VERSION}.parquet'
    manifest = pd.read_parquet(manifest_path, columns=['sample_id', 'image_path']).reset_index(drop=True)
    if manifest.sample_id.astype(str).duplicated().any():
        raise RuntimeError(f'{dataset}: sample_id duplicati nel manifest sorgente.')
    return {
        str(row.sample_id): (int(i), str(row.image_path))
        for i, row in enumerate(manifest.itertuples(index=False))
    }


def ensure_local_copy(job):
    source, target = job
    # Cache creata atomicamente dal 03R nello stesso runtime: evita perfino la
    # lenta chiamata stat() su Drive quando il file locale e' gia' disponibile.
    if target.exists() and target.stat().st_size > 0:
        return False, target.stat().st_size
    try:
        source_size = source.stat().st_size
    except FileNotFoundError:
        raise RuntimeError(f'Immagine mancante: {source}')
    tmp = target.with_name(target.name + '.tmp')
    try:
        if tmp.exists():
            tmp.unlink()
        shutil.copyfile(source, tmp)
        if tmp.stat().st_size != source_size:
            raise RuntimeError(f'Copia incompleta: {source} -> {tmp}')
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return True, source_size


def stage_cohort(frame):
    frame = frame.copy().reset_index(drop=True)
    positions = {dataset: manifest_positions(dataset) for dataset in DATASETS}
    runtime_paths, jobs = [], []
    for row in frame.itertuples(index=False):
        dataset = str(row.dataset)
        sample_id = str(row.sample_id)
        if sample_id not in positions[dataset]:
            raise RuntimeError(f'{dataset}: donor {sample_id} assente dal manifest sorgente.')
        manifest_index, manifest_image_path = positions[dataset][sample_id]
        if manifest_image_path != str(row.image_path):
            raise RuntimeError(f'{dataset}/{sample_id}: image_path OOF diverso dal manifest sorgente.')
        source = PROJECT_ROOT / manifest_image_path
        target_dir = LOCAL_STAGE_ROOT / dataset
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', sample_id)
        suffix = source.suffix.lower() or '.img'
        # Nome identico allo staging del Notebook 03R: cache hit immediato nello stesso runtime.
        target = target_dir / f'{manifest_index:06d}_{safe_id}{suffix}'
        runtime_paths.append(str(target))
        jobs.append((source, target))

    copied = cached = bytes_seen = 0
    stage_started = time.time()
    workers = min(STAGE_COPY_WORKERS, max(1, len(jobs)))
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='cpet-xai-stage')
    futures = [executor.submit(ensure_local_copy, job) for job in jobs]
    try:
        progress = tqdm(as_completed(futures), total=len(futures), desc='Stage donor XAI', mininterval=1.0)
        for future in progress:
            was_copied, size_bytes = future.result()
            copied += int(was_copied)
            cached += int(not was_copied)
            bytes_seen += int(size_bytes)
            elapsed = max(time.time() - stage_started, 1e-6)
            progress.set_postfix(copied=copied, cached=cached, MiB_s=f'{bytes_seen/2**20/elapsed:.1f}')
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    frame['runtime_path'] = runtime_paths
    return frame, copied, cached


cohort_runtime, copied, cached = stage_cohort(cohort)
say(
    f'File donor pronti in cache locale: {len(cohort_runtime):,} | '
    f'copiati ora={copied:,} | riutilizzati da 03R={cached:,}'
)


def decode_preprocess(job):
    index, runtime_path, input_size = job
    image = cv2.imread(str(runtime_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f'Decode fallito: {runtime_path}')
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    image = resize_pad(image, input_size).astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.ascontiguousarray(np.transpose(image, (2, 0, 1)), dtype=np.float32)
    return index, chw


class PreloadedExplanationDataset(Dataset):
    def __init__(self, images, signs):
        self.images = images
        self.signs = signs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.signs[index], index


def preload_fold(frame, input_size, description):
    """Decode and normalize once; all six explainers reuse the same CPU tensor."""
    frame = frame.reset_index(drop=True)
    images = torch.empty((len(frame), 3, input_size, input_size), dtype=torch.float32)
    signs = torch.from_numpy(frame.target_sign.astype(np.float32).to_numpy(copy=True))
    jobs = [(i, path, input_size) for i, path in enumerate(frame.runtime_path.astype(str))]
    workers = min(PRELOAD_DECODE_WORKERS, max(1, len(jobs)))
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='cpet-xai-decode')
    futures = [executor.submit(decode_preprocess, job) for job in jobs]
    try:
        for future in tqdm(as_completed(futures), total=len(futures), desc=description, leave=False):
            index, chw = future.result()
            images[index].copy_(torch.from_numpy(chw))
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return PreloadedExplanationDataset(images, signs)


def make_loader(preloaded_dataset, batch_size):
    return DataLoader(
        preloaded_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
    )


def build_model(checkpoint_path):
    model = regnet_x_400mf(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    # LRP e CAM usano hook: disattivare inplace evita collisioni col backward,
    # senza cambiare la funzione numerica del checkpoint.
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def blurred_baseline(x):
    # Blur forte ma content-aware; evita il riferimento nero fuori distribuzione.
    kernel = 31 if min(x.shape[-2:]) >= 256 else 21
    return F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)


def normalize_positive(maps):
    maps = torch.relu(maps.float())
    flat = maps.flatten(1)
    maxima = flat.amax(dim=1, keepdim=True)
    degenerate = maxima.squeeze(1) <= 1e-12
    flat = torch.where(maxima > 1e-12, flat / maxima.clamp_min(1e-12), torch.zeros_like(flat))
    return flat.reshape_as(maps), degenerate


def channel_reduce_signed(attribution, signs):
    signed = attribution.float() * signs[:, None, None, None]
    return signed.sum(dim=1, keepdim=True)


# ======================================================================================
# 6. IMPLEMENTAZIONE DEI SEI EXPLAINER
# ======================================================================================
step(6, 10, 'Definizione e smoke test degli explainer')


def explain_cam_pair(model, loader, n, size):
    gradcam_maps = np.empty((n, size, size), dtype=np.float16)
    layercam_maps = np.empty((n, size, size), dtype=np.float16)
    deg_g = np.zeros(n, dtype=np.uint8)
    deg_l = np.zeros(n, dtype=np.uint8)
    holder = {}

    def hook(_, __, output):
        holder['activation'] = output
        output.retain_grad()

    handle = model.trunk_output.block4.register_forward_hook(hook)
    try:
        for images, signs, indices in tqdm(loader, desc='Grad-CAM + Layer-CAM', leave=False):
            images = images.to(device, non_blocking=True)
            signs = signs.to(device, non_blocking=True)
            # I pesi sono congelati, quindi l'input deve richiedere gradiente affinche'
            # le attivazioni interne conservino il grafo necessario alle CAM.
            images.requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits = model(images).flatten()
            (logits * signs).sum().backward()
            activation = holder['activation'].float()
            gradient = holder['activation'].grad.float()
            gradcam = torch.relu((gradient.mean(dim=(2, 3), keepdim=True) * activation).sum(dim=1, keepdim=True))
            layercam = torch.relu((torch.relu(gradient) * activation).sum(dim=1, keepdim=True))
            gradcam = F.interpolate(gradcam, (size, size), mode='bilinear', align_corners=False)
            layercam = F.interpolate(layercam, (size, size), mode='bilinear', align_corners=False)
            gradcam, dg = normalize_positive(gradcam)
            layercam, dl = normalize_positive(layercam)
            idx = indices.numpy()
            gradcam_maps[idx] = gradcam[:, 0].detach().cpu().numpy().astype(np.float16)
            layercam_maps[idx] = layercam[:, 0].detach().cpu().numpy().astype(np.float16)
            deg_g[idx] = dg.detach().cpu().numpy().astype(np.uint8)
            deg_l[idx] = dl.detach().cpu().numpy().astype(np.uint8)
            del images, signs, logits, activation, gradient, gradcam, layercam
    finally:
        handle.remove()
    return (gradcam_maps, deg_g), (layercam_maps, deg_l)


def explain_integrated_gradients(model, loader, n, size):
    maps = np.empty((n, size, size), dtype=np.float16)
    deg = np.zeros(n, dtype=np.uint8)
    ig = IntegratedGradients(model)
    for images, signs, indices in tqdm(loader, desc='Integrated Gradients', leave=False):
        images = images.to(device, non_blocking=True)
        signs = signs.to(device, non_blocking=True)
        images.requires_grad_(True)
        baseline = blurred_baseline(images.detach())
        attr = ig.attribute(
            images, baselines=baseline, target=0, n_steps=IG_STEPS,
            method='gausslegendre', internal_batch_size=max(len(images), len(images) * 4),
        )
        saliency, d = normalize_positive(channel_reduce_signed(attr, signs))
        idx = indices.numpy()
        maps[idx] = saliency[:, 0].detach().cpu().numpy().astype(np.float16)
        deg[idx] = d.detach().cpu().numpy().astype(np.uint8)
        del images, signs, baseline, attr, saliency
        torch.cuda.empty_cache()
    return maps, deg


def explain_lrp(model, loader, n, size):
    maps = np.empty((n, size, size), dtype=np.float16)
    deg = np.zeros(n, dtype=np.uint8)
    lrp = LRP(model)
    for images, signs, indices in tqdm(loader, desc='LRP', leave=False):
        images = images.to(device, non_blocking=True)
        signs = signs.to(device, non_blocking=True)
        images.requires_grad_(True)
        attr = lrp.attribute(images, target=0)
        saliency, d = normalize_positive(channel_reduce_signed(attr, signs))
        idx = indices.numpy()
        maps[idx] = saliency[:, 0].detach().cpu().numpy().astype(np.float16)
        deg[idx] = d.detach().cpu().numpy().astype(np.uint8)
        del images, signs, attr, saliency
        torch.cuda.empty_cache()
    return maps, deg


def make_rise_masks(size, count, grid, p_keep, seed):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    cell = int(math.ceil(size / grid))
    up_size = (grid + 1) * cell
    low = (torch.rand((count, 1, grid, grid), generator=generator) < p_keep).float()
    up = F.interpolate(low, size=(up_size, up_size), mode='bilinear', align_corners=False)
    offsets_y = torch.randint(0, cell, (count,), generator=generator)
    offsets_x = torch.randint(0, cell, (count,), generator=generator)
    masks = torch.empty((count, 1, size, size), dtype=torch.float32)
    for i in range(count):
        y, x = int(offsets_y[i]), int(offsets_x[i])
        masks[i] = up[i, :, y:y + size, x:x + size]
    return masks


@torch.no_grad()
def explain_rise(model, loader, n, size, seed):
    maps = np.empty((n, size, size), dtype=np.float16)
    deg = np.zeros(n, dtype=np.uint8)
    masks_cpu = make_rise_masks(size, RISE_MASKS, RISE_GRID, RISE_P_KEEP, seed)
    for images, signs, indices in tqdm(loader, desc='RISE', leave=False):
        # Elaborazione per immagine: la dimensione forward resta controllata su T4.
        for local_i in range(len(images)):
            x = images[local_i:local_i + 1].to(device, non_blocking=True)
            sign = signs[local_i].to(device)
            base = blurred_baseline(x)
            weighted = torch.zeros((1, size, size), device=device)
            score_sum = 0.0
            mask_sum = torch.zeros((1, size, size), device=device)
            total = 0
            for start_mask in range(0, RISE_MASKS, RISE_FORWARD_BATCH):
                masks = masks_cpu[start_mask:start_mask + RISE_FORWARD_BATCH].to(device, non_blocking=True)
                perturbed = base + masks * (x - base)
                scores = torch.sigmoid(model(perturbed).flatten() * sign)
                weighted += (scores[:, None, None, None] * masks).sum(dim=0)
                score_sum += float(scores.sum())
                mask_sum += masks.sum(dim=0)
                total += len(masks)
            # Covarianza empirica score-mask: rimuove il fondo positivo costante di RISE.
            saliency = weighted / total - (score_sum / total) * (mask_sum / total)
            saliency, d = normalize_positive(saliency.unsqueeze(0))
            out_index = int(indices[local_i])
            maps[out_index] = saliency[0, 0].cpu().numpy().astype(np.float16)
            deg[out_index] = int(d.item())
            del x, base, weighted, mask_sum, saliency
    return maps, deg


def total_variation(mask):
    return (mask[:, :, 1:, :] - mask[:, :, :-1, :]).abs().mean() + (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().mean()


def explain_extremal(model, loader, n, size):
    maps = np.empty((n, size, size), dtype=np.float16)
    deg = np.zeros(n, dtype=np.uint8)
    for images, signs, indices in tqdm(loader, desc='Extremal Perturbation', leave=False):
        images = images.to(device, non_blocking=True)
        signs = signs.to(device, non_blocking=True)
        base = blurred_baseline(images)
        init = math.log(EP_AREA / (1.0 - EP_AREA))
        parameters = torch.full((len(images), 1, EP_GRID, EP_GRID), init, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([parameters], lr=0.15)
        for _ in range(EP_STEPS):
            optimizer.zero_grad(set_to_none=True)
            low_mask = torch.sigmoid(parameters)
            mask = F.interpolate(low_mask, (size, size), mode='bilinear', align_corners=False)
            perturbed = base + mask * (images - base)
            margin = model(perturbed).flatten() * signs
            area_penalty = (mask.mean(dim=(1, 2, 3)) - EP_AREA).pow(2).mean()
            loss = -margin.mean() + EP_AREA_WEIGHT * area_penalty + EP_TV_WEIGHT * total_variation(mask)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            mask = F.interpolate(torch.sigmoid(parameters), (size, size), mode='bilinear', align_corners=False)
            saliency, d = normalize_positive(mask)
        idx = indices.numpy()
        maps[idx] = saliency[:, 0].cpu().numpy().astype(np.float16)
        deg[idx] = d.cpu().numpy().astype(np.uint8)
        del images, signs, base, parameters, optimizer, mask, saliency
        torch.cuda.empty_cache()
    return maps, deg


def smoke_explainers():
    dataset = datasets_for_xai[0]
    fold = int(cohort_runtime[cohort_runtime.dataset == dataset].fold.iloc[0])
    frame = cohort_runtime[(cohort_runtime.dataset == dataset) & (cohort_runtime.fold == fold)].head(2).copy()
    size = DATASET_SETTINGS[dataset]['input_size']
    checkpoint = checkpoint_root / dataset / f'fold_{fold}' / 'best.pt'
    smoke_results = {}
    smoke_data = preload_fold(frame, size, 'Preload smoke')
    smoke_loader = make_loader(smoke_data, 2)
    # Replay del classificatore: protegge da divergenze fra preprocessing 03R/04R
    # o dalla conversione hook-safe dei ReLU.
    model = build_model(checkpoint)
    replay_logits = []
    geometry = {}
    handles = []
    for name in ['block2', 'block3', 'block4']:
        module = getattr(model.trunk_output, name)
        handles.append(module.register_forward_hook(
            lambda _, __, output, layer_name=name: geometry.__setitem__(layer_name, list(output.shape))
        ))
    with torch.no_grad():
        for images, _, _ in smoke_loader:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                replay = model(images.to(device)).flatten()
            replay_logits.extend(replay.float().cpu().numpy().tolist())
    for handle in handles:
        handle.remove()
    expected_geometry = {'block2': [40, 40], 'block3': [20, 20], 'block4': [10, 10]}
    geometry_ok = all(geometry.get(name, [])[-2:] == spatial for name, spatial in expected_geometry.items())
    smoke_results['regnet_geometry'] = geometry_ok
    say(
        'RegNet geometry: '
        + ' | '.join(f'{name}={geometry.get(name)}' for name in expected_geometry)
        + f' | {"PASS" if geometry_ok else "FAIL"}'
    )
    replay_errors = np.abs(np.asarray(replay_logits) - frame.logit.astype(float).to_numpy())
    replay_error = float(np.max(replay_errors))
    replay_mean_error = float(np.mean(replay_errors))
    # Le OOF del 03R furono inferite in autocast FP16 con batch=48. Qui il replay
    # usa due donor: convoluzioni FP16 con forma di batch diversa non sono bitwise
    # identiche. Il replay resta un controllo contro divergenze macroscopiche del
    # preprocessing/checkpoint; non pretende un'identita' numerica irrealistica.
    replay_safety_limit = 5e-2
    smoke_results['classifier_replay'] = bool(
        np.isfinite(replay_error) and replay_error <= replay_safety_limit
    )
    say(
        f'Classifier replay: mean |error|={replay_mean_error:.3e} | '
        f'max={replay_error:.3e} | safety limit={replay_safety_limit:.2e} | '
        f'{"PASS" if smoke_results["classifier_replay"] else "FAIL"}'
    )
    del model, replay_logits
    gc.collect(); torch.cuda.empty_cache()
    # Smoke CAM pair.
    model = build_model(checkpoint)
    pair = explain_cam_pair(model, make_loader(smoke_data, 2), len(frame), size)
    smoke_results['gradcam'] = bool(np.isfinite(pair[0][0]).all())
    smoke_results['layercam'] = bool(np.isfinite(pair[1][0]).all())
    del model, pair
    gc.collect(); torch.cuda.empty_cache()
    # Smoke IG e LRP con parametri reali.
    model = build_model(checkpoint)
    result = explain_integrated_gradients(model, make_loader(smoke_data, 1), len(frame), size)
    smoke_results['integrated_gradients'] = bool(np.isfinite(result[0]).all())
    del model, result
    gc.collect(); torch.cuda.empty_cache()
    model = build_model(checkpoint)
    result = explain_lrp(model, make_loader(smoke_data, 1), len(frame), size)
    smoke_results['lrp'] = bool(np.isfinite(result[0]).all())
    del model, result
    gc.collect(); torch.cuda.empty_cache()
    # Anche i due metodi piu' costosi vengono provati con i parametri CORE reali
    # su una sola immagine, prima di avviare qualunque job lungo.
    one_data = PreloadedExplanationDataset(smoke_data.images[:1], smoke_data.signs[:1])
    model = build_model(checkpoint)
    result = explain_rise(model, make_loader(one_data, 1), 1, size, BASE_SEED + 404)
    smoke_results['rise'] = bool(np.isfinite(result[0]).all())
    del model, result
    gc.collect(); torch.cuda.empty_cache()
    model = build_model(checkpoint)
    result = explain_extremal(model, make_loader(one_data, 1), 1, size)
    smoke_results['extremal_perturbation'] = bool(np.isfinite(result[0]).all())
    del model, result
    gc.collect(); torch.cuda.empty_cache()
    if not all(v is True for v in smoke_results.values()):
        raise RuntimeError(f'Smoke explainer fallito: {smoke_results}')
    del smoke_loader, one_data, smoke_data
    gc.collect()
    return smoke_results


smoke = smoke_explainers()
say('Smoke explainer: ' + ' | '.join(f'{k}={v}' for k, v in smoke.items()))


# ======================================================================================
# 7. STORAGE ATOMICO E VALIDAZIONE DELLE MAPPE
# ======================================================================================
step(7, 10, 'Preparazione dello storage HDF5 atomico e del resume')


def cohort_hash(frame):
    return stable_hash(frame[['dataset', 'fold', 'sample_id', 'label', 'prediction']].astype(str).to_dict('records'))


def artifact_paths(dataset, fold, method):
    folder = explanation_root / dataset / f'fold_{fold}'
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f'{method}.h5', folder / f'{method}.json'


def expected_job_identity(dataset, fold, method, frame):
    return {
        'dataset': dataset,
        'fold': int(fold),
        'method': method,
        'config_hash': config_hash,
        'checkpoint_sha256': checkpoint_hashes[f'{dataset}/fold_{fold}'],
        'cohort_hash': cohort_hash(frame),
        'n': int(len(frame)),
        'input_size': int(DATASET_SETTINGS[dataset]['input_size']),
    }


def validate_complete(dataset, fold, method, frame):
    h5_path, meta_path = artifact_paths(dataset, fold, method)
    if not (h5_path.exists() and meta_path.exists()):
        return False, None
    meta = load_json(meta_path)
    expected = expected_job_identity(dataset, fold, method, frame)
    if any(meta.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f'Artefatto incompatibile con il run corrente: {meta_path}')
    if meta.get('h5_sha256') != sha256_file(h5_path):
        raise RuntimeError(f'Hash HDF5 non valido: {h5_path}')
    with h5py.File(h5_path, 'r') as h5:
        if h5['saliency'].shape != (len(frame), expected['input_size'], expected['input_size']):
            raise RuntimeError(f'Shape HDF5 non valida: {h5_path}')
        stored_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in h5['sample_id'][:]]
        if stored_ids != frame.sample_id.astype(str).tolist():
            raise RuntimeError(f'Ordine sample_id non valido: {h5_path}')
    return True, meta


def write_h5_atomic(dataset, fold, method, frame, maps, degenerate, seconds):
    h5_path, meta_path = artifact_paths(dataset, fold, method)
    expected = expected_job_identity(dataset, fold, method, frame)
    if maps.shape != (len(frame), expected['input_size'], expected['input_size']):
        raise RuntimeError(f'{dataset} f{fold} {method}: shape mappe inattesa {maps.shape}.')
    if not np.isfinite(maps).all() or maps.min() < 0 or maps.max() > 1.001:
        raise RuntimeError(f'{dataset} f{fold} {method}: mappe non finite o fuori [0,1].')
    tmp = h5_path.with_suffix('.h5.tmp')
    if tmp.exists():
        tmp.unlink()
    strings = h5py.string_dtype(encoding='utf-8')
    with h5py.File(tmp, 'w') as h5:
        h5.create_dataset('saliency', data=maps, dtype='float16', compression='lzf', chunks=(1, expected['input_size'], expected['input_size']))
        h5.create_dataset('sample_id', data=frame.sample_id.astype(str).to_numpy(dtype=object), dtype=strings)
        h5.create_dataset('label', data=frame.label.astype(np.int8).to_numpy(), dtype='int8')
        h5.create_dataset('prediction', data=frame.prediction.astype(np.int8).to_numpy(), dtype='int8')
        h5.create_dataset('probability_calibrated', data=frame.probability_calibrated.astype(np.float32).to_numpy(), dtype='float32')
        h5.create_dataset('target_sign', data=frame.target_sign.astype(np.float32).to_numpy(), dtype='float32')
        h5.create_dataset('group_id_effective', data=frame.group_id_effective.astype(str).to_numpy(dtype=object), dtype=strings)
        h5.create_dataset('degenerate', data=np.asarray(degenerate, dtype=np.uint8), dtype='uint8')
        h5.attrs['identity_json'] = json.dumps(expected, sort_keys=True)
        h5.flush()
    os.replace(tmp, h5_path)
    h5_hash = sha256_file(h5_path)
    meta = {
        **expected,
        'created_utc': utc_now(),
        'seconds': float(seconds),
        'degenerate_count': int(np.sum(degenerate)),
        'degenerate_fraction': float(np.mean(degenerate)),
        'mean_saliency': float(np.mean(maps, dtype=np.float64)),
        'h5_path': str(h5_path),
        'h5_sha256': h5_hash,
        'xai_role': str(frame.xai_role.iloc[0]),
    }
    atomic_json(meta_path, meta)
    return meta


say(f'Directory mappe: {explanation_root}')
say('Resume: un job viene saltato soltanto se identita, SHA-256, shape e ordine sample_id sono validi.')


# ======================================================================================
# 8. GENERAZIONE COMPLETA DELLE SPIEGAZIONI
# ======================================================================================
step(8, 10, 'Generazione delle spiegazioni OOF')
jobs = []
for dataset in datasets_for_xai:
    for fold in FOLDS:
        frame = cohort_runtime[(cohort_runtime.dataset == dataset) & (cohort_runtime.fold.astype(int) == fold)].copy()
        frame = frame.sort_values('sample_id').reset_index(drop=True)
        if len(frame) == 0:
            raise RuntimeError(f'Coorte vuota: {dataset} fold {fold}')
        for method in EXPLAINERS:
            jobs.append((dataset, fold, method, frame))

total_jobs = len(jobs)
completed_jobs = 0
generated_jobs = 0
job_metas = []


def report_job(meta, resumed, job_started):
    global completed_jobs
    completed_jobs += 1
    elapsed = time.time() - job_started
    generated_average = (time.time() - generation_started) / max(1, generated_jobs)
    remaining_nonresume_upper = total_jobs - completed_jobs
    eta = generated_average * remaining_nonresume_upper
    state = 'RESUME' if resumed else 'GENERATO'
    say(
        f'[{completed_jobs:03d}/{total_jobs:03d}] {state:8s} | {meta["dataset"]:12s} f{meta["fold"]} '
        f'{meta["method"]:24s} | n={meta["n"]:3d} | deg={meta["degenerate_count"]:3d} '
        f'| job={human_seconds(elapsed)} | ETA<={human_seconds(eta)}'
    )


generation_started = time.time()
for dataset in datasets_for_xai:
    size = DATASET_SETTINGS[dataset]['input_size']
    say('\n' + '-' * 112)
    say(f'DATASET {dataset} | role={DATASET_SETTINGS[dataset]["role"]} | input={size}')
    say('-' * 112)
    for fold in FOLDS:
        frame = cohort_runtime[(cohort_runtime.dataset == dataset) & (cohort_runtime.fold.astype(int) == fold)].copy()
        frame = frame.sort_values('sample_id').reset_index(drop=True)
        checkpoint = checkpoint_root / dataset / f'fold_{fold}' / 'best.pt'
        job_status = {method: validate_complete(dataset, fold, method, frame) for method in EXPLAINERS}
        if all(value[0] for value in job_status.values()):
            for method in EXPLAINERS:
                now = time.time()
                meta = job_status[method][1]
                job_metas.append(meta)
                report_job(meta, True, now)
            continue

        # Decodifica e normalizzazione avvengono una sola volta per fold. Il tensore
        # CPU risultante viene riutilizzato da tutti gli explainer mancanti.
        preload_started = time.time()
        preloaded = preload_fold(frame, size, f'Preload {dataset} f{fold}')
        say(
            f'  {dataset} fold {fold}: {len(frame)} immagini pre-processate in RAM '
            f'in {human_seconds(time.time()-preload_started)}; riuso per tutti i metodi.'
        )

        # Grad-CAM e Layer-CAM condividono esattamente lo stesso forward/backward.
        cam_status = {m: job_status[m] for m in ['gradcam', 'layercam']}
        if all(v[0] for v in cam_status.values()):
            for method in ['gradcam', 'layercam']:
                now = time.time(); meta = cam_status[method][1]; job_metas.append(meta); report_job(meta, True, now)
        else:
            if any(v[0] for v in cam_status.values()):
                say(f'  {dataset} fold {fold}: coppia CAM parziale; rigenerazione deterministica di entrambe.')
            now = time.time(); model = build_model(checkpoint)
            grad_result, layer_result = explain_cam_pair(model, make_loader(preloaded, CAM_BATCH), len(frame), size)
            del model; gc.collect(); torch.cuda.empty_cache()
            for method, result in [('gradcam', grad_result), ('layercam', layer_result)]:
                method_start = now
                meta = write_h5_atomic(dataset, fold, method, frame, result[0], result[1], time.time() - now)
                generated_jobs += 1; job_metas.append(meta); report_job(meta, False, method_start)
            del grad_result, layer_result; gc.collect()

        for method in ['integrated_gradients', 'lrp', 'rise', 'extremal_perturbation']:
            job_started = time.time()
            complete, existing_meta = job_status[method]
            if complete:
                job_metas.append(existing_meta); report_job(existing_meta, True, job_started)
                continue
            model = build_model(checkpoint)
            loader = make_loader(preloaded, EXPLAIN_BATCH if method != 'rise' else 1)
            if method == 'integrated_gradients':
                maps, degenerate = explain_integrated_gradients(model, loader, len(frame), size)
            elif method == 'lrp':
                maps, degenerate = explain_lrp(model, loader, len(frame), size)
            elif method == 'rise':
                maps, degenerate = explain_rise(
                    model, loader, len(frame), size,
                    BASE_SEED + DATASETS.index(dataset) * 1000 + fold,
                )
            elif method == 'extremal_perturbation':
                maps, degenerate = explain_extremal(model, loader, len(frame), size)
            else:
                raise AssertionError(method)
            del model, loader; gc.collect(); torch.cuda.empty_cache()
            meta = write_h5_atomic(dataset, fold, method, frame, maps, degenerate, time.time() - job_started)
            generated_jobs += 1; job_metas.append(meta); report_job(meta, False, job_started)
            del maps, degenerate; gc.collect(); torch.cuda.empty_cache()
        del preloaded
        gc.collect()


# ======================================================================================
# 9. AUDIT SCIENTIFICO E COMPLETEZZA
# ======================================================================================
step(9, 10, 'Audit scientifico delle mappe e firma della catena')
audit_rows = []
artifact_hashes = {}
for dataset in datasets_for_xai:
    for fold in FOLDS:
        frame = cohort_runtime[(cohort_runtime.dataset == dataset) & (cohort_runtime.fold.astype(int) == fold)].copy()
        frame = frame.sort_values('sample_id').reset_index(drop=True)
        for method in EXPLAINERS:
            complete, meta = validate_complete(dataset, fold, method, frame)
            if not complete:
                raise RuntimeError(f'Artefatto finale mancante: {dataset} fold {fold} {method}')
            audit_rows.append({
                'dataset': dataset, 'fold': fold, 'method': method, 'n': meta['n'],
                'degenerate_count': meta['degenerate_count'],
                'degenerate_fraction': meta['degenerate_fraction'],
                'mean_saliency': meta['mean_saliency'], 'seconds': meta['seconds'],
                'xai_role': meta['xai_role'],
            })
            artifact_hashes[f'{dataset}/fold_{fold}/{method}'] = meta['h5_sha256']

audit = pd.DataFrame(audit_rows)
audit_path = explanation_root / 'baseline_explainer_audit.csv'
atomic_parquet(explanation_root / 'baseline_explainer_audit.parquet', audit)
audit.to_csv(audit_path, index=False)
say(f'Artefatti completi: {len(audit)}/{len(datasets_for_xai)*len(FOLDS)*len(EXPLAINERS)}')
for dataset in datasets_for_xai:
    for method in EXPLAINERS:
        part = audit[(audit.dataset == dataset) & (audit.method == method)]
        say(
            f'{dataset:12s} | {method:24s} | donor={int(part.n.sum()):4d} | '
            f'degenerate={int(part.degenerate_count.sum()):3d} ({part.degenerate_count.sum()/part.n.sum():.2%}) | '
            f'tempo={human_seconds(part.seconds.sum())}'
        )

chain = stable_hash({
    'config_sha256': sha256_file(config_path),
    'cohort_sha256': sha256_file(cohort_path),
    'checkpoint_hashes': checkpoint_hashes,
    'artifact_hashes': artifact_hashes,
})
say(f'Artifact chain SHA-256: {chain}')


# ======================================================================================
# 10. MANIFEST FINALE E HANDOFF AL NOTEBOOK 06R
# ======================================================================================
step(10, 10, 'Manifest finale e handoff al trasporto RegNet')
stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
manifest_path = run_root / f'baseline_explainer_manifest_{stamp}.json'
latest_path = run_root / 'latest_baseline_explainer_manifest.json'
manifest = {
    'notebook_id': NOTEBOOK_ID,
    'status': 'PASS',
    'protocol_version': PROTOCOL_VERSION,
    'completed_utc': utc_now(),
    'hardware': {'device': torch.cuda.get_device_name(0), 'gpu_required': True},
    'backbone': BACKBONE,
    'classifier_manifest_path': str(classifier_manifest_path),
    'classifier_manifest_sha256': sha256_file(classifier_manifest_path),
    'classifier_artifact_chain_sha256': classifier_chain,
    'config_path': str(config_path),
    'config_sha256': sha256_file(config_path),
    'cohort_path': str(cohort_path),
    'cohort_sha256': sha256_file(cohort_path),
    'cohort_rows': int(len(cohort_runtime)),
    'datasets_for_xai': datasets_for_xai,
    'dataset_roles': {d: DATASET_SETTINGS[d]['role'] for d in datasets_for_xai},
    'fold_performance': fold_performance,
    'explainers': EXPLAINERS,
    'completed_artifacts': int(len(audit)),
    'expected_artifacts': int(len(datasets_for_xai) * len(FOLDS) * len(EXPLAINERS)),
    'generated_this_run': int(generated_jobs),
    'resumed_this_run': int(len(audit) - generated_jobs),
    'artifact_chain_sha256': chain,
    'audit_csv': str(audit_path),
    'audit_csv_sha256': sha256_file(audit_path),
    'elapsed_seconds': float(time.time() - started),
    'io_strategy': '03R cache reuse + 8-thread atomic staging + one in-RAM preprocess per fold',
    'next_notebook': '06R_transport_refinement_regnet_x_400mf.py',
}
atomic_json(manifest_path, manifest)
atomic_json(latest_path, manifest)

say(f'Stato Notebook 04R:      PASS')
say(f'Donatori OOF:            {len(cohort_runtime):,}')
say(f'Explainer:                {len(EXPLAINERS)}')
say(f'Artefatti HDF5:           {len(audit)}/{manifest["expected_artifacts"]}')
say(f'Generati questa run:      {generated_jobs}')
say(f'Ripresi/verificati:       {len(audit)-generated_jobs}')
say(f'Backbone:                 {BACKBONE}')
say(f'Tempo totale:             {human_seconds(time.time()-started)}')
say(f'Manifest:                 {manifest_path}')
say(f'Latest manifest:          {latest_path}')
say(f'Prossimo notebook:        {manifest["next_notebook"]}')
say('\n' + LINE)
say('CPET_NOTEBOOK_04R_STATUS=PASS')
say('NEXT=NOTEBOOK_06R')
say(LINE)
say('Inviare in chat l’intero output testuale di questa cella prima di procedere.')
