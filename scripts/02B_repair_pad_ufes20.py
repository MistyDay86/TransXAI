# CPET — EXTRA CELL 02B — RECUPERO PAD-UFES-20 E CHIUSURA DEL GATE DATI
# Una sola cella Colab. CPU sufficiente; GPU non utilizzata. num_workers=0.
# Eseguire dopo 02_preprocessing_and_splits.py terminato con PARTIAL per il solo errore PAD 403.

import os, sys, re, json, time, shutil, hashlib, subprocess, importlib
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

PROJECT_ROOT = Path('/content/gdrive/MyDrive/Colab Notebooks/CPET')
BASE_SEED = 20260906
PROTOCOL_VERSION = '1.1.0'
NOTEBOOK_ID = '02b_repair_pad_ufes20'
NUM_WORKERS = 0
N_FOLDS = 5
MAX_SOURCE_GIB = 5.0
MAX_SOURCE_BYTES = int(MAX_SOURCE_GIB * 1024**3)
KAGGLE_HANDLE = 'mahdavi1202/skin-cancer'
EXPECTED_IMAGES = 2298
EXPECTED_PATIENTS = 1373
EXPECTED_LESIONS = 1641
EXPECTED_CLASS_COUNTS = {'ACK': 730, 'BCC': 845, 'MEL': 52, 'NEV': 244, 'SCC': 192, 'SEK': 235}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

started = time.time()

def banner(title):
    print('\n' + '=' * 108)
    print(title)
    print('=' * 108, flush=True)

def step(i, n, title):
    print(f'\n[{i}/{n}] {title}', flush=True)

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def human_bytes(n):
    n = float(n)
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if n < 1024 or unit == 'TiB':
            return f'{n:.2f} {unit}'
        n /= 1024

def sha256_file(path, chunk=4*1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def atomic_json(path, obj):
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n')

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def relative(path):
    return str(Path(path).resolve().relative_to(PROJECT_ROOT.resolve()))

def clean_id(value):
    return re.sub(r'\.(png|jpe?g|bmp|tiff?)$', '', str(value).strip(), flags=re.I)

def find_column(df, names):
    normalized = {re.sub(r'[^a-z0-9]+', '', str(c).lower()): c for c in df.columns}
    for name in names:
        key = re.sub(r'[^a-z0-9]+', '', name.lower())
        if key in normalized:
            return normalized[key]
    raise KeyError(f'Colonna richiesta non trovata fra {names}. Disponibili: {list(df.columns)}')

def json_safe(value):
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except Exception:
        pass
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (str, int, float)):
        return value
    return str(value)

TOTAL_STEPS = 9
banner('CPET — EXTRA CELL 02B — RECUPERO PAD-UFES-20 E CHIUSURA DEL GATE DATI')
print('Scopo: recuperare esclusivamente PAD-UFES-20 tramite archivio KaggleHub.')
print('Gli altri tre dataset NON vengono scaricati, decodificati o riauditati.')
print(f'Hardware: CPU | GPU: non necessaria | num_workers={NUM_WORKERS} | protocollo={PROTOCOL_VERSION}')
print(f'Limite sorgente: {MAX_SOURCE_GIB:.1f} GiB | attesi: {EXPECTED_IMAGES} immagini, {EXPECTED_PATIENTS} pazienti, {EXPECTED_LESIONS} lesioni.', flush=True)

# --------------------------------------------------------------------------------------------------
step(1, TOTAL_STEPS, 'Mount di Drive e verifica del PARTIAL precedente')
try:
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=False)
except Exception as exc:
    raise RuntimeError(f'Questa cella deve essere eseguita in Google Colab con Drive: {exc}')

latest_path = PROJECT_ROOT/'runs/preprocessing/latest_preprocessing_manifest.json'
required_configs = [PROJECT_ROOT/f'configs/{name}.json' for name in ['protocol','project','datasets','experiments']]
missing = [str(p) for p in required_configs + [latest_path] if not p.exists()]
if missing:
    raise RuntimeError('Prerequisiti mancanti: ' + ', '.join(missing))
previous = load_json(latest_path)
if previous.get('protocol_version') != PROTOCOL_VERSION:
    raise RuntimeError(f'Protocollo precedente inatteso: {previous.get("protocol_version")}.')
previous_errors = previous.get('errors', [])
non_pad_errors = [e for e in previous_errors if 'pad_ufes20 acquisition' not in e.lower()]
if non_pad_errors:
    raise RuntimeError('Il run precedente contiene errori diversi dal download PAD: ' + ' | '.join(non_pad_errors))
for dataset in ['bus_uclm', 'siim_acr', 'isic2016']:
    path = PROJECT_ROOT/f'data/manifests/{dataset}_manifest_v{PROTOCOL_VERSION}.parquet'
    if not path.exists():
        raise RuntimeError(f'Manca il manifest già validato di {dataset}: {path}')
print(f'Run precedente: {previous.get("status")} | dataset completi: {len(previous.get("dataset_reports", {}))}/4')
print('Unico problema ammesso: download PAD-UFES-20 bloccato con HTTP 403.')

# --------------------------------------------------------------------------------------------------
step(2, TOTAL_STEPS, 'Dipendenze CPU')
deps = {'numpy':'numpy', 'pandas':'pandas', 'cv2':'opencv-python-headless', 'tqdm':'tqdm',
        'sklearn':'scikit-learn', 'pyarrow':'pyarrow', 'kagglehub':'kagglehub'}
missing_packages = []
for module, package in deps.items():
    try:
        importlib.import_module(module)
    except Exception:
        missing_packages.append(package)
if missing_packages:
    print('Installazione:', ', '.join(sorted(set(missing_packages))), flush=True)
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--progress-bar', 'off'] + sorted(set(missing_packages)))
    if result.returncode:
        raise RuntimeError(f'pip terminato con codice {result.returncode}.')
    importlib.invalidate_caches()
else:
    print('Tutte le dipendenze sono disponibili.')
import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedGroupKFold
import kagglehub
print(f'Python={sys.version.split()[0]} | NumPy={np.__version__} | Pandas={pd.__version__} | OpenCV={cv2.__version__}')

# --------------------------------------------------------------------------------------------------
step(3, TOTAL_STEPS, 'Contratto della distribuzione Kaggle')
print(f'Dataset Kaggle: {KAGGLE_HANDLE}')
print('Catalogo: 2.299 file complessivi (2.298 PNG + metadata.csv), circa 4 GB.')
print(f'Il limite rigido di {MAX_SOURCE_GIB:.1f} GiB sarà verificato immediatamente sul payload restituito, prima della persistenza su Drive.')

# --------------------------------------------------------------------------------------------------
step(4, TOTAL_STEPS, 'Download persistente del solo PAD-UFES-20')
pad_root = PROJECT_ROOT/'data/raw/pad_ufes20'
snapshot_root = pad_root/'kaggle_snapshot'
snapshot_root.mkdir(parents=True, exist_ok=True)
partial_hf = pad_root/'hf_snapshot'
if partial_hf.exists():
    partial_count = sum(1 for p in partial_hf.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    print(f'Frammenti Hugging Face rilevati: {partial_count:,} immagini; vengono ignorati e non cancellati.')
print('KaggleHub scarica un unico archivio e ne gestisce automaticamente cache ed estrazione.', flush=True)
cache_path = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
source_bytes = sum(p.stat().st_size for p in cache_path.rglob('*') if p.is_file())
if source_bytes > MAX_SOURCE_BYTES:
    raise RuntimeError(f'Payload Kaggle rifiutato: {human_bytes(source_bytes)} > {MAX_SOURCE_GIB:.1f} GiB.')
source_images = sorted(p for p in cache_path.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
csv_candidates = sorted(cache_path.rglob('metadata.csv'))
if not csv_candidates:
    csv_candidates = sorted(cache_path.rglob('*.csv'), key=lambda p:p.stat().st_size, reverse=True)
if len(source_images) != EXPECTED_IMAGES or not csv_candidates:
    raise RuntimeError(f'Payload Kaggle inatteso: immagini={len(source_images)}/{EXPECTED_IMAGES}, CSV={len(csv_candidates)}.')
source_metadata = csv_candidates[0]
to_copy = source_images + [source_metadata]
for source in tqdm(to_copy, desc='Persistenza PAD-UFES-20 su Drive'):
    target = snapshot_root/source.relative_to(cache_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source,target)
distribution_version = cache_path.name if cache_path.name.isdigit() else str(cache_path)
distribution_fingerprint = hashlib.sha256('\n'.join(
    f'{p.relative_to(cache_path)}:{p.stat().st_size}' for p in sorted(to_copy)
).encode()).hexdigest()
local_images = sorted(p for p in snapshot_root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
metadata_matches = sorted(snapshot_root.rglob(source_metadata.name))
metadata_path = metadata_matches[0] if metadata_matches else snapshot_root/'metadata.csv'
if not metadata_path.exists() or len(local_images) != EXPECTED_IMAGES:
    raise RuntimeError(f'Download incompleto: metadata={metadata_path.exists()}, immagini={len(local_images)}/{EXPECTED_IMAGES}.')
marker = pad_root/'_ACQUIRED.json'
atomic_json(marker, {
    'completed_utc': utc_now(), 'distribution_channel': 'KaggleHub archive', 'repository': KAGGLE_HANDLE,
    'resolved_distribution_version': distribution_version, 'payload_bytes': source_bytes,
    'distribution_fingerprint_sha256':distribution_fingerprint, 'local_images': len(local_images),
    'upstream_dataset': 'PAD-UFES-20', 'upstream_doi': '10.17632/zr7vgbcyr2.1',
    'license': 'CC BY 4.0'
})
print(f'Download PAD completato: {len(local_images):,} immagini | payload={human_bytes(source_bytes)}')
print(f'Fingerprint distribuzione: {distribution_fingerprint}')

# --------------------------------------------------------------------------------------------------
step(5, TOTAL_STEPS, 'Join immagini-metadata e controllo clinico')
meta = pd.read_csv(metadata_path)
img_col = find_column(meta, ['img_id','image_id','image'])
patient_col = find_column(meta, ['patient_id','patient'])
lesion_col = find_column(meta, ['lesion_id','lesion'])
diag_col = find_column(meta, ['diagnostic','diagnosis','label'])
image_map = {clean_id(p.name): p for p in local_images}
if len(image_map) != EXPECTED_IMAGES:
    raise RuntimeError(f'ID immagine non univoci: {len(image_map)} ID per {len(local_images)} file.')

meta = meta.copy()
meta['_image_key'] = meta[img_col].map(clean_id)
if meta['_image_key'].duplicated().any():
    raise RuntimeError(f'Metadata con {int(meta["_image_key"].duplicated().sum())} image_id duplicati.')
missing_files = sorted(set(meta['_image_key']) - set(image_map))
orphan_files = sorted(set(image_map) - set(meta['_image_key']))
if missing_files or orphan_files or len(meta) != EXPECTED_IMAGES:
    raise RuntimeError(f'Join non 1:1: righe={len(meta)}, missing_file={len(missing_files)}, orphan_file={len(orphan_files)}.')

meta['_diag'] = meta[diag_col].astype(str).str.upper().str.strip()
class_counts = {str(k): int(v) for k, v in meta['_diag'].value_counts().sort_index().items()}
if class_counts != EXPECTED_CLASS_COUNTS:
    raise RuntimeError(f'Distribuzione diagnostica inattesa: {class_counts}; attesa {EXPECTED_CLASS_COUNTS}.')
patients = int(meta[patient_col].nunique(dropna=True))
lesions = int(meta[lesion_col].nunique(dropna=True))
if patients != EXPECTED_PATIENTS or lesions != EXPECTED_LESIONS:
    raise RuntimeError(f'Cardinalità cliniche inattese: pazienti={patients}/{EXPECTED_PATIENTS}, lesioni={lesions}/{EXPECTED_LESIONS}.')
if meta[patient_col].isna().any() or meta[lesion_col].isna().any():
    raise RuntimeError('Patient ID o lesion ID mancanti: impossibile garantire split patient-safe.')
print(f'Join 1:1: PASS | pazienti={patients:,} | lesioni={lesions:,}')
print('Classi:', ', '.join(f'{k}={v}' for k,v in class_counts.items()))
print(f'Task binario: cancer BCC+MEL+SCC={sum(class_counts[k] for k in ["BCC","MEL","SCC"]):,}; non-cancer={sum(class_counts[k] for k in ["ACK","NEV","SEK"]):,}.')

# --------------------------------------------------------------------------------------------------
step(6, TOTAL_STEPS, 'Audit immagini e fingerprint')
def phash64(gray):
    small = cv2.resize(gray, (32,32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coeff = cv2.dct(small)[:8,:8]
    median = float(np.median(coeff.flatten()[1:]))
    bits = (coeff.flatten() >= median).astype(np.uint8)
    return f'{int("".join(str(int(x)) for x in bits), 2):016x}'

cancers = {'BCC','MEL','SCC'}
records = []
invalid = []
for _, row in tqdm(meta.iterrows(), total=len(meta), desc='Audit PAD-UFES-20'):
    sid = row['_image_key']
    path = image_map[sid]
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        invalid.append((sid, 'decode_failed'))
        continue
    h, w = raw.shape[:2]
    if min(h,w) < 64:
        invalid.append((sid, 'image_too_small'))
        continue
    channels = 1 if raw.ndim == 2 else raw.shape[2]
    gray = raw if raw.ndim == 2 else cv2.cvtColor(raw[:,:,:3], cv2.COLOR_BGR2GRAY)
    extra = {str(k): json_safe(v) for k,v in row.items() if k not in {img_col,patient_col,lesion_col,diag_col,'_image_key','_diag'}}
    records.append({
        'dataset':'pad_ufes20', 'sample_id':sid, 'image_path':relative(path), 'mask_path':'',
        'label':int(row['_diag'] in cancers), 'class_name':row['_diag'].lower(),
        'patient_id':str(row[patient_col]), 'lesion_id':str(row[lesion_col]),
        'group_id_original':str(row[patient_col]), 'inference_scope':'patient',
        'metadata_json':json.dumps(extra, ensure_ascii=False, sort_keys=True),
        'height':int(h), 'width':int(w), 'channels':int(channels),
        'image_sha256':sha256_file(path), 'phash64':phash64(gray),
        'mask_valid':False, 'mask_nonempty':False, 'mask_area_fraction':np.nan,
        'eligible':True, 'exclusion_reason':''
    })
if invalid:
    raise RuntimeError(f'Audit immagini fallito per {len(invalid)} file; primi casi: {invalid[:5]}')
df = pd.DataFrame(records)
print(f'Audit completato: {len(df):,}/{EXPECTED_IMAGES:,} immagini valide.')

# --------------------------------------------------------------------------------------------------
step(7, TOTAL_STEPS, 'Deduplicazione e cinque split patient-safe')
parent = list(range(len(df)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(a,b):
    a,b = find(a),find(b)
    if a != b:
        parent[b] = a

exact = defaultdict(list)
for i, value in enumerate(df['image_sha256']):
    exact[value].append(i)
for indices in exact.values():
    for j in indices[1:]:
        union(indices[0],j)

# BK-tree per tutte le coppie pHash con distanza di Hamming <=4.
tree = None
for i, value_hex in enumerate(df['phash64']):
    value = int(value_hex,16)
    if tree is None:
        tree = {'v':value,'idx':[i],'children':{}}
        continue
    node = tree
    while True:
        d = (value ^ node['v']).bit_count()
        if d == 0:
            for j in node['idx']: union(i,j)
            node['idx'].append(i)
            break
        if d not in node['children']:
            node['children'][d] = {'v':value,'idx':[i],'children':{}}
            break
        node = node['children'][d]
    stack = [tree]
    while stack:
        node = stack.pop()
        d = (value ^ node['v']).bit_count()
        if d <= 4:
            for j in node['idx']:
                if i != j: union(i,j)
        stack.extend(child for edge,child in node['children'].items() if d-4 <= edge <= d+4)

clusters = defaultdict(list)
for i in range(len(df)):
    clusters[find(i)].append(i)
conflicts = []
for indices in clusters.values():
    if len({int(df.iloc[i]['label']) for i in indices}) > 1:
        conflicts.extend(indices)
if conflicts:
    df.loc[sorted(set(conflicts)), 'eligible'] = False
    df.loc[sorted(set(conflicts)), 'exclusion_reason'] = 'near_duplicate_cluster_label_conflict'

# Componenti di pazienti connesse da duplicati: una componente non può attraversare fold.
patient_graph = defaultdict(set)
for indices in clusters.values():
    pids = {str(df.iloc[i]['patient_id']) for i in indices}
    for pid in pids:
        patient_graph[pid].update(pids)
seen, patient_component = set(), {}
for pid in sorted(patient_graph):
    if pid in seen: continue
    stack, members = [pid], []
    while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x); members.append(x); stack.extend(patient_graph[x]-seen)
    component_id = hashlib.sha1('|'.join(sorted(members)).encode()).hexdigest()[:16]
    for x in members: patient_component[x] = component_id
df['duplicate_cluster'] = [f'dup_{find(i):06d}' for i in range(len(df))]
df['group_id_effective'] = [patient_component[str(pid)] for pid in df['patient_id']]
work = df[df['eligible']].reset_index(drop=True).copy()

best = None
for trial in range(64):
    seed = BASE_SEED + trial
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    assignment = np.full(len(work), -1, dtype=int)
    try:
        for fold,(_,test_idx) in enumerate(splitter.split(work, work['label'], groups=work['group_id_effective'])):
            assignment[test_idx] = fold
    except ValueError:
        continue
    if np.any(assignment < 0):
        continue
    rates = [float(work.loc[assignment==f,'label'].mean()) for f in range(N_FOLDS)]
    sizes = [int(np.sum(assignment==f)) for f in range(N_FOLDS)]
    if any(work.loc[assignment==f,'label'].nunique() != 2 for f in range(N_FOLDS)):
        continue
    score = float(np.std(rates) + .25*np.std(np.array(sizes)/len(work)))
    if best is None or score < best[0]:
        best = (score,seed,assignment.copy(),rates,sizes)
if best is None:
    raise RuntimeError('Impossibile costruire cinque fold stratificati patient-safe.')
_,split_seed,assignment,rates,sizes = best
work['outer_fold'] = assignment
for test_fold in range(N_FOLDS):
    val_fold = (test_fold+1)%N_FOLDS
    work[f'fold_{test_fold}_role'] = np.where(work['outer_fold']==test_fold,'test',np.where(work['outer_fold']==val_fold,'val','train'))
    role_sets = {role:set(work.loc[work[f'fold_{test_fold}_role']==role,'group_id_effective']) for role in ['train','val','test']}
    if role_sets['train']&role_sets['val'] or role_sets['train']&role_sets['test'] or role_sets['val']&role_sets['test']:
        raise RuntimeError(f'Leakage di pazienti nel fold {test_fold}.')

report = {
    'dataset':'pad_ufes20', 'eligible_samples':int(len(work)),
    'groups':int(work['group_id_effective'].nunique()), 'original_patients':int(work['patient_id'].nunique()),
    'lesions':int(work['lesion_id'].nunique()), 'positive_rate':float(work['label'].mean()),
    'split_seed':int(split_seed), 'fold_sizes':sizes, 'fold_positive_rates':rates,
    'duplicate_links':int(sum(len(v)-1 for v in clusters.values() if len(v)>1)),
    'excluded_duplicate_label_conflicts':int(len(df)-len(work)), 'inference_scope':'patient'
}
print(f'PAD finale: n={len(work):,} | pazienti={work["patient_id"].nunique():,} | gruppi effettivi={report["groups"]:,} | pos={report["positive_rate"]:.3f}')
print(f'Fold: {sizes} | positive rates: {[round(x,4) for x in rates]} | seed={split_seed}')

# --------------------------------------------------------------------------------------------------
step(8, TOTAL_STEPS, 'Scrittura manifest PAD e aggiornamento aggregato')
manifest_dir = PROJECT_ROOT/'data/manifests'
split_dir = PROJECT_ROOT/'data/splits'
manifest_dir.mkdir(parents=True, exist_ok=True)
split_dir.mkdir(parents=True, exist_ok=True)
pad_csv = manifest_dir/f'pad_ufes20_manifest_v{PROTOCOL_VERSION}.csv'
pad_parquet = manifest_dir/f'pad_ufes20_manifest_v{PROTOCOL_VERSION}.parquet'
work.to_csv(pad_csv,index=False)
work.to_parquet(pad_parquet,index=False)

frames = []
for dataset in ['bus_uclm','siim_acr','isic2016','pad_ufes20']:
    path = manifest_dir/f'{dataset}_manifest_v{PROTOCOL_VERSION}.parquet'
    if not path.exists():
        raise RuntimeError(f'Impossibile creare aggregato: manca {path}.')
    frames.append(pd.read_parquet(path))
combined = pd.concat(frames,ignore_index=True,sort=False)
combined_csv = split_dir/f'all_datasets_splits_v{PROTOCOL_VERSION}.csv'
combined_parquet = split_dir/f'all_datasets_splits_v{PROTOCOL_VERSION}.parquet'
combined.to_csv(combined_csv,index=False)
combined.to_parquet(combined_parquet,index=False)

# Registrazione non distruttiva della sostituzione del solo canale di distribuzione.
amendment_path = PROJECT_ROOT/'configs/amendments/CPET-A02_pad_distribution_repair.json'
amendment = {
    'amendment_id':'CPET-A02', 'protocol_version':PROTOCOL_VERSION,
    'type':'data_distribution_channel_repair', 'scientific_design_changed':False,
    'reason':'The direct Mendeley cache URL returned HTTP 403 before any PAD bytes were acquired.',
    'replacement_repository':KAGGLE_HANDLE, 'resolved_distribution_version':distribution_version,
    'distribution_fingerprint_sha256':distribution_fingerprint,
    'upstream_dataset':'PAD-UFES-20', 'upstream_doi':'10.17632/zr7vgbcyr2.1',
    'validation':{'images':EXPECTED_IMAGES,'patients':EXPECTED_PATIENTS,'lesions':EXPECTED_LESIONS,'class_counts':EXPECTED_CLASS_COUNTS},
    'created_utc':utc_now()
}
if amendment_path.exists():
    amendment['created_utc'] = load_json(amendment_path).get('created_utc', amendment['created_utc'])
atomic_json(amendment_path,amendment)
datasets_cfg_path = PROJECT_ROOT/'configs/datasets.json'
datasets_cfg = load_json(datasets_cfg_path)
datasets_cfg['pad_ufes20']['source'] = f'Kaggle {KAGGLE_HANDLE}, distribution {distribution_version}; content-validated mirror of DOI 10.17632/zr7vgbcyr2.1'
datasets_cfg['pad_ufes20']['distribution_fingerprint_sha256'] = distribution_fingerprint
atomic_json(datasets_cfg_path,datasets_cfg)
print(f'Manifest PAD: {pad_parquet}')
print(f'Aggregato: {combined_parquet} | righe={len(combined):,}')

# --------------------------------------------------------------------------------------------------
step(9, TOTAL_STEPS, 'Chiusura del gate 02 e manifest PASS')
dataset_reports = dict(previous.get('dataset_reports',{}))
dataset_reports['pad_ufes20'] = report
if set(dataset_reports) != {'bus_uclm','siim_acr','isic2016','pad_ufes20'}:
    raise RuntimeError(f'Report dataset incompleti: {sorted(dataset_reports)}')
artifacts = [PROJECT_ROOT/'configs/protocol.json', datasets_cfg_path, amendment_path,
             pad_csv,pad_parquet,combined_csv,combined_parquet]
hashes = {relative(p):sha256_file(p) for p in artifacts}
chain = hashlib.sha256(''.join(hashes[k] for k in sorted(hashes)).encode()).hexdigest()
manifest = {
    'notebook_id':NOTEBOOK_ID, 'status':'PASS', 'protocol_version':PROTOCOL_VERSION,
    'repairs_previous_artifact_chain_sha256':previous.get('artifact_chain_sha256'),
    'previous_status':previous.get('status'), 'completed_utc':utc_now(), 'project_root':str(PROJECT_ROOT),
    'base_seed':BASE_SEED, 'num_workers':NUM_WORKERS, 'hardware_required':'CPU',
    'max_source_dataset_gib':MAX_SOURCE_GIB, 'pad_distribution_version':distribution_version,
    'pad_distribution_fingerprint_sha256':distribution_fingerprint,
    'dataset_reports':dataset_reports, 'artifact_hashes':hashes, 'artifact_chain_sha256':chain,
    'warnings':[
        'siim_acr: case-level validation; not counted as patient-level replication.',
        'isic2016: case-level validation; not counted as patient-level replication.'
    ],
    'errors':[]
}
run_dir = PROJECT_ROOT/'runs/preprocessing'
stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
run_path = run_dir/f'preprocessing_repair_pad_{stamp}.json'
atomic_json(run_path,manifest)
atomic_json(latest_path,manifest)
print('Stato preprocessing:       PASS')
print('Dataset completi:          4/4')
print('Validazioni patient-level: 2 (BUS-UCLM, PAD-UFES-20)')
print('Repliche case-level:       2 (SIIM-ACR, ISIC 2016)')
print('GPU necessaria:            NO')
print('Errori:                    0')
print(f'Artifact chain SHA-256:    {chain}')
print(f'Tempo cella extra:         {(time.time()-started)/60:.2f} min')
print(f'Manifest PASS:             {run_path}')
print('Prossimo file:             03_train_classifiers.py — GPU richiesta')
banner('CPET_NOTEBOOK_02B_STATUS=PASS')
print('Inviare in chat l’intero output testuale prima di procedere.',flush=True)