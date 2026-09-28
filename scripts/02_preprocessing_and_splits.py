# CPET — NOTEBOOK 02/08 — ACQUISIZIONE, AUDIT, MANIFEST E SPLIT
# Una sola cella. Eseguire in Google Colab e inviare in chat l'intero output testuale.
# Hardware: CPU sufficiente; GPU non utilizzata. num_workers=0.

import os, sys, re, json, time, shutil, hashlib, subprocess, importlib, zipfile, warnings, traceback
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter

# ======================================================================================
# CONFIGURAZIONE BLOCCATA DEL PROGETTO
# ======================================================================================
PROJECT_ROOT = Path('/content/gdrive/MyDrive/Colab Notebooks/CPET')
BASE_SEED = 20260906
NOTEBOOK_ID = '02_preprocessing_and_splits'
PROTOCOL_FROM = '1.0.0'
PROTOCOL_VERSION = '1.1.0'
NUM_WORKERS = 0
N_OUTER_FOLDS = 5
MAX_SOURCE_GIB = 5.0
MAX_SOURCE_BYTES = int(MAX_SOURCE_GIB * 1024**3)
MAX_EXTRACTED_GIB_PER_DATASET = 18.0
DOWNLOAD_CHUNK = 8 * 1024 * 1024
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

started = time.time()
warnings_list, errors_list = [], []
dataset_reports = {}

def banner(title):
    print('\n' + '=' * 108)
    print(title)
    print('=' * 108, flush=True)

def step(i, n, text):
    print(f'\n[{i}/{n}] {text}', flush=True)

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def human_bytes(n):
    n = float(n)
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if n < 1024 or unit == 'TiB':
            return f'{n:.2f} {unit}'
        n /= 1024

def sha256_file(path, chunk_size=4 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(chunk_size)
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

def tree_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())

def relative(path):
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)

def safe_extract(zip_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ('.extracted_' + Path(zip_path).name + '.json')
    archive_sha = sha256_file(zip_path)
    if marker.exists():
        old = load_json(marker)
        if old.get('archive_sha256') == archive_sha:
            print(f'  Estrazione già valida: {Path(zip_path).name}')
            return old
    with zipfile.ZipFile(zip_path, 'r') as zf:
        infos = zf.infolist()
        total_uncompressed = sum(x.file_size for x in infos)
        limit = int(MAX_EXTRACTED_GIB_PER_DATASET * 1024**3)
        if total_uncompressed > limit:
            raise RuntimeError(f'Archivio {Path(zip_path).name}: contenuto dichiarato {human_bytes(total_uncompressed)} oltre il limite di sicurezza {MAX_EXTRACTED_GIB_PER_DATASET:.1f} GiB.')
        root = destination.resolve()
        for info in infos:
            target = (destination / info.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f'Path traversal rilevato nell\'archivio: {info.filename}')
        print(f'  Estrazione {Path(zip_path).name}: {len(infos):,} elementi, {human_bytes(total_uncompressed)} non compressi...', flush=True)
        for idx, info in enumerate(infos, 1):
            zf.extract(info, destination)
            if idx == 1 or idx % 1000 == 0 or idx == len(infos):
                print(f'    estratti {idx:,}/{len(infos):,}', flush=True)
    report = {'archive': str(zip_path), 'archive_sha256': archive_sha, 'members': len(infos),
              'uncompressed_bytes': total_uncompressed, 'completed_utc': utc_now()}
    atomic_json(marker, report)
    return report

def download_http(url, destination, expected_sha256=None):
    import requests
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected_sha256 and sha256_file(destination) == expected_sha256:
        print(f'  Archivio già presente e checksum valido: {destination.name} ({human_bytes(destination.stat().st_size)})')
        return destination
    remote_size = None
    try:
        head = requests.head(url, allow_redirects=True, timeout=60)
        if head.ok and head.headers.get('content-length'):
            remote_size = int(head.headers['content-length'])
            if remote_size > MAX_SOURCE_BYTES:
                raise RuntimeError(f'Fonte rifiutata prima del download: {human_bytes(remote_size)} > {MAX_SOURCE_GIB:.1f} GiB.')
    except RuntimeError:
        raise
    except Exception as exc:
        print(f'  HEAD non conclusivo ({type(exc).__name__}); il limite sarà verificato durante lo streaming.')
    if destination.exists() and remote_size is not None and destination.stat().st_size == remote_size:
        print(f'  Archivio già completo per Content-Length: {destination.name} ({human_bytes(remote_size)})')
        return destination
    existing = destination.stat().st_size if destination.exists() else 0
    headers = {'Range': f'bytes={existing}-'} if existing else {}
    mode = 'ab' if existing else 'wb'
    with requests.get(url, stream=True, allow_redirects=True, headers=headers, timeout=(60, 300)) as response:
        response.raise_for_status()
        if existing and response.status_code != 206:
            print('  Il server non supporta resume: riavvio controllato del singolo archivio.')
            existing, mode = 0, 'wb'
        advertised = response.headers.get('content-length')
        total = existing + int(advertised) if advertised else remote_size
        if total is not None and total > MAX_SOURCE_BYTES:
            raise RuntimeError(f'Fonte rifiutata: {human_bytes(total)} > {MAX_SOURCE_GIB:.1f} GiB.')
        print(f'  Download {destination.name}: ripresa da {human_bytes(existing)}; totale atteso {human_bytes(total) if total else "non dichiarato"}', flush=True)
        written = existing
        last_print = time.time()
        with open(destination, mode) as f:
            for chunk in response.iter_content(DOWNLOAD_CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if written > MAX_SOURCE_BYTES:
                    raise RuntimeError(f'Download interrotto: superato il limite di {MAX_SOURCE_GIB:.1f} GiB.')
                if time.time() - last_print >= 10:
                    pct = f'{100*written/total:5.1f}%' if total else ' n/d '
                    print(f'    {pct} | {human_bytes(written)} ricevuti', flush=True)
                    last_print = time.time()
            f.flush()
            os.fsync(f.fileno())
    if expected_sha256:
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise RuntimeError(f'Checksum errato per {destination.name}: atteso {expected_sha256}, ottenuto {actual}.')
    print(f'  Download completato: {destination.name} ({human_bytes(destination.stat().st_size)})')
    return destination

def all_images(root):
    return sorted(p for p in Path(root).rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

def find_column(df, candidates, required=True):
    normalized = {re.sub(r'[^a-z0-9]+', '', str(c).lower()): c for c in df.columns}
    for candidate in candidates:
        key = re.sub(r'[^a-z0-9]+', '', candidate.lower())
        if key in normalized:
            return normalized[key]
    if required:
        raise KeyError(f'Nessuna colonna fra {candidates}; colonne disponibili: {list(df.columns)}')
    return None

def clean_id(value):
    value = str(value).strip()
    return re.sub(r'\.(png|jpe?g|bmp|tiff?)$', '', value, flags=re.I)

def json_safe(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (str, int, float)):
        return value
    return str(value)

def find_by_clean_stem(paths):
    out = {}
    for p in paths:
        stem = clean_id(p.name)
        stem = re.sub(r'(_segmentation|_mask|mask)$', '', stem, flags=re.I)
        if stem not in out:
            out[stem] = p
    return out

TOTAL_STEPS = 12
banner('CPET — NOTEBOOK 02/08 — ACQUISIZIONE, AUDIT, MANIFEST E SPLIT')
print('Obiettivo: acquisire quattro dataset compatti, verificarli e produrre split leakage-safe.')
print('Hardware richiesto: CPU. La GPU non viene utilizzata in questo notebook.')
print(f'Protocollo: {PROTOCOL_FROM} -> {PROTOCOL_VERSION} | Seed: {BASE_SEED} | num_workers: {NUM_WORKERS}')
print(f'Vincolo storage: ogni sorgente scaricata deve essere <= {MAX_SOURCE_GIB:.1f} GiB.', flush=True)

# --------------------------------------------------------------------------------------
step(1, TOTAL_STEPS, 'Mount di Drive, gate del Notebook 01 e spazio disponibile')
try:
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=False)
except Exception as exc:
    raise RuntimeError(f'Questo file deve essere eseguito in Colab con Drive montabile: {exc}')
required = [PROJECT_ROOT / 'configs' / name for name in ['protocol.json', 'project.json', 'datasets.json', 'experiments.json']]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise RuntimeError('Notebook 01 non completato; file mancanti: ' + ', '.join(missing))
setup_manifest_path = PROJECT_ROOT / 'runs/setup/latest_setup_manifest.json'
if not setup_manifest_path.exists():
    raise RuntimeError('Manca runs/setup/latest_setup_manifest.json: non è possibile verificare il gate del Notebook 01.')
setup_manifest = load_json(setup_manifest_path)
if setup_manifest.get('errors'):
    raise RuntimeError('Il manifest del Notebook 01 contiene errori; interrompo prima di modificare il protocollo.')
disk = shutil.disk_usage(PROJECT_ROOT)
print(f'Root: {PROJECT_ROOT}')
print(f'Gate Notebook 01: PASS | protocol chain={setup_manifest.get("protocol_chain_sha256", "n/d")[:16]}...')
print(f'Spazio libero su Drive: {disk.free/1024**3:.2f} GiB')
if disk.free < 20 * 1024**3:
    raise RuntimeError('Servono almeno 20 GiB liberi per archivi, estrazione e manifest.')

# --------------------------------------------------------------------------------------
step(2, TOTAL_STEPS, 'Dipendenze CPU e controllo ambiente')
dependency_map = {'numpy': 'numpy', 'pandas': 'pandas', 'sklearn': 'scikit-learn', 'cv2': 'opencv-python-headless',
                  'PIL': 'pillow', 'requests': 'requests', 'tqdm': 'tqdm', 'pyarrow': 'pyarrow',
                  'datasets': 'datasets', 'huggingface_hub': 'huggingface-hub', 'kagglehub': 'kagglehub'}
missing_packages = []
for module_name, package_name in dependency_map.items():
    try:
        importlib.import_module(module_name)
    except Exception:
        missing_packages.append(package_name)
if missing_packages:
    print('Installazione pacchetti mancanti:', ', '.join(sorted(set(missing_packages))), flush=True)
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--progress-bar', 'off'] + sorted(set(missing_packages)))
    if result.returncode != 0:
        raise RuntimeError(f'pip terminato con codice {result.returncode}.')
    importlib.invalidate_caches()
else:
    print('Tutte le dipendenze sono già disponibili.')
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedGroupKFold
print(f'Python={sys.version.split()[0]} | NumPy={np.__version__} | Pandas={pd.__version__} | OpenCV={cv2.__version__}')
print('GPU: non interrogata e non necessaria.')

# --------------------------------------------------------------------------------------
step(3, TOTAL_STEPS, 'Emendamento preregistrato 1.1.0: dataset tutti <=5 GiB')
config_dir = PROJECT_ROOT / 'configs'
history_dir = config_dir / 'history'
amendment_dir = config_dir / 'amendments'
history_dir.mkdir(parents=True, exist_ok=True)
amendment_dir.mkdir(parents=True, exist_ok=True)
old_protocol = load_json(config_dir / 'protocol.json')
current_version = old_protocol.get('protocol_version')
if current_version not in {PROTOCOL_FROM, PROTOCOL_VERSION}:
    raise RuntimeError(f'Versione protocollo inattesa: {current_version}; attese {PROTOCOL_FROM} o {PROTOCOL_VERSION}.')
for name in ['protocol', 'project', 'datasets', 'experiments']:
    source = config_dir / f'{name}.json'
    backup = history_dir / f'{name}_v{PROTOCOL_FROM}.json'
    if not backup.exists() and current_version == PROTOCOL_FROM:
        shutil.copy2(source, backup)

amendment = {
    'amendment_id': 'CPET-A01', 'from_version': PROTOCOL_FROM, 'to_version': PROTOCOL_VERSION,
    'decision_time': 'before_dataset_acquisition_and_before_any_model_training',
    'created_utc': utc_now(), 'reason': 'Constrain each source dataset to at most 5 GiB for feasible Colab-scale experimentation.',
    'changes': [
        {'remove': 'isic2017', 'add': 'isic2016', 'reason': 'Official masked replication below the source-size cap.'},
        {'remove': 'isic2020', 'add': 'pad_ufes20', 'reason': 'Patient-level metadata and clinically relevant subgroups below the cap.'},
        {'add_constraint': 'max_source_dataset_gib', 'value': MAX_SOURCE_GIB},
        {'clarification': 'inference_scope', 'value': 'BUS-UCLM and PAD-UFES-20 support patient-level inference; compact SIIM-ACR and ISIC 2016 releases support case-level replication only.'}
    ]
}
amendment_path = amendment_dir / 'CPET-A01_protocol_1.1.0.json'
if amendment_path.exists():
    existing_amendment = load_json(amendment_path)
    amendment['created_utc'] = existing_amendment.get('created_utc', amendment['created_utc'])
atomic_json(amendment_path, amendment)

protocol = old_protocol
protocol['protocol_version'] = PROTOCOL_VERSION
protocol['amended_from'] = PROTOCOL_FROM
protocol['active_amendments'] = ['CPET-A01']
protocol.setdefault('data_policy', {})['max_source_dataset_gib'] = MAX_SOURCE_GIB
protocol['data_policy']['inference_scope_by_dataset'] = {
    'bus_uclm': 'patient', 'pad_ufes20': 'patient', 'siim_acr': 'case_proxy', 'isic2016': 'case_proxy'
}
protocol['primary_unit_of_inference'] = 'patient where patient identifiers are available; otherwise case-level robustness only'
project_cfg = load_json(config_dir / 'project.json')
project_cfg['protocol_version'] = PROTOCOL_VERSION
project_cfg['num_workers'] = NUM_WORKERS
datasets_cfg = {
    'bus_uclm': {'role': 'development_smoke', 'task': 'benign_vs_malignant', 'group_key': 'patient_id',
                 'inference_scope': 'patient', 'has_manual_masks': True, 'input_size': 256,
                 'source': 'MedOtter/BUS-UCLM mirror of DOI 10.17632/7fvgj4jsp7.3', 'expected_images': 683},
    'siim_acr': {'role': 'primary_large_scale', 'task': 'pneumothorax_vs_negative', 'group_key': 'image_case_id',
                 'inference_scope': 'case_proxy', 'has_manual_masks': True, 'input_size': 320,
                 'source': 'Kaggle vbookshelf/pneumothorax-chest-xray-images-and-masks', 'expected_images': 12047},
    'isic2016': {'role': 'masked_replication', 'task': 'malignant_vs_benign', 'group_key': 'image_case_id',
                 'inference_scope': 'case_proxy', 'has_manual_masks': True, 'input_size': 320,
                 'source': 'Official ISIC 2016 Task 3B', 'expected_images': 900},
    'pad_ufes20': {'role': 'external_patient_subgroup', 'task': 'skin_cancer_vs_non_cancer', 'group_key': 'patient_id',
                   'secondary_group_key': 'lesion_id', 'inference_scope': 'patient', 'has_manual_masks': False,
                   'geometry_mask_source': 'segmenter validated independently; not used as ground truth', 'input_size': 320,
                   'source': 'Mendeley DOI 10.17632/zr7vgbcyr2.1', 'expected_images': 2298}
}
experiments_cfg = load_json(config_dir / 'experiments.json')
experiments_cfg['protocol_version'] = PROTOCOL_VERSION
experiments_cfg['masked_replication_dataset'] = 'isic2016'
experiments_cfg['patient_subgroup_dataset'] = 'pad_ufes20'
for path, obj in [(config_dir/'protocol.json', protocol), (config_dir/'project.json', project_cfg),
                  (config_dir/'datasets.json', datasets_cfg), (config_dir/'experiments.json', experiments_cfg)]:
    atomic_json(path, obj)
print('CPET-A01 registrato prima dell’acquisizione: ISIC 2016 e PAD-UFES-20 sostituiscono ISIC 2017/2020.')
print('Validità dichiarata: patient-level su BUS-UCLM/PAD-UFES-20; case-level su SIIM-ACR/ISIC 2016.')

# --------------------------------------------------------------------------------------
step(4, TOTAL_STEPS, 'Acquisizione BUS-UCLM (circa 335 MiB, patient-level)')
raw_root = PROJECT_ROOT / 'data/raw'
raw_root.mkdir(parents=True, exist_ok=True)
bus_root = raw_root / 'bus_uclm'
bus_marker = bus_root / '_ACQUIRED.json'
try:
    if not bus_marker.exists():
        from datasets import load_dataset
        bus_root.mkdir(parents=True, exist_ok=True)
        (bus_root/'images').mkdir(exist_ok=True)
        (bus_root/'masks').mkdir(exist_ok=True)
        print('  Download dello snapshot Hugging Face con cache temporanea di Colab...', flush=True)
        ds = load_dataset('MedOtter/BUS-UCLM', split='train')
        rows = []
        for i, row in enumerate(tqdm(ds, total=len(ds), desc='  Materializzazione BUS-UCLM')):
            image_id = clean_id(row.get('image_id', f'bus_{i:04d}'))
            image_path = bus_root/'images'/f'{image_id}.png'
            mask_path = bus_root/'masks'/f'{image_id}.png'
            if not image_path.exists():
                row['image'].convert('RGB').save(image_path, compress_level=3)
            mask_obj = row.get('mask')
            if mask_obj is not None and not mask_path.exists():
                mask_obj.convert('L').save(mask_path, compress_level=3)
            rows.append({k: row.get(k) for k in ['image_id','patient_id','class_label','has_doppler','has_marks','has_combined']})
        pd.DataFrame(rows).to_csv(bus_root/'source_metadata.csv', index=False)
        if len(rows) != datasets_cfg['bus_uclm']['expected_images']:
            raise RuntimeError(f'BUS-UCLM: attese 683 righe, trovate {len(rows)}.')
        atomic_json(bus_marker, {'completed_utc': utc_now(), 'rows': len(rows), 'bytes': tree_size(bus_root),
                                 'source': datasets_cfg['bus_uclm']['source']})
    print(f'  BUS-UCLM pronto: {human_bytes(tree_size(bus_root))}')
except Exception as exc:
    errors_list.append(f'bus_uclm acquisition: {type(exc).__name__}: {exc}')
    print(f'  ACQUISIZIONE BUS-UCLM NON COMPLETATA: {type(exc).__name__}: {exc}')

# --------------------------------------------------------------------------------------
step(5, TOTAL_STEPS, 'Acquisizione SIIM-ACR PNG+mask (circa 1.4–3 GiB, case-level)')
siim_root = raw_root / 'siim_acr'
siim_marker = siim_root / '_ACQUIRED.json'
try:
    if not siim_marker.exists():
        import kagglehub
        print('  Download pubblico KaggleHub; non vengono stampate né lette credenziali...', flush=True)
        cache_path = Path(kagglehub.dataset_download('vbookshelf/pneumothorax-chest-xray-images-and-masks'))
        downloaded_bytes = tree_size(cache_path)
        if downloaded_bytes > MAX_SOURCE_BYTES:
            raise RuntimeError(f'SIIM-ACR mirror pesa {human_bytes(downloaded_bytes)}, oltre il limite.')
        siim_root.mkdir(parents=True, exist_ok=True)
        copied = 0
        files = [p for p in cache_path.rglob('*') if p.is_file()]
        for source in tqdm(files, desc='  Persistenza SIIM-ACR su Drive'):
            target = siim_root / source.relative_to(cache_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != source.stat().st_size:
                shutil.copy2(source, target)
            copied += 1
        atomic_json(siim_marker, {'completed_utc': utc_now(), 'files': copied, 'source_bytes': downloaded_bytes,
                                  'source': datasets_cfg['siim_acr']['source']})
    print(f'  SIIM-ACR pronto: {human_bytes(tree_size(siim_root))}')
except Exception as exc:
    errors_list.append(f'siim_acr acquisition: {type(exc).__name__}: {exc}')
    print(f'  ACQUISIZIONE SIIM-ACR NON COMPLETATA: {type(exc).__name__}: {exc}')

# --------------------------------------------------------------------------------------
step(6, TOTAL_STEPS, 'Acquisizione ISIC 2016 Task 3B ufficiale (circa 608 MiB, case-level)')
isic_root = raw_root / 'isic2016'
isic_marker = isic_root / '_ACQUIRED.json'
try:
    if not isic_marker.exists():
        isic_root.mkdir(parents=True, exist_ok=True)
        urls = {
            'ISBI2016_ISIC_Part3B_Training_Data.zip': 'https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part3B_Training_Data.zip',
            'ISBI2016_ISIC_Part3B_Training_GroundTruth.csv': 'https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv'
        }
        for filename, url in urls.items():
            path = download_http(url, isic_root/filename)
            if path.suffix.lower() == '.zip':
                safe_extract(path, isic_root/'extracted')
        atomic_json(isic_marker, {'completed_utc': utc_now(), 'source_bytes': sum((isic_root/k).stat().st_size for k in urls),
                                  'source': datasets_cfg['isic2016']['source']})
    print(f'  ISIC 2016 pronto: {human_bytes(tree_size(isic_root))}')
except Exception as exc:
    errors_list.append(f'isic2016 acquisition: {type(exc).__name__}: {exc}')
    print(f'  ACQUISIZIONE ISIC 2016 NON COMPLETATA: {type(exc).__name__}: {exc}')

# --------------------------------------------------------------------------------------
step(7, TOTAL_STEPS, 'Acquisizione PAD-UFES-20 ufficiale (circa 3.35 GiB, patient-level)')
pad_root = raw_root / 'pad_ufes20'
pad_marker = pad_root / '_ACQUIRED.json'
try:
    if not pad_marker.exists():
        pad_root.mkdir(parents=True, exist_ok=True)
        pad_zip = download_http(
            'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip',
            pad_root/'zr7vgbcyr2-1.zip',
            expected_sha256='e8c7e17bac1698c97e44d4096ec20ac1b91c135285c1446b7b2e7ebbc9be933c'
        )
        safe_extract(pad_zip, pad_root/'extracted')
        atomic_json(pad_marker, {'completed_utc': utc_now(), 'source_bytes': pad_zip.stat().st_size,
                                 'source_sha256': sha256_file(pad_zip), 'source': datasets_cfg['pad_ufes20']['source']})
    print(f'  PAD-UFES-20 pronto: {human_bytes(tree_size(pad_root))}')
except Exception as exc:
    errors_list.append(f'pad_ufes20 acquisition: {type(exc).__name__}: {exc}')
    print(f'  ACQUISIZIONE PAD-UFES-20 NON COMPLETATA: {type(exc).__name__}: {exc}')

# --------------------------------------------------------------------------------------
step(8, TOTAL_STEPS, 'Costruzione dei manifest clinici canonici')
manifest_root = PROJECT_ROOT / 'data/manifests'
manifest_root.mkdir(parents=True, exist_ok=True)
canonical_frames = {}

def base_record(dataset, sample_id, image_path, mask_path, label, class_name, patient_id, lesion_id, group_id, scope, metadata=None):
    return {'dataset': dataset, 'sample_id': str(sample_id), 'image_path': relative(image_path),
            'mask_path': relative(mask_path) if mask_path else '', 'label': int(label), 'class_name': str(class_name),
            'patient_id': str(patient_id) if patient_id is not None else '', 'lesion_id': str(lesion_id) if lesion_id is not None else '',
            'group_id_original': str(group_id), 'inference_scope': scope,
            'metadata_json': json.dumps({str(k): json_safe(v) for k, v in (metadata or {}).items()}, ensure_ascii=False, sort_keys=True)}

if bus_marker.exists():
    meta = pd.read_csv(bus_root/'source_metadata.csv')
    records = []
    for _, r in meta.iterrows():
        cls = str(r.get('class_label', '')).lower().strip()
        if cls not in {'benign', 'malignant'}:
            continue
        sid = clean_id(r['image_id'])
        pid = str(r.get('patient_id', '')).strip()
        if not pid or pid.lower() == 'nan':
            pid = re.split(r'[_\-]', sid)[0]
        records.append(base_record('bus_uclm', sid, bus_root/'images'/f'{sid}.png', bus_root/'masks'/f'{sid}.png',
                                   cls == 'malignant', cls, pid, pid, pid, 'patient',
                                   {k: r.get(k) for k in ['has_doppler','has_marks','has_combined']}))
    canonical_frames['bus_uclm'] = pd.DataFrame(records)
    if len(records) != 264 or sum(x['label'] for x in records) != 90:
        raise RuntimeError(f'BUS-UCLM task audit inatteso: n={len(records)}, malignant={sum(x["label"] for x in records)}; attesi 264 e 90.')
    print(f'  BUS-UCLM: {len(records):,} campioni eleggibili benign/malignant; normali esclusi dal task binario principale.')

if siim_marker.exists():
    paths = all_images(siim_root)
    mask_paths = [p for p in paths if 'mask' in str(p.parent).lower() or re.search(r'mask', p.stem, re.I)]
    image_paths = [p for p in paths if p not in set(mask_paths)]
    image_map, mask_map = find_by_clean_stem(image_paths), find_by_clean_stem(mask_paths)
    common = sorted(set(image_map) & set(mask_map))
    if not common:
        raise RuntimeError('SIIM-ACR: nessuna coppia immagine/mask individuata; struttura mirror inattesa.')
    records = []
    for sid in tqdm(common, desc='  Lettura label da mask SIIM'):
        m = cv2.imread(str(mask_map[sid]), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        positive = int(np.any(m > 0))
        records.append(base_record('siim_acr', sid, image_map[sid], mask_map[sid], positive,
                                   'pneumothorax' if positive else 'negative', '', sid, sid, 'case_proxy'))
    canonical_frames['siim_acr'] = pd.DataFrame(records)
    if len(records) != datasets_cfg['siim_acr']['expected_images']:
        raise RuntimeError(f'SIIM-ACR: trovate {len(records)} coppie, attese {datasets_cfg["siim_acr"]["expected_images"]}.')
    print(f'  SIIM-ACR: {len(records):,} coppie; positive={sum(x["label"] for x in records):,}.')

if isic_marker.exists():
    csvs = list(isic_root.rglob('*GroundTruth*.csv'))
    if not csvs:
        raise RuntimeError('ISIC 2016: CSV ground truth non trovato.')
    labels = pd.read_csv(csvs[0], header=None)
    if labels.shape[1] < 2:
        labels = pd.read_csv(csvs[0])
    id_col, label_col = labels.columns[:2]
    images = all_images(isic_root/'extracted')
    mask_candidates = [p for p in images if 'segmentation' in p.stem.lower() or 'mask' in p.stem.lower()]
    source_candidates = [p for p in images if p not in set(mask_candidates)]
    image_map, mask_map = find_by_clean_stem(source_candidates), find_by_clean_stem(mask_candidates)
    records = []
    for _, r in labels.iterrows():
        sid, cls = clean_id(r[id_col]), str(r[label_col]).strip().lower()
        if sid.lower() in {'image_id', 'image', 'id'} or cls not in {'benign', 'malignant'}:
            continue
        if sid not in image_map or sid not in mask_map:
            continue
        records.append(base_record('isic2016', sid, image_map[sid], mask_map[sid], cls == 'malignant', cls,
                                   '', sid, sid, 'case_proxy'))
    canonical_frames['isic2016'] = pd.DataFrame(records)
    if len(records) != datasets_cfg['isic2016']['expected_images']:
        raise RuntimeError(f'ISIC 2016: trovate {len(records)} coppie con label, attese {datasets_cfg["isic2016"]["expected_images"]}.')
    print(f'  ISIC 2016: {len(records):,} immagini classificate con mask abbinate.')

if pad_marker.exists():
    csvs = list((pad_root/'extracted').rglob('*.csv'))
    if not csvs:
        raise RuntimeError('PAD-UFES-20: metadata CSV non trovato.')
    meta_path = max(csvs, key=lambda p: p.stat().st_size)
    meta = pd.read_csv(meta_path)
    img_col = find_column(meta, ['img_id','image_id','image'])
    patient_col = find_column(meta, ['patient_id','patient'])
    lesion_col = find_column(meta, ['lesion_id','lesion'])
    diag_col = find_column(meta, ['diagnostic','diagnosis','label'])
    all_pad_images = all_images(pad_root/'extracted')
    image_map = {clean_id(p.name): p for p in all_pad_images}
    records = []
    cancers = {'bcc','mel','scc'}
    for _, r in meta.iterrows():
        sid = clean_id(r[img_col])
        if sid not in image_map:
            continue
        diag = str(r[diag_col]).strip().lower()
        pid, lid = str(r[patient_col]).strip(), str(r[lesion_col]).strip()
        extra = {str(k): (None if pd.isna(v) else v) for k, v in r.items() if k not in {img_col, patient_col, lesion_col, diag_col}}
        records.append(base_record('pad_ufes20', sid, image_map[sid], None, diag in cancers, diag,
                                   pid, lid, pid, 'patient', extra))
    canonical_frames['pad_ufes20'] = pd.DataFrame(records)
    if len(records) != datasets_cfg['pad_ufes20']['expected_images']:
        raise RuntimeError(f'PAD-UFES-20: trovate {len(records)} immagini, attese {datasets_cfg["pad_ufes20"]["expected_images"]}.')
    print(f'  PAD-UFES-20: {len(records):,} immagini; cancer={sum(x["label"] for x in records):,}.')

# --------------------------------------------------------------------------------------
step(9, TOTAL_STEPS, 'Audit immagini, mask, hash esatti e perceptual hash')
def phash64(gray):
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    values = dct.flatten()[1:]
    med = float(np.median(values))
    bits = (dct.flatten() >= med).astype(np.uint8)
    return f'{int("".join(str(int(b)) for b in bits), 2):016x}'

audited_frames = {}
for dataset, df in canonical_frames.items():
    audit_rows = []
    print(f'  Audit {dataset}: {len(df):,} immagini...', flush=True)
    for row in tqdm(df.to_dict('records'), desc=f'  {dataset}'):
        image_abs = PROJECT_ROOT / row['image_path']
        mask_abs = PROJECT_ROOT / row['mask_path'] if row['mask_path'] else None
        valid, reason, h, w, channels, exact, perceptual = True, '', 0, 0, 0, '', ''
        mask_valid, mask_nonempty, mask_fraction = (False, False, np.nan)
        try:
            raw = cv2.imread(str(image_abs), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError('decode_image_failed')
            h, w = raw.shape[:2]
            channels = 1 if raw.ndim == 2 else raw.shape[2]
            gray = raw if raw.ndim == 2 else cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)
            exact = sha256_file(image_abs)
            perceptual = phash64(gray)
            if min(h, w) < 64:
                raise ValueError('image_too_small')
            if mask_abs:
                mask = cv2.imread(str(mask_abs), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError('decode_mask_failed')
                mask_valid = mask.shape[:2] == (h, w)
                mask_nonempty = bool(np.any(mask > 0))
                mask_fraction = float(np.mean(mask > 0))
                if not mask_valid:
                    raise ValueError('mask_dimension_mismatch')
        except Exception as exc:
            valid, reason = False, str(exc)
        row.update({'height': h, 'width': w, 'channels': channels, 'image_sha256': exact, 'phash64': perceptual,
                    'mask_valid': bool(mask_valid), 'mask_nonempty': bool(mask_nonempty),
                    'mask_area_fraction': mask_fraction, 'eligible': bool(valid), 'exclusion_reason': reason})
        audit_rows.append(row)
    audited_frames[dataset] = pd.DataFrame(audit_rows)

# --------------------------------------------------------------------------------------
step(10, TOTAL_STEPS, 'Deduplicazione leakage-safe e split stratificati a gruppi')
def hamming_hex(a, b):
    return (int(a, 16) ^ int(b, 16)).bit_count()

def build_duplicate_groups(df):
    n = len(df)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    exact_map = defaultdict(list)
    phash_map = defaultdict(list)
    for i, row in df.reset_index(drop=True).iterrows():
        if row['image_sha256']:
            exact_map[row['image_sha256']].append(i)
        if row['phash64']:
            phash_map[row['phash64']].append(i)
    for indices in exact_map.values():
        for j in indices[1:]: union(indices[0], j)
    # BK-tree: ricerca esatta di tutti i pHash entro Hamming <= 4 senza matrice O(n^2).
    tree = None
    nodes = []
    for i, value_hex in enumerate(df.reset_index(drop=True)['phash64']):
        if not value_hex:
            continue
        value = int(value_hex, 16)
        if tree is None:
            tree = {'value': value, 'indices': [i], 'children': {}}
            nodes.append(tree)
            continue
        node = tree
        while True:
            distance = (value ^ node['value']).bit_count()
            if distance == 0:
                for j in node['indices']:
                    union(i, j)
                node['indices'].append(i)
                break
            if distance not in node['children']:
                child = {'value': value, 'indices': [i], 'children': {}}
                node['children'][distance] = child
                nodes.append(child)
                break
            node = node['children'][distance]
        stack = [tree]
        while stack:
            node = stack.pop()
            distance = (value ^ node['value']).bit_count()
            if distance <= 4:
                for j in node['indices']:
                    if j != i:
                        union(i, j)
            lower, upper = distance - 4, distance + 4
            stack.extend(child for edge, child in node['children'].items() if lower <= edge <= upper)
    clusters = defaultdict(list)
    for i in range(n): clusters[find(i)].append(i)
    cluster_name = {}
    conflicts = set()
    for root, indices in clusters.items():
        name = f'dup_{min(indices):06d}'
        labels = set(int(df.iloc[i]['label']) for i in indices)
        if len(labels) > 1:
            conflicts.update(indices)
        for i in indices: cluster_name[i] = name
    return cluster_name, conflicts, sum(len(v)-1 for v in clusters.values() if len(v)>1)

def choose_split(df, dataset):
    eligible = df[df['eligible']].reset_index(drop=True).copy()
    cluster_name, conflicts, duplicate_links = build_duplicate_groups(eligible)
    eligible['duplicate_cluster'] = [cluster_name[i] for i in range(len(eligible))]
    if conflicts:
        eligible.loc[list(conflicts), 'eligible'] = False
        eligible.loc[list(conflicts), 'exclusion_reason'] = 'duplicate_cluster_label_conflict'
    work = eligible[eligible['eligible']].copy().reset_index(drop=True)
    # All exact/perceptual duplicates share a split, even when their original group IDs differ.
    dup_to_groups = work.groupby('duplicate_cluster')['group_id_original'].agg(lambda x: sorted(set(map(str, x))))
    canonical_group = {dup: hashlib.sha1(('|'.join(groups)).encode()).hexdigest()[:16] for dup, groups in dup_to_groups.items()}
    work['group_id_effective'] = [f'{g}__{canonical_group[d]}' for g, d in zip(work['group_id_original'], work['duplicate_cluster'])]
    # If a duplicate spans groups, collapse all involved original groups through union-find at group level.
    graph = defaultdict(set)
    for groups in dup_to_groups:
        for g in groups:
            graph[g].update(groups)
    seen, component = set(), {}
    for g in sorted(graph):
        if g in seen: continue
        stack, members = [g], []
        while stack:
            x = stack.pop()
            if x in seen: continue
            seen.add(x); members.append(x); stack.extend(graph[x]-seen)
        cid = hashlib.sha1(('|'.join(sorted(members))).encode()).hexdigest()[:16]
        for x in members: component[x] = cid
    work['group_id_effective'] = [component[str(g)] for g in work['group_id_original']]
    best = None
    for trial in range(64):
        random_state = BASE_SEED + trial
        splitter = StratifiedGroupKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=random_state)
        assignment = np.full(len(work), -1, dtype=int)
        try:
            for fold, (_, test_idx) in enumerate(splitter.split(work, work['label'], groups=work['group_id_effective'])):
                assignment[test_idx] = fold
        except ValueError:
            continue
        if np.any(assignment < 0):
            continue
        rates = [float(work.loc[assignment == f, 'label'].mean()) for f in range(N_OUTER_FOLDS)]
        sizes = [int(np.sum(assignment == f)) for f in range(N_OUTER_FOLDS)]
        score = np.std(rates) + 0.25*np.std(np.array(sizes)/max(1, len(work)))
        if best is None or score < best[0]:
            best = (score, random_state, assignment.copy(), rates, sizes)
    if best is None:
        raise RuntimeError(f'{dataset}: impossibile ottenere {N_OUTER_FOLDS} fold stratificati a gruppi.')
    _, split_seed, assignment, rates, sizes = best
    work['outer_fold'] = assignment
    for test_fold in range(N_OUTER_FOLDS):
        val_fold = (test_fold + 1) % N_OUTER_FOLDS
        work[f'fold_{test_fold}_role'] = np.where(work['outer_fold'] == test_fold, 'test',
                                                  np.where(work['outer_fold'] == val_fold, 'val', 'train'))
    for f in range(N_OUTER_FOLDS):
        for role in ['train','val','test']:
            groups = set(work.loc[work[f'fold_{f}_role'] == role, 'group_id_effective'])
            for other in ['train','val','test']:
                if role < other:
                    other_groups = set(work.loc[work[f'fold_{f}_role'] == other, 'group_id_effective'])
                    if groups & other_groups:
                        raise AssertionError(f'{dataset} fold {f}: leakage gruppi {role}/{other}.')
    return work, {'dataset': dataset, 'eligible_samples': len(work), 'groups': int(work['group_id_effective'].nunique()),
                  'positive_rate': float(work['label'].mean()), 'split_seed': split_seed,
                  'fold_sizes': sizes, 'fold_positive_rates': rates, 'duplicate_links': duplicate_links,
                  'excluded_invalid_or_conflicting': int(len(df)-len(work))}

split_frames = {}
for dataset, df in audited_frames.items():
    split_df, report = choose_split(df, dataset)
    split_frames[dataset] = split_df
    dataset_reports[dataset] = report
    csv_path = manifest_root/f'{dataset}_manifest_v{PROTOCOL_VERSION}.csv'
    parquet_path = manifest_root/f'{dataset}_manifest_v{PROTOCOL_VERSION}.parquet'
    split_df.to_csv(csv_path, index=False)
    split_df.to_parquet(parquet_path, index=False)
    print(f'  {dataset:12s} n={len(split_df):5d} | gruppi={report["groups"]:5d} | pos={report["positive_rate"]:.3f} | '
          f'fold={report["fold_sizes"]} | seed={report["split_seed"]}')

# --------------------------------------------------------------------------------------
step(11, TOTAL_STEPS, 'Validazione scientifica finale e firma degli artefatti')
split_root = PROJECT_ROOT / 'data/splits'
split_root.mkdir(parents=True, exist_ok=True)
combined = pd.concat(split_frames.values(), ignore_index=True) if split_frames else pd.DataFrame()
if not combined.empty:
    combined.to_csv(split_root/f'all_datasets_splits_v{PROTOCOL_VERSION}.csv', index=False)
    combined.to_parquet(split_root/f'all_datasets_splits_v{PROTOCOL_VERSION}.parquet', index=False)
expected_available = set(k for k, marker in {'bus_uclm':bus_marker, 'siim_acr':siim_marker, 'isic2016':isic_marker, 'pad_ufes20':pad_marker}.items() if marker.exists())
manifest_available = set(split_frames)
if expected_available != manifest_available:
    errors_list.append(f'Dataset acquisiti senza manifest completo: {sorted(expected_available-manifest_available)}')
artifacts = [amendment_path, config_dir/'protocol.json', config_dir/'datasets.json']
artifacts += sorted(manifest_root.glob(f'*_manifest_v{PROTOCOL_VERSION}.*'))
artifacts += sorted(split_root.glob(f'*_v{PROTOCOL_VERSION}.*'))
artifact_hashes = {relative(p): sha256_file(p) for p in artifacts if p.exists()}
chain = hashlib.sha256(''.join(artifact_hashes[k] for k in sorted(artifact_hashes)).encode()).hexdigest()
for dataset, report in dataset_reports.items():
    if report['groups'] < N_OUTER_FOLDS * 2:
        warnings_list.append(f'{dataset}: solo {report["groups"]} gruppi; interpretare le stime fold con cautela.')
    if dataset in {'siim_acr','isic2016'}:
        warnings_list.append(f'{dataset}: validazione case-level, non conteggiata come replica patient-level del claim principale.')
print(f'Dataset con manifest completo: {len(split_frames)}/4')
print(f'Artifact chain SHA-256: {chain}')

# --------------------------------------------------------------------------------------
step(12, TOTAL_STEPS, 'Manifest di esecuzione e gate per il Notebook 03')
status = 'PASS' if len(errors_list) == 0 and len(split_frames) == 4 else 'PARTIAL'
run_manifest = {
    'notebook_id': NOTEBOOK_ID, 'status': status, 'protocol_version': PROTOCOL_VERSION,
    'started_utc': datetime.fromtimestamp(started, timezone.utc).isoformat(), 'completed_utc': utc_now(),
    'project_root': str(PROJECT_ROOT), 'base_seed': BASE_SEED, 'num_workers': NUM_WORKERS,
    'hardware_required': 'CPU', 'max_source_dataset_gib': MAX_SOURCE_GIB,
    'dataset_reports': dataset_reports, 'artifact_hashes': artifact_hashes, 'artifact_chain_sha256': chain,
    'warnings': warnings_list, 'errors': errors_list
}
run_dir = PROJECT_ROOT / 'runs/preprocessing'
run_dir.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
manifest_path = run_dir/f'preprocessing_manifest_{stamp}.json'
atomic_json(manifest_path, run_manifest)
atomic_json(run_dir/'latest_preprocessing_manifest.json', run_manifest)
elapsed = time.time() - started
print(f'Stato preprocessing:       {status}')
print(f'Dataset completi:          {len(split_frames)}/4')
print(f'GPU necessaria:            NO')
print(f'Warning:                   {len(warnings_list)}')
for i, message in enumerate(warnings_list, 1): print(f'  W{i}: {message}')
print(f'Errori:                    {len(errors_list)}')
for i, message in enumerate(errors_list, 1): print(f'  E{i}: {message}')
print(f'Tempo totale:              {elapsed/60:.2f} min')
print(f'Manifest:                  {manifest_path}')
if status == 'PASS':
    print('Prossimo file:              03_train_classifiers.py (GPU richiesta)')
    print('Prossima operazione:        training patient/group-safe dei classificatori congelabili.')
else:
    print('Azione richiesta:           inviare questo output; NON procedere al Notebook 03.')
banner(f'CPET_NOTEBOOK_02_STATUS={status}')
print('Inviare in chat l’intero output testuale di questa cella prima di procedere.', flush=True)
if status != 'PASS':
    raise RuntimeError('Gate Notebook 02 non superato: almeno un dataset o manifest non è completo.')
