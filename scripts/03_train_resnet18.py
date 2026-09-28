# CPET — NOTEBOOK 03/08 — CLASSIFICATORI DI BASE CROSS-VALIDATI
# Una sola cella Colab. GPU richiesta (T4 sufficiente). num_workers=0.
# Addestra ResNet-18 ImageNet su 4 dataset x 5 fold con resume, selezione validation-only e test post-selection.

import os, sys, re, json, time, math, random, hashlib, shutil, subprocess, importlib
from pathlib import Path
from datetime import datetime, timezone

# ======================================================================================
# CONFIGURAZIONE SCIENTIFICA — CORE PAPER RUN
# ======================================================================================
PROJECT_ROOT = Path('/content/gdrive/MyDrive/Colab Notebooks/CPET')
PROTOCOL_VERSION = '1.1.0'
BASE_SEED = 20260906
NOTEBOOK_ID = '03_train_classifiers'
NUM_WORKERS = 0
BACKBONE = 'resnet18'
DATASETS = ['bus_uclm', 'siim_acr', 'isic2016', 'pad_ufes20']
FOLDS = [0, 1, 2, 3, 4]
MAX_EPOCHS = 100
MIN_EPOCHS = 12
EARLY_STOP_PATIENCE = 12
CHECKPOINT_EVERY = 2
HEAD_LR = 5e-4
BACKBONE_LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
WARMUP_EPOCHS = 3
STAGE_TO_LOCAL = True
LOCAL_STAGE_ROOT = Path('/content/cpet_stage')
QUALITY_GATE_MEAN_AUROC = 0.70
QUALITY_GATE_MIN_FOLD_AUROC = 0.60

DATASET_SETTINGS = {
    'bus_uclm':  {'input_size':256, 'batch_size':64, 'modality':'ultrasound', 'scope':'patient'},
    'siim_acr':  {'input_size':320, 'batch_size':48, 'modality':'xray',       'scope':'case_proxy'},
    'isic2016':  {'input_size':320, 'batch_size':48, 'modality':'dermoscopy','scope':'case_proxy'},
    'pad_ufes20':{'input_size':320, 'batch_size':48, 'modality':'clinical',  'scope':'patient'},
}

started = time.time()
errors_list, warnings_list = [], []

def banner(title):
    print('\n' + '='*110)
    print(title)
    print('='*110, flush=True)

def step(i, n, title):
    print(f'\n[{i}/{n}] {title}', flush=True)

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def human_seconds(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'

def sha256_file(path, chunk=4*1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',',':')).encode()).hexdigest()

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)

TOTAL_STEPS = 8
banner('CPET — NOTEBOOK 03/08 — CLASSIFICATORI DI BASE CROSS-VALIDATI')
print('Disegno: ResNet-18 ImageNet | 4 dataset | 5 fold grouped | una seed deterministica per fold.')
print('Selezione: esclusivamente validation AUROC. Il test è valutato solo dopo il caricamento del checkpoint scelto.')
print('Output: checkpoint congelati, history, predizioni OOF, calibrazione e metriche per il paper.')
print(f'Protocollo={PROTOCOL_VERSION} | seed base={BASE_SEED} | num_workers={NUM_WORKERS} | max_epochs={MAX_EPOCHS}', flush=True)

# --------------------------------------------------------------------------------------------------
step(1, TOTAL_STEPS, 'Mount, gate dati e audit hardware')
try:
    from google.colab import drive
    drive.mount('/content/gdrive', force_remount=False)
except Exception as exc:
    raise RuntimeError(f'Questa cella deve essere eseguita in Colab con Drive: {exc}')

data_gate_path = PROJECT_ROOT/'runs/preprocessing/latest_preprocessing_manifest.json'
if not data_gate_path.exists():
    raise RuntimeError('Manca il manifest del Notebook 02.')
data_gate = load_json(data_gate_path)
if data_gate.get('status') != 'PASS' or set(data_gate.get('dataset_reports',{})) != set(DATASETS):
    raise RuntimeError(f'Gate dati non valido: status={data_gate.get("status")}, dataset={sorted(data_gate.get("dataset_reports",{}))}.')
if data_gate.get('protocol_version') != PROTOCOL_VERSION:
    raise RuntimeError(f'Protocollo dati {data_gate.get("protocol_version")} != {PROTOCOL_VERSION}.')

try:
    smi = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'], capture_output=True, text=True, timeout=20)
    gpu_description = smi.stdout.strip()
except Exception:
    gpu_description = ''
print(f'Gate dati: PASS | dataset=4/4 | chain={data_gate.get("artifact_chain_sha256","n/d")[:16]}...')
print(f'GPU nvidia-smi: {gpu_description or "NON RILEVATA"}')

# --------------------------------------------------------------------------------------------------
step(2, TOTAL_STEPS, 'Dipendenze, versioni e smoke GPU')
deps = {'numpy':'numpy','pandas':'pandas','cv2':'opencv-python-headless','sklearn':'scikit-learn',
        'scipy':'scipy','tqdm':'tqdm','torch':'torch','torchvision':'torchvision','pyarrow':'pyarrow'}
missing = []
for module, package in deps.items():
    try: importlib.import_module(module)
    except Exception: missing.append(package)
if missing:
    print('Installazione dipendenze:', ', '.join(sorted(set(missing))), flush=True)
    result = subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--progress-bar','off']+sorted(set(missing)))
    if result.returncode: raise RuntimeError(f'pip fallito con codice {result.returncode}.')
    importlib.invalidate_caches()
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score, confusion_matrix, log_loss, brier_score_loss
from scipy.optimize import minimize_scalar
from tqdm.auto import tqdm

if not torch.cuda.is_available():
    raise RuntimeError('GPU non disponibile: attivare un runtime GPU prima di eseguire il Notebook 03.')
device = torch.device('cuda')
torch.set_num_threads(min(4, os.cpu_count() or 1))
probe = torch.randn(512,512,device=device)
probe = probe @ probe.T
torch.cuda.synchronize()
if not bool(torch.isfinite(probe).all()): raise RuntimeError('Smoke GPU non finito.')
del probe
torch.cuda.empty_cache()
print(f'PyTorch={torch.__version__} | torchvision={torchvision.__version__} | CUDA={torch.version.cuda}')
print(f'Device={torch.cuda.get_device_name(0)} | mixed precision=FP16 | GPU smoke=PASS')

# --------------------------------------------------------------------------------------------------
step(3, TOTAL_STEPS, 'Freeze della configurazione di training e verifica manifest')
training_config = {
    'protocol_version':PROTOCOL_VERSION, 'backbone':BACKBONE, 'pretraining':'ImageNet1K_V1',
    'datasets':DATASETS, 'folds':FOLDS, 'max_epochs':MAX_EPOCHS, 'min_epochs':MIN_EPOCHS,
    'early_stop_patience':EARLY_STOP_PATIENCE, 'checkpoint_every':CHECKPOINT_EVERY,
    'optimizer':'AdamW', 'head_lr':HEAD_LR, 'backbone_lr':BACKBONE_LR, 'weight_decay':WEIGHT_DECAY,
    'scheduler':'linear_warmup_then_cosine', 'warmup_epochs':WARMUP_EPOCHS, 'grad_clip':GRAD_CLIP,
    'loss':'BCEWithLogitsLoss_train_fold_pos_weight', 'selection_metric':'validation_AUROC',
    'calibration':'validation_temperature_scaling', 'threshold':'validation_Youden_J',
    'augmentations':'modality_specific_conservative', 'num_workers':NUM_WORKERS,
    'mixed_precision':'fp16', 'quality_gate':{'mean_auroc':QUALITY_GATE_MEAN_AUROC,'min_fold_auroc':QUALITY_GATE_MIN_FOLD_AUROC},
    'dataset_settings':DATASET_SETTINGS
}
config_hash = stable_hash(training_config)
config_path = PROJECT_ROOT/'configs/classifier_training_v1.1.0.json'
if config_path.exists():
    existing = load_json(config_path)
    if stable_hash(existing) != config_hash:
        raise RuntimeError(f'{config_path} esiste con configurazione diversa: non mescolo run incompatibili.')
else:
    atomic_json(config_path, training_config)
print(f'Config training: {config_path}')
print(f'Config SHA-256: {config_hash}')

manifest_frames, manifest_hashes = {}, {}
for dataset in DATASETS:
    path = PROJECT_ROOT/f'data/manifests/{dataset}_manifest_v{PROTOCOL_VERSION}.parquet'
    if not path.exists(): raise RuntimeError(f'Manifest mancante: {path}')
    df = pd.read_parquet(path)
    required_columns = {'sample_id','image_path','label','group_id_effective'} | {f'fold_{f}_role' for f in FOLDS}
    missing_columns = required_columns-set(df.columns)
    if missing_columns: raise RuntimeError(f'{dataset}: colonne mancanti {sorted(missing_columns)}.')
    if len(df)==0 or set(df['label'].astype(int).unique()) != {0,1}: raise RuntimeError(f'{dataset}: label binarie non valide.')
    for fold in FOLDS:
        roles = set(df[f'fold_{fold}_role'].unique())
        if roles != {'train','val','test'}: raise RuntimeError(f'{dataset} fold {fold}: ruoli {roles}.')
        group_sets = {r:set(df.loc[df[f'fold_{fold}_role']==r,'group_id_effective']) for r in roles}
        if group_sets['train']&group_sets['val'] or group_sets['train']&group_sets['test'] or group_sets['val']&group_sets['test']:
            raise RuntimeError(f'{dataset} fold {fold}: leakage di gruppi.')
    manifest_frames[dataset] = df
    manifest_hashes[dataset] = sha256_file(path)
    print(f'{dataset:12s} n={len(df):5d} | pos={df.label.mean():.3f} | gruppi={df.group_id_effective.nunique():5d} | sha={manifest_hashes[dataset][:12]}')

# --------------------------------------------------------------------------------------------------
step(4, TOTAL_STEPS, 'Definizione preprocessing, modello, metriche e calibrazione')
IMAGENET_MEAN = np.array([0.485,0.456,0.406],dtype=np.float32)
IMAGENET_STD = np.array([0.229,0.224,0.225],dtype=np.float32)

def resize_pad(image, size):
    h,w = image.shape[:2]
    scale = size/max(h,w)
    nh,nw = max(1,int(round(h*scale))),max(1,int(round(w*scale)))
    resized = cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    top=(size-nh)//2; bottom=size-nh-top; left=(size-nw)//2; right=size-nw-left
    return cv2.copyMakeBorder(resized,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(0,0,0))

def augment(image, modality):
    if modality in {'dermoscopy','clinical'}:
        if random.random()<0.5: image=cv2.flip(image,1)
        if random.random()<0.5: image=cv2.flip(image,0)
        angle=random.uniform(-20,20)
    elif modality=='ultrasound':
        if random.random()<0.5: image=cv2.flip(image,1)
        angle=random.uniform(-10,10)
    else:
        angle=random.uniform(-5,5)
    if random.random()<0.7:
        h,w=image.shape[:2]
        matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,random.uniform(0.95,1.05))
        image=cv2.warpAffine(image,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    if random.random()<0.6:
        alpha=random.uniform(0.90,1.10); beta=random.uniform(-10,10)
        image=cv2.convertScaleAbs(image,alpha=alpha,beta=beta)
    return image

class CPETImageDataset(Dataset):
    def __init__(self, frame, input_size, modality, train):
        self.frame=frame.reset_index(drop=True)
        self.input_size=input_size; self.modality=modality; self.train=train
    def __len__(self): return len(self.frame)
    def __getitem__(self,index):
        row=self.frame.iloc[index]
        image=cv2.imread(str(row['runtime_path']),cv2.IMREAD_UNCHANGED)
        if image is None: raise RuntimeError(f'Decode fallito: {row["runtime_path"]}')
        if image.ndim==2: image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        else: image=cv2.cvtColor(image[:,:,:3],cv2.COLOR_BGR2RGB)
        if self.train: image=augment(image,self.modality)
        image=resize_pad(image,self.input_size).astype(np.float32)/255.0
        image=(image-IMAGENET_MEAN)/IMAGENET_STD
        tensor=torch.from_numpy(np.transpose(image,(2,0,1))).float()
        return tensor, torch.tensor(float(row['label']),dtype=torch.float32), index

def build_model():
    model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features=model.fc.in_features
    model.fc=nn.Linear(in_features,1)
    return model

def make_loader(frame, settings, train, seed):
    ds=CPETImageDataset(frame,settings['input_size'],settings['modality'],train)
    generator=torch.Generator(); generator.manual_seed(seed)
    return DataLoader(ds,batch_size=settings['batch_size'],shuffle=train,num_workers=NUM_WORKERS,
                      pin_memory=True,drop_last=False,generator=generator)

@torch.no_grad()
def predict(model,loader):
    model.eval(); logits=np.empty(len(loader.dataset),dtype=np.float32); labels=np.empty(len(loader.dataset),dtype=np.int64)
    for images,target,indices in loader:
        images=images.to(device,non_blocking=True)
        with torch.autocast(device_type='cuda',dtype=torch.float16): out=model(images).squeeze(1)
        idx=indices.numpy(); logits[idx]=out.float().cpu().numpy(); labels[idx]=target.numpy().astype(np.int64)
    return logits,labels

def sigmoid_np(x):
    x=np.clip(x,-50,50)
    return 1/(1+np.exp(-x))

def fit_temperature(logits,labels):
    def objective(log_t):
        probs=sigmoid_np(logits/np.exp(log_t))
        return log_loss(labels,np.clip(probs,1e-7,1-1e-7),labels=[0,1])
    result=minimize_scalar(objective,bounds=(-3,3),method='bounded',options={'xatol':1e-5})
    return float(np.exp(result.x))

def choose_threshold(labels,probs):
    candidates=np.unique(np.concatenate(([0.0],probs,[1.0])))
    if len(candidates)>4000: candidates=np.quantile(candidates,np.linspace(0,1,4000))
    best=(float('-inf'),0.5)
    for threshold in candidates:
        pred=(probs>=threshold).astype(int)
        score=balanced_accuracy_score(labels,pred)
        if score>best[0] or (score==best[0] and abs(threshold-.5)<abs(best[1]-.5)): best=(score,float(threshold))
    return best[1]

def expected_calibration_error(labels,probs,bins=15):
    edges=np.linspace(0,1,bins+1); ece=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(probs>=lo)&(probs<(hi if hi<1 else hi+1e-12))
        if mask.any(): ece += mask.mean()*abs(labels[mask].mean()-probs[mask].mean())
    return float(ece)

def metrics(labels,probs,threshold):
    pred=(probs>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(labels,pred,labels=[0,1]).ravel()
    return {'n':int(len(labels)),'prevalence':float(np.mean(labels)),'auroc':float(roc_auc_score(labels,probs)),
            'auprc':float(average_precision_score(labels,probs)),'accuracy':float(accuracy_score(labels,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(labels,pred)),'f1':float(f1_score(labels,pred,zero_division=0)),
            'sensitivity':float(tp/max(1,tp+fn)),'specificity':float(tn/max(1,tn+fp)),
            'brier':float(brier_score_loss(labels,probs)),'ece15':expected_calibration_error(labels,probs),
            'threshold':float(threshold),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

# --------------------------------------------------------------------------------------------------
step(5, TOTAL_STEPS, 'Staging locale delle immagini per evitare I/O Drive durante il training')
def stage_dataset(dataset,frame):
    frame=frame.copy().reset_index(drop=True)
    if not STAGE_TO_LOCAL:
        frame['runtime_path']=[str(PROJECT_ROOT/p) for p in frame['image_path']]
        return frame
    stage_dir=LOCAL_STAGE_ROOT/dataset
    stage_dir.mkdir(parents=True,exist_ok=True)
    runtime_paths=[]
    copied=0
    for i,row in tqdm(frame.iterrows(),total=len(frame),desc=f'Stage {dataset}'):
        source=PROJECT_ROOT/row['image_path']
        suffix=source.suffix.lower() or '.img'
        safe_id=re.sub(r'[^A-Za-z0-9_.-]+','_',str(row['sample_id']))
        target=stage_dir/f'{i:06d}_{safe_id}{suffix}'
        if not target.exists() or target.stat().st_size!=source.stat().st_size:
            shutil.copy2(source,target); copied+=1
        runtime_paths.append(str(target))
    frame['runtime_path']=runtime_paths
    print(f'  {dataset}: {len(frame):,} file pronti, {copied:,} copiati in questa esecuzione.')
    return frame

for dataset in DATASETS:
    manifest_frames[dataset]=stage_dataset(dataset,manifest_frames[dataset])
print('Staging completato. I file locali sono cache effimera; gli originali su Drive restano invariati.')

# --------------------------------------------------------------------------------------------------
step(6, TOTAL_STEPS, 'Training 4 dataset x 5 fold con resume')
checkpoint_root=PROJECT_ROOT/'checkpoints/classifiers'/BACKBONE
results_root=PROJECT_ROOT/'results/classifiers'/BACKBONE
run_root=PROJECT_ROOT/'runs/training/classifiers'/BACKBONE
for p in [checkpoint_root,results_root,run_root]: p.mkdir(parents=True,exist_ok=True)
all_fold_metrics=[]
total_jobs=len(DATASETS)*len(FOLDS); completed_jobs=0

for dataset_index,dataset in enumerate(DATASETS):
    df=manifest_frames[dataset]
    settings=DATASET_SETTINGS[dataset]
    print('\n'+'-'*110)
    print(f'DATASET {dataset_index+1}/{len(DATASETS)}: {dataset} | n={len(df):,} | size={settings["input_size"]} | batch={settings["batch_size"]}')
    print('-'*110,flush=True)
    for fold in FOLDS:
        job_started=time.time(); seed=BASE_SEED+dataset_index*100+fold
        fold_dir=checkpoint_root/dataset/f'fold_{fold}'
        result_dir=results_root/dataset/f'fold_{fold}'
        fold_dir.mkdir(parents=True,exist_ok=True); result_dir.mkdir(parents=True,exist_ok=True)
        best_path=fold_dir/'best.pt'; last_path=fold_dir/'last.pt'; done_path=fold_dir/'COMPLETE.json'
        expected_identity={'dataset':dataset,'fold':fold,'backbone':BACKBONE,'config_hash':config_hash,
                           'manifest_hash':manifest_hashes[dataset],'protocol_version':PROTOCOL_VERSION}
        if done_path.exists():
            done=load_json(done_path)
            if all(done.get(k)==v for k,v in expected_identity.items()) and (result_dir/'test_predictions.parquet').exists():
                all_fold_metrics.append(done['test_metrics']); completed_jobs+=1
                print(f'[{completed_jobs}/{total_jobs}] {dataset} fold {fold}: GIÀ COMPLETO | test AUROC={done["test_metrics"]["auroc"]:.4f}')
                continue
            raise RuntimeError(f'{done_path} esiste ma appartiene a una configurazione diversa.')

        seed_everything(seed)
        role_col=f'fold_{fold}_role'
        train_df=df[df[role_col]=='train'].reset_index(drop=True)
        val_df=df[df[role_col]=='val'].reset_index(drop=True)
        test_df=df[df[role_col]=='test'].reset_index(drop=True)
        for role,part in [('train',train_df),('val',val_df),('test',test_df)]:
            if set(part.label.astype(int).unique())!={0,1}: raise RuntimeError(f'{dataset} fold {fold} {role}: manca una classe.')
        train_loader=make_loader(train_df,settings,True,seed)
        val_loader=make_loader(val_df,settings,False,seed+1)
        test_loader=make_loader(test_df,settings,False,seed+2)
        model=build_model().to(device)
        backbone_params=[p for name,p in model.named_parameters() if not name.startswith('fc.')]
        head_params=list(model.fc.parameters())
        optimizer=torch.optim.AdamW([{'params':backbone_params,'lr':BACKBONE_LR},{'params':head_params,'lr':HEAD_LR}],weight_decay=WEIGHT_DECAY)
        def lr_factor(epoch):
            if epoch<WARMUP_EPOCHS: return float(epoch+1)/WARMUP_EPOCHS
            progress=(epoch-WARMUP_EPOCHS)/max(1,MAX_EPOCHS-WARMUP_EPOCHS-1)
            return 0.5*(1+math.cos(math.pi*min(1.0,progress)))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_factor)
        scaler=torch.amp.GradScaler('cuda',enabled=True)
        positives=float(train_df.label.sum()); negatives=float(len(train_df)-positives)
        criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives/max(1,positives)],device=device))
        start_epoch=0; best_auc=-float('inf'); best_epoch=-1; patience=0; history=[]
        if last_path.exists():
            state=torch.load(last_path,map_location=device,weights_only=False)
            if all(state.get(k)==v for k,v in expected_identity.items()):
                model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
                scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
                start_epoch=int(state['epoch'])+1; best_auc=float(state['best_auc']); best_epoch=int(state['best_epoch'])
                patience=int(state['patience']); history=list(state['history'])
                print(f'{dataset} fold {fold}: RESUME da epoca {start_epoch+1}, best val AUROC={best_auc:.4f} @ {best_epoch+1}.')
            else:
                raise RuntimeError(f'{last_path} incompatibile con il run corrente.')
        else:
            print(f'{dataset} fold {fold}: START | train={len(train_df)} val={len(val_df)} test={len(test_df)} pos_weight={negatives/max(1,positives):.3f}')

        for epoch in range(start_epoch,MAX_EPOCHS):
            epoch_started=time.time(); model.train(); loss_sum=0.0; seen=0
            progress=tqdm(train_loader,desc=f'{dataset} f{fold} ep {epoch+1:03d}',leave=False)
            for batch_idx,(images,target,_) in enumerate(progress):
                images=images.to(device,non_blocking=True); target=target.to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.float16):
                    logits=model(images).squeeze(1); loss=criterion(logits,target)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
                scaler.step(optimizer); scaler.update()
                bs=len(target); loss_sum+=float(loss.detach())*bs; seen+=bs
                if batch_idx%20==0: progress.set_postfix(loss=f'{loss_sum/max(1,seen):.4f}',lr=f'{optimizer.param_groups[0]["lr"]:.1e}')
            scheduler.step()
            val_logits,val_labels=predict(model,val_loader)
            val_auc=float(roc_auc_score(val_labels,val_logits))
            val_auprc=float(average_precision_score(val_labels,val_logits))
            row={'epoch':epoch+1,'train_loss':loss_sum/max(1,seen),'val_auroc':val_auc,'val_auprc':val_auprc,
                 'lr_backbone':optimizer.param_groups[0]['lr'],'seconds':time.time()-epoch_started}
            history.append(row)
            improved=val_auc>best_auc+1e-5
            if improved:
                best_auc=val_auc; best_epoch=epoch; patience=0
                atomic_torch_save({**expected_identity,'epoch':epoch,'model':model.state_dict(),'val_auroc':val_auc},best_path)
            else: patience+=1
            print(f'  {dataset} f{fold} | ep {epoch+1:03d}/{MAX_EPOCHS} | loss={row["train_loss"]:.4f} | '
                  f'val AUROC={val_auc:.4f} AUPRC={val_auprc:.4f} | best={best_auc:.4f}@{best_epoch+1:03d} | '
                  f'patience={patience:02d}/{EARLY_STOP_PATIENCE} | {human_seconds(row["seconds"])}',flush=True)
            if (epoch+1)%CHECKPOINT_EVERY==0 or improved:
                atomic_torch_save({**expected_identity,'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                                   'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'best_auc':best_auc,
                                   'best_epoch':best_epoch,'patience':patience,'history':history},last_path)
                pd.DataFrame(history).to_csv(result_dir/'history.csv',index=False)
            if epoch+1>=MIN_EPOCHS and patience>=EARLY_STOP_PATIENCE:
                print(f'  Early stopping: nessun miglioramento per {EARLY_STOP_PATIENCE} epoche dopo il minimo di {MIN_EPOCHS}.')
                break
        if not best_path.exists(): raise RuntimeError(f'Checkpoint best non creato: {dataset} fold {fold}.')

        # Solo ora il test viene aperto: checkpoint già selezionato esclusivamente via validation.
        best=torch.load(best_path,map_location=device,weights_only=False); model.load_state_dict(best['model'])
        val_logits,val_labels=predict(model,val_loader)
        temperature=fit_temperature(val_logits,val_labels)
        val_probs=sigmoid_np(val_logits/temperature)
        threshold=choose_threshold(val_labels,val_probs)
        val_metrics=metrics(val_labels,val_probs,threshold)
        test_logits,test_labels=predict(model,test_loader)
        test_probs=sigmoid_np(test_logits/temperature)
        test_metrics=metrics(test_labels,test_probs,threshold)
        test_metrics.update({'dataset':dataset,'fold':fold,'seed':seed,'best_epoch':int(best['epoch'])+1,
                             'temperature':temperature,'selection_val_auroc':float(best['val_auroc']),
                             'inference_scope':settings['scope']})
        prediction_frame=test_df.drop(columns=['runtime_path'],errors='ignore').copy()
        prediction_frame['logit']=test_logits; prediction_frame['probability_calibrated']=test_probs
        prediction_frame['prediction']=(test_probs>=threshold).astype(int)
        prediction_frame['fold']=fold; prediction_frame['temperature']=temperature; prediction_frame['threshold']=threshold
        prediction_frame.to_parquet(result_dir/'test_predictions.parquet',index=False)
        pd.DataFrame({'sample_id':val_df.sample_id,'label':val_labels,'logit':val_logits,
                      'probability_calibrated':val_probs}).to_parquet(result_dir/'validation_predictions.parquet',index=False)
        pd.DataFrame(history).to_csv(result_dir/'history.csv',index=False)
        atomic_json(result_dir/'metrics.json',{'identity':expected_identity,'validation':val_metrics,'test':test_metrics})
        done={**expected_identity,'completed_utc':utc_now(),'test_metrics':test_metrics,'validation_metrics':val_metrics,
              'best_checkpoint_sha256':sha256_file(best_path)}
        atomic_json(done_path,done)
        all_fold_metrics.append(test_metrics); completed_jobs+=1
        elapsed_job=time.time()-job_started
        average=(time.time()-started)/completed_jobs
        print(f'[{completed_jobs}/{total_jobs}] COMPLETO {dataset} fold {fold} | test AUROC={test_metrics["auroc"]:.4f} '
              f'AUPRC={test_metrics["auprc"]:.4f} BAcc={test_metrics["balanced_accuracy"]:.4f} | '
              f'{human_seconds(elapsed_job)} | ETA grezza={human_seconds(average*(total_jobs-completed_jobs))}',flush=True)
        del model,optimizer,scheduler,scaler,train_loader,val_loader,test_loader
        torch.cuda.empty_cache()

# --------------------------------------------------------------------------------------------------
step(7, TOTAL_STEPS, 'Aggregazione OOF e gate di qualità dei classificatori')
metrics_df=pd.DataFrame(all_fold_metrics).sort_values(['dataset','fold']).reset_index(drop=True)
metrics_path=results_root/'all_fold_metrics.csv'; metrics_df.to_csv(metrics_path,index=False)
oof_frames=[]
for dataset in DATASETS:
    parts=[pd.read_parquet(results_root/dataset/f'fold_{f}/test_predictions.parquet') for f in FOLDS]
    oof=pd.concat(parts,ignore_index=True)
    if len(oof)!=len(manifest_frames[dataset]) or oof['sample_id'].duplicated().any():
        raise RuntimeError(f'{dataset}: OOF non è una partizione 1:1 del manifest ({len(oof)} vs {len(manifest_frames[dataset])}).')
    oof.to_parquet(results_root/f'{dataset}_oof_predictions.parquet',index=False)
    oof_frames.append(oof.assign(dataset=dataset))
all_oof=pd.concat(oof_frames,ignore_index=True); all_oof.to_parquet(results_root/'all_oof_predictions.parquet',index=False)

summary=[]; gate_failures=[]
for dataset in DATASETS:
    part=metrics_df[metrics_df.dataset==dataset]
    row={'dataset':dataset,'folds':int(len(part))}
    for metric in ['auroc','auprc','balanced_accuracy','f1','sensitivity','specificity','brier','ece15']:
        row[f'{metric}_mean']=float(part[metric].mean()); row[f'{metric}_sd']=float(part[metric].std(ddof=1))
    row['auroc_min']=float(part.auroc.min()); summary.append(row)
    if row['auroc_mean']<QUALITY_GATE_MEAN_AUROC or row['auroc_min']<QUALITY_GATE_MIN_FOLD_AUROC:
        gate_failures.append(f'{dataset}: mean={row["auroc_mean"]:.4f}, min={row["auroc_min"]:.4f}')
    print(f'{dataset:12s} AUROC={row["auroc_mean"]:.4f}±{row["auroc_sd"]:.4f} | min={row["auroc_min"]:.4f} | '
          f'AUPRC={row["auprc_mean"]:.4f}±{row["auprc_sd"]:.4f} | BAcc={row["balanced_accuracy_mean"]:.4f}')
summary_df=pd.DataFrame(summary); summary_path=results_root/'dataset_summary.csv'; summary_df.to_csv(summary_path,index=False)

# --------------------------------------------------------------------------------------------------
step(8, TOTAL_STEPS, 'Manifest finale e prossima operazione')
status='PASS' if not gate_failures else 'QUALITY_GATE_FAIL'
artifact_paths=[config_path,metrics_path,summary_path,results_root/'all_oof_predictions.parquet']
for dataset in DATASETS:
    artifact_paths.append(results_root/f'{dataset}_oof_predictions.parquet')
    for fold in FOLDS:
        artifact_paths.extend([checkpoint_root/dataset/f'fold_{fold}/best.pt',checkpoint_root/dataset/f'fold_{fold}/COMPLETE.json'])
artifact_hashes={str(p.relative_to(PROJECT_ROOT)):sha256_file(p) for p in artifact_paths}
chain=hashlib.sha256(''.join(artifact_hashes[k] for k in sorted(artifact_hashes)).encode()).hexdigest()
manifest={'notebook_id':NOTEBOOK_ID,'status':status,'protocol_version':PROTOCOL_VERSION,'completed_utc':utc_now(),
          'hardware':'GPU','gpu':torch.cuda.get_device_name(0),'base_seed':BASE_SEED,'num_workers':NUM_WORKERS,
          'training_config':training_config,'config_hash':config_hash,'data_gate_chain':data_gate.get('artifact_chain_sha256'),
          'manifest_hashes':manifest_hashes,'dataset_summary':summary,'quality_gate_failures':gate_failures,
          'artifact_hashes':artifact_hashes,'artifact_chain_sha256':chain,'warnings':warnings_list,'errors':errors_list}
stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
manifest_path=run_root/f'classifier_manifest_{stamp}.json'
atomic_json(manifest_path,manifest); atomic_json(run_root/'latest_classifier_manifest.json',manifest)
print(f'Stato training:             {status}')
print(f'Run completati:             {len(metrics_df)}/{total_jobs}')
print(f'Quality gate mean AUROC:    >= {QUALITY_GATE_MEAN_AUROC:.2f}')
print(f'Quality gate min fold:      >= {QUALITY_GATE_MIN_FOLD_AUROC:.2f}')
for failure in gate_failures: print(f'  FAIL: {failure}')
print(f'Artifact chain SHA-256:     {chain}')
print(f'Tempo totale:               {human_seconds(time.time()-started)}')
print(f'Manifest:                   {manifest_path}')
if status=='PASS':
    print('Prossimo file:              04_baseline_explainers.py — GPU richiesta')
else:
    print('Azione:                     inviare l’output; NON procedere agli explainer.')
banner(f'CPET_NOTEBOOK_03_STATUS={status}')
print('Inviare in chat l’intero output testuale e dataset_summary.csv prima di procedere.',flush=True)
if status!='PASS': raise RuntimeError('I classificatori non superano il gate minimo preregistrato per gli esperimenti XAI.')