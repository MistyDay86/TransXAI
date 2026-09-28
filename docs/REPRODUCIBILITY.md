# Reproducibility contract

## Frozen design

- Protocol: `1.1.0`
- Random seed: `20260906`
- Outer folds: 5, grouped by the strongest available patient/case identifier
- Donors: 128 per dataset and fold
- Confirmatory datasets: SIIM-ACR, ISIC 2016, PAD-UFES-20
- Backbones: ResNet-18 and RegNet-X-400MF
- Explainers: Grad-CAM, LayerCAM, Integrated Gradients, LRP, RISE, Extremal Perturbation
- Spatial budgets: 10% and 20%
- Bootstrap replicates: 10,000
- Paired sign flips: 100,000

## Tested runtime

The experiments were executed in Google Colab with Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, and an NVIDIA Tesla T4. Training loaders use `num_workers=0`. GPU scripts use deterministic seeds and persistent, resume-aware artifacts.

## Artifact policy

Every major stage writes an immutable manifest containing protocol identity, source hashes, output hashes, and a chained SHA-256 digest. The public `results/manifests/` directory contains the final manifests required to audit the released aggregate results.

Large or sensitive intermediates are not distributed:

- raw clinical images and metadata;
- trained checkpoints;
- HDF5 saliency maps;
- latent feature caches;
- sample-level OOF predictions;
- donor-level masks and metrics.

Their expected hashes remain in the manifests so an authorized local reproduction can be checked against the frozen study.

## Statistical unit

Recipient scores are averaged within donor. Donor contrasts are aggregated within dataset-fold-explainer cells, and the principal estimand gives equal weight to cells. Confidence intervals use a paired hierarchical bootstrap over effective patient/case clusters. Primary one-sided inference uses paired cluster sign flips.

## Release validation

Run:

```bash
python tests/validate_release.py
```

The check compiles every public script, validates all JSON files, verifies the 360-cell aggregate table, checks headline estimates and confirms zero exact-k and local-faithfulness violations.
