# Datasets

The confirmatory study uses three public medical-imaging datasets. Images are not redistributed by this repository.

| Dataset | Eligible images used | Endpoint | Grouping unit |
|---|---:|---|---|
| SIIM-ACR | 11,840 | Pneumothorax | Case-level proxy |
| ISIC 2016 | 898 | Melanoma | Lesion/case identifier |
| PAD-UFES-20 | 2,284 | Cancer vs non-cancer | Patient |

The eligible counts follow validation, conflict removal, and duplicate/group auditing. The preprocessing manifest in `results/manifests/preprocessing.json` records the exclusions and fold sizes.

## Acquisition

- **SIIM-ACR Pneumothorax Segmentation:** obtain the original competition data under its applicable terms.
- **ISIC 2016:** obtain the official challenge images and metadata from the ISIC Archive/challenge distribution.
- **PAD-UFES-20:** the executed pipeline used the KaggleHub distribution identified in `configs/datasets.json`, content-validated against the referenced PAD-UFES-20 release.

The scripts expect the persistent project structure created by stage 1 and record cryptographic hashes of derived manifests. Do not mix alternative dataset revisions within a run.

BUS-UCLM appears in the initial registry and ResNet development scripts only. It was a development smoke dataset and is not part of the confirmatory panel or reported TransXAI results.

## Privacy

Do not commit raw images, clinical metadata, sample-level predictions, donor-level metrics, patient/case identifiers, checkpoints, or cached tensors. The supplied `.gitignore` excludes the standard generated locations.
