# Internal and External Validation of a Vision Transformer for Chest X-ray Pneumonia Screening

## Abstract

### Background

Chest radiography is frequently used in pneumonia evaluation, but interpretation demand and reader variability motivate investigation of computer-aided screening tools. This study evaluates a binary deep learning model for pneumonia screening on chest X-ray images.

### Methods

We trained and internally validated convolutional and transformer-based image classifiers using a patient-level validation strategy where available. The primary model was a Vision Transformer (ViT-B/16). Performance was assessed on an internal held-out test set and an external RSNA validation set using a threshold fixed from internal validation. We report discrimination, threshold-based clinical metrics, calibration, decision curve analysis, explainability examples, and subgroup performance.

### Results

To be completed after final analysis.

### Conclusions

To be completed after final analysis. The model should be interpreted as a research prototype for screening support and not as a standalone diagnostic device.

## Introduction

Pneumonia remains a major cause of morbidity and mortality worldwide. Chest X-ray imaging is widely available, but interpretation can be affected by clinical setting, reader expertise, and workload. Deep learning methods have shown promise for triage and screening, yet models require transparent validation, calibration assessment, subgroup analysis, and external evaluation before clinical translation.

This study aims to evaluate a Vision Transformer for binary chest X-ray pneumonia screening under a CLAIM/TRIPOD+AI-aligned workflow. The primary objectives are:

1. To compare ViT-B/16 against common CNN baselines under a unified training policy.
2. To estimate internal test performance with 95% confidence intervals.
3. To externally validate the locked model and threshold on RSNA data without re-tuning.
4. To assess calibration, decision curves, explainability cases, and subgroup performance.

## Methods

### Study Design

Retrospective diagnostic/prognostic model development and validation study using publicly available chest X-ray datasets. The analysis follows an internal training/validation/test workflow and a separate external validation workflow.

### Data Sources

Internal dataset: Kermany pediatric chest X-ray dataset organized as `NORMAL` and `PNEUMONIA`.

External dataset: RSNA Pneumonia Detection Challenge data, converted to binary labels using patient-level maximum `Target`.

### Participants and Images

Inclusion criteria:

- Frontal chest X-ray images available in the source dataset.
- Binary label mappable to `NORMAL` or `PNEUMONIA`.

Exclusion criteria:

- Missing or unreadable image files.
- Labels not mappable to the binary task.
- Duplicate or near-duplicate records flagged during audit may be excluded or reported separately.

### Reference Standard

The reference label is the public dataset-provided binary class. For RSNA, patient-level pneumonia status is derived by collapsing bounding-box annotations to a binary target.

### Dataset Audit

Dataset integrity was assessed using:

- Split/class counts.
- Patient-level distribution by parsed patient identifier.
- Duplicate filenames.
- Exact SHA-256 duplicate image hashes.
- Average-hash near-duplicate candidates.
- Patient overlap across train/validation/test splits.

### Model Development

All models use:

- Input size: `224 x 224`.
- Normalization: ImageNet mean/std.
- Optimizer: AdamW.
- Scheduler: cosine annealing.
- Class order: `NORMAL`, `PNEUMONIA`.
- Threshold decision score: `P(PNEUMONIA)`.

Evaluated models:

- ResNet18.
- ResNet50.
- DenseNet121.
- EfficientNet-B0.
- ViT-B/16.

Each model is trained with three random seeds. The primary model is ViT-B/16 unless the final model comparison indicates otherwise.

### Threshold Selection

The operating threshold is selected only on the internal validation data. The external validation set is evaluated once using the locked threshold, with no re-tuning.

### Statistical Analysis

Primary discrimination metric:

- ROC AUC with 95% bootstrap confidence interval.

Threshold-based metrics:

- Accuracy.
- Sensitivity.
- Specificity.
- PPV.
- NPV.
- F1 score.

Calibration metrics:

- Brier score.
- Expected calibration error (ECE).
- Calibration curve by probability bins.

Model comparison:

- Paired bootstrap ROC AUC difference versus the reference model.
- Two-sided paired bootstrap p value.

Decision analysis:

- Decision curve analysis across threshold probabilities from 0.01 to 0.99.

Subgroup analysis:

- Sex.
- Age group.
- View position.
- Sensitivity and specificity with 95% Wilson confidence intervals.

### Explainability

Grad-CAM heatmaps are generated for pneumonia-focused target class activation. Representative examples include TP, TN, FP, and FN cases, five per category where available.

### Software and Reproducibility

Primary software stack:

- Python 3.10.
- PyTorch.
- timm.
- scikit-learn.
- pandas.
- matplotlib/seaborn.
- pytorch-grad-cam.

Core scripts:

- `audit_dataset.py`.
- `train_all_baselines.py`.
- `evaluate_binary.py`.
- `explain_vit.py`.
- `bias_analysis.py`.

## Results

### Dataset Characteristics

| Dataset | Split | NORMAL images | PNEUMONIA images | Total images | Unique patients | PNEUMONIA:NORMAL ratio | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Kermany | Train | TBD | TBD | TBD | TBD | TBD | TBD |
| Kermany | Validation | TBD | TBD | TBD | TBD | TBD | Threshold selection only |
| Kermany | Internal test | TBD | TBD | TBD | TBD | TBD | Final internal evaluation |
| RSNA | External validation | TBD | TBD | TBD | TBD | TBD | Locked-threshold evaluation |

### Dataset Audit Findings

| Audit item | Internal train | Internal validation | Internal test | External validation | Action |
| --- | ---: | ---: | ---: | ---: | --- |
| Duplicate filenames | TBD | TBD | TBD | TBD | TBD |
| Exact duplicate hashes | TBD | TBD | TBD | TBD | TBD |
| Near-duplicate candidates | TBD | TBD | TBD | TBD | TBD |
| Patient overlap across splits | TBD | TBD | TBD | TBD | TBD |

### Internal Test Performance by Model

| Model | Seed(s) | Threshold | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | PPV (95% CI) | NPV (95% CI) | ROC AUC (95% CI) | PR AUC (95% CI) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 42/1337/2025 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ResNet50 | 42/1337/2025 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DenseNet121 | 42/1337/2025 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| EfficientNet-B0 | 42/1337/2025 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ViT-B/16 | 42/1337/2025 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Paired ROC AUC Comparison

| Reference model | Comparison model | ROC AUC difference | 95% CI lower | 95% CI upper | Paired bootstrap p value |
| --- | --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | ResNet18 | TBD | TBD | TBD | TBD |
| ViT-B/16 | ResNet50 | TBD | TBD | TBD | TBD |
| ViT-B/16 | DenseNet121 | TBD | TBD | TBD | TBD |
| ViT-B/16 | EfficientNet-B0 | TBD | TBD | TBD | TBD |

### External Validation

| Dataset | Model | Locked threshold source | Threshold | Accuracy (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | ROC AUC (95% CI) | PR AUC (95% CI) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RSNA | ViT-B/16 | Internal validation | TBD | TBD | TBD | TBD | TBD | TBD |

### Calibration

| Dataset | Model | Brier score (95% CI) | ECE (95% CI) | Calibration slope | Calibration intercept | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Internal test | ViT-B/16 | TBD | TBD | TBD | TBD | TBD |
| External validation | ViT-B/16 | TBD | TBD | TBD | TBD | TBD |

### Decision Curve Analysis

| Dataset | Clinically relevant threshold range | Model net benefit summary | Treat-all comparison | Treat-none comparison | Notes |
| --- | --- | --- | --- | --- | --- |
| Internal test | TBD | TBD | TBD | TBD | TBD |
| External validation | TBD | TBD | TBD | TBD | TBD |

### Explainability

| Case type | Number displayed | Selection rule | Figure |
| --- | ---: | --- | --- |
| True positive | Up to 5 | Highest `P(PNEUMONIA)` among TP | `results/gradcam_representative_cases.png` |
| True negative | Up to 5 | Lowest `P(PNEUMONIA)` among TN | `results/gradcam_representative_cases.png` |
| False positive | Up to 5 | Highest `P(PNEUMONIA)` among FP | `results/gradcam_representative_cases.png` |
| False negative | Up to 5 | Lowest `P(PNEUMONIA)` among FN | `results/gradcam_representative_cases.png` |

### Subgroup Analysis

| Dataset | Subgroup column | Subgroup value | n | Sensitivity (95% CI) | Specificity (95% CI) | ROC AUC | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Internal test | Sex | TBD | TBD | TBD | TBD | TBD | TBD |
| Internal test | Age group | TBD | TBD | TBD | TBD | TBD | TBD |
| Internal test | View | TBD | TBD | TBD | TBD | TBD | TBD |
| External validation | Sex | TBD | TBD | TBD | TBD | TBD | TBD |
| External validation | Age group | TBD | TBD | TBD | TBD | TBD | TBD |
| External validation | View | TBD | TBD | TBD | TBD | TBD | TBD |

## Discussion

### Principal Findings

To be completed after final analysis. Summarize model discrimination, clinical operating point performance, calibration, external validation shift, and subgroup differences.

### Comparison With Prior Work

To be completed with relevant literature. Discuss how the model compares with published chest X-ray pneumonia classifiers and whether external validation narrows performance estimates.

### Clinical and Research Implications

The model may be useful as a research prototype for prioritizing images or demonstrating end-to-end evaluation methodology. It should not be used as a standalone diagnostic system.

### Strengths

- Unified training and evaluation policy across multiple architectures.
- Patient-level validation strategy where available.
- Locked-threshold external validation.
- Calibration, decision curve, explainability, and subgroup analyses.
- Reproducible scripts for audit, training, evaluation, and reporting.

### Limitations

- Public dataset labels may not match prospective clinical reference standards.
- Kermany and RSNA populations differ in age, acquisition context, and label construction.
- Metadata availability may limit subgroup analysis.
- Grad-CAM provides qualitative localization support but does not prove causal reasoning.
- Retrospective performance does not establish clinical utility or safety.

## Conclusion

To be completed after final analysis. The conclusion should clearly state the validated operating point, external validation findings, calibration behavior, and the remaining evidence needed before any clinical deployment.

## Data Availability

The study uses publicly available datasets. Exact local paths, preprocessing steps, and exclusion decisions should be documented in the final reproducibility appendix.

## Code Availability

Analysis code is contained in this repository. Final commit hash and environment details should be added before submission.

## Ethics Statement

This analysis uses publicly available de-identified datasets. Any use with institutional or patient data would require appropriate ethics, privacy, security, and regulatory review.

## Reporting Checklist Notes

CLAIM/TRIPOD+AI items to verify before submission:

- Dataset source, inclusion/exclusion criteria, and participant flow.
- Reference standard definition.
- Missing data handling.
- Model architecture and training details.
- Hyperparameter selection.
- Threshold selection method.
- Internal and external validation design.
- Uncertainty intervals.
- Calibration and clinical utility assessment.
- Subgroup/fairness analysis.
- Intended use and limitations.
