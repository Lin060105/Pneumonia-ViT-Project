# Internal and External Validation of a Vision Transformer for Chest X-ray Pneumonia Screening

# Vision Transformer 於胸腔 X 光肺炎篩檢之內部與外部驗證

## Abstract

## 摘要

### Background

Chest radiography is widely used for pneumonia assessment, but image interpretation can be affected by reader workload, local practice, and patient population. Deep learning systems may support screening workflows, yet their apparent performance on a single internal dataset may not generalize to external clinical data. This study developed and validated a Vision Transformer (ViT-B/16) for binary chest X-ray pneumonia screening, with comparison against convolutional neural network baselines and external validation on the RSNA Pneumonia Detection Challenge dataset.

### 背景

胸腔 X 光常用於肺炎評估，但判讀結果可能受到醫師工作量、醫療場域習慣與病人族群差異影響。深度學習系統可能協助篩檢流程，但模型在單一內部資料集上的高表現，不一定能推廣到外部臨床資料。本研究開發並驗證一個用於胸腔 X 光二元肺炎篩檢的 Vision Transformer（ViT-B/16），並與卷積神經網路基準模型比較，同時使用 RSNA Pneumonia Detection Challenge 資料集進行外部驗證。

### Methods

We conducted a retrospective model development and validation study using the Kermany pediatric chest X-ray dataset as the internal dataset and RSNA chest radiographs as the external validation dataset. Images were classified as NORMAL or PNEUMONIA. Five architectures were trained under a unified policy: ResNet18, ResNet50, DenseNet121, EfficientNet-B0, and ViT-B/16. Each architecture was trained with three random seeds, ImageNet normalization, 224 x 224 input size, AdamW optimization, and a fixed class order. The primary model was the best ViT-B/16 checkpoint selected by internal validation performance. Final internal and external evaluations used a locked probability threshold of 0.5 without retuning on the test or external set. Metrics included accuracy, sensitivity, specificity, positive predictive value (PPV), negative predictive value (NPV), F1 score, ROC AUC, PR AUC, Brier score, expected calibration error (ECE), and 95% bootstrap confidence intervals. Paired bootstrap comparisons were used for ROC AUC differences between models.

### 方法

本研究為回溯性模型開發與驗證研究，使用 Kermany 兒童胸腔 X 光資料集作為內部資料集，並使用 RSNA 胸腔 X 光資料作為外部驗證資料集。影像被分類為 NORMAL 或 PNEUMONIA。研究在統一訓練策略下訓練五種架構：ResNet18、ResNet50、DenseNet121、EfficientNet-B0 與 ViT-B/16。每種架構皆以三個 random seed 訓練，使用 ImageNet normalization、224 x 224 輸入尺寸、AdamW 最佳化器，以及固定類別順序。主要模型為依內部驗證表現選出的最佳 ViT-B/16 checkpoint。最終內部與外部評估皆使用鎖定的機率門檻 0.5，不在測試集或外部資料集重新調整門檻。評估指標包含 accuracy、sensitivity、specificity、positive predictive value（PPV）、negative predictive value（NPV）、F1 score、ROC AUC、PR AUC、Brier score、expected calibration error（ECE）與 95% bootstrap confidence intervals。模型間 ROC AUC 差異以 paired bootstrap 進行比較。

### Results

The internal Kermany dataset contained 5,216 training images, 16 validation images, and 624 held-out test images. The held-out internal test set had 390 pneumonia images and 234 normal images, with pneumonia prevalence of 62.5%. On internal testing, the primary ViT-B/16 model achieved accuracy 0.856 (95% CI, 0.833-0.881), sensitivity 0.997 (95% CI, 0.992-1.000), specificity 0.620 (95% CI, 0.566-0.677), PPV 0.814 (95% CI, 0.781-0.850), NPV 0.993 (95% CI, 0.979-1.000), ROC AUC 0.980 (95% CI, 0.969-0.989), PR AUC 0.986 (95% CI, 0.976-0.994), Brier score 0.133 (95% CI, 0.107-0.155), and ECE 0.148 (95% CI, 0.121-0.170). On RSNA external validation, using the same locked threshold, the model was evaluated on 26,684 images with pneumonia prevalence of 22.5%. External performance decreased to accuracy 0.473 (95% CI, 0.467-0.479), sensitivity 0.955 (95% CI, 0.950-0.960), specificity 0.333 (95% CI, 0.327-0.340), PPV 0.294 (95% CI, 0.287-0.301), NPV 0.962 (95% CI, 0.958-0.966), ROC AUC 0.801 (95% CI, 0.795-0.807), PR AUC 0.513 (95% CI, 0.499-0.527), Brier score 0.489 (95% CI, 0.483-0.495), and ECE 0.511 (95% CI, 0.505-0.517). In internal model comparison, ViT-B/16 had ROC AUC 0.980; ResNet18, ResNet50, DenseNet121, and EfficientNet-B0 achieved ROC AUC values of 0.971, 0.970, 0.968, and 0.979, respectively.

### 結果

Kermany 內部資料集包含 5,216 張訓練影像、16 張驗證影像與 624 張保留測試影像。內部保留測試集包含 390 張肺炎影像與 234 張正常影像，肺炎盛行率為 62.5%。在內部測試中，主要 ViT-B/16 模型達到 accuracy 0.856（95% CI, 0.833-0.881）、sensitivity 0.997（95% CI, 0.992-1.000）、specificity 0.620（95% CI, 0.566-0.677）、PPV 0.814（95% CI, 0.781-0.850）、NPV 0.993（95% CI, 0.979-1.000）、ROC AUC 0.980（95% CI, 0.969-0.989）、PR AUC 0.986（95% CI, 0.976-0.994）、Brier score 0.133（95% CI, 0.107-0.155）與 ECE 0.148（95% CI, 0.121-0.170）。在 RSNA 外部驗證中，使用相同鎖定門檻，模型於 26,684 張影像上評估，肺炎盛行率為 22.5%。外部表現下降為 accuracy 0.473（95% CI, 0.467-0.479）、sensitivity 0.955（95% CI, 0.950-0.960）、specificity 0.333（95% CI, 0.327-0.340）、PPV 0.294（95% CI, 0.287-0.301）、NPV 0.962（95% CI, 0.958-0.966）、ROC AUC 0.801（95% CI, 0.795-0.807）、PR AUC 0.513（95% CI, 0.499-0.527）、Brier score 0.489（95% CI, 0.483-0.495）與 ECE 0.511（95% CI, 0.505-0.517）。在內部模型比較中，ViT-B/16 的 ROC AUC 為 0.980；ResNet18、ResNet50、DenseNet121 與 EfficientNet-B0 的 ROC AUC 分別為 0.971、0.970、0.968 與 0.979。

### Conclusions

The ViT-B/16 model showed strong internal discrimination and high sensitivity for pediatric chest X-ray pneumonia screening, but performance, calibration, specificity, and PPV decreased substantially under RSNA external validation. These findings emphasize that internal performance alone is insufficient for clinical claims. The model may have value as a high-sensitivity research screening prototype, but external calibration, threshold adaptation, prospective validation, and clinical workflow evaluation are required before clinical deployment.

### 結論

ViT-B/16 模型在兒童胸腔 X 光肺炎篩檢的內部測試中具有良好辨識能力與高敏感度，但在 RSNA 外部驗證下，整體表現、校準、特異度與 PPV 皆明顯下降。這些結果強調，僅憑內部資料集表現不足以支持臨床宣稱。本模型可視為高敏感度研究型篩檢原型，但在臨床部署前仍需要外部校準、門檻調整、前瞻性驗證與臨床流程評估。

## Introduction

## 前言

Pneumonia remains an important cause of morbidity worldwide, and chest radiography is often the first-line imaging examination when pneumonia is suspected. In routine care, chest X-rays are interpreted across heterogeneous settings, including emergency departments, outpatient clinics, pediatric hospitals, and general hospitals. Variability in disease appearance, image acquisition, patient age, and reader expertise can affect diagnostic consistency.

肺炎仍是全球重要的疾病負擔之一，而當臨床懷疑肺炎時，胸腔 X 光通常是第一線影像檢查。在日常醫療中，胸腔 X 光會出現在急診、門診、兒童醫院與一般醫院等不同場域。疾病影像表現、拍攝方式、病人年齡與判讀者經驗的差異，都可能影響診斷一致性。

Deep learning has become a promising approach for chest X-ray triage and screening. Convolutional neural networks (CNNs) have historically dominated medical image classification, while Vision Transformers (ViTs) have introduced a different modeling strategy based on global attention across image patches. However, a common pitfall in medical imaging AI is overinterpreting internal performance: a model may perform well on the dataset it was trained around but fail when exposed to a different hospital, age distribution, disease definition, scanner protocol, or label source.

深度學習已成為胸腔 X 光分流與篩檢的潛在方法。過去醫學影像分類多由卷積神經網路（CNN）主導，而 Vision Transformer（ViT）則提供了不同的建模方式：它透過影像 patch 之間的 global attention 來理解影像。然而，醫學影像 AI 常見陷阱是過度解讀內部資料集表現：模型可能在自己熟悉的資料集上表現良好，但換到不同醫院、年齡分布、疾病定義、掃描流程或標籤來源後，表現便明顯下降。

This study follows a CLAIM/TRIPOD+AI-oriented workflow by reporting dataset structure, model development, internal and external validation, confidence intervals, calibration metrics, decision curve analysis outputs, model comparison, and explainability cases. The main objective was not to claim clinical readiness, but to quantify how a ViT-based pneumonia screening model behaves when moved from a pediatric internal dataset to an adult-dominant external RSNA dataset.

本研究採 CLAIM/TRIPOD+AI 導向流程，報告資料集結構、模型開發、內部與外部驗證、信賴區間、校準指標、decision curve analysis 輸出、模型比較與解釋性案例。研究主要目的不是宣稱模型已可臨床使用，而是量化一個 ViT 肺炎篩檢模型從兒童內部資料集移轉到成人為主的 RSNA 外部資料集時，表現會如何改變。

## Methods

## 方法

### Study Design

We performed a retrospective diagnostic model development and validation study using publicly available chest radiograph datasets. The internal dataset was used for model development, internal validation, and held-out internal testing. The RSNA dataset was reserved for external validation using a locked model and fixed decision threshold.

### 研究設計

本研究為使用公開胸腔 X 光資料集的回溯性診斷模型開發與驗證研究。內部資料集用於模型開發、內部驗證與保留測試。RSNA 資料集則作為外部驗證資料，使用鎖定後的模型與固定決策門檻進行評估。

### Data Sources

The internal dataset was the Kermany pediatric chest X-ray dataset, organized into NORMAL and PNEUMONIA folders. The external dataset was the RSNA Pneumonia Detection Challenge dataset. For RSNA, DICOM images and the stage 2 training labels were used; bounding-box annotations were collapsed into a binary pneumonia label at the image level.

### 資料來源

內部資料為 Kermany 兒童胸腔 X 光資料集，資料夾結構分為 NORMAL 與 PNEUMONIA。外部資料為 RSNA Pneumonia Detection Challenge 資料集。RSNA 使用 DICOM 影像與 stage 2 training labels；原本的 bounding-box 標註被轉換為影像層級的二元肺炎標籤。

### Dataset Characteristics

| Dataset / split | NORMAL | PNEUMONIA | Total | Pneumonia prevalence |
|---|---:|---:|---:|---:|
| Kermany train | 1,341 | 3,875 | 5,216 | 74.3% |
| Kermany validation | 8 | 8 | 16 | 50.0% |
| Kermany internal test | 234 | 390 | 624 | 62.5% |
| RSNA external validation | 20,672 | 6,012 | 26,684 | 22.5% |

### 資料集特徵

| 資料集 / 分割 | NORMAL | PNEUMONIA | 總數 | 肺炎盛行率 |
|---|---:|---:|---:|---:|
| Kermany 訓練集 | 1,341 | 3,875 | 5,216 | 74.3% |
| Kermany 驗證集 | 8 | 8 | 16 | 50.0% |
| Kermany 內部測試集 | 234 | 390 | 624 | 62.5% |
| RSNA 外部驗證集 | 20,672 | 6,012 | 26,684 | 22.5% |

Patient-level audit of the Kermany dataset found 1,211 unique normal patients and 1,635 unique pneumonia patients in the training split, 225 normal and 202 pneumonia patients in the internal test split, and a higher mean number of images per pneumonia patient than per normal patient. Exact and near-duplicate image candidates were detected by SHA-256 and perceptual hash checks; these findings were retained as audit outputs and considered in interpretation.

Kermany 資料集的 patient-level audit 顯示，訓練集中 NORMAL 有 1,211 位 unique patients，PNEUMONIA 有 1,635 位 unique patients；內部測試集中 NORMAL 有 225 位，PNEUMONIA 有 202 位。肺炎病人平均影像張數高於正常病人。資料審計也透過 SHA-256 與 perceptual hash 偵測到 exact duplicate 與 near-duplicate 候選影像；這些結果保留於 audit 輸出中，並納入結果詮釋。

### Model Development

Five architectures were trained: ResNet18, ResNet50, DenseNet121, EfficientNet-B0, and ViT-B/16. Each model used 224 x 224 input images, ImageNet mean and standard deviation normalization, AdamW optimization, a unified class order of NORMAL then PNEUMONIA, and three random seeds. The primary model was ViT-B/16 seed 42, selected as the best ViT run by internal validation balanced accuracy.

### 模型開發

本研究訓練五種模型架構：ResNet18、ResNet50、DenseNet121、EfficientNet-B0 與 ViT-B/16。所有模型皆使用 224 x 224 輸入影像、ImageNet mean/std 正規化、AdamW 最佳化器、固定類別順序 NORMAL 再 PNEUMONIA，並各自訓練三個 random seed。主要模型為 ViT-B/16 seed 42，依內部驗證 balanced accuracy 選為最佳 ViT run。

### Outcome and Threshold

The model output was the predicted probability of PNEUMONIA. A fixed threshold of 0.5 was used for final binary classification in both internal and external evaluation. No threshold tuning was performed on the internal test set or RSNA external validation set.

### 輸出與門檻

模型輸出為 PNEUMONIA 的預測機率。內部與外部最終二元分類皆使用固定門檻 0.5。研究未在內部測試集或 RSNA 外部驗證集重新調整門檻。

### Statistical Analysis

Performance metrics included accuracy, sensitivity, specificity, PPV, NPV, F1 score, ROC AUC, PR AUC, Brier score, and ECE. Confidence intervals were estimated using 200 bootstrap resamples. ROC AUC differences between the primary ViT-B/16 model and CNN baselines were assessed by paired bootstrap comparison.

### 統計分析

模型表現以 accuracy、sensitivity、specificity、PPV、NPV、F1 score、ROC AUC、PR AUC、Brier score 與 ECE 呈現。信賴區間以 200 次 bootstrap resampling 估計。主要 ViT-B/16 模型與 CNN 基準模型之 ROC AUC 差異以 paired bootstrap comparison 評估。

## Results

## 結果

### Internal Test Performance

On the Kermany internal test set, ViT-B/16 produced 389 true positives, 145 true negatives, 89 false positives, and 1 false negative at threshold 0.5. This operating point favored sensitivity and NPV, with sensitivity 0.997 and NPV 0.993, but specificity was lower at 0.620.

### 內部測試表現

在 Kermany 內部測試集中，ViT-B/16 於門檻 0.5 下產生 389 個 true positives、145 個 true negatives、89 個 false positives 與 1 個 false negative。此操作點偏向高 sensitivity 與高 NPV，sensitivity 為 0.997、NPV 為 0.993，但 specificity 較低，為 0.620。

| Metric | Internal value | 95% CI |
|---|---:|---:|
| Accuracy | 0.856 | 0.833-0.881 |
| Sensitivity | 0.997 | 0.992-1.000 |
| Specificity | 0.620 | 0.566-0.677 |
| PPV | 0.814 | 0.781-0.850 |
| NPV | 0.993 | 0.979-1.000 |
| F1 score | 0.896 | 0.877-0.918 |
| ROC AUC | 0.980 | 0.969-0.989 |
| PR AUC | 0.986 | 0.976-0.994 |
| Brier score | 0.133 | 0.107-0.155 |
| ECE | 0.148 | 0.121-0.170 |

| 指標 | 內部測試數值 | 95% CI |
|---|---:|---:|
| Accuracy | 0.856 | 0.833-0.881 |
| Sensitivity | 0.997 | 0.992-1.000 |
| Specificity | 0.620 | 0.566-0.677 |
| PPV | 0.814 | 0.781-0.850 |
| NPV | 0.993 | 0.979-1.000 |
| F1 score | 0.896 | 0.877-0.918 |
| ROC AUC | 0.980 | 0.969-0.989 |
| PR AUC | 0.986 | 0.976-0.994 |
| Brier score | 0.133 | 0.107-0.155 |
| ECE | 0.148 | 0.121-0.170 |

### Model Comparison

Internal model comparison showed that ViT-B/16 and EfficientNet-B0 had the highest ROC AUC values. The AUC difference between ViT-B/16 and EfficientNet-B0 was small and not statistically significant by paired bootstrap testing, whereas differences versus ResNet18, ResNet50, and DenseNet121 were small but statistically detectable.

### 模型比較

內部模型比較顯示，ViT-B/16 與 EfficientNet-B0 具有最高的 ROC AUC。ViT-B/16 與 EfficientNet-B0 的 AUC 差異很小，paired bootstrap test 未達統計顯著；相較於 ResNet18、ResNet50 與 DenseNet121，差異雖小但可被統計偵測。

| Model | Accuracy | Sensitivity | Specificity | ROC AUC | PR AUC |
|---|---:|---:|---:|---:|---:|
| ViT-B/16 | 0.856 | 0.997 | 0.620 | 0.980 | 0.986 |
| ResNet18 | 0.878 | 0.992 | 0.688 | 0.971 | 0.980 |
| ResNet50 | 0.889 | 0.992 | 0.718 | 0.970 | 0.978 |
| DenseNet121 | 0.854 | 0.995 | 0.620 | 0.968 | 0.974 |
| EfficientNet-B0 | 0.880 | 1.000 | 0.679 | 0.979 | 0.985 |

| 模型 | Accuracy | Sensitivity | Specificity | ROC AUC | PR AUC |
|---|---:|---:|---:|---:|---:|
| ViT-B/16 | 0.856 | 0.997 | 0.620 | 0.980 | 0.986 |
| ResNet18 | 0.878 | 0.992 | 0.688 | 0.971 | 0.980 |
| ResNet50 | 0.889 | 0.992 | 0.718 | 0.970 | 0.978 |
| DenseNet121 | 0.854 | 0.995 | 0.620 | 0.968 | 0.974 |
| EfficientNet-B0 | 0.880 | 1.000 | 0.679 | 0.979 | 0.985 |

| Comparison vs ViT-B/16 | AUC difference | 95% CI | p value |
|---|---:|---:|---:|
| ResNet18 | -0.0087 | -0.0205 to -0.0003 | 0.03 |
| ResNet50 | -0.0094 | -0.0188 to -0.0007 | 0.04 |
| DenseNet121 | -0.0115 | -0.0219 to -0.0023 | 0.02 |
| EfficientNet-B0 | -0.0007 | -0.0094 to 0.0058 | 0.86 |

| 與 ViT-B/16 比較 | AUC 差異 | 95% CI | p 值 |
|---|---:|---:|---:|
| ResNet18 | -0.0087 | -0.0205 至 -0.0003 | 0.03 |
| ResNet50 | -0.0094 | -0.0188 至 -0.0007 | 0.04 |
| DenseNet121 | -0.0115 | -0.0219 至 -0.0023 | 0.02 |
| EfficientNet-B0 | -0.0007 | -0.0094 至 0.0058 | 0.86 |

### External Validation on RSNA

When the same ViT-B/16 model and threshold were applied to RSNA, discrimination and calibration worsened. ROC AUC decreased from 0.980 internally to 0.801 externally. PPV decreased from 0.814 to 0.294, while NPV remained high at 0.962. The confusion matrix showed 5,740 true positives, 6,888 true negatives, 13,784 false positives, and 272 false negatives.

### RSNA 外部驗證

當同一個 ViT-B/16 模型與相同門檻套用到 RSNA 時，辨識能力與校準皆變差。ROC AUC 從內部的 0.980 下降到外部的 0.801。PPV 從 0.814 下降到 0.294，而 NPV 仍維持在 0.962。混淆矩陣顯示 true positives 為 5,740、true negatives 為 6,888、false positives 為 13,784、false negatives 為 272。

| Metric | RSNA external value | 95% CI |
|---|---:|---:|
| Accuracy | 0.473 | 0.467-0.479 |
| Sensitivity | 0.955 | 0.950-0.960 |
| Specificity | 0.333 | 0.327-0.340 |
| PPV | 0.294 | 0.287-0.301 |
| NPV | 0.962 | 0.958-0.966 |
| F1 score | 0.450 | 0.441-0.458 |
| ROC AUC | 0.801 | 0.795-0.807 |
| PR AUC | 0.513 | 0.499-0.527 |
| Brier score | 0.489 | 0.483-0.495 |
| ECE | 0.511 | 0.505-0.517 |

| 指標 | RSNA 外部驗證數值 | 95% CI |
|---|---:|---:|
| Accuracy | 0.473 | 0.467-0.479 |
| Sensitivity | 0.955 | 0.950-0.960 |
| Specificity | 0.333 | 0.327-0.340 |
| PPV | 0.294 | 0.287-0.301 |
| NPV | 0.962 | 0.958-0.966 |
| F1 score | 0.450 | 0.441-0.458 |
| ROC AUC | 0.801 | 0.795-0.807 |
| PR AUC | 0.513 | 0.499-0.527 |
| Brier score | 0.489 | 0.483-0.495 |
| ECE | 0.511 | 0.505-0.517 |

### Calibration and Decision Curve Analysis

The internal calibration error was moderate (ECE 0.148), whereas external calibration was poor (ECE 0.511). On RSNA, the highest predicted-probability bin (0.9-1.0) had a mean predicted probability of 0.994 but observed pneumonia fraction of 0.323, indicating substantial overconfidence. Decision curve outputs were generated for internal and external datasets; these should be interpreted as exploratory because the fixed threshold was not clinically optimized for RSNA.

### 校準與 Decision Curve Analysis

內部校準誤差為中等程度（ECE 0.148），但外部校準明顯不佳（ECE 0.511）。在 RSNA 中，最高預測機率區間（0.9-1.0）的平均預測機率為 0.994，但實際肺炎比例只有 0.323，顯示模型在外部資料上明顯過度自信。研究已產生內部與外部 decision curve outputs；由於固定門檻並未針對 RSNA 臨床情境最佳化，這些結果應視為探索性分析。

### Explainability

Grad-CAM representative cases were generated for true positive, true negative, false positive, and false negative examples. These visualizations are intended for qualitative review of model attention and failure modes, not as proof that the model reasons like a radiologist.

### 解釋性分析

本研究產生 true positive、true negative、false positive 與 false negative 的 Grad-CAM 代表案例。這些視覺化結果主要用於定性檢視模型注意區域與錯誤模式，不能被解讀為模型已經像放射科醫師一樣進行推理。

## Discussion

## 討論

This study demonstrates two simultaneous truths. First, a ViT-B/16 model can achieve strong internal discrimination on a pediatric pneumonia chest X-ray benchmark. Second, strong internal performance does not guarantee external reliability. The external RSNA results showed clinically important degradation in discrimination, calibration, specificity, and PPV, despite preservation of high sensitivity and NPV.

本研究同時呈現兩件事。第一，ViT-B/16 模型可以在兒童肺炎胸腔 X 光 benchmark 上達到很強的內部辨識能力。第二，強大的內部表現並不保證外部可靠性。RSNA 外部結果顯示，儘管 sensitivity 與 NPV 仍高，但 discrimination、calibration、specificity 與 PPV 都出現具有臨床意義的下降。

### Domain Gap Between Kermany and RSNA

The performance drop is consistent with a substantial domain gap. The Kermany dataset is pediatric and originates from a curated pneumonia classification task, whereas RSNA is adult-dominant and derived from a detection challenge built on a broader chest radiograph population. Pediatric and adult chest radiographs differ in body size, lung volume, thymic appearance, disease patterns, positioning, acquisition protocols, and the relative frequency of alternative abnormalities. A model trained around pediatric pneumonia patterns may learn features that are useful internally but less specific in adult radiographs.

### Kermany 與 RSNA 之間的 Domain Gap

表現下降符合明顯 domain gap 的現象。Kermany 資料集為兒童資料，且來自經整理的肺炎分類任務；RSNA 則以成人為主，來自建立於更廣泛胸腔 X 光族群上的偵測挑戰。兒童與成人胸腔 X 光在身體大小、肺容量、胸腺外觀、疾病型態、擺位、拍攝流程與其他非肺炎異常的比例上都不同。模型若主要學到兒童肺炎影像特徵，這些特徵在內部資料中可能有效，但在成人影像上可能變得不夠特異。

A second domain gap concerns label definition. Kermany uses image-level NORMAL/PNEUMONIA classes, whereas RSNA labels were derived from radiologist-annotated lung opacity bounding boxes collapsed into a binary target. RSNA also includes images that are abnormal but not pneumonia. Such cases can visually resemble infection or opacity and may increase false positives when evaluated with a pediatric classifier.

第二個 domain gap 來自標籤定義。Kermany 使用影像層級 NORMAL/PNEUMONIA 類別；RSNA 則由放射科醫師標註的 lung opacity bounding boxes 轉換成二元目標。RSNA 也包含「異常但非肺炎」的影像，這類影像可能在視覺上類似感染或陰影，因此當使用兒童資料訓練出的分類器時，false positives 可能增加。

### Prevalence Shift and PPV Collapse

The decrease in PPV is not only a model problem; it is also a mathematical consequence of prevalence shift. The internal test set had pneumonia prevalence of 62.5%, whereas RSNA external validation had prevalence of 22.5%. When disease prevalence drops, even a model with good sensitivity may produce a lower fraction of true positives among all positive predictions, especially if specificity is imperfect. In this study, the external PPV decreased to 0.294 because the model produced many false positives in a lower-prevalence setting.

### 盛行率變化與 PPV 下降

PPV 下降不只是模型問題，也是一個盛行率改變造成的數學結果。內部測試集肺炎盛行率為 62.5%，而 RSNA 外部驗證只有 22.5%。當疾病盛行率下降時，即使模型 sensitivity 不錯，只要 specificity 不夠好，所有被判為陽性的病例中真正陽性的比例就會下降。本研究外部 PPV 降至 0.294，原因是模型在低盛行率情境下產生大量 false positives。

Clinically, this means that the model's positive predictions should not be treated as diagnoses. A positive result may be useful for prioritization or review, but it would require clinician interpretation and likely additional context. Conversely, the high external NPV suggests potential value as a rule-out or triage-support tool, provided that the acceptable miss rate, workflow consequences, and calibration are prospectively validated.

從臨床角度來看，這表示模型的陽性預測不能被當作診斷。陽性結果可能有助於排序或提醒需要審查，但仍需要臨床人員判讀與其他臨床資訊。相對地，外部 NPV 仍高，表示模型可能具有 rule-out 或輔助分流價值，但前提是可接受的漏診率、流程影響與校準都必須經過前瞻性驗證。

### Calibration and Clinical Use

The external ECE of 0.511 and high-probability overconfidence indicate that raw model probabilities are not reliable risk estimates on RSNA. A probability of 0.99 should not be interpreted as a 99% chance of pneumonia in an adult external population. Before clinical use, recalibration using representative local data would be necessary, and decision thresholds should be selected based on the intended workflow, such as high-sensitivity triage, second-reader support, or prioritization of radiologist worklists.

### 校準與臨床使用

外部 ECE 為 0.511，且高機率區間明顯過度自信，表示模型在 RSNA 上輸出的原始機率不能被視為可靠風險估計。模型輸出 0.99 不代表成人外部族群中真的有 99% 機率是肺炎。若要臨床使用，必須用具有代表性的在地資料重新校準，並根據實際流程選擇門檻，例如高敏感度分流、第二讀者輔助，或放射科工作清單排序。

### Clinical Value

Despite external degradation, the model may still have research value. High sensitivity and NPV could support screening experiments where the goal is to avoid missing pneumonia at the cost of more false alarms. However, the large number of false positives on RSNA would increase downstream review burden. Therefore, the most realistic near-term role is not autonomous diagnosis but human-in-the-loop decision support, combined with local validation and recalibration.

### 真實臨床價值

儘管外部表現下降，模型仍具有研究價值。高 sensitivity 與高 NPV 可支援以「盡量不要漏掉肺炎」為目標的篩檢實驗，但代價是更多 false alarms。RSNA 上大量 false positives 會增加後續人工審查負擔。因此，較合理的近期定位不是自動診斷，而是 human-in-the-loop 的決策輔助，並且必須搭配在地驗證與重新校準。

### Limitations

This study has several limitations. First, the internal validation split was small, and the Kermany validation folder contained only 16 images; model selection therefore relied on grouped splitting and available validation outputs but remains sensitive to dataset structure. Second, public dataset labels are imperfect proxies for clinical ground truth. Third, duplicate and near-duplicate candidates were detected in the internal dataset, which may inflate apparent performance if not fully controlled. Fourth, RSNA labels reflect opacity detection annotations rather than a complete clinical diagnosis of pneumonia. Fifth, subgroup fairness analysis was limited by available metadata; sex, age group, and view-position analyses require complete and reliable metadata.

### 限制

本研究有幾項限制。第一，內部 validation split 很小，Kermany 原始 validation folder 僅有 16 張影像；因此模型選擇雖使用 grouped splitting 與可用驗證輸出，但仍可能受資料集結構影響。第二，公開資料集標籤並不等同於完整臨床 ground truth。第三，內部資料集中偵測到 duplicate 與 near-duplicate 候選影像，若未完全控制，可能高估表現。第四，RSNA 標籤反映的是 opacity detection annotation，而非完整臨床肺炎診斷。第五，subgroup fairness analysis 受限於可用 metadata；sex、age group 與 view-position 分析需要完整可靠的 metadata。

## Conclusion

## 結論

A ViT-B/16 model trained for chest X-ray pneumonia screening achieved strong internal discrimination on a pediatric test set but showed substantial performance and calibration degradation under RSNA external validation. The results support the importance of external validation, prevalence-aware interpretation, and calibration assessment before clinical claims. The model should be considered a research prototype for screening support rather than a standalone diagnostic device.

一個用於胸腔 X 光肺炎篩檢的 ViT-B/16 模型，在兒童內部測試集中達到良好辨識能力，但在 RSNA 外部驗證下出現明顯表現與校準下降。本研究結果支持：在提出臨床宣稱之前，必須進行外部驗證、考慮盛行率對指標的影響，並評估模型校準。本模型應被視為篩檢輔助研究原型，而不是獨立診斷醫療器材。

## Reproducibility Notes

## 可重現性註記

Primary model checkpoint: `saved_models/baselines_full/vit-b_16_seed42_best.pth`.

主要模型 checkpoint：`saved_models/baselines_full/vit-b_16_seed42_best.pth`。

Primary internal/external evaluation outputs: `results/baselines_full_eval/vit_rsna_external/`.

主要內部與外部評估輸出：`results/baselines_full_eval/vit_rsna_external/`。

Model comparison outputs: `results/baselines_full_eval/internal_test_performance_summary.csv` and `results/baselines_full_eval/model_comparison_auc_report.csv`.

模型比較輸出：`results/baselines_full_eval/internal_test_performance_summary.csv` 與 `results/baselines_full_eval/model_comparison_auc_report.csv`。

## References

## 參考資料

1. Kermany DS, Goldbaum M, Cai W, et al. Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell. 2018;172(5):1122-1131.
2. Kermany DS, Zhang K, Goldbaum M. Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. Mendeley Data. 2018.
3. RSNA Pneumonia Detection Challenge. Radiological Society of North America. https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018
4. Checklist for Artificial Intelligence in Medical Imaging (CLAIM). Radiological Society of North America. https://pubs.rsna.org/page/ai/claim
5. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378.

