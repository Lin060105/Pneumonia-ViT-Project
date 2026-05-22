# 肺炎 X 光分類專案操作手冊

本文件提供第一次接觸本專案的人一條可以照著走的路徑：從理解專案、建立環境、準備資料，到訓練、評估、推論與啟動 Streamlit 介面。

> 重要提醒：本專案是研究與作品集原型，不是醫療器材，也不能作為診斷依據。任何臨床用途都需要獨立驗證、校準、隱私審查、法規審查與臨床人員監督。

## 1. 專案在做什麼

本專案使用胸腔 X 光影像做二元分類：

- `NORMAL`：正常
- `PNEUMONIA`：肺炎

模型主要使用 Vision Transformer，預設架構是 `vit_base_patch16_224`。輸入影像會被縮放到 `224 x 224`，並使用 ImageNet 的 mean/std 做標準化。模型輸出兩個機率：

- `P(NORMAL)`
- `P(PNEUMONIA)`

系統再用肺炎機率門檻值做篩檢決策：

- `P(PNEUMONIA) >= threshold`：判為 `PNEUMONIA`
- `P(PNEUMONIA) < threshold`：判為 `NORMAL`
- 若肺炎機率落在門檻附近的人工覆核區間，判為 `REVIEW`

核心決策邏輯位於 `model_utils.py` 的 `decide_screening_status()`。

## 2. 主要功能

- 訓練二元肺炎分類模型
- 使用乾淨的 validation split，避免用 test set 做模型選擇
- 單張影像命令列推論
- Streamlit 批次上傳介面與 CSV 匯出
- Grad-CAM 熱區圖
- SHAP 解釋範例
- Test set 臨床指標評估：accuracy、sensitivity、specificity、PPV、NPV、ROC AUC、PR AUC、Brier score、bootstrap CI
- 有真實 metadata 時可做 fairness/bias analysis
- Docker 部署
- Pytest 基礎測試

## 3. 專案結構

```text
.
├── app_binary.py                  # Streamlit 批次篩檢介面
├── model_utils.py                 # 模型建立、checkpoint、前處理、推論與決策共用工具
├── train_binary.py                # 二元 ViT 訓練腳本
├── evaluate_binary.py             # test set 評估與臨床指標輸出
├── predict.py                     # 單張影像推論
├── explain_vit.py                 # Grad-CAM 熱區圖
├── explain_shap.py                # SHAP 解釋圖
├── bias_analysis.py               # 需真實 metadata 的公平性分析
├── data_preprocessing.py          # ImageFolder dataloader 輔助工具
├── setup_dataset.py               # Kermany chest X-ray dataset 下載與解壓
├── restructure_dataset.py         # 舊三分類實驗用資料重整工具，預設 dry-run
├── preprocess_rsna.py             # RSNA DICOM 外部驗證資料前處理
├── saved_models/                  # 已訓練模型權重
├── results/                       # 評估報告與圖表
├── chest_xray/                    # 資料集
└── tests/                         # 單元測試
```

目前本機資料集結構如下：

```text
chest_xray/
├── train/
│   ├── NORMAL/       1342 images
│   └── PNEUMONIA/    3876 images
├── val/
│   ├── NORMAL/       9 images
│   └── PNEUMONIA/    9 images
└── test/
    ├── NORMAL/       234 images
    └── PNEUMONIA/    390 images
```

目前已存在的主要模型：

```text
saved_models/pneumonia_binary_best.pth
saved_models/pneumonia_vit_best.pth
saved_models/pneumonia_vit_weighted.pth
saved_models/pneumonia_resnet18_best.pth
```

一般使用請優先使用：

```text
saved_models/pneumonia_binary_best.pth
```

## 4. 環境建置

建議使用 Python 3.10 的乾淨虛擬環境，不建議直接混用大型 Anaconda base 環境。

### Windows PowerShell

```powershell
cd D:\Pneumonia_Classification_PyTorch_L2_forCodex
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

如果 PowerShell 不允許啟動虛擬環境，先執行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### macOS/Linux

```bash
cd /path/to/Pneumonia_Classification_PyTorch_L2_forCodex
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

## 5. 資料集格式

本專案使用 PyTorch `ImageFolder` 格式。類別資料夾名稱必須剛好是：

- `NORMAL`
- `PNEUMONIA`

標準資料夾格式：

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

注意：

- `train_binary.py` 預設不使用 `chest_xray/val`，而是從 `chest_xray/train` 依 patient id 做 grouped validation split。
- `chest_xray/test` 應只在最後評估時使用。
- 不要用 test set 調 threshold 或挑模型。

## 6. 如果需要下載資料集

如果 `chest_xray/` 不存在，可以使用：

```powershell
python setup_dataset.py
```

注意：`setup_dataset.py` 會下載約 1.2GB 的 Kermany chest X-ray dataset，解壓後建立 `chest_xray/`。

依照本專案目前的檔案操作規範，不建議自行批次刪除資料夾。若下載或解壓中斷而留下暫存資料夾，請先確認路徑，再手動處理。

## 7. 啟動 Streamlit 批次篩檢介面

最容易上手的入口是 Streamlit：

```powershell
streamlit run app_binary.py
```

開啟瀏覽器中的網址，通常是：

```text
http://localhost:8501
```

介面功能：

- 上傳一張或多張 `jpg/jpeg/png` 胸腔 X 光影像
- 調整肺炎機率門檻值
- 調整人工覆核區間
- 選擇詳細報告或快速批次模式
- 顯示每張圖的正常/肺炎機率
- 詳細模式會產生 pneumonia-focused Grad-CAM
- 匯出 CSV 批次報告

預設模型路徑：

```text
saved_models/pneumonia_binary_best.pth
```

如需改用其他模型，可以設定 `MODEL_PATH`：

```powershell
$env:MODEL_PATH="saved_models/pneumonia_vit_best.pth"
streamlit run app_binary.py
```

## 8. 單張影像推論

使用 `predict.py` 對單張影像輸出模型決策與機率：

```powershell
python predict.py chest_xray/test/PNEUMONIA/person10_virus_35.jpeg --threshold 0.5
```

常用參數：

```powershell
python predict.py `
  chest_xray/test/PNEUMONIA/person10_virus_35.jpeg `
  --model-path saved_models/pneumonia_binary_best.pth `
  --threshold 0.5 `
  --uncertainty-margin 0.05
```

輸出範例會包含：

```text
Model: vit_base_patch16_224
Image: ...
Decision: PNEUMONIA / NORMAL / REVIEW
P(Normal): ...
P(Pneumonia): ...
Threshold: ...
```

## 9. 訓練模型

基本訓練：

```powershell
python train_binary.py --epochs 10 --batch-size 32 --lr 1e-4
```

常用參數：

```powershell
python train_binary.py `
  --data-dir chest_xray `
  --epochs 10 `
  --batch-size 32 `
  --lr 1e-4 `
  --val-split 0.15 `
  --output-path saved_models/pneumonia_binary_best.pth
```

Windows 若遇到 dataloader worker 問題，可以改成：

```powershell
python train_binary.py --epochs 10 --batch-size 32 --lr 1e-4 --num-workers 0
```

訓練流程重點：

- 使用 `timm` 建立 `vit_base_patch16_224`
- 輸出類別數為 2
- 預設載入 pretrained backbone
- 使用 `WeightedRandomSampler` 處理類別不平衡
- optimizer 是 `AdamW`
- scheduler 是 `CosineAnnealingLR`
- 以 validation balanced accuracy 選 best checkpoint
- checkpoint 會保存模型權重與 metadata

## 10. 最終評估

使用固定 checkpoint 在 test set 上評估：

```powershell
python evaluate_binary.py `
  --model-path saved_models/pneumonia_binary_best.pth `
  --test-dir chest_xray/test `
  --threshold 0.5 `
  --output-dir results
```

輸出：

```text
results/clinical_metrics_report.csv
results/clinical_evaluation_plots.png
```

報告包含：

- Accuracy
- Sensitivity
- Specificity
- Precision/PPV
- NPV
- F1
- ROC AUC
- PR AUC
- Brier score
- Confusion matrix
- Bootstrap 95% confidence intervals

## 11. 在 validation set 上選 threshold

如果要依目標 sensitivity 選 threshold，必須使用獨立 validation folder，不要用 test set：

```powershell
python evaluate_binary.py `
  --model-path saved_models/pneumonia_binary_best.pth `
  --threshold-from-dir chest_xray/val `
  --target-sensitivity 0.95 `
  --test-dir chest_xray/test `
  --output-dir results
```

輸出會多一份：

```text
results/threshold_selection_report.csv
```

## 12. 產生 Grad-CAM 熱區圖

```powershell
python explain_vit.py chest_xray/test/PNEUMONIA/person10_virus_35.jpeg --output heatmap_result.jpg
```

輸出：

```text
heatmap_result.jpg
```

這張圖會顯示模型針對 `PNEUMONIA` 類別關注的區域。它可用於研究展示，但不能當作臨床依據。

## 13. 產生 SHAP 解釋圖

```powershell
python explain_shap.py --data-dir chest_xray/test --output shap_explanation.png
```

若覺得太慢，可以降低 `--max-evals`：

```powershell
python explain_shap.py --data-dir chest_xray/test --output shap_explanation.png --max-evals 200
```

## 14. Fairness / Bias Analysis

`bias_analysis.py` 不會製造假的人口學資料。你必須提供真實 metadata CSV。

CSV 至少需要：

- 影像檔名欄位，例如 `filename`
- 分組欄位，例如 `age_group`、`sex`、`scanner_site`、`hospital_id`

執行範例：

```powershell
python bias_analysis.py `
  --metadata-csv metadata.csv `
  --image-column filename `
  --group-column age_group `
  --output results/bias_analysis_report.csv
```

輸出會按群組列出：

- Accuracy
- False Negative Rate
- Selection Rate

## 15. Docker 使用

建立 image：

```powershell
docker build -t pneumonia-vit-app .
```

啟動容器：

```powershell
docker run -p 8501:8501 pneumonia-vit-app
```

開啟：

```text
http://localhost:8501
```

## 16. 執行測試

```powershell
pytest -q
```

如果在 Anaconda base 環境遇到 `torch` 或 `numpy` 載入時 abort，請改用本文件第 4 節的乾淨 Python 3.10 虛擬環境。

## 17. 常見問題

### 找不到模型

錯誤：

```text
Model file not found: saved_models/pneumonia_binary_best.pth
```

處理方式：

- 確認 `saved_models/pneumonia_binary_best.pth` 是否存在
- 或設定 `MODEL_PATH`
- 或重新訓練產生 checkpoint

### 找不到資料

錯誤通常會提到：

```text
Training folder not found
Expected classes ('NORMAL', 'PNEUMONIA')
```

處理方式：

- 確認資料夾是 `chest_xray/train/NORMAL` 與 `chest_xray/train/PNEUMONIA`
- 類別資料夾名稱大小寫必須一致
- 不要把 `BACTERIA`、`VIRUS` 當作主訓練類別，除非你要做舊版三分類實驗

### 訓練很慢

可能原因：

- ViT 模型較大
- 使用 CPU 訓練會很慢
- SHAP 也會很慢

處理方式：

- 有 GPU 時確認 PyTorch 能偵測 CUDA
- 減少 epochs
- 降低 batch size
- SHAP 降低 `--max-evals`

### Windows dataloader 卡住

嘗試：

```powershell
python train_binary.py --num-workers 0
```

或：

```powershell
python evaluate_binary.py --num-workers 0
```

## 18. 新人建議操作路線

第一次接觸專案時，建議照以下順序：

1. 建立 Python 3.10 虛擬環境
2. 安裝 `requirements.txt` 與 `requirements-dev.txt`
3. 確認 `chest_xray/` 與 `saved_models/` 是否存在
4. 先跑單張推論，確認模型可載入
5. 啟動 Streamlit，熟悉使用者流程
6. 跑 final evaluation，理解模型指標
7. 需要時再訓練新模型
8. 最後再做 Grad-CAM、SHAP、fairness analysis

推薦最小驗證指令：

```powershell
python predict.py chest_xray/test/PNEUMONIA/person10_virus_35.jpeg --threshold 0.5
streamlit run app_binary.py
python evaluate_binary.py --model-path saved_models/pneumonia_binary_best.pth --test-dir chest_xray/test --threshold 0.5 --output-dir results
```

## 19. 交付或展示前檢查清單

- 使用乾淨 validation strategy 訓練
- test set 只用於最終評估
- threshold 不從 test set 調整
- `results/clinical_metrics_report.csv` 已重新產生
- `results/clinical_evaluation_plots.png` 已重新產生
- Demo 使用去識別化影像
- 明確標註此專案不是醫療診斷工具
- 若有公平性分析，metadata 必須是真實且合規取得

