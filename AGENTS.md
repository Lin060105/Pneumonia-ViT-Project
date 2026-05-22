# AGENTS.md

本檔案供 Codex / 自動化代理在本專案中工作時快速理解背景與限制。請以繁體中文維護；若更新專案資訊，保持格式清晰、可掃讀。

## 最高優先安全規則

- 禁止批量刪除文件或目錄。
- 不要使用：`del /s`、`rd /s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。
- 需要刪除文件時，只能一次刪除一個明確路徑的文件，例如：`Remove-Item "C:\path\to\file.txt"`。
- 如果需要批量刪除文件，停止操作並請用戶手動刪除。
- 開始修改前先檢查 `git status`，不要覆蓋或還原使用者已有的未提交變更。

## 專案概覽

- 專案名稱與用途：AI Pneumonia Screening Research Prototype，使用胸腔 X 光影像做二元肺炎篩檢研究原型。
- 任務型態：`NORMAL` vs `PNEUMONIA` 二分類；決策以肺炎機率 `P(PNEUMONIA)` 與可調門檻為核心。
- 主要模型：`timm` 的 `vit_base_patch16_224` Vision Transformer，輸入尺寸 `224x224`，使用 ImageNet mean/std 正規化。
- 應用入口：`app_binary.py` 提供 Streamlit 批次上傳、CSV 匯出、門檻調整與 Grad-CAM 視覺化。
- 重要聲明：這是研究與作品集原型，不是醫療器材，也不應用於臨床診斷；任何臨床用途都需要獨立驗證、校準、隱私與法規審查、臨床人員監督。

## 技術棧

- Python 版本：專案標示使用 Python `3.10`，`.python-version` 也是 `3.10`。
- 深度學習：PyTorch、torchvision、timm。
- 介面：Streamlit。
- 解釋性：`pytorch-grad-cam`、SHAP。
- 評估與分析：scikit-learn、pandas、matplotlib、seaborn、fairlearn。
- 影像與醫學資料輔助：Pillow、OpenCV headless、pydicom。
- 測試：pytest。
- Docker：以 `python:3.10-slim` 建置，執行 Streamlit 於 `8501`。

## 重要檔案

| 路徑 | 角色 |
| --- | --- |
| `README.md` | 專案說明、快速開始、訓練與評估指令。 |
| `model_utils.py` | 共用模型建立、checkpoint 載入/儲存、影像前處理、機率預測與門檻決策。 |
| `train_binary.py` | 二元 ViT 訓練流程；預設從 `train/` 做 grouped validation split，避免使用 test set 做模型選擇。 |
| `evaluate_binary.py` | 固定 checkpoint 的最終測試評估，輸出臨床指標 CSV 與 ROC/PR/calibration/confusion matrix 圖。 |
| `evaluate.py` | 相容包裝器，直接呼叫 `evaluate_binary.py`。 |
| `predict.py` | 單張影像 CLI 推論。 |
| `app_binary.py` | Streamlit 批次篩檢 UI。 |
| `explain_vit.py` | 對單張影像產生肺炎類別 Grad-CAM heatmap。 |
| `explain_shap.py` | 從資料集中各挑一張圖產生 SHAP 範例圖。 |
| `bias_analysis.py` | 使用真實 metadata 做公平性分析；不會合成敏感屬性。 |
| `data_preprocessing.py` | 以 ImageFolder 格式建立 train/val/test DataLoader。 |
| `setup_dataset.py` | 下載並解壓 Kermany chest X-ray dataset。注意此腳本內部會清理暫存 zip 與解壓資料夾，執行前需確認符合安全規則。 |
| `restructure_dataset.py` | 舊三分類實驗工具；預設 dry-run，`--apply` 會移動檔案並可能刪除空資料夾，使用前需特別小心。 |
| `preprocess_rsna.py` | RSNA 外部驗證資料準備工具，預期 `data/rsna` 下有 Kaggle RSNA DICOM 與 label CSV。檔內部分註解/訊息有亂碼。 |
| `tests/test_basic.py` | 基本測試：模型 forward、門檻決策、checkpoint roundtrip。 |

## 資料集結構

- 預設資料根目錄：`chest_xray/`。
- 必須符合 torchvision `ImageFolder` 結構，且類別名稱必須剛好是 `NORMAL` 與 `PNEUMONIA`。
- 目前本機資料量概況：
  - `chest_xray/train/NORMAL`：1341 張。
  - `chest_xray/train/PNEUMONIA`：3875 張。
  - `chest_xray/val/NORMAL`：8 張。
  - `chest_xray/val/PNEUMONIA`：8 張。
  - `chest_xray/test/NORMAL`：234 張。
  - `chest_xray/test/PNEUMONIA`：390 張。
- `.gitignore` 忽略 `chest_xray/`、`data/`、暫存 zip、虛擬環境與 log；資料集不應直接提交。

## 模型與輸出

- 預設 checkpoint：`saved_models/pneumonia_binary_best.pth`。
- `saved_models/*.pth` 由 Git LFS 管理。
- 其他既有 checkpoint：
  - `saved_models/pneumonia_resnet18_best.pth`
  - `saved_models/pneumonia_vit_best.pth`
  - `saved_models/pneumonia_vit_weighted.pth`
- `model_utils.load_model_checkpoint()` 支援新格式 metadata checkpoint，也支援舊的 plain `state_dict`。
- checkpoint metadata 會保存 model name、class order、image size、mean/std、seed、validation strategy、best validation metrics 等資訊。
- 評估輸出預設寫到 `results/`：
  - `results/clinical_metrics_report.csv`
  - `results/clinical_evaluation_plots.png`
  - `results/confusion_matrix.png`
  - `results/confusion_matrix_vit.png`
- 目前 `results/clinical_metrics_report.csv` 中 threshold `0.5` 的主要指標：Accuracy 約 `0.864`、Sensitivity 約 `0.933`、Specificity 約 `0.748`、AUC-ROC 約 `0.931`、AUC-PR 約 `0.947`。

## 常用指令

安裝依賴：

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

下載資料集：

```bash
python setup_dataset.py
```

啟動 Streamlit app：

```bash
streamlit run app_binary.py
```

訓練：

```bash
python train_binary.py --epochs 10 --batch-size 32 --lr 1e-4
```

最終測試評估：

```bash
python evaluate_binary.py --model-path saved_models/pneumonia_binary_best.pth --test-dir chest_xray/test --threshold 0.5 --output-dir results
```

單張影像推論：

```bash
python predict.py chest_xray/test/PNEUMONIA/person10_virus_35.jpeg --threshold 0.5
```

Grad-CAM：

```bash
python explain_vit.py chest_xray/test/PNEUMONIA/person10_virus_35.jpeg
```

SHAP 範例：

```bash
python explain_shap.py --data-dir chest_xray/test --output shap_explanation.png
```

公平性分析，需要真實 metadata：

```bash
python bias_analysis.py --metadata-csv metadata.csv --image-column filename --group-column age_group --output results/bias_analysis_report.csv
```

測試：

```bash
pytest -q
```

Docker：

```bash
docker build -t pneumonia-vit-app .
docker run -p 8501:8501 pneumonia-vit-app
```

## 訓練與評估注意事項

- 不要使用 `chest_xray/test` 做模型選擇、門檻調整或調參；test set 只留給最終評估。
- `train_binary.py` 預設從 `chest_xray/train` 依 patient id 做 grouped stratified validation split，降低同一病患影像跨 split 的資料洩漏風險。
- 若使用 `--use-existing-val`，需確認 `chest_xray/val` 的來源與用途，不要把 test 資訊帶入訓練決策。
- 若要選擇高 sensitivity 門檻，使用 `evaluate_binary.py --threshold-from-dir <validation_dir> --target-sensitivity <value>`，不要在 test set 上調 threshold。
- `WeightedRandomSampler` 用於處理類別不平衡；改訓練流程時注意不要移除平衡策略或破壞 class order。
- `CLASS_NAMES` 固定為 `("NORMAL", "PNEUMONIA")`；所有資料夾、metadata、輸出報告都應保持同一順序。

## Streamlit App 注意事項

- `app_binary.py` 透過環境變數 `MODEL_PATH` 指定模型，預設為 `saved_models/pneumonia_binary_best.pth`。
- UI 支援多圖上傳、`Pneumonia probability threshold`、`Manual review band`、`Detailed report` / `Fast batch`。
- `Detailed report` 會為每張圖產生 Grad-CAM，速度較慢。
- app 會顯示研究用途警告；涉及真實醫療影像時，必須要求去識別化與隱私保護。
- `decision_badge()` 目前含有疑似亂碼符號；若修改 UI，建議一併改成乾淨的文字或可正確顯示的圖示。

## 已知風險與維護提示

- 部分檔案有亂碼註解或字串，例如 `.dockerignore`、`CONTRIBUTING.md`、`preprocess_rsna.py`、`app_binary.py` 的 badge 字串；新增或整理文件時請使用 UTF-8 與繁體中文。
- `setup_dataset.py` 會下載約 1.2GB 資料，且需要網路；在受限環境中可能失敗。
- `setup_dataset.py` 的 `finally` 區塊會移除暫存 zip，並以遞迴方式清理解壓資料夾；在本工作區要特別遵守最高優先安全規則。
- `restructure_dataset.py --apply` 會移動影像到 `BACTERIA` / `VIRUS` 舊三分類資料夾，這不符合目前二分類主流程；除非明確需要舊實驗，否則只用 dry-run。
- 大型模型檔與資料集不宜納入一般 Git diff；模型權重走 Git LFS，資料集留在本機。
- 修改模型、前處理、class order、threshold 決策或 checkpoint 格式後，至少執行 `pytest -q`，並考慮重新跑 `evaluate_binary.py` 產生新的結果。

## 協作風格

- 優先延續現有程式風格與 helper，例如模型/前處理相關邏輯應集中使用 `model_utils.py`。
- 變更醫療 AI 相關輸出時，要保守表述，不要暗示可直接診斷。
- 對資料、模型與結果的說明需區分「目前本機檔案狀態」與「可重現流程」。
- 需要新增依賴時，同步更新 `requirements.txt`；測試依賴放在 `requirements-dev.txt`。
