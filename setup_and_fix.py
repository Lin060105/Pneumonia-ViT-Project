"""Repair local encoding issues and set conservative runtime defaults.

This script is intentionally small and repeatable. It rewrites the Traditional
Chinese project manual as UTF-8 and normalizes the Streamlit decision badge text
if mojibake or unsupported glyphs appear again.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANUAL_PATH = ROOT / "docs" / "PROJECT_MANUAL.zh-TW.md"
APP_PATH = ROOT / "app_binary.py"


PROJECT_MANUAL_CONTENT = r"""# 胸腔 X 光肺炎分類研究原型操作手冊

本專案是以胸腔 X 光影像進行 `NORMAL` 與 `PNEUMONIA` 二分類的研究與作品集原型。模型輸出應被解讀為肺炎篩檢風險分數，不是臨床診斷結論。

> 重要聲明：本系統不是醫療器材，也不應直接用於臨床診斷、治療或病患管理。任何真實部署都需要獨立外部驗證、校準、資料隱私與法規審查，並由合格臨床人員監督。

## 1. 專案核心

- 任務：`NORMAL` vs `PNEUMONIA` 二分類。
- 主要模型：`timm` 的 `vit_base_patch16_224` Vision Transformer。
- 影像輸入：`224 x 224`，使用 ImageNet mean/std 正規化。
- 主要決策分數：`P(PNEUMONIA)`。
- 預設門檻：`0.5`，可在 CLI 或 Streamlit app 中調整。
- 不確定區間：若分數接近門檻，可標示為 `REVIEW`，建議人工複核。

## 2. 重要檔案

| 路徑 | 角色 |
| --- | --- |
| `model_utils.py` | 模型建立、checkpoint 載入、前處理、推論與門檻決策。 |
| `train_binary.py` | 單一 ViT 二元分類訓練流程。 |
| `train_all_baselines.py` | 多模型、多 seed baseline 訓練與結果彙整。 |
| `evaluate_binary.py` | 內部/外部測試、bootstrap CI、校準、decision curve 與模型比較。 |
| `audit_dataset.py` | 資料數量、病患層級分佈、檔名重複與影像 hash 審計。 |
| `app_binary.py` | Streamlit 批次上傳與 Grad-CAM 視覺化介面。 |
| `explain_vit.py` | 單張或 TP/TN/FP/FN 代表案例 Grad-CAM 輸出。 |
| `bias_analysis.py` | 使用真實 metadata 執行 subgroup/fairness analysis。 |

## 3. 資料結構

資料夾需符合 torchvision `ImageFolder` 格式，類別名稱必須剛好為 `NORMAL` 與 `PNEUMONIA`。

```text
chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

請勿使用 `chest_xray/test` 做模型選擇、超參數調整或 threshold tuning。test set 只保留給最終評估。

## 4. 安裝與測試

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
.\run_pytest.ps1
```

`run_pytest.ps1` 會設定保守的 OpenMP/MKL 環境變數，降低 Windows/Anaconda 環境中重複載入 OpenMP runtime 的衝突風險。

## 5. 常用指令

啟動 Streamlit app：

```powershell
streamlit run app_binary.py
```

單張推論：

```powershell
python predict.py chest_xray/test/PNEUMONIA/person10_virus_35.jpeg --threshold 0.5
```

資料審計：

```powershell
python audit_dataset.py --data-dir chest_xray --output-dir results
```

多模型 baseline 訓練：

```powershell
python train_all_baselines.py --epochs 10 --batch-size 32 --lr 1e-4
```

最終測試評估：

```powershell
python evaluate_binary.py --model-path saved_models/pneumonia_binary_best.pth --test-dir chest_xray/test --threshold 0.5 --output-dir results
```

使用內部 validation set 鎖定高 sensitivity 門檻後再評估 test/external set：

```powershell
python evaluate_binary.py `
  --model-path saved_models/pneumonia_binary_best.pth `
  --threshold-from-dir chest_xray/val `
  --target-sensitivity 0.95 `
  --test-dir chest_xray/test `
  --external-dir data/rsna_processed `
  --output-dir results
```

TP/TN/FP/FN Grad-CAM 代表案例圖：

```powershell
python explain_vit.py --auto-cases --dataset-dir chest_xray/test --output-dir results
```

Subgroup analysis：

```powershell
python bias_analysis.py `
  --metadata-csv metadata.csv `
  --image-column filename `
  --group-columns sex age_group view `
  --output results/bias_analysis_report.csv
```

## 6. 維護原則

- 保留 `CLASS_NAMES = ("NORMAL", "PNEUMONIA")` 的順序。
- 修改模型、前處理、threshold、checkpoint 或評估邏輯後，至少執行 `.\run_pytest.ps1`。
- 醫療 AI 相關描述應保持保守，不暗示可直接診斷。
- 新增依賴時同步更新 `requirements.txt` 或 `requirements-dev.txt`。
- 大型資料與模型權重不要納入一般 Git diff；資料留在本機，模型權重走 Git LFS。
"""


BADGE_FUNCTION = '''def decision_badge(decision):
    badges = {
        "NORMAL": "Normal",
        "PNEUMONIA": "Pneumonia alert",
        "REVIEW": "Needs manual review",
    }
    return badges[decision]
'''


def configure_openmp_environment() -> None:
    """Set runtime defaults that make local Windows testing less fragile."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def rewrite_project_manual() -> bool:
    MANUAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    old_text = MANUAL_PATH.read_text(encoding="utf-8", errors="replace") if MANUAL_PATH.exists() else ""
    if old_text == PROJECT_MANUAL_CONTENT:
        return False
    MANUAL_PATH.write_text(PROJECT_MANUAL_CONTENT, encoding="utf-8", newline="\n")
    return True


def fix_app_badges() -> bool:
    text = APP_PATH.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"def decision_badge\(decision\):\n"
        r"    badges = \{\n"
        r".*?"
        r"    \}\n"
        r"    return badges\[decision\]\n",
        flags=re.DOTALL,
    )
    if not pattern.search(text):
        raise RuntimeError("Could not locate decision_badge() in app_binary.py")

    updated = pattern.sub(BADGE_FUNCTION, text, count=1)
    if updated == text:
        return False
    APP_PATH.write_text(updated, encoding="utf-8", newline="\n")
    return True


def main() -> None:
    configure_openmp_environment()
    manual_changed = rewrite_project_manual()
    app_changed = fix_app_badges()
    print(f"Manual rewritten: {manual_changed}")
    print(f"App badges fixed: {app_changed}")
    print("OpenMP test environment defaults configured for this process.")


if __name__ == "__main__":
    main()
