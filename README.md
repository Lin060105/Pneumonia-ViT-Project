# 🫁 肺炎 X 光片分類系統 (Level 2: Vision Transformer)

本專案使用最前沿的 **Vision Transformer (ViT)** 模型，將原本的二分類任務升級為更具臨床意義的 **三分類任務**。

## 🎯 專案目標
將胸腔 X 光片自動分類為以下三種：
1.  **NORMAL**: 正常肺部
2.  **BACTERIA**: 細菌性肺炎 (常見特徵：局部實變)
3.  **VIRUS**: 病毒性肺炎 (常見特徵：間質性浸潤)

## 🛠️ 技術架構
* **核心模型**: `vit_base_patch16_224` (Pre-trained on ImageNet)
* **框架**: PyTorch, timm
* **資料集**: Chest X-Ray Images (Pneumonia) - 分割為三類

## 📂 檔案說明
* `data_preprocessing_v2.py`: 資料載入與增強 (Data Augmentation)
* `train_vit.py`: 模型訓練腳本 (Transfer Learning)
* `evaluate_vit.py`: 評估模型效能 (混淆矩陣)
* `predict_vit.py`: 單張圖片推論 (Inference)

## 📊 效能指標 (待填寫)
* **Accuracy**: [執行 evaluate_vit.py 後填入]
* **Recall (Bacteria)**: [執行 evaluate_vit.py 後填入]
* **Recall (Virus)**: [執行 evaluate_vit.py 後填入]

## 📊 效能指標 (Level 2 結果)
* **Overall Accuracy**: 83%
* **Recall (Bacteria)**: 93% (表現最優，極少漏判)
* **Recall (Virus)**: 71% (主要誤判來源，易與細菌混淆)
* **Precision (Normal)**: 93% (高可信度的健康判斷)