"""
Pneumonia Binary Classification Evaluator
此腳本用於全面評估訓練好的 ViT 模型，計算醫療 AI 關鍵指標：
包含 Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, AUC-ROC，並繪製混淆矩陣。
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. 基本設定 ---
BATCH_SIZE = 32
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
MODEL_PATH = 'saved_models/pneumonia_binary_best.pth'
TEST_DIR = os.path.join('chest_xray', 'test') # 使用測試集進行最終評估
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"🔍 開始執行專業醫療指標評估... 使用裝置: {device}")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型檔案 {MODEL_PATH}，請先訓練模型。")
        return

    # --- 2. 載入資料 ---
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- 3. 載入模型 ---
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # --- 4. 進行預測 ---
    all_labels = []
    all_preds = []
    all_probs = [] # 用於計算 AUC-ROC

    print("⏳ 正在掃描測試集影像...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) # 取出 PNEUMONIA 的機率

    # --- 5. 計算醫療核心指標 ---
    acc = accuracy_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    # 提取混淆矩陣數值 (TN, FP, FN, TP)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) # 敏感度 (Recall) -> 真正生病且被抓出來的比例
    specificity = tn / (tn + fp) # 特異度 -> 真沒病且被確認沒病的比例

    print("\n" + "="*40)
    print(" 🏥 醫療 AI 核心指標分析報告 (Test Set)")
    print("="*40)
    print(f"🔸 準確率 (Accuracy):    {acc:.4f}")
    print(f"🔸 敏感度 (Sensitivity): {sensitivity:.4f} (極重要：漏診率的指標)")
    print(f"🔸 特異度 (Specificity): {specificity:.4f}")
    print(f"🔸 AUC-ROC 分數:         {auc_roc:.4f}")
    print("\n📊 詳細分類報告 (Precision / Recall / F1-Score):")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # --- 6. 繪製並儲存混淆矩陣 ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - Pneumonia Screening')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("💾 混淆矩陣圖表已儲存為 'confusion_matrix.png'")

if __name__ == '__main__':
    main()