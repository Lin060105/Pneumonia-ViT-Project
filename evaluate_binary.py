"""
Pneumonia Binary Classification Evaluator (Clinical Grade)
進階醫療評估腳本：包含 Bootstrapping 置信區間、校準曲線、詳細指標 CSV 匯出。
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BATCH_SIZE = 32
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
MODEL_PATH = os.getenv('MODEL_PATH', 'saved_models/pneumonia_binary_best.pth') # 支援環境變數
TEST_DIR = os.path.join('chest_xray', 'test')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_confidence_interval(y_true, y_pred, metric_func, n_bootstraps=1000, ci=95):
    """使用 Bootstrapping 計算 95% 置信區間"""
    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = resample(np.arange(len(y_pred)), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    lower_bound = np.percentile(sorted_scores, (100 - ci) / 2.0)
    upper_bound = np.percentile(sorted_scores, 100 - (100 - ci) / 2.0)
    return lower_bound, upper_bound

def main():
    logging.info(f"開始執行臨床級醫療指標評估... 使用裝置: {device}")

    if not os.path.exists(MODEL_PATH):
        logging.error(f"找不到模型檔案 {MODEL_PATH}。")
        return

    # 載入資料與模型
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    logging.info("正在掃描測試集影像以獲取預測結果...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 計算基本指標
    acc = accuracy_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # 計算 95% 置信區間 (CI)
    logging.info("正在計算 95% 置信區間 (Bootstrapping)...")
    sens_lower, sens_upper = calculate_confidence_interval(all_labels, all_preds, lambda y, p: confusion_matrix(y, p).ravel()[3] / (confusion_matrix(y, p).ravel()[3] + confusion_matrix(y, p).ravel()[2]))
    spec_lower, spec_upper = calculate_confidence_interval(all_labels, all_preds, lambda y, p: confusion_matrix(y, p).ravel()[0] / (confusion_matrix(y, p).ravel()[0] + confusion_matrix(y, p).ravel()[1]))

    # 建立報告 DataFrame 並匯出 CSV
    metrics_data = {
        "Metric": ["Accuracy", "Sensitivity (Recall)", "Specificity", "AUC-ROC"],
        "Value": [acc, sensitivity, specificity, auc_roc],
        "95% CI Lower": [np.nan, sens_lower, spec_lower, np.nan],
        "95% CI Upper": [np.nan, sens_upper, spec_upper, np.nan]
    }
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv("clinical_metrics_report.csv", index=False)
    logging.info("✅ 臨床指標數值報告已匯出至 clinical_metrics_report.csv")

    # 繪製圖表 (混淆矩陣與校準曲線)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 混淆矩陣
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # 2. 校準曲線 (Calibration Curve) - 醫療 AI 必備，看預測機率是否可信
    prob_true, prob_pred = calibration_curve(all_labels, all_probs, n_bins=10)
    axes[1].plot(prob_pred, prob_true, marker='o', linewidth=2, label='ViT Model')
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    axes[1].set_title('Calibration Curve')
    axes[1].set_xlabel('Mean Predicted Probability')
    axes[1].set_ylabel('Fraction of Positives')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('clinical_evaluation_plots.png', dpi=300)
    logging.info("✅ 醫療評估圖表已儲存為 clinical_evaluation_plots.png")

if __name__ == '__main__':
    main()