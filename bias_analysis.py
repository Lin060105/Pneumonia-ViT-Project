"""
Pneumonia ViT - Bias & Fairness Analysis (using Fairlearn)
用於分析模型在不同病患群體 (如年齡、性別) 中是否存在預測偏差 (Demographic Parity)。
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, false_negative_rate, selection_rate
from sklearn.metrics import accuracy_score
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'saved_models/pneumonia_binary_best.pth'
TEST_DIR = os.path.join('chest_xray', 'test')

def main():
    logging.info("⚖️ 開始執行 Fairlearn 醫療偏差與公平性分析...")
    
    # 1. 載入模型
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 2. 載入測試集
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 3. 模擬敏感特徵 (因為 Kermany 沒有開源年齡性別，我們生成模擬資料來展示架構能力)
    np.random.seed(42)
    # 假設測試集中，病患被隨機分為 'Under 18', '18-65', 'Over 65' 三個年齡層
    age_groups = np.random.choice(['Under 18', '18-65', 'Over 65'], size=len(y_true))
    
    # 4. 使用 Fairlearn 計算指標
    metrics = {
        'Accuracy': accuracy_score,
        'False Negative Rate (漏診率)': false_negative_rate,
        'Selection Rate (確診比例)': selection_rate
    }
    
    mf = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=age_groups)
    
    # 5. 匯出報告
    report_df = mf.by_group
    report_df.to_csv('bias_analysis_report.csv')
    
    logging.info("\n📊 群體公平性指標 (By Age Group):")
    print(report_df)
    
    dpd = mf.difference(method='between_groups')['Selection Rate (確診比例)']
    logging.info(f"\n⚠️ Demographic Parity Difference (群組最大差異): {dpd:.4f}")
    logging.info("✅ 偏差分析報告已匯出至 bias_analysis_report.csv")

if __name__ == '__main__':
    main()