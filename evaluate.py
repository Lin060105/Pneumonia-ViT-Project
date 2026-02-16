import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
# 從我們寫好的 data_preprocessing.py 引入測試資料集與類別名稱
from data_preprocessing import test_loader, class_names

def evaluate_model(model, dataloader):
    """
    在測試集上評估模型性能
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 設置為評估模式

    all_preds = []
    all_labels = []

    with torch.no_grad():  # 在評估時不計算梯度，省記憶體
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """
    繪製並儲存混淆矩陣圖
    """
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    
    plt.figure(figsize=(8, 6))
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
        
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    
    # 儲存圖片
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(save_path)
    print(f"混淆矩陣圖已儲存至 {save_path}")

if __name__ == '__main__':
    # --- 1. 建立模型結構 (與訓練時相同) ---
    # 這裡 weights=None，因為我們要載入自己的權重，而不是線上的
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 類別數為 2

    # --- 2. 載入我們訓練好的模型權重 ---
    # 注意：這個檔案要等回家跑完 train.py 才會出現
    model_path = 'saved_models/pneumonia_resnet18_best.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("成功載入模型權重！")
        
        # --- 3. 執行評估 ---
        print("正在評估模型...")
        predictions, true_labels = evaluate_model(model, test_loader)

        # --- 4. 打印分類報告 ---
        report = classification_report(true_labels, predictions, target_names=class_names)
        print("Classification Report:")
        print(report)

        # --- 5. 繪製並儲存混淆矩陣 ---
        plot_confusion_matrix(true_labels, predictions, class_names, save_path='results/confusion_matrix.png')
    else:
        print(f"錯誤：找不到模型檔案 {model_path}")
        print("請先執行 python train.py 完成訓練後再來執行此腳本。")