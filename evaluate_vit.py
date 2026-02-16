import torch
import torch.nn as nn
import timm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from data_preprocessing_v2 import test_loader, class_names # 注意這裡是用 v2

def evaluate_model(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    
    plt.figure(figsize=(10, 8)) # 圖稍微大一點
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
        
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('ViT Confusion Matrix (3-Class)')
    
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(save_path)
    print(f"混淆矩陣圖已儲存至 {save_path}")

if __name__ == '__main__':
    # --- 1. 建立 ViT 模型結構 (必須與訓練時完全一致) ---
    model = timm.create_model('vit_base_patch16_224', pretrained=False) # 這裡 pretrained=False 沒關係，因為我們會載入自己的權重
    
    # 修改分類頭為 3 類
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, len(class_names))

   # model_path = 'saved_models/pneumonia_vit_best.pth'
    model_path = 'saved_models/pneumonia_vit_weighted.pth' # 改成讀取加權版
    
    if os.path.exists(model_path):
        # 這裡加上 weights_only=True 避免警告
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        print("成功載入 ViT 模型權重！")
        
        print("正在評估模型 (ViT)...")
        # --- 3. 執行評估 ---
        predictions, true_labels = evaluate_model(model, test_loader)

        # --- 4. 打印分類報告 ---
        print("Classification Report:")
        print(classification_report(true_labels, predictions, target_names=class_names))

        # --- 5. 繪製混淆矩陣 ---
        plot_confusion_matrix(true_labels, predictions, class_names, save_path='results/confusion_matrix_vit.png')
    else:
        print(f"錯誤：找不到模型檔案 {model_path}")
        print("請等待 train_vit.py 跑完喔！")