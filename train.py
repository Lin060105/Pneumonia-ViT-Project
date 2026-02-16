import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
import copy
import os
# 從我們剛剛寫好的 data_preprocessing.py 引入資料載入器
from data_preprocessing import train_loader, val_loader

def train_model(model, criterion, optimizer, num_epochs=10):
    """
    模型訓練函數
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 檢查是否有可用的 GPU，若有則使用，否則使用 CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"使用的設備: {device}")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每個 epoch 都有一個訓練和驗證階段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 設置模型為訓練模式
                dataloader = train_loader
            else:
                model.eval()   # 設置模型為評估模式
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # 遍歷資料
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向傳播
                # 只有在訓練階段才追蹤梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 只有在訓練階段才進行反向傳播和優化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 統計
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度複製模型權重 (如果這是目前最好的表現)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'訓練耗時 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳驗證準確率: {best_acc:.4f}')

    # 載入最佳模型權重
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # --- 1. 載入預訓練的 ResNet18 模型 ---
    # weights='IMAGENET1K_V1' 會自動下載在 ImageNet 上預訓練好的權重
    model_ft = models.resnet18(weights='IMAGENET1K_V1')

    # --- 2. 凍結所有層的權重 (遷移學習) ---
    for param in model_ft.parameters():
        param.requires_grad = False

    # --- 3. 替換最後的全連接層 ---
    # ResNet18 最後一層的特徵數是 512
    num_ftrs = model_ft.fc.in_features
    # 我們有 2 個類別 (NORMAL, PNEUMONIA)
    model_ft.fc = nn.Linear(num_ftrs, 2)

    # --- 4. 定義損失函數和優化器 ---
    criterion = nn.CrossEntropyLoss()
    
    # 優化器只更新我們剛剛替換掉的全連接層參數 (model_ft.fc)
    optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=0.001)

    # --- 5. 開始訓練 ---
    # 這裡設定訓練 10 個週期 (epochs)
    trained_model = train_model(model_ft, criterion, optimizer_ft, num_epochs=10)

    # --- 6. 儲存模型 ---
    # 確保 saved_models 資料夾存在
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
        
    torch.save(trained_model.state_dict(), 'saved_models/pneumonia_resnet18_best.pth')
    print("最佳模型已儲存至 saved_models/pneumonia_resnet18_best.pth")