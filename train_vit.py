import torch
import torch.nn as nn
import torch.optim as optim
import timm  # 引入 timm 函式庫來載入 ViT
import time
import copy
import os
from data_preprocessing_v2 import train_loader, val_loader, class_names # 從 v2 匯入

def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"使用的設備: {device}")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'訓練耗時 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳驗證準確率: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # --- 1. 使用 timm 載入預訓練的 ViT 模型 ---
    model_vit = timm.create_model('vit_base_patch16_224', pretrained=True)

    # --- 2. 凍結所有層的權重 ---
    for param in model_vit.parameters():
        param.requires_grad = False

    # --- 3. 替換 ViT 的分類頭 ---
    num_ftrs = model_vit.head.in_features
    num_classes = len(class_names)  # 這裡會自動抓到 3 個類別
    model_vit.head = nn.Linear(num_ftrs, num_classes)
    print(f"已將 ViT 模型的分類頭修改為 {num_classes} 個輸出。")

    # --- 4. 定義損失函數和優化器 ---
    # --- 4. 定義損失函數 (加入類別權重，解決病毒被忽視的問題) ---
    # 根據資料量計算權重: Bacteria(~2530), Normal(~1341), Virus(~1345)
    # 權重設定為: [1.0, 1.9, 1.9] (讓 AI 更重視正常和病毒)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor([1.0, 1.9, 1.9]).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights) # <--- 這裡把權重放進去
    optimizer_vit = optim.Adam(model_vit.head.parameters(), lr=0.001)

    # --- 5. 開始訓練 ---
    trained_model = train_model(model_vit, criterion, optimizer_vit, num_epochs=10)

# --- 6. 儲存模型 ---
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    # 改個名字，叫做 weighted 版本
    torch.save(trained_model.state_dict(), 'saved_models/pneumonia_vit_weighted.pth')
    print("ViT 最佳模型 (加權版) 已儲存至 saved_models/pneumonia_vit_weighted.pth")