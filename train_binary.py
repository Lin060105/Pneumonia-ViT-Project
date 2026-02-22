import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
import os
import numpy as np
from tqdm import tqdm

# --- 1. 設定參數 ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
CLASS_NAMES = ['NORMAL', 'PNEUMONIA'] # 這裡只剩下兩類囉！
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"🚀 開始二分類訓練 (Binary Classification)... 使用裝置: {device}")

    # --- 2. 資料增強與載入 ---
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 設定資料夾路徑 (對應您剛剛整理好的乾淨結構)
    train_dir = os.path.join('chest_xray', 'train')
    val_dir = os.path.join('chest_xray', 'test') # 用 test 當驗證集看最真實的分數

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    print(f"📁 訓練集類別: {train_dataset.classes}")

    # --- 3. 處理資料不平衡 (肺炎圖片比正常圖片多很多) ---
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_dataset.targets]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- 4. 建立模型 (ViT) ---
    print("🧠 載入 ViT 模型...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    
    # 關鍵修改：輸出層從 3 類改為 2 類
    model.head = nn.Linear(model.head.in_features, 2) 
    model = model.to(device)

    # --- 5. 定義損失函數與優化器 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # --- 6. 開始訓練 ---
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())

        train_acc = correct / total
        scheduler.step()

        # --- 驗證 ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        print(f"📊 Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            if not os.path.exists('saved_models'):
                os.makedirs('saved_models')
            # 存成新的權重檔名，避免蓋掉您之前的紀錄
            torch.save(model.state_dict(), 'saved_models/pneumonia_binary_best.pth')
            print(f"💾 模型已儲存 (Acc: {best_acc:.4f})")

    print(f"🎉 訓練完成！最佳準確率: {best_acc:.4f}")

if __name__ == '__main__':
    main()