import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- 1. 定義資料路徑 ---
# 確保 chest_xray 資料夾與此腳本位於同一個主專案資料夾下
data_dir = './chest_xray'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# --- 2. 定義資料轉換 (Transformations) ---
# ResNet18 模型期望的輸入尺寸是 224x224
# 訓練集使用資料增強 (Data Augmentation) 來增加模型泛化能力
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 調整圖片大小
    transforms.RandomHorizontalFlip(), # 隨機水平翻轉
    transforms.RandomRotation(10),     # 隨機旋轉 (-10 到 +10 度)
    transforms.ToTensor(),             # 將圖片轉換為 PyTorch Tensor
    # 使用 ImageNet 的平均值和標準差進行標準化
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 驗證集與測試集不做資料增強，只需做標準化處理
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. 建立 PyTorch Datasets ---
# ImageFolder 會自動從資料夾名稱讀取標籤 (NORMAL, PNEUMONIA)
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

# --- 4. 建立 DataLoaders ---
# DataLoader 能以批次 (batch) 的方式加載資料，並可選擇是否打亂順序
BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# ...前面省略...
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 【移到這裡！】讓所有程式都能共用這個類別名稱
class_names = train_dataset.classes

# ---5. 驗證與資訊顯示(可選)---
if __name__ == '__main__':
    print(f"訓練集樣本數: {len(train_dataset)}")
    print(f"驗證集樣本數: {len(val_dataset)}")
    print(f"測試集樣本數: {len(test_dataset)}")
    
    # 顯示類別與索引的對應關係
    print(f"類別名稱: {class_names}")
    print(f"類別名稱: {class_names}")
    print(f"類別索引: {train_dataset.class_to_idx}")
    
    # 取一個批次的資料來看看
    images, labels = next(iter(train_loader))
    print(f"一個批次的圖片 Tensor 形狀: {images.shape}") # 預期: torch.Size([32, 3, 224, 224])
    print(f"一個批次的標籤 Tensor 形狀: {labels.shape}") # 預期: torch.Size([32])