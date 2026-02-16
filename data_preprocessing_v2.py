import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

data_dir = './chest_xray'
train_dir = os.path.join(data_dir, 'train')
# 原始 val 資料夾太爛了，我們暫時用 test 資料夾來做驗證
val_dir = os.path.join(data_dir, 'test')
test_dir = os.path.join(data_dir, 'test')

# --- 修改這一段 ---
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # 角度稍微加大
    # 新增：隨機調整亮度與對比，模擬不同品質的 X 光片
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    # 新增：輕微的位移與縮放
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# ------------------

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 讓其他檔案可以匯入類別名稱
class_names = train_dataset.classes

if __name__ == '__main__':
    print(f"訓練集樣本數: {len(train_dataset)}")
    print(f"測試集樣本數: {len(test_dataset)}")
    print(f"類別名稱: {class_names}")