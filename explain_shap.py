"""
Pneumonia ViT - Advanced Explainability using SHAP (SHapley Additive exPlanations)
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import timm
import shap
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'saved_models/pneumonia_binary_best.pth'

def main():
    print("🔬 初始化 SHAP 進階模型解釋器...")
    
    # 載入模型
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 準備一小部分資料作為 SHAP 的背景分佈
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(os.path.join('chest_xray', 'test'), transform=transform)
    
    # 取 5 張圖當背景，2 張圖來解釋 (因為 SHAP 運算極度消耗資源)
    background_images = torch.stack([test_dataset[i][0] for i in range(5)]).to(device)
    test_images = torch.stack([test_dataset[i][0] for i in range(5, 7)]).to(device)

    # 建立 SHAP GradientExplainer
    explainer = shap.GradientExplainer(model, background_images)
    shap_values, indexes = explainer.shap_values(test_images, ranked_outputs=1)

    # 整理 Tensor 回 Numpy 以便畫圖
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)
    
    # 繪製 SHAP 圖表並存檔
    shap.image_plot(shap_numpy, -test_numpy, show=False)
    plt.savefig('shap_explanation.png', dpi=300, bbox_inches='tight')
    print("✅ SHAP 解釋圖表已儲存為 shap_explanation.png")

if __name__ == '__main__':
    main()