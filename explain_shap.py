"""
Pneumonia ViT - Advanced Explainability using SHAP (Partition Explainer)
專為 Vision Transformer 設計的黑盒遮罩法，解決梯度斷裂導致熱圖空白的問題。
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import timm
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'saved_models/pneumonia_binary_best.pth'
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

def main():
    logging.info("🔬 初始化 SHAP 進階模型解釋器 (使用原始 Logits 避免機率飽和)...")
    
    # 1. 載入模型
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 2. 建立預測函數
    def predict_fn(images_np):
        images_tensor = torch.tensor(images_np).permute(0, 3, 1, 2).float().to(device)
        
        # 手動進行 ImageNet 正規化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        images_tensor = (images_tensor - mean) / std
        
        with torch.no_grad():
            outputs = model(images_tensor)
            # 🌟 終極修改：把 probs = softmax(...) 刪掉！
            # 直接回傳 outputs (模型原始得分 Logits)，這樣 SHAP 數值就會被放大，顏色會超明顯！
        return outputs.cpu().numpy()

    # 3. 載入測試集
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_dataset = datasets.ImageFolder(os.path.join('chest_xray', 'test'), transform=base_transform)
    
    idx_normal = 0
    idx_pneumonia = len(test_dataset) - 1
    test_images_tensor = torch.stack([test_dataset[idx_normal][0], test_dataset[idx_pneumonia][0]])
    test_images_np = test_images_tensor.permute(0, 2, 3, 1).numpy()

    # 4. 建立 SHAP Explainer
    masker = shap.maskers.Image("blur(128,128)", test_images_np[0].shape)
    explainer = shap.Explainer(predict_fn, masker, output_names=CLASS_NAMES)

    # 5. 計算 SHAP 值
    logging.info("⏳ 正在計算像素貢獻度 (使用 Logits)...")
    shap_values = explainer(test_images_np, max_evals=500, outputs=shap.Explanation.argsort.flip[:1])

    # 6. 繪製圖表並存檔
    shap.image_plot(shap_values, show=False)
    fig = plt.gcf()
    fig.savefig('shap_explanation.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info("✅ SHAP 解釋圖表已重新生成並儲存為 shap_explanation.png")

if __name__ == '__main__':
    main()