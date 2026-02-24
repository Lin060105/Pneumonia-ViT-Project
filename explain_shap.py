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
    logging.info("🔬 初始化 SHAP 進階模型解釋器 (使用黑盒遮罩法，完美支援 ViT)...")
    
    # 1. 載入模型
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 2. 建立預測函數 (讓 SHAP 傳入乾淨的 numpy 圖片，我們在內部進行正規化並預測)
    def predict_fn(images_np):
        # images_np 是 SHAP 傳進來的陣列，形狀為 (N, 224, 224, 3)
        images_tensor = torch.tensor(images_np).permute(0, 3, 1, 2).float().to(device)
        
        # 手動進行 ImageNet 正規化 (為了預測)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        images_tensor = (images_tensor - mean) / std
        
        with torch.no_grad():
            outputs = model(images_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    # 3. 載入測試集 (這裡「不使用」正規化，保持圖片原本的樣子讓 SHAP 畫圖)
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_dataset = datasets.ImageFolder(os.path.join('chest_xray', 'test'), transform=base_transform)
    
    # 取 2 張圖來解釋 (1張正常，1張肺炎)
    # 測試集通常前面是 Normal，後面是 Pneumonia。我們各取一張確保多樣性。
    idx_normal = 0
    idx_pneumonia = len(test_dataset) - 1
    test_images_tensor = torch.stack([test_dataset[idx_normal][0], test_dataset[idx_pneumonia][0]])
    
    # 轉換為 (2, 224, 224, 3) 的 Numpy 陣列供 SHAP 使用
    test_images_np = test_images_tensor.permute(0, 2, 3, 1).numpy()

    # 4. 建立 SHAP Explainer
    # 使用影像遮罩法 (把區塊模糊化代表拿掉特徵)
    masker = shap.maskers.Image("blur(128,128)", test_images_np[0].shape)
    explainer = shap.Explainer(predict_fn, masker, output_names=CLASS_NAMES)

    # 5. 計算 SHAP 值 (max_evals 控制細緻度，設為 500 取得平衡)
    logging.info("⏳ 正在計算像素貢獻度，這可能需要幾分鐘，請稍候...")
    shap_values = explainer(test_images_np, max_evals=500, outputs=shap.Explanation.argsort.flip[:1])

    # 6. 繪製圖表並存檔 (修正存白圖的問題)
    # 加入 show=False，讓它在背景畫圖就好，不要彈出視窗清空畫布
    shap.image_plot(shap_values, show=False)
    
    # 抓取目前的畫布並直接存檔
    fig = plt.gcf()
    fig.savefig('shap_explanation.png', dpi=300, bbox_inches='tight')
    
    # 存檔完成後，把畫布從記憶體中關閉
    plt.close(fig)
    logging.info("✅ SHAP 解釋圖表已重新生成並儲存為 shap_explanation.png")

if __name__ == '__main__':
    main()