import torch
import cv2
import numpy as np
import timm
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
from torchvision import transforms
import sys
import os

# 類別名稱 (必須與訓練時順序一致)
# 根據資料夾排序通常是: BACTERIA (0), NORMAL (1), VIRUS (2)
CLASS_NAMES = ['BACTERIA', 'NORMAL', 'VIRUS']

# --- 1. 定義 ViT 的 Reshape Transform ---
# 這是最關鍵的一步！因為 ViT 輸出的不是圖片而是序列，必須手動轉回 2D 圖片格式
def reshape_transform(tensor, height=14, width=14):
    # 去掉 class token (第一個)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # 將通道轉到正確維度 (N, C, H, W)
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def run_explanation(image_path, model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # --- 2. 載入模型 ---
    print(f"載入模型: {model_path}...")
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # --- 3. 設定 Grad-CAM ---
    # ViT 的目標層通常是最後一個 Block 的 Normalization 層
    target_layers = [model.blocks[-1].norm1]

    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    # --- 4. 讀取與處理圖片 ---
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1] # BGR to RGB
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img_float = np.float32(rgb_img) / 255.0

    # 轉成 Tensor
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(Image.fromarray(rgb_img)).unsqueeze(0).to(device)

    # --- 5. 產生熱力圖 ---
    print("正在生成熱力圖...")
    # 這裡 targets=None 表示讓 AI 解釋它覺得「機率最高」的那個類別
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # 將熱力圖疊加在原圖上
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

    # --- 6. 顯示預測結果文字 ---
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, idx = torch.max(probs, 1)
        pred_label = CLASS_NAMES[idx.item()]
        
    print(f"AI 診斷: {pred_label} (信心度: {conf.item():.2%})")
    
    # 存檔
    save_name = "heatmap_result.jpg"
    cv2.imwrite(save_name, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"熱力圖已儲存為: {save_name} (請打開它來看看！)")

if __name__ == '__main__':
    model_file = 'saved_models/pneumonia_vit_best.pth'
    
    # 預設測試圖片
    test_img = 'chest_xray/test/BACTERIA/person1_bacteria_1.jpeg' # 找張細菌的來試試
    
    if len(sys.argv) > 1:
        test_img = sys.argv[1]

    if os.path.exists(test_img):
        run_explanation(test_img, model_file)
    else:
        # 如果找不到細菌圖，試試看正常的
        backup_img = 'chest_xray/test/NORMAL/IM-0001-0001.jpeg'
        if os.path.exists(backup_img):
             run_explanation(backup_img, model_file)
        else:
             print(f"錯誤：找不到圖片 {test_img}")