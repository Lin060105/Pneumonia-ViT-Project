import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import sys
import os

# --- 設定類別名稱 (手動對應比較保險) ---
# 根據資料夾排序，通常是 BACTERIA, NORMAL, VIRUS，但建議跑 evaluate.py 確認
# 這裡我們先動態讀取，或者您稍後確認 output
CLASS_NAMES = ['BACTERIA', 'NORMAL', 'VIRUS'] 

def predict_image(image_path, model_path):
    if not os.path.exists(image_path):
        return "錯誤：找不到圖片檔案"

    # 1. 準備模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, len(CLASS_NAMES))
    
    # 載入權重
    if not os.path.exists(model_path):
        return "錯誤：找不到模型檔案"
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 2. 圖片處理 (ViT 需要 224x224)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
    except Exception as e:
        return f"圖片讀取錯誤: {e}"

    # 3. 推論
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        score = confidence.item()

    return predicted_class, score

if __name__ == '__main__':
    model_file = 'saved_models/pneumonia_vit_best.pth'
    
    if len(sys.argv) < 2:
        print("使用方式: python predict_vit.py <圖片路徑>")
        # 預設測試一張 (請確保路徑存在)
        test_img = 'chest_xray/test/NORMAL/IM-0001-0001.jpeg'
        if os.path.exists(test_img):
            print(f"範例測試: {test_img}")
            cls, conf = predict_image(test_img, model_file)
            print(f"預測結果: {cls} (信心度: {conf:.2%})")
    else:
        img_path = sys.argv[1]
        cls, conf = predict_image(img_path, model_file)
        print(f"預測結果: {cls} (信心度: {conf:.2%})")