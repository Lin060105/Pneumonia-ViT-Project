import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

# --- 1. 定義預測函數 ---
def predict_image(image_path, model, class_names):
    """
    對單張圖片進行預測
    """
    # 檢查檔案是否存在
    if not os.path.exists(image_path):
        return "Error: Image path does not exist.", 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 設置為評估模式

    # 圖片預處理 (必須和訓練時的驗證/測試集轉換一致)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 載入圖片並處理
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image)
        # PyTorch 模型需要一個 batch 維度，所以我們用 unsqueeze(0) 增加一個維度
        # 變成 [1, 3, 224, 224]
        image_tensor = image_tensor.unsqueeze(0).to(device)
    except Exception as e:
        return f"Error processing image: {e}", 0.0

    with torch.no_grad():
        outputs = model(image_tensor)
        # 使用 softmax 獲取機率
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # 獲取最大機率的索引和值
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = class_names[predicted_idx.item()]

    return predicted_class, confidence.item()

if __name__ == '__main__':
    # --- 2. 建立模型並載入權重 (與 evaluate.py 相同) ---
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # 模型權重路徑 (回家跑完訓練才會出現這個檔案)
    model_path = 'saved_models/pneumonia_resnet18_best.pth'
    class_names = ['NORMAL', 'PNEUMONIA']

    if os.path.exists(model_path):
        # 載入訓練好的參數
        model.load_state_dict(torch.load(model_path))
        
        # --- 3. 處理命令行輸入 ---
        if len(sys.argv) != 2:
            print("\n[使用方法]")
            print(f"python {sys.argv[0]} <圖片路徑>\n")
            print("[範例]")
            # 提供一個存在的範例圖片路徑 (測試集中的第一張正常圖片)
            sample_image = os.path.join('chest_xray', 'test', 'NORMAL', 'IM-0001-0001.jpeg')
            if os.path.exists(sample_image):
                print(f"python {sys.argv[0]} {sample_image}")
            else:
                print(f"python {sys.argv[0]} chest_xray/test/PNEUMONIA/person10_virus_35.jpeg")
        else:
            image_to_predict = sys.argv[1]
            print(f"正在預測圖片: {image_to_predict} ...")
            
            predicted_class, confidence = predict_image(image_to_predict, model, class_names)

            if "Error" in predicted_class:
                print(predicted_class)
            else:
                print("-" * 30)
                print(f"預測結果: {predicted_class}")
                print(f"信心度:   {confidence:.4f} ({confidence*100:.2f}%)")
                print("-" * 30)
    else:
        print(f"錯誤：找不到模型檔案 {model_path}")
        print("請先執行 python train.py 完成訓練後再來執行此腳本。")