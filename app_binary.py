import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

# --- 1. 初始化設定 ---
st.set_page_config(page_title="AI 肺炎影像篩檢系統 (批次精準版)", layout="wide")
CLASS_NAMES = ['NORMAL', 'PNEUMONIA'] # 🌟 核心改變：只剩兩類！

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# --- 2. 載入模型 ---
@st.cache_resource
def load_vit_model(path):
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    # 🌟 核心改變：輸出層改為 2
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

# --- 3. UI 介面設計 ---
st.title("🏥 智慧醫療 AI 肺炎篩檢系統 (批次版)")
st.markdown("本系統採用 Vision Transformer 模型，提供**高精準度**的肺炎風險二元篩檢。支援一次上傳多張影像進行批次分析。")

st.sidebar.header("⚙️ 篩檢設定")
confidence_threshold = st.sidebar.slider("AI 警示信心度門檻", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# --- 4. 主程式邏輯 ---
model_path = 'saved_models/pneumonia_binary_best.pth' 

if not os.path.exists(model_path):
    st.error(f"找不到模型檔案：{model_path}，請確認是否訓練成功。")
    st.stop()

model = load_vit_model(model_path)

# 🌟 批次升級：啟用多檔案上傳
uploaded_files = st.file_uploader("📂 請上傳胸部 X 光片 (可一次框選多張 JPG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"📥 成功接收 {len(uploaded_files)} 張影像，AI 醫師已完成初步建檔！")
    st.markdown("---")
    
    # 🌟 批次升級：使用迴圈處理每一張圖片
    for i, file in enumerate(uploaded_files):
        
        # 🌟 批次升級：使用折疊面板 (Expander) 保持版面整潔
        # 預設只展開第一份報告 (i==0)，其他的自動收合
        with st.expander(f"📋 檢測報告 #{i+1} - 檔案名稱: {file.name}", expanded=(i==0)):
            img = Image.open(file).convert('RGB')
            
            # 影像前處理
            rgb_img = np.array(img.resize((224, 224)))
            rgb_img_float = np.float32(rgb_img) / 255.0
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = preprocess(img.resize((224, 224))).unsqueeze(0)

            # Grad-CAM 熱力圖設定
            target_layers = [model.blocks[-1].norm1]
            cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
            
            with st.spinner(f'正在精密分析 {file.name}...'):
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    conf, idx = torch.max(probs, 1)
                    
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
                visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

            # --- 5. 顯示專業報告 ---
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(img, use_container_width=True, caption="原始 X 光片") 
            
            with col2:
                pred_class = CLASS_NAMES[idx.item()]
                confidence = conf.item()
                
                if confidence >= confidence_threshold:
                    if pred_class == "NORMAL":
                        st.success("🟢 **檢測結果：正常 (Normal)**")
                        st.write("影像清澈，未發現明顯肺部浸潤。")
                    else:
                        st.error("🔴 **檢測結果：發現肺炎徵兆 (Pneumonia Detected)**")
                        st.warning("請尋求專業醫師進一步診斷。")
                        
                    st.metric("AI 判斷把握度", f"{confidence:.2%}")
                else:
                    st.warning("⚠️ 影像特徵模糊")
                    st.write("AI 把握度不足，建議人工判讀。")
                    
            with col3:
                st.image(visualization, use_container_width=True, caption="AI 病灶關注區域 (Grad-CAM)")