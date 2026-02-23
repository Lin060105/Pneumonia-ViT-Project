import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import logging

# 🌟 新增：設定日誌記錄 (Logging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="AI 肺炎影像篩檢系統 (臨床版)", layout="wide")
CLASS_NAMES = ['NORMAL', 'PNEUMONIA'] 

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

@st.cache_resource
def load_vit_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model, device

st.title("🏥 智慧醫療 AI 肺炎篩檢系統")

# 🌟 新增：醫療隱私與合規警語 (符合 HIPAA 精神)
st.warning("🔒 **隱私與安全提示**：本系統僅供學術測試與輔助篩檢。請確保上傳之影像已去除個人識別資訊 (De-identified) 以符合相關醫療隱私法規。系統不會在伺服器端永久儲存您的影像。")

st.sidebar.header("⚙️ 篩檢設定")
confidence_threshold = st.sidebar.slider("AI 警示信心度門檻", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
run_mode = st.sidebar.radio("🔍 選擇檢測模式", ["詳細報告模式 (含熱力圖)", "快速批次模式 (僅產出 Excel)"])

model_path = os.getenv('MODEL_PATH', 'saved_models/pneumonia_binary_best.pth')

if not os.path.exists(model_path):
    st.error(f"找不到模型檔案：{model_path}")
    st.stop()

model, device = load_vit_model(model_path)
uploaded_files = st.file_uploader("📂 請上傳胸部 X 光片 (支援 JPG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"📥 成功接收 {len(uploaded_files)} 張影像，開始判讀...")
    results_list = []
    progress_bar = st.progress(0)

    for i, file in enumerate(uploaded_files):
        # 🌟 新增：Robust 錯誤處理 (防呆機制)
        try:
            img = Image.open(file).convert('RGB')
            rgb_img = np.array(img.resize((224, 224)))
            rgb_img_float = np.float32(rgb_img) / 255.0
            
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = preprocess(img.resize((224, 224))).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, idx = torch.max(probs, 1)
                
            pred_class = CLASS_NAMES[idx.item()]
            confidence = conf.item()
            status = "🟢 正常 (Normal)" if (pred_class == "NORMAL" and confidence >= confidence_threshold) else "🔴 發現肺炎徵兆"

            results_list.append({"檔案名稱": file.name, "AI 診斷結果": status, "信心度": f"{confidence:.2%}"})

            if "詳細" in run_mode:
                with st.expander(f"📋 報告 #{i+1} - {file.name} ({status})", expanded=(i==0)):
                    # 將模型與 tensor 轉回 CPU 給 Grad-CAM 計算
                    model_cpu = model.cpu()
                    input_tensor_cpu = input_tensor.cpu()
                    target_layers = [model_cpu.blocks[-1].norm1]
                    cam = GradCAM(model=model_cpu, target_layers=target_layers, reshape_transform=reshape_transform)
                    grayscale_cam = cam(input_tensor=input_tensor_cpu, targets=None)[0, :]
                    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
                    
                    # 計算完記得把模型推回 GPU 以備下一張圖使用
                    model = model.to(device)

                    col1, col2, col3 = st.columns(3)
                    with col1: st.image(img, use_container_width=True, caption="原始 X 光片") 
                    with col2: 
                        st.write(f"**診斷結果:** {status}")
                        st.metric("AI 判斷把握度", f"{confidence:.2%}")
                    with col3: st.image(visualization, use_container_width=True, caption="AI 關注區域 (Grad-CAM)")
                    
        except Exception as e:
            logging.error(f"處理檔案 {file.name} 時發生錯誤: {str(e)}")
            st.error(f"⚠️ 檔案 {file.name} 處理失敗，可能不是有效的影像格式。")
            results_list.append({"檔案名稱": file.name, "AI 診斷結果": "❌ 處理失敗", "信心度": "N/A"})

        progress_bar.progress((i + 1) / len(uploaded_files))

    st.markdown("### 📊 批次篩檢數據總表")
    df = pd.DataFrame(results_list)
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(label="📥 下載 Excel (CSV) 篩檢報告", data=csv, file_name="肺炎篩檢報告.csv", mime="text/csv")