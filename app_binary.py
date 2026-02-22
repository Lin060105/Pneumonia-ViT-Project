import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd  # 🌟 新增：處理表格與 Excel 輸出
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

# --- 1. 初始化設定 ---
st.set_page_config(page_title="AI 肺炎影像篩檢系統 (專業批次版)", layout="wide")
CLASS_NAMES = ['NORMAL', 'PNEUMONIA'] 

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# --- 2. 載入模型 ---
@st.cache_resource
def load_vit_model(path):
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

# --- 3. UI 介面設計 ---
st.title("🏥 智慧醫療 AI 肺炎篩檢系統")
st.markdown("支援單張深入分析與**大批量數據匯出 (Excel)**，大幅提升臨床篩檢效率。")

st.sidebar.header("⚙️ 篩檢設定")
confidence_threshold = st.sidebar.slider("AI 警示信心度門檻", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# 🌟 新增功能：模式切換器
st.sidebar.markdown("---")
run_mode = st.sidebar.radio(
    "🔍 選擇檢測模式", 
    ["詳細報告模式 (含熱力圖與影像)", "快速批次模式 (僅產出 Excel 總表)"],
    help="如果一次上傳超過 10 張圖片，建議使用『快速批次模式』以節省等待時間。"
)

# --- 4. 主程式邏輯 ---
model_path = 'saved_models/pneumonia_binary_best.pth' 

if not os.path.exists(model_path):
    st.error(f"找不到模型檔案：{model_path}，請確認是否訓練成功。")
    st.stop()

model = load_vit_model(model_path)
uploaded_files = st.file_uploader("📂 請上傳胸部 X 光片 (可一次框選多張 JPG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"📥 成功接收 {len(uploaded_files)} 張影像，AI 醫師開始執行判讀...")
    st.markdown("---")
    
    # 用來收集所有預測結果的清單
    results_list = []
    
    # 建立一個進度條，批次處理時讓使用者知道進度
    progress_bar = st.progress(0)

    for i, file in enumerate(uploaded_files):
        img = Image.open(file).convert('RGB')
        
        # 影像前處理
        rgb_img = np.array(img.resize((224, 224)))
        rgb_img_float = np.float32(rgb_img) / 255.0
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(img.resize((224, 224))).unsqueeze(0)

        # 核心預測邏輯 (兩種模式都需要跑)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, idx = torch.max(probs, 1)
            
        pred_class = CLASS_NAMES[idx.item()]
        confidence = conf.item()

        # 判斷狀態
        if confidence >= confidence_threshold:
            status = "🟢 正常 (Normal)" if pred_class == "NORMAL" else "🔴 發現肺炎徵兆"
        else:
            status = "⚠️ 影像模糊/信心度不足"

        # 將結果存入清單，準備最後轉成 Excel
        results_list.append({
            "檔案名稱": file.name,
            "AI 診斷結果": status,
            "信心度": f"{confidence:.2%}"
        })

        # ---------------------------------------------------------
        # 模式一：如果選擇「詳細報告模式」，就繼續畫熱力圖並顯示圖片
        # ---------------------------------------------------------
        if "詳細" in run_mode:
            with st.expander(f"📋 檢測報告 #{i+1} - {file.name} ({status})", expanded=(i==0)):
                # 計算 Grad-CAM (最耗時的步驟，所以在快速模式中被跳過了)
                target_layers = [model.blocks[-1].norm1]
                cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
                visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(img, use_container_width=True, caption="原始 X 光片") 
                with col2:
                    st.write(f"**診斷結果:** {status}")
                    st.metric("AI 判斷把握度", f"{confidence:.2%}")
                with col3:
                    st.image(visualization, use_container_width=True, caption="AI 病灶關注區域 (Grad-CAM)")
                    
        # 更新進度條
        progress_bar.progress((i + 1) / len(uploaded_files))

    # ---------------------------------------------------------
    # 模式二：批次處理完成後，永遠在最下方顯示總表並提供下載
    # ---------------------------------------------------------
    st.markdown("### 📊 批次篩檢數據總表")
    
    # 轉換成 Pandas 表格
    df = pd.DataFrame(results_list)
    
    # 在網頁上顯示表格
    st.dataframe(df, use_container_width=True)

    # 準備下載用的 CSV 檔案 (加入 utf-8-sig 編碼確保 Excel 打開中文不會亂碼)
    csv = df.to_csv(index=False).encode('utf-8-sig')
    
    st.download_button(
        label="📥 下載 Excel (CSV) 篩檢報告",
        data=csv,
        file_name="肺炎篩檢報告_AI_Report.csv",
        mime="text/csv",
    )