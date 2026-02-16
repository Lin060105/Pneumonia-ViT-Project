import streamlit as st
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os  # 必要的路徑處理模組

# --- 1. 初始化設定 ---
st.set_page_config(page_title="AI 肺炎篩檢系統 v3.0", layout="wide")
CLASS_NAMES = ['BACTERIA', 'NORMAL', 'VIRUS']

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# --- 2. 載入模型 (快取) ---
@st.cache_resource
def load_vit_model(path):
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, len(CLASS_NAMES))
    # 這裡載入加權訓練後的模型
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

# --- 3. UI 介面與側邊欄 ---
st.title("🏥 智慧醫療影像篩檢系統 (Vision Transformer)")

# --- 側邊欄：參數設定 ---
st.sidebar.header("⚙️ 參數設定")
confidence_threshold = st.sidebar.slider(
    "信心度門檻 (Confidence Threshold)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.6, 
    step=0.05,
    help="只有當 AI 的信心度高於此數值時，才會顯示診斷結果。"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 快速測試範例")

# !!! 請注意：請確認以下檔名存在於您的電腦中，如果找不到檔案會報錯 !!!
# (這是您之前手動修正過的部分，請依據您電腦實際有的檔案修改)
example_normal = 'chest_xray/test/NORMAL/IM-0001-0001.jpeg'
# 下面這兩個請換成您電腦裡真正存在的檔名
example_bacteria = 'chest_xray/test/BACTERIA/person78_bacteria_378.jpeg' 
example_virus = 'chest_xray/test/VIRUS/person1_virus_11.jpeg' 

# 使用 Session State 來記住現在要顯示哪張圖
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None

col_demo1, col_demo2, col_demo3 = st.sidebar.columns(3)

if col_demo1.button("🟢 正常"):
    st.session_state.selected_image = example_normal

if col_demo2.button("🔴 細菌"):
    st.session_state.selected_image = example_bacteria
    
if col_demo3.button("🟣 病毒"):
    st.session_state.selected_image = example_virus

st.sidebar.info("本系統採二元篩檢模式：區分「健康」與「肺炎風險」。")

# --- 4. 主程式邏輯 ---
st.markdown("### ⚡ 自動化肺炎風險評估")
st.markdown("上傳胸部 X 光片，AI 將自動偵測肺部異常徵兆並標示病灶區域。")

# 設定模型路徑 (指向加權訓練後的新模型)
model_path = 'saved_models/pneumonia_vit_weighted.pth'

if not os.path.exists(model_path):
    st.error(f"找不到模型檔案：{model_path}，請確認是否已完成訓練。")
    st.stop()

# 載入模型
model = load_vit_model(model_path)

# 檔案上傳區
uploaded_file = st.file_uploader("選擇 X 光片圖片...", type=["jpg", "jpeg", "png"])

# --- 圖片來源判斷邏輯 ---
pil_image = None

# 1. 優先使用使用者上傳的圖片
if uploaded_file:
    pil_image = Image.open(uploaded_file).convert('RGB')
# 2. 如果沒上傳，檢查有沒有按範例按鈕
elif st.session_state.selected_image:
    if os.path.exists(st.session_state.selected_image):
        pil_image = Image.open(st.session_state.selected_image).convert('RGB')
    else:
        st.sidebar.error(f"找不到範例圖片: {st.session_state.selected_image}，請檢查程式碼中的路徑。")

# --- 如果有圖片 (不管是上傳的還是範例)，就開始分析 ---
if pil_image is not None:
    img = pil_image
    
    # 圖片預處理
    rgb_img = np.array(img.resize((224, 224)))
    rgb_img_float = np.float32(rgb_img) / 255.0
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img.resize((224, 224))).unsqueeze(0)

    # 設定 Grad-CAM
    target_layers = [model.blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    
    with st.spinner('AI 醫師正在分析影像特徵...'):
        # 預測
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, idx = torch.max(probs, 1)
        
        # 產生熱力圖
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

    # --- 5. 顯示結果 (三欄配置) ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("原始影像")
        st.image(img, use_container_width=True) 
    
    with col2:
        st.subheader("AI 診斷報告")
        
        # --- 修改後的顯示邏輯：嚴格執行二元分類顯示 ---
        if conf.item() > confidence_threshold:
            # 取得 AI 原始預測
            raw_pred = CLASS_NAMES[idx.item()]
            
            # --- 合併邏輯 ---
            if raw_pred == "NORMAL":
                # 如果是正常 -> 顯示綠色
                st.success(f"🟢 檢測結果：正常 (Normal)")
                st.metric("健康信心度", f"{conf.item():.2%}")
                st.info("肺野清晰，未發現顯著浸潤或實變。")
            else:
                # 如果是 BACTERIA 或 VIRUS -> 統一顯示紅色「肺炎」
                # 這裡只告訴使用者有風險，不顯示具體是細菌還是病毒
                st.error(f"🔴 檢測結果：發現肺炎徵兆 (Pneumonia)")
                st.metric("風險信心度", f"{conf.item():.2%}")
                
                # 移除了原本顯示 "AI 內部特徵分析" 的程式碼，避免誤導
                st.warning("建議立即轉診放射科醫師進行鑑別診斷。")
            # ----------------

        else:
            # 信心不足
            st.warning("⚠️ 影像特徵不明顯")
            st.write(f"AI 把握度僅 {conf.item():.2%}，低於門檻 {confidence_threshold}。")
            st.write("建議由醫師進行人工判讀。")
        # ---------------------
            
    with col3:
        st.subheader("病灶熱力圖")
        st.image(visualization, use_container_width=True) 
        st.caption("紅黃色區域為 AI 重點判讀部位")