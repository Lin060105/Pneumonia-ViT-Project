import logging
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model_utils import (
    CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    PNEUMONIA_INDEX,
    decide_screening_status,
    image_to_tensor,
    load_model_checkpoint,
    predict_probabilities,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

st.set_page_config(page_title="AI Pneumonia Screening", layout="wide")


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


@st.cache_resource
def load_vit_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_model_checkpoint(path, device=device)
    return model, device, metadata


def decision_label(decision):
    labels = {
        "NORMAL": "Normal",
        "PNEUMONIA": "Pneumonia alert",
        "REVIEW": "Needs manual review",
    }
    return labels[decision]


def decision_badge(decision):
    badges = {
        "NORMAL": "Normal",
        "PNEUMONIA": "Pneumonia alert",
        "REVIEW": "Needs manual review",
    }
    return badges[decision]


def build_gradcam(model, image, input_tensor):
    rgb_img = np.array(image.resize(DEFAULT_IMAGE_SIZE))
    rgb_img_float = np.float32(rgb_img) / 255.0
    target_layers = [model.blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(PNEUMONIA_INDEX)],
    )[0, :]
    return show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)


st.title("AI Pneumonia Screening")

with st.expander("Model Evaluation", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("results/clinical_evaluation_plots.png"):
            st.image("results/clinical_evaluation_plots.png", caption="Clinical evaluation plots")
        elif os.path.exists("clinical_evaluation_plots.png"):
            st.image("clinical_evaluation_plots.png", caption="Clinical evaluation plots")
    with col2:
        if os.path.exists("shap_explanation.png"):
            st.image("shap_explanation.png", caption="SHAP explanation example")

st.warning(
    "Research prototype only. Uploaded images should be de-identified, and any deployment "
    "environment must be reviewed for privacy, retention, access control, and regulatory compliance."
)

st.sidebar.header("Screening Settings")
confidence_threshold = st.sidebar.slider(
    "Pneumonia probability threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
)
uncertainty_margin = st.sidebar.slider(
    "Manual review band",
    min_value=0.0,
    max_value=0.20,
    value=0.05,
    step=0.01,
)
run_mode = st.sidebar.radio("Mode", ["Detailed report", "Fast batch"])

model_path = os.getenv("MODEL_PATH", "saved_models/pneumonia_binary_best.pth")

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

model, device, metadata = load_vit_model(model_path)
st.sidebar.caption(f"Model: {metadata.get('model_name', 'unknown')}")
st.sidebar.caption(f"Classes: {', '.join(metadata.get('class_names', CLASS_NAMES))}")

uploaded_files = st.file_uploader(
    "Upload chest X-ray images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.success(f"Loaded {len(uploaded_files)} image(s).")
    results_list = []
    progress_bar = st.progress(0)

    for i, file in enumerate(uploaded_files):
        try:
            image = Image.open(file).convert("RGB")
            input_tensor = image_to_tensor(image, device)
            probabilities = predict_probabilities(model, input_tensor)
            screening = decide_screening_status(
                probabilities,
                threshold=confidence_threshold,
                uncertainty_margin=uncertainty_margin,
            )

            result_text = decision_badge(screening["decision"])
            results_list.append(
                {
                    "File": file.name,
                    "Decision": decision_label(screening["decision"]),
                    "P(Normal)": f"{screening['normal_probability']:.2%}",
                    "P(Pneumonia)": f"{screening['pneumonia_probability']:.2%}",
                    "Threshold": f"{confidence_threshold:.2f}",
                }
            )

            if run_mode == "Detailed report":
                with st.expander(f"Report #{i + 1} - {file.name} ({result_text})", expanded=(i == 0)):
                    visualization = build_gradcam(model, image, input_tensor)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(image, use_container_width=True, caption="Original image")
                    with col2:
                        st.metric("Decision", decision_label(screening["decision"]))
                        st.metric("Pneumonia probability", f"{screening['pneumonia_probability']:.2%}")
                        st.metric("Normal probability", f"{screening['normal_probability']:.2%}")
                    with col3:
                        st.image(visualization, use_container_width=True, caption="Pneumonia-focused Grad-CAM")

        except Exception as exc:
            logging.exception("Failed to process %s", file.name)
            st.error(f"Could not process {file.name}: {exc}")
            results_list.append(
                {
                    "File": file.name,
                    "Decision": "Processing error",
                    "P(Normal)": "N/A",
                    "P(Pneumonia)": "N/A",
                    "Threshold": f"{confidence_threshold:.2f}",
                }
            )

        progress_bar.progress((i + 1) / len(uploaded_files))

    st.markdown("### Batch Report")
    df = pd.DataFrame(results_list)
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download CSV report",
        data=csv,
        file_name="pneumonia_screening_report.csv",
        mime="text/csv",
    )
