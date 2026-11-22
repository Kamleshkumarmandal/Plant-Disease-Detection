from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from plant_disease_labels import IDX_TO_CLASSES

st.set_page_config(
    page_title="PlantCare AI | Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DISEASE_INFO_PATH = Path("disease_info.csv")


@st.cache_data(show_spinner=False)
def load_disease_information(path: str, modified_time: float) -> dict[int, dict]:
    del modified_time
    frame = pd.read_csv(path, encoding="utf-8")
    required = {"index", "disease_name", "description", "Possible Steps"}
    if not required.issubset(frame.columns):
        raise ValueError("Disease information CSV is missing required columns.")
    frame["index"] = pd.to_numeric(frame["index"], errors="raise").astype(int)
    return frame.set_index("index").to_dict(orient="index")


def configured_model_path() -> Path | None:
    configured = os.getenv("PLANT_DISEASE_MODEL_PATH", "").strip()
    if not configured:
        try:
            configured = str(st.secrets.get("PLANT_DISEASE_MODEL_PATH", "")).strip()
        except Exception:
            configured = ""
    if configured:
        candidate = Path(configured).expanduser()
        return candidate if candidate.is_file() else None

    candidates = [
        Path("plant_disease_model_1_latest.pt"),
        Path("models/plant_disease_model_1_latest.pt"),
        Path("Plant-Disease-Detection-main/Flask Deployed App/plant_disease_model_1_latest.pt"),
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


@st.cache_resource(show_spinner="Loading the trained CNN model...")
def load_cached_model(model_path: str, modified_time: float):
    # Only model files supplied by the project owner should be loaded.
    del modified_time
    from inference import load_model

    return load_model(model_path, device="cpu")


def readable_label(class_name: str) -> str:
    if class_name == "Background_without_leaves":
        return "Background without leaves"
    parts = class_name.split("___", maxsplit=1)
    if len(parts) == 1:
        return parts[0].replace("_", " ").replace(",", "")
    crop, condition = parts
    crop = crop.replace(",", "").replace("_", " ").strip()
    condition = condition.replace("_", " ").replace(",", "").strip()
    return f"{crop} — {condition}"


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#f5f8f4;color:#1e2e24}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e5ece4}
.brand{font:800 25px Manrope;letter-spacing:-1.2px;color:#20372a;margin:4px 0 3px}
.brand span{color:#2e8b57}
.brand-sub{font-size:8px;letter-spacing:1.25px;color:#91a092;font-weight:800;margin-bottom:21px}
.hero{background:linear-gradient(112deg,#173c2a,#286d48);border-radius:16px;padding:26px 29px;color:#fff;margin:7px 0 17px}
.hero .eyebrow{color:#bce7c9;font-size:9px;letter-spacing:1.6px;font-weight:800;margin-bottom:10px}
.hero h1{font:800 31px Manrope;color:#fff;letter-spacing:-1.5px;margin:0 0 7px}
.hero p{color:#dceee1;font-size:12px;line-height:1.65;max-width:760px;margin:0}
.section-title{font:800 19px Manrope;letter-spacing:-.4px;color:#23372a;margin:10px 0 4px}
.section-sub{font-size:10px;color:#8a988b;margin-bottom:13px}
div[data-testid="stMetric"]{background:#fff;border:1px solid #e4ebe3;border-radius:11px;padding:13px 15px;box-shadow:0 3px 14px #1b352408}
div[data-testid="stMetricLabel"]{font-size:10px;color:#839184}
div[data-testid="stMetricValue"]{font:800 23px Manrope;color:#20372a}
.notice{background:#eaf4ec;border:1px solid #dbeadd;border-radius:10px;padding:12px 15px;color:#32694a;font-size:11px;line-height:1.65;margin:10px 0 15px}
.disclaimer{background:#fff7e7;border:1px solid #f1e2bf;border-radius:10px;padding:12px 15px;color:#745a2a;font-size:10px;line-height:1.7;margin:12px 0}
.smallcaps{font-size:8px;letter-spacing:1.1px;color:#98a698;font-weight:800}
.footer{font-size:9px;color:#9aa69a;text-align:center;padding:20px 0 8px}
.stButton>button{border-radius:8px;font-weight:600}
.stButton>button[kind="primary"]{background:#2e8b57;border-color:#2e8b57}
div[data-testid="stDataFrame"]{border:1px solid #e4ebe3;border-radius:10px;overflow:hidden}
hr{border-color:#e4ebe3}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand">plant<span>care</span> ai</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">LEAF HEALTH · CLASSIFICATION DEMO</div>', unsafe_allow_html=True)
    page = st.radio("WORKSPACE", ["Home", "Analyze a leaf", "Disease library", "About & setup"])
    st.divider()
    st.caption("39 PlantVillage dataset labels")
    st.caption("CNN · PyTorch · CPU inference")
    st.divider()
    st.caption("Educational agriculture demo; not a substitute for expert crop diagnosis.")

model_path = configured_model_path()
model = None
model_error = None
if model_path is not None:
    try:
        model = load_cached_model(str(model_path), model_path.stat().st_mtime)
    except Exception as exc:
        model_error = f"{type(exc).__name__}: {exc}"

try:
    if not DISEASE_INFO_PATH.is_file():
        raise FileNotFoundError("disease_info.csv is missing from the repository root.")
    disease_info = load_disease_information(
        str(DISEASE_INFO_PATH), DISEASE_INFO_PATH.stat().st_mtime
    )
    data_error = None
except Exception as exc:
    disease_info = {}
    data_error = f"{type(exc).__name__}: {exc}"

st.markdown(
    '<div class="hero"><div class="eyebrow">UPLOAD · CLASSIFY · LEARN</div>'
    '<h1>Healthier leaves start with better insight.</h1>'
    '<p>Explore a 39-class PlantVillage leaf-image classification project. Upload a clear leaf photo, '
    'preview it, and inspect the trained CNN output when the model weights are installed.</p></div>',
    unsafe_allow_html=True,
)

if data_error:
    st.error("Disease information could not be loaded: " + data_error)

if page == "Home":
    total_classes = len(IDX_TO_CLASSES)
    healthy_count = sum("healthy" in label.lower() for label in IDX_TO_CLASSES.values())
    a,b,c=st.columns(3)
    a.metric("Dataset classes", total_classes, "PlantVillage labels")
    b.metric("Healthy labels", healthy_count, "Within the class mapping")
    c.metric("Model status", "Connected" if model else "Preview only",
             "39-class CNN loaded" if model else "Weights not supplied")
    if model:
        st.success("The trained CNN checkpoint loaded successfully on CPU.")
    elif model_error:
        st.error("A model file was found but could not be loaded: " + model_error)
    else:
        st.markdown(
            '<div class="notice"><b>Preview mode is active</b><br/>The trained .pt model file is not stored in this public repository. '
            'Image upload and preview work now; predictions become available after you add your trusted checkpoint and configure '
            'PLANT_DISEASE_MODEL_PATH. The app will not generate a fake prediction.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="section-title">How to use the demo</div>',unsafe_allow_html=True)
    st.write("1. Choose **Analyze a leaf** from the sidebar.")
    st.write("2. Upload a clear JPG, JPEG or PNG image of a single plant leaf.")
    st.write("3. Review the model output if trained weights are connected.")
    st.write("4. Open the disease library for the dataset's informational descriptions.")
    st.markdown(
        '<div class="disclaimer"><b>Important:</b> Predictions are experimental dataset classifications. '
        'They are not verified diagnoses, and top-class scores are not calibrated probabilities. '
        'Confirm suspected diseases with a qualified local agricultural expert before taking action.</div>',
        unsafe_allow_html=True,
    )

elif page == "Analyze a leaf":
    st.markdown('<div class="section-title">Leaf image analysis</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Use one well-lit leaf image with the leaf filling most of the frame.</div>',unsafe_allow_html=True)
    uploaded=st.file_uploader("Upload a leaf image",type=["jpg","jpeg","png"],help="Maximum file size: 10 MB.")
    if uploaded is not None:
        raw=uploaded.getvalue()
        if len(raw)>MAX_UPLOAD_BYTES:
            st.error("This file is larger than 10 MB. Please upload a smaller image.")
        else:
            try:
                with Image.open(uploaded) as check:
                    check.verify()
                uploaded.seek(0)
                with Image.open(uploaded) as opened:
                    image=opened.convert("RGB")
                left,right=st.columns([1.1,1])
                with left:
                    st.image(image,caption="Uploaded leaf image",width="stretch")
                with right:
                    st.markdown("**Image details**")
                    st.write(f"Dimensions: {image.width} × {image.height} pixels")
                    st.write(f"Format: {uploaded.type or 'Image'}")
                    st.write(f"File size: {len(raw)/1024:.1f} KB")
                    st.write("Model preprocessing: resize shortest edge to 255, center crop to 224, normalize pixels to 0–1.")
                    if model is not None:
                        from inference import predict_image
                        try:
                            predictions=predict_image(model,image,top_k=3)
                            best=predictions[0]
                            label=IDX_TO_CLASSES[best["index"]]
                            st.markdown('<div class="section-title">Model output</div>',unsafe_allow_html=True)
                            st.success(readable_label(label))
                            st.caption(f"Model score: {best['score']*100:.1f}% · Not a calibrated probability.")
                            if label == "Background_without_leaves":
                                st.info("The model's top class indicates that no leaf is visible in the image. Try a close-up leaf photo.")
                            if len(predictions)>1:
                                st.markdown("**Other model-ranked classes**")
                                for item in predictions[1:]:
                                    st.write(f"{readable_label(IDX_TO_CLASSES[item['index']])}: {item['score']*100:.1f}%")
                            row=disease_info.get(best["index"])
                            if row:
                                st.markdown('<div class="section-title">Dataset information</div>',unsafe_allow_html=True)
                                st.write(row.get("description",""))
                                st.caption("The following notes come from the repository's reference CSV. They are informational, not personalized treatment advice.")
                                st.write(row.get("Possible Steps",""))
                        except Exception as exc:
                            st.error(f"Inference could not be completed: {type(exc).__name__}: {exc}")
                    elif model_error:
                        st.error("Model file load failed: " + model_error)
                    else:
                        st.info("Image preview is ready. Add the trained model checkpoint to enable CNN predictions.")
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("This file could not be read as an image. Please choose a valid JPG, JPEG or PNG.")
    else:
        st.info("Upload an image to preview it.")

elif page == "Disease library":
    st.markdown('<div class="section-title">PlantVillage disease library</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Descriptions from the reference CSV included with this repository.</div>',unsafe_allow_html=True)
    if disease_info:
        choices=list(IDX_TO_CLASSES.items())
        selected=st.selectbox("Choose a dataset class",choices,format_func=lambda item:readable_label(item[1]))
        row=disease_info.get(selected[0])
        if row:
            st.markdown(f"### {row.get('disease_name', readable_label(selected[1]))}")
            st.write(row.get("description",""))
            st.markdown("**Reference notes**")
            st.write(row.get("Possible Steps",""))
            st.caption("Check local crop guidance and product labels before applying any treatment.")
        else:
            st.warning("No reference description was found for this class.")
    else:
        st.warning("The disease information table is unavailable.")

elif page == "About & setup":
    st.markdown('<div class="section-title">About this project</div>',unsafe_allow_html=True)
    st.write("This repository contains a 39-class convolutional neural network (CNN) originally implemented with PyTorch for PlantVillage leaf images. This Streamlit interface retains the original CNN layer layout and class-index mapping.")
    st.markdown("**What has been improved**")
    st.write("- Shared, explicit 39-class label mapping.")
    st.write("- CPU-safe checkpoint loading with state-dictionary validation.")
    st.write("- Training-compatible image preprocessing.")
    st.write("- Upload size and image-format validation.")
    st.write("- Optional model loading so preview mode works without external weights.")
    st.write("- No upload filename is written to the server filesystem.")
    st.markdown("**Connect your trained model**")
    st.code("PLANT_DISEASE_MODEL_PATH=/path/to/plant_disease_model_1_latest.pt",language="text")
    st.write("For the local training/inference environment install the optional model dependencies from requirements-model.txt. Do not load .pt files from unknown sources.")
    st.markdown(
        '<div class="disclaimer"><b>Agriculture note</b><br/>The classifier is trained on a particular image dataset and may not generalize to field conditions, other cameras, cultivars or regions. A prediction should be treated as a screening aid for learning only, not a definitive diagnosis or pesticide recommendation.</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.markdown('<div class="smallcaps">PLANTCARE AI · 39-CLASS LEAF IMAGE CLASSIFICATION</div>',unsafe_allow_html=True)
st.markdown('<div class="footer">Educational demonstration · Sample-dataset categories · No real-world diagnosis or treatment guarantee.</div>',unsafe_allow_html=True)
