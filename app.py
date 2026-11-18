import os
import glob
from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input

# ---------------------------------------------------------
# Page Setup & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Banana Ripeness AI",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stage-card {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        color: white;
        text-align: center;
    }
    .stage-unripe { background-color: #2e7d32; }
    .stage-half-ripe { background-color: #fbc02d; color: #333 !important; }
    .stage-ripe { background-color: #f57f17; }
    .stage-overripe { background-color: #5d4037; }
    .metric-value { font-size: 1.8rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Constants & Metadata
# ---------------------------------------------------------
CLASS_NAMES = ["Unripe", "Half-ripe", "Ripe", "Overripe"]
STAGE_INFO = {
    "Unripe": {
        "class_css": "stage-unripe",
        "emoji": "🟢",
        "description": "Green and firm. High in starch, low in sugar. Great for cooking or green smoothies."
    },
    "Half-ripe": {
        "class_css": "stage-half-ripe",
        "emoji": "🟡",
        "description": "Yellow with green tips. Mild sweetness with a firmer texture."
    },
    "Ripe": {
        "class_css": "stage-ripe",
        "emoji": "🍌",
        "description": "Fully yellow with slight sugar speckles. Peak sweetness and flavor for direct eating."
    },
    "Overripe": {
        "class_css": "stage-overripe",
        "emoji": "🟤",
        "description": "Heavy brown spots or fully brown. Soft and very sweet. Ideal for banana bread, pancakes, or freezing."
    }
}

ORDINAL_THRESHOLDS = ["P(> Unripe)", "P(> Half-ripe)", "P(> Ripe)"]

# ---------------------------------------------------------
# Model Discovery & Loading
# ---------------------------------------------------------
def find_available_models():
    """Scans models/ and project root for .keras and .h5 files."""
    paths = glob.glob("models/*.keras") + glob.glob("models/*.h5") + \
            glob.glob("*.keras") + glob.glob("*.h5")
    # Clean unique paths
    return sorted(list(set(paths)))

@st.cache_resource
def load_model_from_path(model_path: str):
    """Loads a Keras model without compiling (supports custom losses)."""
    return tf.keras.models.load_model(model_path, compile=False)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.title("🍌 Model Settings")

available_models = find_available_models()

if not available_models:
    st.sidebar.error("No model files found in `models/` or root directory.")
    selected_model_path = None
else:
    model_labels = {path: os.path.basename(path) for path in available_models}
    selected_model_path = st.sidebar.selectbox(
        "Select Model Architecture:",
        options=available_models,
        format_func=lambda x: model_labels[x]
    )

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About This Project
This application classifies banana ripeness using deep convolutional networks (**ResNet152**):
- **Ordinal Regression:** Respects natural stage ordering ($Unripe \\rightarrow Overripe$).
- **Categorical Softmax:** Standard 4-class classification.
""")

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.title("🍌 Banana Ripeness Classification & Grading")
st.write("Upload a photo or capture an image using your camera to predict banana ripeness stage.")

if not selected_model_path:
    st.info("Please place your `.keras` models into the `models/` folder to begin.")
    st.stop()

# Load the selected model
try:
    with st.spinner(f"Loading {os.path.basename(selected_model_path)}..."):
        model = load_model_from_path(selected_model_path)
    output_dim = model.output_shape[-1]
    is_ordinal = (output_dim == 3)
    st.sidebar.success(f"Loaded: `{os.path.basename(selected_model_path)}`")
    st.sidebar.caption(f"Architecture Type: {'Ordinal Model (3 thresholds)' if is_ordinal else 'Categorical Model (4 classes)'}")
except Exception as e:
    st.error(f"Failed to load model `{selected_model_path}`: {e}")
    st.stop()

# ---------------------------------------------------------
# Image Input
# ---------------------------------------------------------
input_mode = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"], horizontal=True)

uploaded_file = None
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Select an image of a banana...", type=["jpg", "jpeg", "png", "webp"])
else:
    uploaded_file = st.camera_input("Take a photo of a banana")

# ---------------------------------------------------------
# Inference & Display
# ---------------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col_img, col_pred = st.columns([1, 1], gap="medium")
    
    with col_img:
        st.subheader("Input Image")
        st.image(image, caption="Analyzed Image", use_container_width=True)
    
    # Preprocess image for ResNet152 (224x224, standard preprocess_input)
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_preprocessed = preprocess_input(img_array)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    
    with col_pred:
        st.subheader("Prediction Results")
        with st.spinner("Classifying ripeness..."):
            raw_output = model.predict(img_batch, verbose=0)[0]
        
        if is_ordinal:
            # Ordinal logic: number of thresholds >= 0.5
            pred_idx = int((raw_output >= 0.5).sum())
            pred_stage = CLASS_NAMES[pred_idx]
            info = STAGE_INFO[pred_stage]
            
            st.markdown(
                f"""
                <div class="stage-card {info['class_css']}">
                    <div style="font-size: 2.2rem;">{info['emoji']} <b>{pred_stage.upper()}</b></div>
                    <p style="margin-top: 0.5rem;">{info['description']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("#### Cumulative Threshold Probabilities:")
            st.caption("Each threshold represents the probability that the ripeness exceeds that stage.")
            for name, prob in zip(ORDINAL_THRESHOLDS, raw_output):
                prob_pct = float(prob) * 100
                st.write(f"**{name}**: {prob_pct:.1f}%")
                st.progress(float(prob))
                
        else:
            # Categorical logic: argmax over softmax probabilities
            pred_idx = int(np.argmax(raw_output))
            confidence = raw_output[pred_idx] * 100
            pred_stage = CLASS_NAMES[pred_idx]
            info = STAGE_INFO[pred_stage]
            
            st.markdown(
                f"""
                <div class="stage-card {info['class_css']}">
                    <div style="font-size: 2.2rem;">{info['emoji']} <b>{pred_stage.upper()}</b></div>
                    <div class="metric-value">{confidence:.1f}% Confidence</div>
                    <p style="margin-top: 0.5rem;">{info['description']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("#### Class Confidence Breakdown:")
            for name, prob in zip(CLASS_NAMES, raw_output):
                prob_pct = float(prob) * 100
                st.write(f"**{name}**: {prob_pct:.1f}%")
                st.progress(float(prob))

    # Ripeness Spectrum Visualizer
    st.markdown("---")
    st.subheader("Ripeness Progression Spectrum")
    cols = st.columns(4)
    for i, name in enumerate(CLASS_NAMES):
        with cols[i]:
            is_active = (i == pred_idx)
            border_style = "3px solid #ff9800" if is_active else "1px solid #ddd"
            opacity = "1.0" if is_active else "0.5"
            active_badge = "👈 Current Stage" if is_active else ""
            st.markdown(f"""
            <div style="border: {border_style}; border-radius: 8px; padding: 10px; opacity: {opacity}; text-align: center;">
                <h4>{STAGE_INFO[name]['emoji']} {name}</h4>
                <small>{active_badge}</small>
            </div>
            """, unsafe_allow_html=True)
