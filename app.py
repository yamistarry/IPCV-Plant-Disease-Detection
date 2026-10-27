"""
IPCV Final Year Project: Tomato Leaf Disease Detection with Explainable AI & Classical Segmentation
Module: app.py
Streamlit Interactive Dashboard for Live Diagnosis, Explainability (Grad-CAM),
Classical IPCV (K-Means), and Model Performance Evaluation.
"""

import json
import os
import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F

from utils import (
    AGRO_REMEDIES,
    DEFAULT_CLASSES,
    DISEASE_DISPLAY_NAMES,
    GradCAM,
    get_inference_transforms,
    kmeans_color_segmentation,
    load_trained_model,
)

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Tomato & Solanaceae Disease AI | IPCV Project",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Academic Grade Interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f4fbf7 0%, #e8f5e9 100%);
        border: 1px solid #c8e6c9;
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }
    .disease-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #2e7d32;
        margin-bottom: 0.3rem;
    }
    .confidence-badge {
        display: inline-block;
        background-color: #2e7d32;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .remedy-box {
        background-color: #fffde7;
        border-left: 5px solid #fbc02d;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        margin-top: 0.8rem;
        font-size: 0.95rem;
        color: #424242;
    }
    .stat-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1b5e20;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# MODEL CACHING & INITIALIZATION
# ==============================================================================

@st.cache_resource(show_spinner="Loading PyTorch MobileNetV2 Architecture & Weights...")
def load_cached_model():
    """Caches the model in memory to ensure millisecond inference latency."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "mobilenet_tomato.pth"
    classes_path = "classes.json"
    model, classes = load_trained_model(
        model_path=model_path,
        device=device,
        classes_json_path=classes_path
    )
    return model, classes, device


model, class_names, device = load_cached_model()
inference_transform = get_inference_transforms()


# ==============================================================================
# SIDEBAR CONTROLS & PROJECT METADATA
# ==============================================================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=75)
    st.title("IPCV Diagnostic")
    st.caption("Solanaceae Disease Analysis System")
    
    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Parameters")
    
    k_clusters = st.slider("K-Means Clusters (K)", min_value=2, max_value=5, value=3,
                           help="Standard: K=3 for [Background, Healthy Green, Diseased Lesion]")
    color_space = st.selectbox("Color Space", ["LAB", "HSV"], index=0,
                               help="CIELAB provides perceptual uniformity and decouples luminance from chromaticity.")
    
    alpha_cam = st.slider("Grad-CAM Overlay Opacity", min_value=0.1, max_value=0.9, value=0.4, step=0.05,
                          help="Specifies blend weight: (1-a)*Original + (a)*Heatmap")
    
    st.markdown("---")
    st.markdown("### 📋 System Status")
    st.write(f"• **Hardware Device**: `{str(device).upper()}`")
    st.write(f"• **Loaded Classes**: `{len(class_names)}`")
    model_exists = os.path.exists("mobilenet_tomato.pth")
    if model_exists:
        st.success("• Weights: `mobilenet_tomato.pth` (Loaded)")
    else:
        st.warning("• Weights: Default Pretrained (Run train.py)")


# ==============================================================================
# HEADER
# ==============================================================================

st.markdown('<div class="main-header">🌿 Tomato & Solanaceae Leaf Disease Diagnosis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Dual-Modality Computer Vision: Classical K-Means Color Segmentation & Deep Explainable AI (Grad-CAM) with MobileNetV2'
    '</div>',
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["🔍 Live Diagnosis & Inspection", "📊 Model Performance & Metrics"])


# ==============================================================================
# TAB 1: LIVE DIAGNOSIS & INSPECTION
# ==============================================================================

with tab1:
    col_input, col_options = st.columns([2, 1])

    with col_input:
        uploaded_file = st.file_uploader(
            "Upload Leaf Specimen Image (JPEG / PNG):",
            type=["jpg", "jpeg", "png"],
            help="Upload a leaf photo from Tomato, Potato, or Bell Pepper."
        )

    with col_options:
        # Check if sample images exist
        sample_images = glob.glob("dataset/*/*.jpg") + glob.glob("dataset/*/*.png")
        sample_choice = None
        if sample_images:
            st.markdown("**...or select a sample test leaf:**")
            sample_options = ["None"] + sorted(sample_images[:18])
            chosen_sample = st.selectbox("Quick Demo Samples", sample_options)
            if chosen_sample != "None":
                sample_choice = chosen_sample

    # Determine input image source
    input_image_pil = None
    if uploaded_file is not None:
        input_image_pil = Image.open(uploaded_file).convert("RGB")
    elif sample_choice is not None:
        input_image_pil = Image.open(sample_choice).convert("RGB")

    if input_image_pil is None:
        # Default Welcome State
        st.info("👋 Upload a leaf image above or select a sample image to initiate the live dual IPCV-XAI inspection pipeline.")
        
        # Display sample workflow diagram
        st.markdown("#### 🔄 Dual-Modality Pipeline Architecture")
        col_wf1, col_wf2, col_wf3 = st.columns(3)
        with col_wf1:
            st.markdown("""
            **1. Input Preprocessing**
            - 224x224 RGB image normalization
            - ImageNet Channel Standardization
            - CIELAB / HSV Color Space Mapping
            """)
        with col_wf2:
            st.markdown("""
            **2. Classical IPCV (K-Means)**
            - Unsupervised pixel clustering (K=3)
            - Background vs Foliage vs Lesion separation
            - Morphological mask cleanup & Severity %
            """)
        with col_wf3:
            st.markdown("""
            **3. Deep Learning + XAI (Grad-CAM)**
            - MobileNetV2 Inverted Residual Feature maps
            - Backward gradient attribution ($\partial y_c / \partial A$)
            - JET Colormap blended activation overlay
            """)
    else:
        # ----------------------------------------------------------------------
        # PIPELINE EXECUTION
        # ----------------------------------------------------------------------
        image_np_rgb = np.array(input_image_pil)
        
        # Ensure dimensions are standard for processing
        if max(image_np_rgb.shape[:2]) > 800:
            image_np_rgb = cv2.resize(image_np_rgb, (512, 512))

        # A. Classical IPCV: K-Means Color Clustering
        with st.spinner("Executing Classical IPCV: K-Means Color Segmentation..."):
            kmeans_results = kmeans_color_segmentation(
                image_rgb=image_np_rgb,
                k=k_clusters,
                color_space=color_space
            )

        # B. Deep Learning Inference & Grad-CAM Generation
        with st.spinner("Executing MobileNetV2 Forward Pass & Grad-CAM Backpropagation..."):
            # Transform for PyTorch inference
            img_tensor = inference_transform(Image.fromarray(image_np_rgb)).unsqueeze(0).to(device)

            grad_cam = GradCAM(model)
            heatmap_2d, pred_idx, confidence = grad_cam.generate_cam(img_tensor)
            grad_cam.close()

            # Generate blended Grad-CAM overlay
            cam_overlay = grad_cam.generate_overlay(
                original_rgb=image_np_rgb,
                heatmap_2d=heatmap_2d,
                alpha_img=1.0 - alpha_cam,
                alpha_cam=alpha_cam
            )

            # Get full model output probabilities for top-k bar chart
            with torch.no_grad():
                logits = model(img_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Predicted Class Name and Friendly Display Name
        pred_raw_class = class_names[pred_idx] if pred_idx < len(class_names) else f"Class {pred_idx}"
        pred_display_name = DISEASE_DISPLAY_NAMES.get(pred_raw_class, pred_raw_class)
        remedy_text = AGRO_REMEDIES.get(
            pred_raw_class,
            "Consult an agricultural extension specialist for integrated pest management guidelines."
        )

        # ----------------------------------------------------------------------
        # PREDICTION CARD & REMEDY
        # ----------------------------------------------------------------------
        conf_pct = confidence * 100.0
        st.markdown(f"""
        <div class="prediction-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="disease-title">{pred_display_name}</div>
                <div><span class="confidence-badge">Confidence: {conf_pct:.2f}%</span></div>
            </div>
            <div style="font-size: 0.9rem; color: #444; margin-top: 0.2rem;">
                Internal Class Identifier: <code>{pred_raw_class}</code> | 
                Estimated Lesion Severity: <strong>{kmeans_results['severity_pct']}%</strong> of leaf area
            </div>
            <div class="remedy-box">
                <strong>🌾 Agro-Agronomist Recommendation:</strong><br/>
                {remedy_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # 3-COLUMN SIDE-BY-SIDE VISUAL COMPARISON
        # ----------------------------------------------------------------------
        st.markdown("### 🔬 Multimodal Inspection Panel")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**1. Original Leaf Specimen**")
            st.image(image_np_rgb, use_container_width=True, caption=f"Raw RGB Input ({image_np_rgb.shape[1]}x{image_np_rgb.shape[0]})")
            st.caption("Standard color leaf capture prior to transformation.")

        with col2:
            st.markdown(f"**2. K-Means Segmented Lesions (IPCV)**")
            st.image(kmeans_results["segmented_lesion"], use_container_width=True,
                     caption=f"Isolated Necrotic Spots (Severity: {kmeans_results['severity_pct']}%)")
            st.caption("Extracted via OpenCV cv2.kmeans() in CIELAB color space.")

        with col3:
            st.markdown("**3. Grad-CAM Explainability (XAI)**")
            st.image(cam_overlay, use_container_width=True,
                     caption=f"Final Conv Layer Activation (JET Heatmap, α={alpha_cam:.2f})")
            st.caption("Gradient attribution on MobileNetV2 `features[-1]`.")

        # ----------------------------------------------------------------------
        # DETAILED EXPANDERS: ADVANCED IPCV & PROBABILITY SPECTRUM
        # ----------------------------------------------------------------------
        st.markdown("---")
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            with st.expander("📊 Top Predicted Class Probabilities", expanded=True):
                # Sort probabilities in descending order
                top_indices = np.argsort(probs)[::-1][:min(5, len(class_names))]
                top_data = []
                for idx in top_indices:
                    raw_name = class_names[idx]
                    short_name = DISEASE_DISPLAY_NAMES.get(raw_name, raw_name)
                    top_data.append({
                        "Diagnosis": short_name,
                        "Probability (%)": round(float(probs[idx]) * 100.0, 2)
                    })
                df_probs = pd.DataFrame(top_data)
                st.dataframe(df_probs, hide_index=True, use_container_width=True)
                st.bar_chart(df_probs.set_index("Diagnosis")["Probability (%)"])

        with exp_col2:
            with st.expander("🧩 K-Means Cluster Breakdown", expanded=True):
                st.image(
                    kmeans_results["cluster_vis"],
                    use_container_width=True,
                    caption="False-Color Cluster Map: [Green=Healthy Foliage, Red=Lesions, Slate=Background]"
                )
                st.markdown(f"""
                - **Cluster K**: {k_clusters} partitions
                - **Color Space**: `{color_space}`
                - **Healthy Leaf Area**: `~{100.0 - kmeans_results['severity_pct']:.1f}%`
                - **Infected Lesion Area**: `~{kmeans_results['severity_pct']:.1f}%`
                """)


# ==============================================================================
# TAB 2: MODEL PERFORMANCE & EVALUATION
# ==============================================================================

with tab2:
    st.markdown("### 📈 Quantitative Performance & Test Evaluation")
    st.markdown("Comprehensive evaluation on the independent 15% held-out test partition.")

    # 1. Summary Cards
    metrics_file = "metrics_summary.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{metrics_data.get('test_accuracy', 0.0)}%</div>
                <div class="stat-label">Test Set Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{metrics_data.get('test_loss', 0.0):.3f}</div>
                <div class="stat-label">Cross-Entropy Loss</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{metrics_data.get('total_test_samples', 0)}</div>
                <div class="stat-label">Evaluated Test Leaves</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{len(metrics_data.get('classes', []))}</div>
                <div class="stat-label">Target Classes</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # 2. Confusion Matrix & Classification Table
        cm_col, tbl_col = st.columns([1.1, 1.2])

        with cm_col:
            st.markdown("#### 🎯 Multiclass Confusion Matrix")
            if os.path.exists("confusion_matrix.png"):
                st.image("confusion_matrix.png", use_container_width=True,
                         caption="Normalized Seaborn Heatmap across Solanaceae Classes")
            else:
                st.info("Run `python train.py` to generate `confusion_matrix.png`.")

        with tbl_col:
            st.markdown("#### 📋 Per-Class Classification Report")
            rep = metrics_data.get("classification_report", {})
            rows = []
            for k, v in rep.items():
                if isinstance(v, dict):
                    display_k = DISEASE_DISPLAY_NAMES.get(k, k)
                    rows.append({
                        "Class Name": display_k,
                        "Precision": round(v.get("precision", 0.0), 3),
                        "Recall": round(v.get("recall", 0.0), 3),
                        "F1-Score": round(v.get("f1-score", 0.0), 3),
                        "Support": int(v.get("support", 0))
                    })
            if rows:
                df_rep = pd.DataFrame(rows)
                st.dataframe(df_rep, hide_index=True, use_container_width=True)
    else:
        st.warning("⚠️ `metrics_summary.json` not found. Execute `python train.py` to generate test metrics.")
