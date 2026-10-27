"""
IPCV Final Year Project: Tomato Leaf Disease Detection with Explainable AI & Classical Segmentation
Module: utils.py
Contains:
  1. Classical IPCV: K-Means Color-Based Clustering & Lesion Segmentation (OpenCV)
  2. Explainable AI: Grad-CAM (Gradient-weighted Class Activation Mapping) on PyTorch MobileNetV2
  3. Preprocessing, Transformations & Model Loading Helpers
  4. Solanaceae Agro-Remedy & Diagnostic Knowledge Base
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# ==============================================================================
# 1. CANONICAL CLASS NAMES & AGRO-DIAGNOSTIC KNOWLEDGE BASE
# ==============================================================================

# Canonical Solanaceae classes (PlantVillage subset)
DEFAULT_CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
]

DISEASE_DISPLAY_NAMES = {
    "Pepper__bell___Bacterial_spot": "Bell Pepper - Bacterial Spot",
    "Pepper__bell___healthy": "Bell Pepper - Healthy Foliage",
    "Potato___Early_blight": "Potato - Early Blight (Alternaria solani)",
    "Potato___Late_blight": "Potato - Late Blight (Phytophthora infestans)",
    "Potato___healthy": "Potato - Healthy Foliage",
    "Tomato___Early_blight": "Tomato - Early Blight (Alternaria solani)",
    "Tomato___Late_blight": "Tomato - Late Blight (Phytophthora infestans)",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato - Yellow Leaf Curl Virus (TYLCV)",
    "Tomato___healthy": "Tomato - Healthy Foliage",
    # Optional 10th class support if included in custom datasets
    "Tomato___Bacterial_spot": "Tomato - Bacterial Spot (Xanthomonas spp.)",
    "Tomato___Septoria_leaf_spot": "Tomato - Septoria Leaf Spot",
}

AGRO_REMEDIES = {
    "Pepper__bell___Bacterial_spot": (
        "Spray copper bactericide mixed with Mancozeb weekly; avoid overhead sprinkler irrigation "
        "to prevent splash dispersal and rogue out severely infected plants."
    ),
    "Pepper__bell___healthy": (
        "Crop condition is optimal. Maintain balanced nitrogen-potassium fertigation, clean drip lines, "
        "and inspect weekly for early aphid/thrip vectors."
    ),
    "Potato___Early_blight": (
        "Apply preventative fungicides (Chlorothalonil or Mancozeb); prune senescing lower foliage, "
        "enforce a 3-year crop rotation away from other Solanaceae species."
    ),
    "Potato___Late_blight": (
        "CRITICAL: Apply systemic fungicides immediately (Metalaxyl / Dimethomorph); destroy blighted haulms "
        "to prevent sporangia wash into tubers, and harvest only in dry conditions."
    ),
    "Potato___healthy": (
        "Crop foliage is robust and disease-free. Ensure uniform soil moisture to prevent tuber cracking "
        "and maintain prophylactic scouting."
    ),
    "Tomato___Early_blight": (
        "Apply copper-based or Azoxystrobin fungicide at first spot detection; stake plants to improve airflow, "
        "and mulch soil surface to block Alternaria soil splash."
    ),
    "Tomato___Late_blight": (
        "URGENT: Eradicate infected plants immediately; spray systemic curative fungicides (Cymoxanil + Mancozeb), "
        "and reduce canopy humidity through trellising."
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Eradicate Bemisia tabaci (whitefly) vectors using yellow sticky traps and systemic Imidacloprid/Neem oil sprays; "
        "install fine insect netting and plant TYLCV-resistant cultivars."
    ),
    "Tomato___healthy": (
        "Plant is healthy and vigorous. Maintain regular drip fertigation, balanced micronutrients (Calcium/Magnesium), "
        "and scout underside of leaves bi-weekly."
    ),
    "Tomato___Bacterial_spot": (
        "Apply copper hydroxide + Mancozeb; sanitize pruning shears with 10% bleach, and avoid field work while leaves are wet."
    ),
    "Tomato___Septoria_leaf_spot": (
        "Remove lower diseased leaves; spray Chlorothalonil or copper soap fungicide, and maintain drip irrigation."
    ),
}


# ==============================================================================
# 2. IMAGE PREPROCESSING & NORMALIZATION
# ==============================================================================

# Standard ImageNet normalization parameters required by MobileNetV2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_inference_transforms():
    """
    Returns standard evaluation transforms for MobileNetV2:
    Resize to 224x224, convert to Tensor, and normalize with ImageNet mean/std.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ==============================================================================
# 3. CLASSICAL IPCV MODULE: K-MEANS COLOR SEGMENTATION
# ==============================================================================

def kmeans_color_segmentation(
    image_rgb: np.ndarray,
    k: int = 3,
    color_space: str = "LAB"
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Classical IPCV Pipeline: Unsupervised K-Means Color-Based Clustering (OpenCV).
    
    Theoretical IPCV Basis:
      1. Converts RGB input to CIELAB color space. CIELAB decouples luminance (L*)
         from chromaticity (a* = green-red, b* = blue-yellow), providing perceptual uniformity.
         Healthy plant foliage exhibits strong negative a* values (green), while necrotic
         and chlorotic disease lesions shift toward positive a* (brown/red) and b* (yellow).
      2. Reshapes 2D spatial pixel lattice (H, W, 3) into a 1D feature array (N, 3).
      3. Minimizes within-cluster sum of squares (WCSS) via cv2.kmeans().
      4. Semantically assigns clusters into:
         - Background (specimen board / ambient void)
         - Healthy Leaf Tissue (high greenness)
         - Diseased Lesion / Necrotic Tissue (chlorosis/necrosis)
      5. Generates isolated binary mask and bitwise-AND segmented lesion visualization.

    Args:
        image_rgb: Input image in RGB uint8 format [H, W, 3].
        k: Number of clusters (default 3: background, healthy green, diseased lesion).
        color_space: "LAB" (recommended for vegetation) or "HSV".

    Returns:
        Dictionary containing:
          - "segmented_lesion": RGB image showing only the isolated diseased spots.
          - "segmented_leaf": RGB image showing entire leaf isolated from background.
          - "cluster_vis": False-color map showing the K distinct cluster regions.
          - "lesion_mask": Binary uint8 mask (255 for lesion pixels, 0 elsewhere).
          - "leaf_mask": Binary uint8 mask (255 for all leaf pixels).
          - "severity_pct": Estimated percentage of leaf area affected by disease.
    """
    h, w, _ = image_rgb.shape
    
    # 1. Color space conversion
    if color_space.upper() == "LAB":
        converted = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    elif color_space.upper() == "HSV":
        converted = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    else:
        converted = image_rgb.copy()

    # 2. Reshape into 2D feature matrix (N, 3) of float32
    pixel_features = converted.reshape((-1, 3)).astype(np.float32)

    # 3. K-Means termination criteria: max 100 iterations or epsilon = 0.2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    attempts = 10
    flags = cv2.KMEANS_PP_CENTERS

    _, labels, centers = cv2.kmeans(
        pixel_features, k, None, criteria, attempts, flags
    )
    labels = labels.flatten()

    # Reshape labels back to 2D image plane (H, W)
    label_matrix = labels.reshape((h, w))

    # 4. Semantic Cluster Identification Heuristic
    # In PlantVillage datasets, the background is predominantly at the 4 borders/corners.
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:5, :] = True
    border_mask[-5:, :] = True
    border_mask[:, :5] = True
    border_mask[:, -5:] = True
    
    border_labels = label_matrix[border_mask]
    # Background cluster is the most frequent cluster on image borders
    counts = np.bincount(border_labels, minlength=k)
    bg_cluster = int(np.argmax(counts))

    non_bg_clusters = [i for i in range(k) if i != bg_cluster]

    # Between the remaining non-background clusters:
    # In LAB: a* channel (index 1) represents green (<128) to magenta/red (>128).
    # Healthy green tissue has the LOWEST a* value (most negative/green).
    # Diseased lesion (chlorosis/necrosis) has HIGHER a* (brown, yellow, dark spot).
    if len(non_bg_clusters) == 2:
        c1, c2 = non_bg_clusters[0], non_bg_clusters[1]
        a1 = centers[c1][1]
        a2 = centers[c2][1]
        
        if a1 < a2:
            healthy_cluster = c1
            lesion_cluster = c2
        else:
            healthy_cluster = c2
            lesion_cluster = c1
    elif len(non_bg_clusters) == 1:
        healthy_cluster = non_bg_clusters[0]
        lesion_cluster = non_bg_clusters[0]
    else:
        healthy_cluster = 0
        lesion_cluster = 1

    # 5. Build Binary Masks & Segmented Output
    lesion_mask = np.uint8((label_matrix == lesion_cluster) * 255)
    healthy_mask = np.uint8((label_matrix == healthy_cluster) * 255)
    leaf_mask = np.bitwise_or(lesion_mask, healthy_mask)

    # Optional Morphological cleaning (OpenCV Opening to eliminate single-pixel impulse noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lesion_mask_clean = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Bitwise-AND extraction of the diseased lesion spots
    segmented_lesion = cv2.bitwise_and(image_rgb, image_rgb, mask=lesion_mask_clean)
    segmented_leaf = cv2.bitwise_and(image_rgb, image_rgb, mask=leaf_mask)

    # 6. Generate False-Color Cluster Visualization for Academic Defense
    cluster_vis = np.zeros_like(image_rgb)
    # Background: Slate Dark Gray [40, 44, 52]
    cluster_vis[label_matrix == bg_cluster] = [40, 44, 52]
    # Healthy Green Tissue: Vibrant Emerald Green [46, 204, 113]
    cluster_vis[label_matrix == healthy_cluster] = [46, 204, 113]
    # Diseased Lesion Tissue: Vivid Crimson/Orange [231, 76, 60]
    cluster_vis[label_matrix == lesion_cluster] = [231, 76, 60]

    # 7. Compute Quantitative Disease Severity Metric
    total_leaf_pixels = max(1, int(np.sum(leaf_mask > 0)))
    total_lesion_pixels = int(np.sum(lesion_mask_clean > 0))
    severity_pct = round((total_lesion_pixels / total_leaf_pixels) * 100.0, 2)

    return {
        "segmented_lesion": segmented_lesion,
        "segmented_leaf": segmented_leaf,
        "cluster_vis": cluster_vis,
        "lesion_mask": lesion_mask_clean,
        "leaf_mask": leaf_mask,
        "severity_pct": severity_pct,
        "centers": centers,
        "clusters": {
            "background": bg_cluster,
            "healthy": healthy_cluster,
            "lesion": lesion_cluster,
        },
    }


# ==============================================================================
# 4. EXPLAINABLE AI (XAI) MODULE: GRAD-CAM IMPLEMENTATION
# ==============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Targeted specifically for PyTorch MobileNetV2 final convolutional feature extractor.
    
    Academic IPCV Formulation:
      1. Forward propagation extracts spatial activation maps A_k from the target layer:
         A = model.features[-1](x)  [Shape: 1, 1280, 7, 7]
      2. Backward propagation computes partial derivatives of class score y_c:
         g_k = d(y_c) / d(A_k)
      3. Neuron importance weights alpha_k^c obtained via Global Average Pooling (GAP):
         alpha_k^c = (1 / Z) * sum_i sum_j ( d(y_c) / d(A_k(i, j)) )
      4. Heatmap is linear combination followed by Rectified Linear Unit (ReLU):
         L_GradCAM^c = ReLU( sum_k alpha_k^c * A_k )
      5. Heatmap normalized to [0, 1] and blended with original RGB image using JET colormap.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        
        # In torchvision MobileNetV2, features[-1] is the final 1x1 Conv2dNormActivation (1280 channels)
        if target_layer is None:
            self.target_layer = self.model.features[-1]
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0] is dL / d(output)
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Computes 2D class activation heatmap for the specified or top predicted class.

        Args:
            input_tensor: Torch tensor of shape [1, 3, 224, 224].
            target_class: Integer index of target class; if None, uses argmax of predictions.

        Returns:
            heatmap_2d: 2D numpy array [224, 224] in range [0, 1].
            predicted_class: Integer index of predicted class.
            confidence: Probability score [0.0 - 1.0] of predicted class.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(probabilities, dim=1).item())

        confidence = float(probabilities[0, target_class].item())

        # One-hot score for backpropagation
        score = output[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients  # [1, 1280, H, W]
        activations = self.activations  # [1, 1280, H, W]

        # Spatial Global Average Pooling across H, W
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, 1280, 1, 1]

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H, W]

        # Apply ReLU to focus only on features that have a positive influence on target class
        cam = F.relu(cam)

        # Interpolate bilinearly to 224x224 input spatial resolution
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-Max Normalization to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, confidence

    def generate_overlay(
        self,
        original_rgb: np.ndarray,
        heatmap_2d: np.ndarray,
        alpha_img: float = 0.6,
        alpha_cam: float = 0.4
    ) -> np.ndarray:
        """
        Blends 2D Grad-CAM heatmap with original RGB image using OpenCV JET colormap.

        Blend formula: Blended = (alpha_img * Original) + (alpha_cam * JET_Heatmap)
        Default alphas: 0.6 original + 0.4 heatmap as specified in project requirements.
        """
        h, w, _ = original_rgb.shape
        # Resize heatmap to match original leaf dimensions
        heatmap_resized = cv2.resize(heatmap_2d, (w, h))

        # Convert to 8-bit scale [0, 255] and apply JET colormap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_jet_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_jet_rgb = cv2.cvtColor(heatmap_jet_bgr, cv2.COLOR_BGR2RGB)

        # Linear alpha blending
        blended = (alpha_img * original_rgb.astype(np.float32) + 
                   alpha_cam * heatmap_jet_rgb.astype(np.float32))
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return blended

    def close(self):
        """Removes PyTorch forward/backward hooks to prevent memory leaks."""
        self.forward_handle.remove()
        self.backward_handle.remove()


# ==============================================================================
# 5. MOBILENET_V2 MODEL BUILDER & CHECKPOINT LOADER
# ==============================================================================

def build_mobilenet_v2(num_classes: int = len(DEFAULT_CLASSES), pretrained: bool = True) -> nn.Module:
    """
    Constructs torchvision MobileNetV2 architecture with transfer learning.
    Replaces final linear classifier layer (1280 -> num_classes).
    
    Academic Engineering Rationale:
      MobileNetV2 introduces Inverted Residuals and Linear Bottlenecks, utilizing
      depthwise separable convolutions to reduce parameters (3.5M) and compute (300M FLOPs),
      making it ideal for low-latency agricultural deployment on edge/embedded IoT devices.
    """
    try:
        model = models.mobilenet_v2(weights=None)
    except Exception:
        model = models.mobilenet_v2(pretrained=False)

    if pretrained:
        loaded = False
        # 1. Try local torch hub cache first (prevents offline network error)
        cache_candidates = [
            os.path.expanduser("~/.cache/torch/hub/checkpoints/mobilenet_v2-7ebf99e0.pth"),
            os.path.expanduser("~/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth"),
        ]
        for cpath in cache_candidates:
            if os.path.exists(cpath):
                try:
                    state_dict = torch.load(cpath, map_location="cpu")
                    model.load_state_dict(state_dict)
                    print(f"[Info] Loaded ImageNet pretrained weights directly from cache: {os.path.basename(cpath)}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"[Warning] Failed loading cached {cpath}: {e}")

        # 2. Try online weights download if not in cache
        if not loaded:
            try:
                from torchvision.models import MobileNet_V2_Weights
                model_pre = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
                model.load_state_dict(model_pre.state_dict())
                loaded = True
            except Exception as e1:
                try:
                    model_pre = models.mobilenet_v2(pretrained=True)
                    model.load_state_dict(model_pre.state_dict())
                    loaded = True
                except Exception as e2:
                    print(f"[Notice] Offline environment: Pretrained weights could not be loaded ({e2}). Initializing architecture.")

    in_features = model.classifier[1].in_features
    # Replace final classifier layer
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_trained_model(
    model_path: str = "mobilenet_tomato.pth",
    num_classes: Optional[int] = None,
    device: Optional[torch.device] = None,
    classes_json_path: str = "classes.json"
) -> Tuple[nn.Module, List[str]]:
    """
    Safely loads trained MobileNetV2 weights and associated class names.
    Falls back cleanly to DEFAULT_CLASSES and initialized weights if model file is missing.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class names if saved
    classes = DEFAULT_CLASSES
    if os.path.exists(classes_json_path):
        try:
            with open(classes_json_path, "r") as f:
                classes = json.load(f)
        except Exception as e:
            print(f"[Warning] Failed loading {classes_json_path}: {e}")

    effective_num_classes = num_classes if num_classes is not None else len(classes)
    model = build_mobilenet_v2(num_classes=effective_num_classes, pretrained=False)

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                if "classes" in checkpoint and checkpoint["classes"]:
                    classes = checkpoint["classes"]
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                model = checkpoint
            print(f"[Info] Successfully loaded weights from {model_path} ({len(classes)} classes)")
        except Exception as e:
            print(f"[Error] Could not load checkpoint from {model_path}: {e}. Initializing fresh weights.")
    else:
        print(f"[Warning] Weights file {model_path} not found. Running with initialized weights.")

    model.to(device)
    model.eval()
    return model, classes
