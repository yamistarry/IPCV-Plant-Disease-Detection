"""
Dataset Setup & Verification Helper for PlantVillage Solanaceae Family.
Allows checking your folder structure and gives direct instructions for downloading
the real high-resolution PlantVillage dataset.
"""

import os
import sys

SOLANACEAE_CLASSES = [
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

def check_dataset_status(data_dir="dataset"):
    print("\n" + "=" * 70)
    print(f"      PLANTVILLAGE SOLANACEAE DATASET AUDIT: '{data_dir}'")
    print("=" * 70)

    if not os.path.exists(data_dir):
        print(f"[!] Directory '{data_dir}' does not exist.")
        return

    total_images = 0
    class_counts = {}

    for c in SOLANACEAE_CLASSES:
        c_path = os.path.join(data_dir, c)
        if os.path.exists(c_path):
            imgs = [f for f in os.listdir(c_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            class_counts[c] = len(imgs)
            total_images += len(imgs)
        else:
            class_counts[c] = 0

    print(f"{'Class Name':<42} | {'Image Count':<12} | Status")
    print("-" * 70)
    for c, count in class_counts.items():
        status = "[OK] Ready" if count >= 100 else ("[WARN] Small (<100)" if count > 0 else "[MISSING]")
        print(f"{c:<42} | {count:<12} | {status}")
    print("-" * 70)
    print(f"Total Images in Dataset: {total_images}")
    
    if total_images < 500:
        print("\n[NOTE FOR FINAL-YEAR DEFENSE]")
        print("Your dataset currently contains synthetic/starter placeholder samples.")
        print("To achieve 96% - 99% benchmark accuracy, download the real PlantVillage dataset:")
        print("1. Kaggle: 'kaggle datasets download -d emmarex/plantdisease'")
        print("2. Extract the Solanaceae folders into your 'dataset/' directory.")
        print("3. Re-run: python train.py --epochs 10 --batch-size 16")
    else:
        print("\n[SUCCESS] Full-scale dataset detected. Ready for high-accuracy training (>95%).")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    check_dataset_status()
