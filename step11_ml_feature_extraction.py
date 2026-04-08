import cv2
import numpy as np
import os
import pandas as pd
from scipy.stats import skew, kurtosis
import math

# ---------------- PATHS ----------------
PATHS = {
    "Cover": "dataset",
    "Entropy": "results/stego_entropy",
    "Chaotic": "results/stego_chaotic",
    "Hybrid": "results/stego_hybrid"
}

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(image_path, label, method_type):
    img = cv2.imread(image_path)
    if img is None: return None
    
    # Convert to RGB and take Red channel (since embedding is in R)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    R = img[:, :, 0].flatten()
    
    # Statistical Features
    mean_val = np.mean(R)
    var_val = np.var(R)
    skew_val = skew(R)
    kurt_val = kurtosis(R)
    
    # Entropy Feature
    hist = np.histogram(R, 256, (0,256))[0]
    hist = hist / np.sum(hist)
    entropy_val = -sum(p * math.log2(p) for p in hist if p > 0)
    
    return [method_type, mean_val, var_val, skew_val, kurt_val, entropy_val, label]

# ---------------- MAIN LOOP ----------------
data = []
columns = ["Method", "Mean", "Variance", "Skewness", "Kurtosis", "Entropy", "Label"]

print("⏳ Extracting features for ML Analysis...")

# 1. Process Cover Images (Label 0)
for img_name in os.listdir(PATHS["Cover"]):
    path = os.path.join(PATHS["Cover"], img_name)
    # We add "Cover" entries for all three comparison sets to balance data later
    data.append(extract_features(path, 0, "Entropy")) 
    data.append(extract_features(path, 0, "Chaotic"))
    data.append(extract_features(path, 0, "Hybrid"))

# 2. Process Stego Images (Label 1)
for method, folder in PATHS.items():
    if method == "Cover": continue
    
    if os.path.exists(folder):
        for img_name in os.listdir(folder):
            path = os.path.join(folder, img_name)
            data.append(extract_features(path, 1, method))

# ---------------- SAVE CSV ----------------
df = pd.DataFrame([d for d in data if d is not None], columns=columns)
df.to_csv("stego_features.csv", index=False)

print(f"✅ Feature extraction complete! Saved {len(df)} rows to 'stego_features.csv'.")
print("   Features extracted: Mean, Variance, Skewness, Kurtosis, Entropy")