import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------- PATHS ----------------
DATASET_PATH = "dataset"              # 23 RGB cover images
STEGO_PATH   = "results/stego_hybrid" # stego RGB images
OUTPUT_PATH  = "results/histograms_rgb"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ---------------- PROCESS EACH IMAGE ----------------
for img_name in sorted(os.listdir(DATASET_PATH)):

    cover_path = os.path.join(DATASET_PATH, img_name)
    stego_path = os.path.join(STEGO_PATH, img_name)

    if not os.path.exists(stego_path):
        continue

    # ✅ Read images in COLOR
    cover = cv2.imread(cover_path)
    stego = cv2.imread(stego_path)

    # Convert BGR → RGB
    cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
    stego = cv2.cvtColor(stego, cv2.COLOR_BGR2RGB)

    # Extract RED channel (as per your method)
    R_cover = cover[:,:,0]
    R_stego = stego[:,:,0]

    # ---------------- PLOT ----------------
    plt.figure(figsize=(10,4))

    # RGB Cover Image
    plt.subplot(1,3,1)
    plt.imshow(cover)
    plt.title("Cover Image (RGB)")
    plt.axis("off")

    # Cover Histogram (R channel)
    plt.subplot(1,3,2)
    plt.hist(R_cover.flatten(), bins=256, range=(0,256))
    plt.title("Cover Histogram (R channel)")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")

    # Stego Histogram (R channel)
    plt.subplot(1,3,3)
    plt.hist(R_stego.flatten(), bins=256, range=(0,256))
    plt.title("Stego Histogram (R channel)")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(
        OUTPUT_PATH, img_name.split('.')[0] + "_rgb_hist.png"
    )
    plt.savefig(save_path)
    plt.close()

    print(f"✅ RGB histogram generated for {img_name}")

print("\n🎯 RGB histogram analysis completed for all images")