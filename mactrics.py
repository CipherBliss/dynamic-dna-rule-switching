import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------- PATHS ----------------
DATASET_PATH = "dataset"              # 23 cover images (RGB)
STEGO_PATH   = "results/stego_hybrid" # 23 stego images
OUTPUT_PATH  = "results/difference_visuals"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ---------------- PROCESS EACH IMAGE ----------------
for img_name in sorted(os.listdir(DATASET_PATH)):

    cover_path = os.path.join(DATASET_PATH, img_name)
    stego_path = os.path.join(STEGO_PATH, img_name)

    if not os.path.exists(stego_path):
        print(f"⚠️ Stego not found for {img_name}")
        continue

    # Read images
    cover = cv2.imread(cover_path)
    stego = cv2.imread(stego_path)

    if cover is None or stego is None:
        continue

    # Convert BGR → RGB
    cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
    stego = cv2.cvtColor(stego, cv2.COLOR_BGR2RGB)

    # Extract R channel
    R_cover = cover[:, :, 0].astype(np.int16)
    R_stego = stego[:, :, 0].astype(np.int16)

    # Difference matrix
    diff = R_stego - R_cover           # values: -1, 0, +1
    diff_vis = diff + 1                # for visualization

    # Zoomed region (fixed for consistency)
    x, y, size = 100, 100, 30
    zoom_diff = diff[x:x+size, y:y+size]

    # ---------------- PLOT ----------------
    plt.figure(figsize=(12, 8))

    # Cover image
    plt.subplot(2, 2, 1)
    plt.imshow(cover)
    plt.title("Cover Image")
    plt.axis("off")

    # Stego image
    plt.subplot(2, 2, 2)
    plt.imshow(stego)
    plt.title("Stego Image")
    plt.axis("off")

    # Difference map
    plt.subplot(2, 2, 3)
    plt.imshow(diff_vis, cmap="hot")
    plt.title("Difference Map (LSB Changes)")
    plt.colorbar(label="Δ Pixel Value")
    plt.axis("off")

    # Zoomed pixel difference
    plt.subplot(2, 2, 4)
    plt.imshow(zoom_diff, cmap="bwr")
    plt.title("Zoomed Pixel-Level Difference")
    plt.colorbar(label="Δ Pixel Value")
    plt.axis("off")

    plt.tight_layout()

    # Save figure
    save_name = img_name.split('.')[0] + "_difference.png"
    plt.savefig(os.path.join(OUTPUT_PATH, save_name))
    plt.close()

    print(f"✅ Visual difference generated for {img_name}")

print("\n🎯 Difference visualization completed for all images")