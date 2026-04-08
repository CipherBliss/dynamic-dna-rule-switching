import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

os.makedirs("final_reports", exist_ok=True)

img_name = "img1.png" 
paths = {
    "Original": os.path.join("dataset", img_name),
    "Hybrid Stego": os.path.join("results/stego_hybrid", img_name)
}

def get_pixel_clusters(image_path, k=5):
    img = cv2.imread(image_path)
    if img is None: return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = np.float32(img.reshape((-1, 3)))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    return np.uint8(kmeans.cluster_centers_), np.unique(kmeans.labels_, return_counts=True)[1]

centers_orig, counts_orig = get_pixel_clusters(paths["Original"])
centers_stego, counts_stego = get_pixel_clusters(paths["Hybrid Stego"])

if centers_orig is not None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].bar(range(5), counts_orig, color=centers_orig / 255.0)
    ax[0].set_title("Original Image Clusters")
    
    ax[1].bar(range(5), counts_stego, color=centers_stego / 255.0)
    ax[1].set_title("Hybrid Stego Image Clusters")
    
    plt.tight_layout()
    plt.savefig("final_reports/kmeans_analysis.png")
    print("✅ K-Means graph saved to 'final_reports/kmeans_analysis.png'")
else:
    print("❌ Error: Could not read images. Make sure 'dataset/img1.png' exists.")