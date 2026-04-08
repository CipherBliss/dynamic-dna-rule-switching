import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ORIGINAL_PATH = "dataset"
ENTROPY_PATH  = "results/stego_entropy"
CHAOTIC_PATH  = "results/stego_chaotic"
HYBRID_PATH   = "results/stego_hybrid"

valid_ext = (".png",".jpg",".jpeg",".bmp")

# --------------------------------------------------
# METRIC FUNCTION
# --------------------------------------------------
def compute_metrics(original, stego):

    original = original.astype(np.float64)
    stego = stego.astype(np.float64)

    mse = np.mean((original - stego) ** 2)

    if mse < 1e-10:
        psnr = 100
    else:
        psnr = 10 * math.log10((255 ** 2) / mse)

    ssim_val = ssim(
        original,
        stego,
        channel_axis=2,
        data_range=255
    )

    return mse, psnr, ssim_val


# --------------------------------------------------
# STORE RESULTS
# --------------------------------------------------
mse_entropy, psnr_entropy, ssim_entropy = [], [], []
mse_chaotic, psnr_chaotic, ssim_chaotic = [], [], []
mse_hybrid,  psnr_hybrid,  ssim_hybrid  = [], [], []


# --------------------------------------------------
# PROCESS DATASET
# --------------------------------------------------
for img_name in sorted(os.listdir(ORIGINAL_PATH)):

    if not img_name.lower().endswith(valid_ext):
        continue

    orig_path = os.path.join(ORIGINAL_PATH, img_name)
    e_path    = os.path.join(ENTROPY_PATH, img_name)
    c_path    = os.path.join(CHAOTIC_PATH, img_name)
    h_path    = os.path.join(HYBRID_PATH, img_name)

    if not (os.path.exists(e_path) and os.path.exists(c_path) and os.path.exists(h_path)):
        continue

    orig = cv2.imread(orig_path)
    e    = cv2.imread(e_path)
    c    = cv2.imread(c_path)
    h    = cv2.imread(h_path)

    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    e    = cv2.cvtColor(e, cv2.COLOR_BGR2RGB)
    c    = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    h    = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)

    # ensure same size
    if orig.shape != e.shape:
        e = cv2.resize(e,(orig.shape[1],orig.shape[0]))
    if orig.shape != c.shape:
        c = cv2.resize(c,(orig.shape[1],orig.shape[0]))
    if orig.shape != h.shape:
        h = cv2.resize(h,(orig.shape[1],orig.shape[0]))

    mse, psnr, ssim_v = compute_metrics(orig, e)
    mse_entropy.append(mse)
    psnr_entropy.append(psnr)
    ssim_entropy.append(ssim_v)

    mse, psnr, ssim_v = compute_metrics(orig, c)
    mse_chaotic.append(mse)
    psnr_chaotic.append(psnr)
    ssim_chaotic.append(ssim_v)

    mse, psnr, ssim_v = compute_metrics(orig, h)
    mse_hybrid.append(mse)
    psnr_hybrid.append(psnr)
    ssim_hybrid.append(ssim_v)


# --------------------------------------------------
# AVERAGES
# --------------------------------------------------
avg = lambda x: float(np.mean(x))

avg_mse  = [avg(mse_entropy), avg(mse_chaotic), avg(mse_hybrid)]
avg_psnr = [avg(psnr_entropy), avg(psnr_chaotic), avg(psnr_hybrid)]
avg_ssim = [avg(ssim_entropy), avg(ssim_chaotic), avg(ssim_hybrid)]

methods = ["Entropy","Chaotic","Hybrid (Proposed)"]


# --------------------------------------------------
# PRINT RESULTS (HIGH PRECISION)
# --------------------------------------------------
print("\n📊 DATASET AVERAGE RESULTS\n")

for i,m in enumerate(methods):
    print(m)
    print(f"  MSE  : {avg_mse[i]:.8e}")
    print(f"  PSNR : {avg_psnr[i]:.4f} dB")
    print(f"  SSIM : {avg_ssim[i]:.10f}")
    print("-"*40)


# --------------------------------------------------
# GRAPH PLOT
# --------------------------------------------------
plt.figure(figsize=(14,4))

# PSNR
plt.subplot(1,3,1)
plt.bar(methods, avg_psnr, color=['#4CAF50','#2196F3','#FF9800'])
plt.title("PSNR Comparison")
plt.ylabel("PSNR (dB)")
plt.grid(axis='y', linestyle='--', alpha=0.5)

# MSE
plt.subplot(1,3,2)
plt.bar(methods, avg_mse, color=['#4CAF50','#2196F3','#FF9800'])
plt.title("MSE Comparison")
plt.ylabel("MSE")
plt.yscale('log')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# SSIM (zoomed)
plt.subplot(1,3,3)
plt.bar(methods, avg_ssim, color=['#4CAF50','#2196F3','#FF9800'])
plt.title("SSIM Comparison")
plt.ylabel("SSIM")
plt.ylim(0.9999,1.0)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle("Image Quality Comparison of Steganography Methods", fontsize=14)

plt.tight_layout()

# save figure for paper
plt.savefig("results/quality_comparison.png", dpi=300)

plt.show()

# --------------------------------------------------
# LINE GRAPH PLOTTING
# --------------------------------------------------

# plt.figure(figsize=(14,4))

# x = np.arange(len(methods))

# # PSNR LINE GRAPH
# plt.subplot(1,3,1)
# plt.plot(methods, avg_psnr, marker='o', linewidth=2, color='#FF9800')
# plt.title("PSNR Comparison")
# plt.ylabel("PSNR (dB)")
# plt.grid(True, linestyle='--', alpha=0.6)

# # MSE LINE GRAPH
# plt.subplot(1,3,2)
# plt.plot(methods, avg_mse, marker='o', linewidth=2, color='#2196F3')
# plt.title("MSE Comparison")
# plt.ylabel("MSE")
# plt.yscale('log')
# plt.grid(True, linestyle='--', alpha=0.6)

# # SSIM LINE GRAPH
# plt.subplot(1,3,3)
# plt.plot(methods, avg_ssim, marker='o', linewidth=2, color='#4CAF50')
# plt.title("SSIM Comparison")
# plt.ylabel("SSIM")
# plt.ylim(0.9999,1.0)
# plt.grid(True, linestyle='--', alpha=0.6)

# plt.suptitle("Image Quality Comparison of DNA Steganography Methods", fontsize=14)

# plt.tight_layout()

# # Save figure for paper
# plt.savefig("results/line_graph_comparison.png", dpi=300)

# plt.show()