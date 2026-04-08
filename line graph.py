import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------
# METHODS & VALUES
# ----------------------------------------

methods = ["Entropy", "Chaotic", "Hybrid (Proposed)"]

avg_psnr = [96.5, 102.3, 95.8]
avg_mse  = [2.1e-5, 5.3e-6, 2.2e-5]
avg_ssim = [0.9999, 0.99995, 0.9999]

# ----------------------------------------
# PSNR LINE GRAPH
# ----------------------------------------

plt.figure(figsize=(7,5))
plt.plot(methods, avg_psnr, marker='o', linewidth=2)

# Highlight Proposed
plt.scatter(methods[-1], avg_psnr[-1], color='red', s=120, zorder=5)
plt.text(methods[-1], avg_psnr[-1], "  Proposed", color='red')

plt.title("PSNR Comparison of DNA Steganography Methods")
plt.xlabel("Method")
plt.ylabel("PSNR (dB)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ----------------------------------------
# MSE LINE GRAPH
# ----------------------------------------

plt.figure(figsize=(7,5))
plt.plot(methods, avg_mse, marker='o', linewidth=2)

plt.scatter(methods[-1], avg_mse[-1], color='red', s=120, zorder=5)
plt.text(methods[-1], avg_mse[-1], "  Proposed", color='red')

plt.title("MSE Comparison of DNA Steganography Methods")
plt.xlabel("Method")
plt.ylabel("MSE")
plt.yscale("log")   # Important for small MSE values
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ----------------------------------------
# SSIM LINE GRAPH
# ----------------------------------------

plt.figure(figsize=(7,5))
plt.plot(methods, avg_ssim, marker='o', linewidth=2)

plt.scatter(methods[-1], avg_ssim[-1], color='red', s=120, zorder=5)
plt.text(methods[-1], avg_ssim[-1], "  Proposed", color='red')

plt.title("SSIM Comparison of DNA Steganography Methods")
plt.xlabel("Method")
plt.ylabel("SSIM")
plt.ylim(0.99, 1.0)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()