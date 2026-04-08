import cv2
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# IMAGE PATHS (Representative Image)
# --------------------------------------------------
img_name = "img1.png"

orig_path   = os.path.join("dataset", img_name)
entropy_path = os.path.join("results/stego_entropy", img_name)
chaotic_path = os.path.join("results/stego_chaotic", img_name)
hybrid_path  = os.path.join("results/stego_hybrid", img_name)

# --------------------------------------------------
# READ IMAGES
# --------------------------------------------------
orig = cv2.imread(orig_path)
entropy = cv2.imread(entropy_path)
chaotic = cv2.imread(chaotic_path)
hybrid = cv2.imread(hybrid_path)

# Convert BGR → RGB
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
entropy = cv2.cvtColor(entropy, cv2.COLOR_BGR2RGB)
chaotic = cv2.cvtColor(chaotic, cv2.COLOR_BGR2RGB)
hybrid = cv2.cvtColor(hybrid, cv2.COLOR_BGR2RGB)

# --------------------------------------------------
# EXTRACT RED CHANNEL
# --------------------------------------------------
orig_R = orig[:,:,0]
entropy_R = entropy[:,:,0]
chaotic_R = chaotic[:,:,0]
hybrid_R = hybrid[:,:,0]

# --------------------------------------------------
# PLOT HISTOGRAM
# --------------------------------------------------
plt.figure(figsize=(10,6))

plt.hist(orig_R.flatten(), bins=256, color='black', alpha=0.5, label='Original')
plt.hist(entropy_R.flatten(), bins=256, color='green', alpha=0.5, label='Entropy-Based')
plt.hist(chaotic_R.flatten(), bins=256, color='blue', alpha=0.5, label='Chaotic-Based')
plt.hist(hybrid_R.flatten(), bins=256, color='red', alpha=0.5, label='Hybrid-Based')

plt.title("Histogram Comparison (Red Channel)")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
