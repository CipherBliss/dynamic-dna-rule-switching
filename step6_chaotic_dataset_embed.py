import cv2
import numpy as np
import os
import hashlib

# ---------------- PATHS ----------------
DATASET_PATH = "dataset"
OUTPUT_PATH = "results/stego_chaotic"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ---------------- DNA RULES ----------------
dna_rules = {
    1: {"00":"A","01":"C","10":"G","11":"T"},
    2: {"00":"A","01":"G","10":"C","11":"T"},
    3: {"00":"C","01":"A","10":"T","11":"G"},
    4: {"00":"C","01":"T","10":"A","11":"G"},
    5: {"00":"G","01":"A","10":"T","11":"C"},
    6: {"00":"G","01":"T","10":"A","11":"C"},
    7: {"00":"T","01":"C","10":"G","11":"A"},
    8: {"00":"T","01":"G","10":"C","11":"A"}
}

# ---------------- SHA-256 SEED ----------------
def get_sha256_seed(image):
    h = hashlib.sha256(image.tobytes()).hexdigest()
    seed = int(h[:8], 16) / (2**32)
    if seed == 0:
        seed = 0.123456
    return seed

# ---------------- LOGISTIC MAP ----------------
def logistic_map(x0, r, n):
    x = x0
    seq = []
    for _ in range(n):
        x = r * x * (1 - x)
        seq.append(x)
    return seq

# ---------------- SECRET MESSAGE ----------------
secret_text = input("Enter secret message: ")
binary_data = ''.join(format(ord(c), '08b') for c in secret_text)
print("Total bits:", len(binary_data))

# ---------------- EMBEDDING ----------------
for img_name in sorted(os.listdir(DATASET_PATH)):

    img_path = os.path.join(DATASET_PATH, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(img)

    # SHA-based chaotic seed
    x0 = get_sha256_seed(img)
    chaos = logistic_map(x0, 3.99, len(binary_data)//2)

    bit_idx = 0
    chaos_idx = 0

    for i in range(R.shape[0]):
        for j in range(0, R.shape[1]-1, 2):

            if bit_idx + 2 > len(binary_data):
                break

            rule = int(chaos[chaos_idx] * 8) % 8 + 1
            table = dna_rules[rule]
            reverse = {v:k for k,v in table.items()}

            dna = table[binary_data[bit_idx:bit_idx+2]]
            enc_bits = reverse[dna]

            R[i, j]   = (R[i, j]   & 254) | int(enc_bits[0])
            R[i, j+1] = (R[i, j+1] & 254) | int(enc_bits[1])

            bit_idx += 2
            chaos_idx += 1

        if bit_idx + 2 > len(binary_data):
            break

    stego = cv2.merge((R, G, B))
    cv2.imwrite(
        os.path.join(OUTPUT_PATH, img_name),
        cv2.cvtColor(stego, cv2.COLOR_RGB2BGR)
    )

    print(f"✅ Embedded → {img_name}")

print("\n🎯 Chaotic DNA embedding with SHA-256 completed")
