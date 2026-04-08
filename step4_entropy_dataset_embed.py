import cv2
import numpy as np
import math
import os

DATASET_PATH = "dataset"
OUTPUT_PATH = "results/stego_entropy"
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

# ---------------- ENTROPY ----------------
def entropy(block):
    hist = np.histogram(block.flatten(),256,(0,256))[0]
    hist = hist / np.sum(hist)
    return -sum(p*math.log2(p) for p in hist if p>0)

# ---------------- SECRET ----------------
secret = input("Enter secret message: ")
binary = ''.join(format(ord(c),'08b') for c in secret)
total_bits = len(binary)

# ---------------- DATASET PROCESS ----------------
for img_name in os.listdir(DATASET_PATH):

    img = cv2.imread(os.path.join(DATASET_PATH,img_name))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    R,G,B = cv2.split(img)

    bit_idx = 0

    for i in range(0,R.shape[0],8):
        for j in range(0,R.shape[1],8):

            if bit_idx >= total_bits:
                break

            block = R[i:i+8,j:j+8]

            # Compute entropy
            e = entropy(block)

            # Map entropy → DNA rule
            rule = min(int(e) + 1,8)

            table = dna_rules[rule]
            rev = {v:k for k,v in table.items()}

            bits = binary[bit_idx:bit_idx+64].ljust(64,'0')

            enc = "".join(rev[table[bits[k:k+2]]] for k in range(0,64,2))

            idx = 0
            for x in range(8):
                for y in range(8):

                    if idx >= len(enc):
                        break

                    bit = int(enc[idx])

                    # Only modify pixel if needed (IMPORTANT FOR PSNR)
                    if (block[x,y] & 1) != bit:
                        block[x,y] ^= 1

                    idx += 1

            R[i:i+8,j:j+8] = block
            bit_idx += 64

    stego = cv2.merge((R,G,B))

    cv2.imwrite(
        os.path.join(OUTPUT_PATH,img_name),
        cv2.cvtColor(stego,cv2.COLOR_RGB2BGR)
    )

    print(f"Entropy embedded → {img_name}")

print("\nEntropy embedding completed")