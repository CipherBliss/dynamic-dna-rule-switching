import cv2
import numpy as np
import os
import hashlib

STEGO_PATH="results/stego_chaotic"
DATASET_PATH="dataset"

dna_rules={
    1:{"00":"A","01":"C","10":"G","11":"T"},
    2:{"00":"A","01":"G","10":"C","11":"T"},
    3:{"00":"C","01":"A","10":"T","11":"G"},
    4:{"00":"C","01":"T","10":"A","11":"G"},
    5:{"00":"G","01":"A","10":"T","11":"C"},
    6:{"00":"G","01":"T","10":"A","11":"C"},
    7:{"00":"T","01":"C","10":"G","11":"A"},
    8:{"00":"T","01":"G","10":"C","11":"A"}
}

def get_sha256_seed(image):
    h=hashlib.sha256(image.tobytes()).hexdigest()
    seed=int(h[:8],16)/(2**32)
    return seed if seed !=0 else 0.123456

def logistic_map(x0,r,n):
    x=x0
    seq=[]
    for _ in range(n):
        x=r*x*(1-x)
        seq.append(x)
    return seq

msg_len=int(input("Enter secret message length: "))
total_bits=msg_len*8

for img_name in sorted(os.listdir(STEGO_PATH)):

    stego=cv2.imread(os.path.join(STEGO_PATH,img_name))
    cover=cv2.imread(os.path.join(DATASET_PATH,img_name))

    if stego is None or cover is None:
        continue

    stego=cv2.cvtColor(stego,cv2.COLOR_BGR2RGB)
    cover=cv2.cvtColor(cover,cv2.COLOR_BGR2RGB)

    R,_,_=cv2.split(stego)

    x0=get_sha256_seed(cover)
    chaos=logistic_map(x0,3.99,total_bits//2)

    binary=""
    bit_idx=0
    chaos_idx=0

    for i in range(R.shape[0]):
        for j in range(0,R.shape[1]-1,2):

            if bit_idx >= total_bits:
                break

            rule=int(chaos[chaos_idx]*8)%8 + 1
            table=dna_rules[rule]
            rev={v:k for k,v in table.items()}

            bits=str(R[i,j]&1)+str(R[i,j+1]&1)
            dna=table[bits]
            binary+=rev[dna]

            bit_idx+=2
            chaos_idx+=1

        if bit_idx >= total_bits:
            break

    message=""
    for i in range(0,total_bits,8):
        message+=chr(int(binary[i:i+8],2))

    print(f"Extracted from {img_name}: {message}")

print("Chaotic extraction completed")