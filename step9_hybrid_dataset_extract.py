import cv2
import numpy as np
import math
import hashlib
import os

DATASET_PATH="dataset"
STEGO_PATH="results/stego_hybrid"

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

def sha_seed(img):
    h=hashlib.sha256(img.tobytes()).hexdigest()
    seed=(int(h[:8],16)%1000)/1000.0
    return seed if seed!=0 else 0.123

def logistic_map(x0,r,n):
    x=x0
    seq=[]
    for _ in range(n):
        x=r*x*(1-x)
        seq.append(x)
    return seq

def entropy(block):
    hist=np.histogram(block.flatten(),256,(0,256))[0]
    hist=hist/np.sum(hist)
    return -sum(p*np.log2(p) for p in hist if p>0)

msg_len=int(input("Enter secret message length: "))
total_bits=msg_len*8
num_blocks=math.ceil(total_bits/64)

for img_name in os.listdir(STEGO_PATH):

    cover=cv2.imread(os.path.join(DATASET_PATH,img_name))
    stego=cv2.imread(os.path.join(STEGO_PATH,img_name))

    cover=cv2.cvtColor(cover,cv2.COLOR_BGR2RGB)
    stego=cv2.cvtColor(stego,cv2.COLOR_BGR2RGB)

    R0,_,_=cv2.split(cover)
    R,_,_=cv2.split(stego)

    chaos=logistic_map(sha_seed(cover),3.99,num_blocks)

    bits=""
    block_idx=0

    for i in range(0,R.shape[0],8):
        for j in range(0,R.shape[1],8):

            if len(bits) >= total_bits:
                break

            block_o=R0[i:i+8,j:j+8]
            block_s=R[i:i+8,j:j+8]

            e=entropy(block_o)

            if e < 2.5:
                continue

            r_entropy=min(int(e)+1,8)
            r_chaos=int(chaos[block_idx]*8)%8+1
            rule=(r_entropy ^ r_chaos)%8+1

            table=dna_rules[rule]
            rev={v:k for k,v in table.items()}

            block_bits=""
            for x in range(8):
                for y in range(8):
                    block_bits+=str(block_s[x,y]&1)

            for k in range(0,64,2):
                dna=table[block_bits[k:k+2]]
                bits+=rev[dna]
                if len(bits)>=total_bits:
                    break

            block_idx+=1

    msg=""
    for i in range(0,total_bits,8):
        msg+=chr(int(bits[i:i+8],2))

    print(f"Extracted from {img_name}: {msg}")

print("Hybrid extraction completed")