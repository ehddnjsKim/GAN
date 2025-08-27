# C:\Dw\GAN\GAN_iris\preprocess_split.py
import os, glob, random
from pathlib import Path
from PIL import Image

# 원본 iris 이미지 폴더
RAW = r"C:\Dw\GAN\GAN_iris\data\raw"
REAL = r"C:\Dw\GAN\GAN_iris\data\real"
SPLIT = r"C:\Dw\GAN\GAN_iris\data\splits"
SIZE = 256
random.seed(42)

os.makedirs(REAL, exist_ok=True)
os.makedirs(SPLIT, exist_ok=True)

paths = []
for ext in ("*.jpg","*.jpeg","*.png"):
    paths += glob.glob(os.path.join(RAW, "**", ext), recursive=True)
random.shuffle(paths)

out_paths = []
for i, p in enumerate(paths):
    try:
        img = Image.open(p).convert("RGB")
        img = img.resize((SIZE, SIZE))
        out = os.path.join(REAL, f"iris_{i:05d}.jpg")
        img.save(out, quality=95)
        out_paths.append(out)
    except Exception as e:
        print("skip:", p, e)

n = len(out_paths)
tr = out_paths[:int(n*0.7)]
va = out_paths[int(n*0.7):int(n*0.85)]
te = out_paths[int(n*0.85):]

def dump(name, arr):
    with open(os.path.join(SPLIT, f"{name}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(arr))

dump("train", tr); dump("val", va); dump("test", te)
print(f"done: total={n}, train={len(tr)}, val={len(va)}, test={len(te)}")
