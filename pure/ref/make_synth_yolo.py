"""Generate a tiny dataset in the STANDARD Ultralytics/YOLO layout to exercise the C++
standard-YOLO ingestion + mosaic path:
  data_yolo/images/{train,val}/*.jpg   (RECTANGULAR images, to test letterboxing)
  data_yolo/labels/{train,val}/*.txt   (one "cls xc yc w h" per line, normalised [0,1])
Usage: python make_synth_yolo.py [ntrain]"""
import os, sys, random
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
NTR  = int(sys.argv[1]) if len(sys.argv) > 1 else 40
D = os.path.join(HERE, "data_yolo")
random.seed(0)
COLORS = [(220, 40, 40), (40, 200, 40), (50, 90, 230)]     # class 0,1,2

def gen(split, name):
    W = random.randint(96, 200); H = random.randint(96, 200)   # non-square -> real letterbox
    im = Image.new("RGB", (W, H), (114, 114, 114)); dr = ImageDraw.Draw(im)
    lines = []
    for _ in range(random.randint(1, 3)):
        c = random.randint(0, 2)
        w = random.randint(W // 5, W // 3); h = random.randint(H // 5, H // 3)
        x1 = random.randint(2, W - w - 2); y1 = random.randint(2, H - h - 2)
        dr.rectangle([x1, y1, x1 + w, y1 + h], fill=COLORS[c])
        xc = (x1 + w / 2) / W; yc = (y1 + h / 2) / H            # normalised cx cy w h
        lines.append(f"{c} {xc:.6f} {yc:.6f} {w / W:.6f} {h / H:.6f}")
    idir = os.path.join(D, "images", split); ldir = os.path.join(D, "labels", split)
    os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
    im.save(os.path.join(idir, name + ".jpg"))
    open(os.path.join(ldir, name + ".txt"), "w").write("\n".join(lines) + "\n")

for i in range(NTR):                 gen("train", f"tr{i:03d}")
for i in range(max(4, NTR // 4)):    gen("val",   f"va{i:03d}")
print(f"wrote {NTR} train + {max(4, NTR // 4)} val images (standard YOLO layout) to {D}")
