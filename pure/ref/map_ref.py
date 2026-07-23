"""Reference for the pure COCO-mAP: generate a synthetic detection set (GT + predictions
with jitter, wrong classes and false positives), dump it for the C++ evaluator, and
compute mAP@0.50 and mAP@0.50:0.95 with pycocotools."""
import os, numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_map"); os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(0)
NIMG, NCLS = 6, 3

def rbox():
    x, y = rng.uniform(0, 60, 2); w, h = rng.uniform(15, 40, 2)
    return [float(x), float(y), float(x + w), float(y + h)]

images = []
for _ in range(NIMG):
    G = rng.integers(1, 5)
    gts = [(int(rng.integers(0, NCLS)), *rbox()) for _ in range(G)]
    dts = []
    for (c, x1, y1, x2, y2) in gts:
        if rng.random() < 0.85:                        # a (jittered) detection for this GT
            j = rng.normal(0, 3, 4)
            cc = c if rng.random() < 0.9 else int(rng.integers(0, NCLS))   # sometimes wrong class
            dts.append((cc, x1 + j[0], y1 + j[1], x2 + j[2], y2 + j[3], float(rng.uniform(0.3, 0.99))))
    for _ in range(int(rng.integers(0, 3))):           # random false positives
        dts.append((int(rng.integers(0, NCLS)), *rbox(), float(rng.uniform(0.1, 0.6))))
    images.append((gts, dts))

# dump for C++
with open(os.path.join(D, "set.txt"), "w") as f:
    f.write(f"{NIMG} {NCLS}\n")
    for gts, dts in images:
        f.write(f"{len(gts)}\n")
        for (c, x1, y1, x2, y2) in gts: f.write(f"{c} {x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f}\n")
        f.write(f"{len(dts)}\n")
        for (c, x1, y1, x2, y2, s) in dts: f.write(f"{c} {x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f} {s:.6f}\n")

# pycocotools
anns, dts_coco, aid = [], [], 1
for i, (gts, dts) in enumerate(images):
    for (c, x1, y1, x2, y2) in gts:
        anns.append({"id": aid, "image_id": i + 1, "category_id": c + 1,
                     "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0}); aid += 1
    for (c, x1, y1, x2, y2, s) in dts:
        dts_coco.append({"image_id": i + 1, "category_id": c + 1,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
gt = COCO()
gt.dataset = {"images": [{"id": i + 1} for i in range(NIMG)], "annotations": anns,
              "categories": [{"id": c + 1} for c in range(NCLS)]}
gt.createIndex()
dt = gt.loadRes(dts_coco)
E = COCOeval(gt, dt, "bbox"); E.evaluate(); E.accumulate(); E.summarize()
map5095, map50 = float(E.stats[0]), float(E.stats[1])
open(os.path.join(D, "ref.txt"), "w").write(f"{map50:.6f} {map5095:.6f}\n")
print(f"\npycocotools: mAP@0.50 = {map50:.6f}   mAP@0.50:0.95 = {map5095:.6f}")
