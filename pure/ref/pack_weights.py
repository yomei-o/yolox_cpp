"""Pack data_net (per-conv w{i}/b{i} + manifest + io) into a single shippable blob at
weights/<model>/ so inference runs from a checkout (no Python export needed).
Usage: python pack_weights.py <model>   (run after export_yolox.py <img> <model>)"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_net")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "yolox_tiny"
OUT = os.path.join(HERE, "..", "..", "weights", MODEL); os.makedirs(OUT, exist_ok=True)

lines = open(os.path.join(D, "manifest.txt")).read().splitlines()
n = int(lines[0])
blob = []
for i in range(n):
    blob.append(np.fromfile(os.path.join(D, f"w{i}.bin"), np.float32))
    blob.append(np.fromfile(os.path.join(D, f"b{i}.bin"), np.float32))
np.concatenate(blob).astype(np.float32).tofile(os.path.join(OUT, "weights.bin"))
open(os.path.join(OUT, "manifest.txt"), "w").write("\n".join(lines) + "\n")
# io.txt: BD/DW (imgsz irrelevant for weights; demo picks its own input size)
io = open(os.path.join(D, "io.txt")).readline().split()
open(os.path.join(OUT, "io.txt"), "w").write(" ".join(io) + "\n")
print(f"packed {n} convs -> {os.path.relpath(OUT)} (weights.bin + manifest.txt + io.txt)")
