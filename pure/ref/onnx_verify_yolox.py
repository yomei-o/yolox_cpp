"""Verify yolox_tiny.onnx in onnxruntime against the pure-engine reference (data_net)."""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data_net")
import onnxruntime as ort
IMG = int(open(os.path.join(D, "io.txt")).readline())
x = np.fromfile(os.path.join(D, "x.bin"), np.float32).reshape(1, 3, IMG, IMG)
sess = ort.InferenceSession(os.path.join(HERE, "..", "..", "yolox_tiny.onnx"), providers=["CPUExecutionProvider"])
outs = sess.run(None, {"images": x})
names = [o.name for o in sess.get_outputs()]
d = {n: o for n, o in zip(names, outs)}
worst = 0.0
for i in range(3):
    ref = np.fromfile(os.path.join(D, f"ref_L{i}.bin"), np.float32)
    o = d[f"out{i}"].ravel()
    mx = float(np.abs(o - ref).max()); worst = max(worst, mx)
    print(f"out{i} {d[f'out{i}'].shape}  max|diff|={mx:.3e}")
print(f"\n{'OK' if worst < 1e-3 else 'FAIL'}  onnxruntime == pure yolox forward (worst {worst:.3e})")
