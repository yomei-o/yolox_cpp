# yolox_cpp

Training **YOLOX** (Megvii) in C++ with the same from-scratch, dependency-free autograd
engine used in [yolov8_cpp](https://github.com/yomei-o/yolov8_cpp) /
[yolov5_cpp](https://github.com/yomei-o/yolov5_cpp) /
[yolov11_cpp](https://github.com/yomei-o/yolov11_cpp). CPU/OpenMP with `g++`, real CUDA
with `nvcc -DUSE_CUDA` (single-header `pure/backend.hpp` seam).

YOLOX is the most different of the family: **anchor-free with a decoupled head** and
**SimOTA** dynamic label assignment (no anchors, no DFL/TAL). See
[`pure/NOTES_yolox.md`](pure/NOTES_yolox.md) for the full architecture blueprint.

## Status
- ✅ M0: oracle (`torch.hub` yolox-tiny) + architecture blueprint + **Focus op** (space-to-depth), gradient-checked.
- ✅ M1: full forward parity vs PyTorch (`net_yolox.hpp` + `export_yolox.py`) — L0 1.8e-4 / L1 4e-5 / L2 1.9e-5.
- ✅ M2: loss — **SimOTA** (plain, no-grad) == YOLOX `get_assignments`; IoU/BCE forward 1.9e-6, **grads 3e-8** vs PyTorch.
- ✅ M3: **end-to-end training** (`m3_train.cpp`, forward→SimOTA→loss→backward→SGD) — loss 24.1→3.2.

- ✅ **.pt write-back**: train unfused conv+BN in C++ → drop into a standard YOLOX `.pt`
  (re-loads with 0 missing/unexpected keys, runs). `net_yolox_unfused.hpp` + `ref/writeback_yolox.py`.
- ✅ **ONNX export/import**: `onnx_export_yolox.cpp` writes `yolox_tiny.onnx` (Focus = strided
  Slice + Concat) — onnxruntime-verified (~9e-5); `m_onnx_run.cpp` runs a `.onnx` graph-driven
  in the pure engine (~1.8e-4). Hand-rolled protobuf, no deps.

conv routes through the single-header `bk::` device seam, so `nvcc -DUSE_CUDA` trains on a
real GPU (verified for the sibling repos on a Colab T4).

## Build (engine self-test, no deps)
```sh
g++ -std=c++20 -O2 -fopenmp pure/gradcheck2.cpp -o gc2 && ./gc2   # incl. Focus
```
