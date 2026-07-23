# yolox_cpp

Training **YOLOX** (Megvii) in C++ with the same from-scratch, dependency-free autograd
engine used in [yolov8_cpp](https://github.com/yomei-o/yolov8_cpp) /
[yolov5_cpp](https://github.com/yomei-o/yolov5_cpp) /
[yolov11_cpp](https://github.com/yomei-o/yolov11_cpp). CPU/OpenMP with `g++`, real CUDA
with `nvcc -DUSE_CUDA` (single-header `pure/backend.hpp` seam).

YOLOX is the most different of the family: **anchor-free with a decoupled head** and
**SimOTA** dynamic label assignment (no anchors, no DFL/TAL). See
[`pure/NOTES_yolox.md`](pure/NOTES_yolox.md) for the full architecture blueprint.

## Status (WIP)
- ✅ M0: oracle (`torch.hub` yolox-tiny) + architecture blueprint + **Focus op** (the only
  new primitive; space-to-depth stem), gradient-checked.
- ⏳ M1: full forward parity vs PyTorch (`net_yolox.hpp` + `export_yolox.py`).
- ⏳ M2: loss — SimOTA (plain, no-grad) + IoU/BCE (forward + backward) vs PyTorch.
- ⏳ M3: training loop; then GPU via the `bk::` seam.

## Build (engine self-test, no deps)
```sh
g++ -std=c++20 -O2 -fopenmp pure/gradcheck2.cpp -o gc2 && ./gc2   # incl. Focus
```
