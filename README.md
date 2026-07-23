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

- ✅ **inference**: anchor-free decode + class-aware NMS (`pure/infer_yolox.hpp`);
  `pure/m_demo.cpp` runs on a real image straight from a checkout (shipped
  `weights/yolox_tiny/`) — `bus.jpg` → bus + 3-4 people. `pure/m_synth.cpp` is an
  end-to-end test (train a few dozen 128×128 synthetic images → detect).
- ✅ **all sizes** t/n/s/m/l/x (nano = depthwise) across forward / training / .pt / ONNX.

conv routes through the single-header `bk::` device seam, so `nvcc -DUSE_CUDA` trains and
runs on a real GPU. **Colab GPU check: [colab/gpu_check.ipynb](colab/gpu_check.ipynb)**
(https://colab.research.google.com/github/yomei-o/yolox_cpp/blob/main/colab/gpu_check.ipynb).

## Run the demo
```sh
g++ -std=c++20 -O2 -fopenmp -Ipure/third_party pure/m_demo.cpp -o m_demo   # or nvcc -DUSE_CUDA
./m_demo assets/bus.jpg out.png 640
```

| `assets/bus.jpg` → | `assets/zidane.jpg` → |
|---|---|
| ![bus](assets/bus_detected.png) | ![zidane](assets/zidane_detected.png) |

```
assets/bus.jpg  810x1080          assets/zidane.jpg  1280x720
  bus     conf=0.94                 person  conf=0.87
  person  conf=0.87                 person  conf=0.84
  person  conf=0.86                 tie     conf=0.75
  person  conf=0.83                 tie     conf=0.48
```
(YOLOX-tiny, shipped `weights/yolox_tiny/`; decode + NMS in `pure/infer_yolox.hpp`.)

## Build (engine self-test, no deps)
```sh
g++ -std=c++20 -O2 -fopenmp pure/gradcheck2.cpp -o gc2 && ./gc2   # incl. Focus
```
