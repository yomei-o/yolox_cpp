# RESUME — remaining work

Status of the pure-C++ YOLOX training toolchain and what's left to make it a full
replacement for reference-quality training. Verified items live in [README.md](README.md);
this file is the forward-looking TODO.

## Done (pure C++, no Python at run time)
- Engine + YOLOX (Focus stem, SimOTA, decoupled anchor-free head), all sizes
  (nano/tiny/s/m/l/x): forward / loss / train / infer / mAP / `.pt`.
- Real training CLI (`pure/train_cli.cpp`): dataset scan → shuffled mini-batches → epochs →
  SimOTA → loss → Adam(warmup+cosine+wd) → per-epoch val mAP@0.5 → `best.pt`/`last.pt`.
- Initial-weight `.pt` generated in C++ (`pure/make_init_pt.cpp`, `rand`/`from`), all sizes,
  zero-Python bootstrap; `best.pt` loads back into the reference YOLOX (0 unexpected).
- **Standard-YOLO dataset ingestion** — directory scan (`images/`↔`labels/`), normalised
  `cls xc yc w h` labels, arbitrary-size images letterboxed (`pure/dataset.hpp`
  `read_yolo_dataset` / `load_boxes_orig`).
- **Augmentation** — mosaic + mixup + random-affine (rotate/scale/shear/translate) + HSV +
  flip, with **close-mosaic** (disable for last N epochs). `AugCfg` / CLI flags.
- **Unified `yolo` CLI** (`pure/yolo.cpp`) reading `data.yaml`: `train` / `val` / `detect`
  (`export` delegates to the standalone ONNX exporter — see remaining #3). Val reports
  **mAP@0.5 and mAP@0.5:0.95**.
- GPU/CUDA seam (`pure/backend.hpp`); conv/matmul route through `bk::`.

## Remaining (roughly in priority order)
1. **Real-dataset convergence parity** — train on COCO128 (or similar) and compare final
   mAP@0.5:0.95 against the reference YOLOX. Only synthetic data checked so far.
2. **Custom `nc`** — head is fixed at 80 classes; `nc != 80` needs the cls head resized +
   re-initialised. Today class ids must be < 80.
3. **`export` in the unified CLI** — fold BN from the `.pt` and emit ONNX in-CLI (today
   `yolo export` points at the standalone, onnxruntime-verified exporter).
4. **Training-quality features** — EMA weights, resume-from-checkpoint, multi-scale, label
   smoothing, warmup-bias-lr. (mAP@0.5:0.95 in val — done.)
5. **Speed** — YOLOX forward is **per-image, summed per mini-batch** (SimOTA/loss are
   per-image); batch it like yolov8 for a real speedup.
6. **CPU speed on Apple Silicon** — CUDA seam doesn't help on Mac (Metal≠CUDA); add a BLAS
   path (Apple Accelerate / OpenBLAS) to `bk::gemm_hosted` for a big no-GPU CPU speedup.

## Notes / gotchas
- Coords: everything is xyxy in the **letterboxed SxS pixel** space; GT and detections share
  it. `load_boxes_orig` reads either label format into original pixels, then `lb_map` applies
  the letterbox transform.
- YOLOX eval uses a low conf threshold (~0.05) since `score = obj·cls` runs lower than v8.
- `.pth` init: `yolox_tiny.pth` is a Megvii `{'model': state_dict}` checkpoint — read in C++
  via `pt::load_pt_state_under("model")`. Non-ASCII filesystem paths mangle native argv, so
  keep the `.pth` at an ASCII path (or use `rand` init, which needs no path).
- Build: MSVC via `C:/prog/claude/cc11.sh`; `scratch/` must pre-exist.

7. **Verify the unified `train_cli`/`yolo.cpp` under a CUDA build** — compile with `nvcc -DUSE_CUDA` and run COCO128 end-to-end on a (free-Colab) T4. The CUDA seam + a training loop were verified on T4, but the new dataset-ingestion + augmentation CLI path has not been built/run under nvcc yet (aug/dataset are host-side; conv/matmul auto-route to `bk::` on GPU). Est. COCO128/640px/100ep: T4 GPU ~7-20 min; CPU ~a day (measured ~5.7 s/image fwd+bwd at 640px, naive GEMM) so a real GPU is the fix.
