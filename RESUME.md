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
- **Mosaic augmentation** (`make_mosaic`) + horizontal flip + brightness.
  `train_cli … <imgsz> <mosaic>`.
- GPU/CUDA seam (`pure/backend.hpp`); conv/matmul route through `bk::`.

## Remaining (roughly in priority order)
1. **Real-dataset convergence parity** — train on COCO128 (or similar) and compare final
   mAP@0.5:0.95 against the reference YOLOX. Only synthetic data checked so far.
2. **Richer augmentation** — HSV colour jitter, random affine (scale/translate/rotate/shear),
   mixup, and "close mosaic for the last N epochs". Only flip + brightness + mosaic today.
3. **`data.yaml` + unified CLI** — parse `data.yaml` (train/val paths, `nc`, `names`) and add
   `train`/`val`/`detect`/`export` subcommands.
4. **Training-quality features** — EMA weights, resume-from-checkpoint, multi-scale, label
   smoothing, mAP@0.5:0.95 in the val loop (only mAP@0.5 printed now).
5. **Speed** — YOLOX forward is **per-image, summed per mini-batch** (SimOTA/loss are
   per-image); batch it like yolov8 for a real speedup. Verify the GPU path on hardware.

## Notes / gotchas
- Coords: everything is xyxy in the **letterboxed SxS pixel** space; GT and detections share
  it. `load_boxes_orig` reads either label format into original pixels, then `lb_map` applies
  the letterbox transform.
- YOLOX eval uses a low conf threshold (~0.05) since `score = obj·cls` runs lower than v8.
- `.pth` init: `yolox_tiny.pth` is a Megvii `{'model': state_dict}` checkpoint — read in C++
  via `pt::load_pt_state_under("model")`. Non-ASCII filesystem paths mangle native argv, so
  keep the `.pth` at an ASCII path (or use `rand` init, which needs no path).
- Build: MSVC via `C:/prog/claude/cc11.sh`; `scratch/` must pre-exist.
