# YOLOX-tiny — architecture blueprint (for the pure C++ port)

Source of truth: `torch.hub.load("Megvii-BaseDetection/YOLOX", "yolox_tiny", pretrained=True)`
(needs `pip install loguru tabulate thop`). 5.06M params, nc=80, strides [8,16,32],
base_channels=24, base_depth=1 (width 0.375, depth 0.33). Act = SiLU. BN folds into conv
for inference (same as v8/v11); keep unfused for training/round-trip later.

## New op (only one) — implemented
- **Focus** (space-to-depth): (N,C,H,W)→(N,4C,H/2,W/2). YOLOX channel order
  `[tl, bl, tr, br] = [(::2,::2),(1::2,::2),(::2,1::2),(1::2,1::2)]`. In `autograd.hpp`,
  gradcheck OK. Everything else (conv/silu/maxpool/upsample/concat/slice) already exists.

## Building blocks
- **BaseConv** = conv(bias=False) + BN + SiLU  → fused conv + SiLU.
- **Bottleneck**(c,c_,shortcut): conv1(c→c_,1) + conv2(c_→c,3); `out = x+... if shortcut`.
  shortcut = True in backbone, **False** in neck (PAFPN).
- **CSPLayer / C3**(c1,c2,n,shortcut): conv1(c1→c_,1), conv2(c1→c_,1), m=Sequential(n×Bottleneck),
  conv3(2c_→c2,1). forward: `conv3(cat([ m(conv1(x)), conv2(x) ], 1))`.  c_ = c2*0.5.
- **SPPBottleneck**(c1,c2): conv1(c1→c_,1), maxpool k∈{5,9,13} s1 (pad k//2),
  conv2(c_*4→c2,1). forward: `conv2(cat([x, p5, p9, p13], 1))`, x=conv1(in). c_=c1//2.
- **Focus stem**: focus(3→12) → BaseConv(12→24,3,1).

## Backbone: CSPDarknet  (out: dark3=s8, dark4=s16, dark5=s32)
- stem   : Focus → 24
- dark2  : BaseConv(24→48,3,s2)  , CSPLayer(48,48,n=1,shortcut=T)
- dark3  : BaseConv(48→96,3,s2)  , CSPLayer(96,96,n=3,shortcut=T)      → **C3 out (96, s8)**
- dark4  : BaseConv(96→192,3,s2) , CSPLayer(192,192,n=3,shortcut=T)    → **C4 out (192, s16)**
- dark5  : BaseConv(192→384,3,s2), SPPBottleneck(384→384), CSPLayer(384,384,n=1,shortcut=T)
           → **C5 out (384, s32)**

## Neck: YOLOPAFPN  (canonical forward)
```
x2,x1,x0 = C3(s8=96), C4(s16=192), C5(s32=384)
fpn0 = lateral_conv0(x0)          # 384->192
u    = upsample(fpn0) x2          # 192, s16
p4   = C3_p4(cat[u, x1])          # 384->192      (shortcut=False)
red  = reduce_conv1(p4)           # 192->96
u2   = upsample(red) x2           # 96, s8
pan2 = C3_p3(cat[u2, x2])         # 192->96       → head level0 (s8, 96)
d0   = bu_conv2(pan2)             # 96->96, s2
pan1 = C3_n3(cat[d0, red])        # 192->192      → head level1 (s16, 192)
d1   = bu_conv1(pan1)             # 192->192, s2
pan0 = C3_n4(cat[d1, fpn0])       # 384->384      → head level2 (s32, 384)
```
(neck CSPLayers use shortcut=False)

## Head: decoupled  (per level i, in = [96,192,384])
- stem_i = BaseConv(in→96,1)
- cls tower = 2× BaseConv(96→96,3);  reg tower = 2× BaseConv(96→96,3)
- cls_pred_i = Conv2d(96→80,1, bias)   [plain, no BN/act]
- reg_pred_i = Conv2d(96→4,1, bias)    [plain]
- obj_pred_i = Conv2d(96→1,1, bias)    [plain]
- per-level output = `cat([reg(4), obj(1), cls(80)], 1)` → (b, 85, h, w)

## Decode (anchor-free, inference / for loss box)
per grid cell (gx,gy) at stride s:
  `x = (reg_x + gx)*s`, `y = (reg_y + gy)*s`, `w = exp(reg_w)*s`, `h = exp(reg_h)*s` (cx,cy,w,h).
  score = sigmoid(obj) * sigmoid(cls).

## Loss (SimOTA) — the big new piece (M2)
- **SimOTA** dynamic assignment (no-grad, plain C++): cost = cls_cost + 3*iou_cost + 1e5*(~center_prior);
  candidate mask = center/‘in box or in center radius’; dynamic-k = sum of top-k IoU per gt; pick lowest-cost.
- losses: **IoU loss** (1-IoU or GIoU) for box (positives), **BCE** for obj (all) and cls (positives).
  optional L1 on raw reg. Reuse `bce_logits`, CIoU/IoU from engine.

## Milestones
- M0 oracle + blueprint + Focus op  — ✅ done
- M1 forward parity (net_yolox.hpp + export_yolox.py, CPU-forced) vs torch  — ✅ done
  (L0 1.8e-4 / L1 4e-5 / L2 1.9e-5). `pure/m1_forward.cpp`, debug `pure/m_dbg.cpp`.
- M2 loss: SimOTA (plain) + IoU/BCE (fwd+bwd) vs torch  — ✅ done
  - M2a `simota.hpp` + `m2a_simota.cpp`: SimOTA == YOLOX get_assignments (num_fg/fg_mask/
    matched_gt/matched_cls/pred_ious all exact). M2b `yolox_loss.hpp` + `m2b_loss.cpp`:
    loss fwd 1.9e-6, grads 3e-8 vs PyTorch. Targets treated as constants (soft cls label
    = onehot*pred_iou detached; forward value identical to YOLOX get_losses).
  - loss formulas: IoU loss = 1-iou², cost = clsBCE + 3·(-log iou) + 1e6·(~center1.5),
    dynamic-k = int(sum(top10 iou)), reg_weight=5. bboxes_iou/IOUloss use cxcywh.
- M3 training loop  — ✅ done. `m3_train.cpp`: forward→SimOTA→loss→backward→SGD,
  loss 24.1→3.2 on a synthetic batch. conv routes through the `bk::` seam so a
  `nvcc -DUSE_CUDA` build trains on GPU (same as v5/v8/v11).
- .pt write-back  — ✅ done. Unfused conv+BN path (`net_yolox_unfused.hpp`, uses `bn.hpp`)
  verified vs YOLOX (`m_unfused.cpp`, ~1e-4). `m_writeback.cpp` trains it (loss 23.9→3.5,
  lr 1e-4; higher lr → exp() overflow → NaN) and dumps weights; `ref/writeback_yolox.py`
  drops them into a YOLOX model (canonical `yolox_walk` order) → `torch.save({"model":...})`
  → re-loads with 0 missing/unexpected keys and runs. Train BN in eval (running stats frozen)
  so they round-trip unchanged.

## Gotchas to remember
- export must **force CPU** (`.cpu()`) — torch.hub loads to GPU on GPU hosts (learned on v5).
- YOLOX head output channel order is **[reg, obj, cls]** (not cls-first).
- Focus channel order [tl,bl,tr,br] — must match exactly.
- **dark5's CSPLayer has shortcut=False** (it sits after SPP); dark2/3/4 CSPLayers are True.
  Neck (PAFPN) CSPLayers are all False. Getting this wrong diverges only from c5 onward.
