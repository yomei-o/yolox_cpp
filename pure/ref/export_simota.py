"""M2a: dump SimOTA inputs + YOLOX's get_assignments result, to verify the C++ SimOTA.
Also dumps raw per-level head outputs + labels for the M2b loss step."""
import os, sys, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data_loss"); os.makedirs(D, exist_ok=True)
IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 64
torch.manual_seed(3)

m = torch.hub.load("Megvii-BaseDetection/YOLOX", "yolox_tiny", pretrained=True, trust_repo=True, verbose=False).eval().cpu().float()
head, neck = m.head, m.backbone
NC = head.num_classes
def save(n, t): t.detach().contiguous().float().cpu().numpy().tofile(os.path.join(D, n))

x = torch.randn(1, 3, IMG, IMG)
# raw per-level head outputs (leaves for M2b)
raw = []
with torch.no_grad():
    fpn = neck(x)
    for i, f in enumerate(fpn):
        xx = head.stems[i](f); cf = head.cls_convs[i](xx); rf = head.reg_convs[i](xx)
        raw.append(torch.cat([head.reg_preds[i](rf), head.obj_preds[i](rf), head.cls_preds[i](cf)], 1))
for i, r in enumerate(raw): save(f"raw_L{i}.bin", r)

# decode + build grids exactly like YOLOX forward
outputs, x_shifts, y_shifts, exp_strides = [], [], [], []
for i, (r, stride) in enumerate(zip(raw, head.strides)):
    out, grid = head.get_output_and_grid(r.clone(), i, stride, x.type())   # decodes reg
    x_shifts.append(grid[:, :, 0]); y_shifts.append(grid[:, :, 1])
    exp_strides.append(torch.full((1, grid.shape[1]), float(stride)))
    outputs.append(out)
outputs = torch.cat(outputs, 1)                       # (1, A, 85) decoded
xs = torch.cat(x_shifts, 1); ys = torch.cat(y_shifts, 1); es = torch.cat(exp_strides, 1)
bbox_preds = outputs[:, :, :4]; obj_preds = outputs[:, :, 4:5]; cls_preds = outputs[:, :, 5:]
A = outputs.shape[1]

# labels: (1, G, 5) = [cls, cx, cy, w, h] absolute px
labels = torch.tensor([[[5., 20, 22, 24, 20], [17., 44, 40, 28, 30], [3., 12, 46, 16, 24]]])
G = labels.shape[1]
gt_boxes = labels[0, :, 1:5]; gt_cls = labels[0, :, 0]

with torch.no_grad():
    gt_matched_classes, fg_mask, pred_ious, matched_gt_inds, num_fg = head.get_assignments(
        0, G, gt_boxes, gt_cls, bbox_preds[0], es, xs, ys, cls_preds, obj_preds)

save("bbox_preds.bin", bbox_preds[0]); save("cls_preds.bin", cls_preds[0]); save("obj_preds.bin", obj_preds[0])
save("xs.bin", xs[0]); save("ys.bin", ys[0]); save("strides.bin", es[0])
save("gt_boxes.bin", gt_boxes); save("gt_cls.bin", gt_cls)
save("fg_mask.bin", fg_mask.float()); save("matched_gt.bin", matched_gt_inds.float())
save("gt_matched_cls.bin", gt_matched_classes.float()); save("pred_ious.bin", pred_ious)
save("x.bin", x)
open(os.path.join(D, "meta.txt"), "w").write(f"{IMG} {A} {NC} {G} {int(num_fg)}\n")
open(os.path.join(D, "levels.txt"), "w").write("\n".join(f"{int(IMG//s)} {int(IMG//s)} {int(s)}" for s in head.strides) + "\n")
print(f"A={A} G={G} num_fg={int(num_fg)} fg_sum={int(fg_mask.sum())}")

# ---- M2b: loss (forward) + grads wrt raw head outputs, detached SimOTA targets ----
import torch.nn.functional as F
raw_leaf = [r.detach().clone().requires_grad_(True) for r in raw]
outs = []
for i, (r, stride) in enumerate(zip(raw_leaf, head.strides)):
    b, ch, h, w = r.shape
    o = r.view(1, ch, h*w).permute(0, 2, 1)             # (1,hw,85)
    yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    gx = xv.reshape(-1).float(); gy = yv.reshape(-1).float()
    cx = (o[..., 0] + gx) * stride; cy = (o[..., 1] + gy) * stride
    ww = torch.exp(o[..., 2]) * stride; hh = torch.exp(o[..., 3]) * stride
    outs.append(torch.cat([torch.stack([cx, cy, ww, hh], -1), o[..., 4:]], -1))
outputs2 = torch.cat(outs, 1)                            # (1,A,85) differentiable
bp = outputs2[0, :, :4]; ob = outputs2[0, :, 4:5]; cl = outputs2[0, :, 5:]
fgb = fg_mask.bool()
reg_target = gt_boxes[matched_gt_inds]                  # const
cls_target = (F.one_hot(gt_matched_classes.long(), NC).float() * pred_ious.unsqueeze(-1)).detach()
obj_target = fg_mask.float().unsqueeze(-1)
def iou_loss_yolox(p, t):
    tl = torch.max(p[:, :2]-p[:, 2:]/2, t[:, :2]-t[:, 2:]/2)
    br = torch.min(p[:, :2]+p[:, 2:]/2, t[:, :2]+t[:, 2:]/2)
    en = (tl < br).type(tl.type()).prod(1)
    area_i = torch.prod(br-tl, 1)*en
    area_u = torch.prod(p[:, 2:], 1)+torch.prod(t[:, 2:], 1)-area_i
    return 1 - (area_i/(area_u+1e-16))**2
nf = max(float(num_fg), 1)
loss_iou = iou_loss_yolox(bp[fgb], reg_target).sum()/nf
loss_obj = F.binary_cross_entropy_with_logits(ob, obj_target, reduction="none").sum()/nf
loss_cls = F.binary_cross_entropy_with_logits(cl[fgb], cls_target, reduction="none").sum()/nf
total = 5.0*loss_iou + loss_obj + loss_cls
total.backward()
for i, r in enumerate(raw_leaf): save(f"grad_L{i}.bin", r.grad)
import numpy as np
np.array([loss_iou.item(), loss_obj.item(), loss_cls.item(), total.item()], np.float32).tofile(os.path.join(D, "loss.bin"))
print(f"loss iou={loss_iou.item():.6f} obj={loss_obj.item():.6f} cls={loss_cls.item():.6f} total={total.item():.6f}")
print("matched_gt:", matched_gt_inds.tolist(), " classes:", gt_matched_classes.int().tolist())
print("pred_ious:", [round(v,4) for v in pred_ious.tolist()])
