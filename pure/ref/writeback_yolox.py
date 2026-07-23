"""A-1 write-back (yolox): drop the C++-trained weights (data_wb/) into a YOLOX model
in the SAME canonical order, save a standard checkpoint, and verify it re-loads."""
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); DW = os.path.join(HERE, "data_wb")
def r(n): return np.fromfile(os.path.join(DW, n), np.float32)

m = torch.hub.load("Megvii-BaseDetection/YOLOX", (sys.argv[1] if len(sys.argv)>1 else "yolox_tiny"), pretrained=True, trust_repo=True, verbose=False).eval().cpu().float()
head, neck, bb = m.head, m.backbone, m.backbone.backbone
def csp(c): return [c.conv1, c.conv2] + sum([[b.conv1, b.conv2] for b in c.m], []) + [c.conv3]
mods = [bb.stem.conv] + [bb.dark2[0]] + csp(bb.dark2[1]) + [bb.dark3[0]] + csp(bb.dark3[1]) \
     + [bb.dark4[0]] + csp(bb.dark4[1]) + [bb.dark5[0], bb.dark5[1].conv1, bb.dark5[1].conv2] + csp(bb.dark5[2]) \
     + [neck.lateral_conv0] + csp(neck.C3_p4) + [neck.reduce_conv1] + csp(neck.C3_p3) \
     + [neck.bu_conv2] + csp(neck.C3_n3) + [neck.bu_conv1] + csp(neck.C3_n4)
for i in range(3):
    mods += [head.stems[i], head.cls_convs[i][0], head.cls_convs[i][1],
             head.reg_convs[i][0], head.reg_convs[i][1],
             head.cls_preds[i], head.reg_preds[i], head.obj_preds[i]]
_flat = []
for _mm in mods:
    _flat += [_mm.dconv, _mm.pconv] if hasattr(_mm, "dconv") else [_mm]
mods = _flat

def load_(p, a):
    with torch.no_grad(): p.copy_(torch.from_numpy(a.astype(np.float32)).reshape(p.shape))
for i, mod in enumerate(mods):
    if hasattr(mod, "bn"):
        load_(mod.conv.weight, r(f"cw{i}.bin")); load_(mod.bn.weight, r(f"bg{i}.bin")); load_(mod.bn.bias, r(f"bb{i}.bin"))
        load_(mod.bn.running_mean, r(f"rm{i}.bin")); load_(mod.bn.running_var, r(f"rv{i}.bin"))
    else:
        load_(mod.weight, r(f"cw{i}.bin")); load_(mod.bias, r(f"cb{i}.bin"))

# serialization exact check
serr = 0.0
for i, mod in enumerate(mods):
    if hasattr(mod, "bn"):
        for a, p in [(r(f"cw{i}.bin"), mod.conv.weight), (r(f"bg{i}.bin"), mod.bn.weight), (r(f"bb{i}.bin"), mod.bn.bias),
                     (r(f"rm{i}.bin"), mod.bn.running_mean), (r(f"rv{i}.bin"), mod.bn.running_var)]:
            serr = max(serr, float(np.abs(a.reshape(p.shape) - p.detach().numpy()).max()))
    else:
        for a, p in [(r(f"cw{i}.bin"), mod.weight), (r(f"cb{i}.bin"), mod.bias)]:
            serr = max(serr, float(np.abs(a.reshape(p.shape) - p.detach().numpy()).max()))
print(f"serialization max|diff| = {serr:.3e}  {'OK' if serr < 1e-6 else 'FAIL'}")

# save a standard YOLOX checkpoint and verify it re-loads into a fresh model
out = os.path.join(HERE, "..", (sys.argv[1] if len(sys.argv)>1 else "yolox_tiny")+"_cpp.pth")
torch.save({"model": m.state_dict()}, out)
m2 = torch.hub.load("Megvii-BaseDetection/YOLOX", (sys.argv[1] if len(sys.argv)>1 else "yolox_tiny"), pretrained=False, trust_repo=True, verbose=False).eval().cpu().float()
missing, unexpected = m2.load_state_dict(torch.load(out, map_location="cpu")["model"], strict=False)
with torch.no_grad():
    y = m2(torch.randn(1, 3, 64, 64))
print(f"reloaded YOLOX .pt: missing={len(missing)} unexpected={len(unexpected)} forward_ok={tuple(y.shape) if hasattr(y,'shape') else type(y)}")
print("saved:", os.path.abspath(out))
