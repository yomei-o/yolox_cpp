"""A-1 (yolox): dump RAW (unfused) conv + BN params in canonical order, so the pure
C++ can train them and write them back into a standard YOLOX .pt. Also a reference
forward (BN eval) to verify the unfused net. CPU-forced."""
import os, sys, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data_unf"); os.makedirs(D, exist_ok=True)
IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 64

m = torch.hub.load("Megvii-BaseDetection/YOLOX", "yolox_tiny", pretrained=True, trust_repo=True, verbose=False).eval().cpu().float()
head, neck, bb = m.head, m.backbone, m.backbone.backbone
NC = head.num_classes

def yolox_walk(m):
    def csp(c): return [c.conv1, c.conv2] + sum([[b.conv1, b.conv2] for b in c.m], []) + [c.conv3]
    mods = [bb.stem.conv]
    mods += [bb.dark2[0]] + csp(bb.dark2[1])
    mods += [bb.dark3[0]] + csp(bb.dark3[1])
    mods += [bb.dark4[0]] + csp(bb.dark4[1])
    mods += [bb.dark5[0], bb.dark5[1].conv1, bb.dark5[1].conv2] + csp(bb.dark5[2])
    mods += [neck.lateral_conv0] + csp(neck.C3_p4) + [neck.reduce_conv1] + csp(neck.C3_p3)
    mods += [neck.bu_conv2] + csp(neck.C3_n3) + [neck.bu_conv1] + csp(neck.C3_n4)
    for i in range(3):
        mods += [head.stems[i], head.cls_convs[i][0], head.cls_convs[i][1],
                 head.reg_convs[i][0], head.reg_convs[i][1],
                 head.cls_preds[i], head.reg_preds[i], head.obj_preds[i]]
    return mods

mods = yolox_walk(m)
def save(n, t): t.detach().contiguous().float().cpu().numpy().tofile(os.path.join(D, n))
lines = [str(len(mods))]
for i, mod in enumerate(mods):
    if hasattr(mod, "bn"):        # BaseConv: conv(no bias)+BN+SiLU
        c, b = mod.conv, mod.bn
        save(f"cw{i}.bin", c.weight); save(f"bg{i}.bin", b.weight); save(f"bb{i}.bin", b.bias)
        save(f"rm{i}.bin", b.running_mean); save(f"rv{i}.bin", b.running_var)
        lines.append(f"1 {c.weight.shape[0]} {c.weight.shape[1]} {c.kernel_size[0]} {c.stride[0]} {b.eps}")
    else:                         # plain Conv2d (bias)
        save(f"cw{i}.bin", mod.weight); save(f"cb{i}.bin", mod.bias)
        lines.append(f"0 {mod.weight.shape[0]} {mod.weight.shape[1]} {mod.kernel_size[0]} {mod.stride[0]} 0")
open(os.path.join(D, "manifest_unfused.txt"), "w").write("\n".join(lines) + "\n")

# reference forward (BN eval == unfused C++ with training=false)
x = torch.randn(1, 3, IMG, IMG).cpu()
with torch.no_grad():
    fpn = neck(x)
    for i, f in enumerate(fpn):
        xx = head.stems[i](f); cf = head.cls_convs[i](xx); rf = head.reg_convs[i](xx)
        out = torch.cat([head.reg_preds[i](rf), head.obj_preds[i](rf), head.cls_preds[i](cf)], 1)
        save(f"ref_L{i}.bin", out)
save("x.bin", x)
open(os.path.join(D, "io.txt"), "w").write(f"{IMG}\n" + "\n".join(f"{int(IMG//s)} {int(IMG//s)} {int(s)}" for s in head.strides) + "\n")
print(f"unfused: {len(mods)} layers, img {IMG}")
