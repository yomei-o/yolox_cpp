"""Dump the architecture description (manifest_unfused.txt + names.txt) for EVERY YOLOX
size (nano/tiny/s/m/l/x), built from the exp system in CODE (random init, NO checkpoint
download and NO Megvii .pth). These tiny text files let the pure-C++ make_init_pt generate
initial weights for any size with zero Python. Writes pure/ref/arch/<model>/{manifest,names}.
The YOLOX repo (cached under torch hub) provides the exp/get_model definitions.
Usage: python export_arch_all.py [/path/to/YOLOX/repo]"""
import os, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
# locate the cached YOLOX repo (or take it as argv[1])
YHUB = sys.argv[1] if len(sys.argv) > 1 else None
if not YHUB:
    c = glob.glob(os.path.expanduser("~/.cache/torch/hub/*YOLOX*"))
    if not c: sys.exit("YOLOX repo not found under ~/.cache/torch/hub — pass its path as argv[1]")
    YHUB = c[0]
sys.path.insert(0, YHUB)
from yolox.exp import get_exp

SIZES = ["yolox_nano", "yolox_tiny", "yolox_s", "yolox_m", "yolox_l", "yolox_x"]

def yolox_walk(m):
    """Same canonical order as export_unfused_yolox.py, but parametrised by the model."""
    bb, neck, head = m.backbone.backbone, m.backbone, m.head
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

for name in SIZES:
    exp = get_exp(exp_file=os.path.join(YHUB, "exps", "default", name + ".py"))
    m = exp.get_model().eval().float()
    mods = yolox_walk(m)
    flat = []                                             # DWConv -> (dconv, pconv)
    for mm in mods: flat += [mm.dconv, mm.pconv] if hasattr(mm, "dconv") else [mm]
    mods = flat
    qn = {id(mm): nm for nm, mm in m.named_modules()}
    lines = [str(len(mods))]; names = []
    for mod in mods:
        p = qn[id(mod)]
        if hasattr(mod, "bn"):                            # BaseConv: conv(no bias)+BN+SiLU
            c, b = mod.conv, mod.bn
            lines.append(f"1 {c.weight.shape[0]} {c.weight.shape[1]} {c.kernel_size[0]} {c.stride[0]} {b.eps} {c.groups}")
            names += [f"{p}.conv.weight", f"{p}.bn.weight", f"{p}.bn.bias", f"{p}.bn.running_mean", f"{p}.bn.running_var"]
        else:                                             # plain Conv2d (bias)
            lines.append(f"0 {mod.weight.shape[0]} {mod.weight.shape[1]} {mod.kernel_size[0]} {mod.stride[0]} 0 {mod.groups}")
            names += [f"{p}.weight", f"{p}.bias"]
    D = os.path.join(HERE, "arch", name); os.makedirs(D, exist_ok=True)
    open(os.path.join(D, "manifest_unfused.txt"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(D, "names.txt"), "w").write("\n".join(names) + "\n")
    print(f"{name}: {len(mods)} layers, {len(names)} tensors -> arch/{name}/")
