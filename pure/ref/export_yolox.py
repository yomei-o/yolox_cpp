"""M1: dump yolox-tiny convs (BN-folded) in the exact order net_yolox.hpp consumes,
plus a fixed input and per-level reference head outputs. Forces CPU (torch.hub loads to
GPU on GPU hosts). Needs: pip install loguru tabulate thop."""
import os, sys, torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data_net"); os.makedirs(D, exist_ok=True)
IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 64
MODEL = sys.argv[2] if len(sys.argv) > 2 else "yolox_tiny"

m = torch.hub.load("Megvii-BaseDetection/YOLOX", MODEL, pretrained=True, trust_repo=True, verbose=False)
m = m.eval().cpu().float()
BD = len(m.backbone.backbone.dark2[1].m)   # base_depth (CSPLayer repeats)
DW = 1 if any(type(mm).__name__ == "DWConv" for mm in m.modules()) else 0

convs = []  # (w, b, k, stride, act)
def fuse(bc):
    conv, bn = bc.conv, bc.bn
    std = torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
    b = bn.bias - bn.weight * bn.running_mean / std
    return (w, b, conv.kernel_size[0], conv.stride[0], 1, conv.groups)
def emit(mod):
    if hasattr(mod, "dconv"):   # DWConv = depthwise(dconv) + pointwise(pconv)
        emit(mod.dconv); emit(mod.pconv)
    else:
        convs.append(fuse(mod))
def emit_plain(c): convs.append((c.weight, c.bias, c.kernel_size[0], c.stride[0], 0, c.groups))
def emit_csp(csp):
    emit(csp.conv1); emit(csp.conv2)
    for b in csp.m: emit(b.conv1); emit(b.conv2)
    emit(csp.conv3)
def emit_spp(spp): emit(spp.conv1); emit(spp.conv2)

bb, neck, head = m.backbone.backbone, m.backbone, m.head
emit(bb.stem.conv)                                   # Focus stem conv
emit(bb.dark2[0]); emit_csp(bb.dark2[1])
emit(bb.dark3[0]); emit_csp(bb.dark3[1])
emit(bb.dark4[0]); emit_csp(bb.dark4[1])
emit(bb.dark5[0]); emit_spp(bb.dark5[1]); emit_csp(bb.dark5[2])
emit(neck.lateral_conv0); emit_csp(neck.C3_p4); emit(neck.reduce_conv1); emit_csp(neck.C3_p3)
emit(neck.bu_conv2); emit_csp(neck.C3_n3); emit(neck.bu_conv1); emit_csp(neck.C3_n4)
for i in range(3):
    emit(head.stems[i])
    emit(head.cls_convs[i][0]); emit(head.cls_convs[i][1])
    emit(head.reg_convs[i][0]); emit(head.reg_convs[i][1])
    emit_plain(head.cls_preds[i]); emit_plain(head.reg_preds[i]); emit_plain(head.obj_preds[i])

def save(n, t): t.detach().contiguous().float().cpu().numpy().tofile(os.path.join(D, n))
lines = [str(len(convs))]
for i, (w, b, k, s, act, g) in enumerate(convs):
    save(f"w{i}.bin", w); save(f"b{i}.bin", b)
    lines.append(f"{w.shape[0]} {w.shape[1]} {k} {s} {act} {g}")
open(os.path.join(D, "manifest.txt"), "w").write("\n".join(lines) + "\n")

# reference: raw per-level head outputs [reg, obj, cls] (b, 85, h, w)
x = torch.randn(1, 3, IMG, IMG).cpu()
with torch.no_grad():
    # debug intermediates
    st = bb.stem(x); save("dbg_stem.bin", st)
    feats = bb(x)                                     # dict {dark3,dark4,dark5}
    for kk in ["dark3","dark4","dark5"]: save(f"dbg_{kk}.bin", feats[kk])
    d50 = bb.dark5[0](feats["dark4"]); save("dbg_d50.bin", d50)
    d51 = bb.dark5[1](d50); save("dbg_d51.bin", d51)   # after SPP
    print("dbg stem", tuple(st.shape), "c3", tuple(feats["dark3"].shape), "c5", tuple(feats["dark5"].shape))
    fpn = neck(x)                                     # (s8=96, s16=192, s32=384)
    shapes = []
    for i, f in enumerate(fpn):
        xx = head.stems[i](f)
        cf = head.cls_convs[i](xx); rf = head.reg_convs[i](xx)
        reg = head.reg_preds[i](rf); obj = head.obj_preds[i](rf); cls = head.cls_preds[i](cf)
        out = torch.cat([reg, obj, cls], 1)
        save(f"ref_L{i}.bin", out); shapes.append(tuple(out.shape))
save("x.bin", x)
open(os.path.join(D, "io.txt"), "w").write(f"{IMG} {BD} {DW}\n" + "\n".join(f"{s[1]} {s[2]} {s[3]}" for s in shapes) + "\n")
print(f"{MODEL}: convs={len(convs)} img={IMG} base_depth={BD} level_shapes={shapes}")
