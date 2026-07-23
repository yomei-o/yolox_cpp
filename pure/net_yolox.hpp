// Full yolox-tiny forward on the pure engine. Consumes convs in the exact order
// export_yolox.py emits them. CSPDarknet + PAFPN + decoupled head; Focus stem.
#pragma once
#include "autograd.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cstdlib>

struct ConvW { Tensor w, b; int64_t stride, act, groups = 1; };
struct Provider { std::vector<ConvW> convs; size_t i = 0; ConvW& next() { return convs[i++]; } };

inline std::vector<float> rd(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { printf("cannot open %s\n", p.c_str()); std::exit(1); }
  auto n = f.tellg(); f.seekg(0);
  std::vector<float> v(n / sizeof(float)); f.read((char*)v.data(), n); return v;
}
inline Provider load_net(const std::string& D) {
  std::ifstream f(D + "manifest.txt"); std::string h; std::getline(f, h); int n = std::stoi(h);
  Provider p;
  for (int i = 0; i < n; ++i) {
    std::string line; std::getline(f, line); std::istringstream ss(line);
    int64_t Co, Ci, k, s, act, g = 1; ss >> Co >> Ci >> k >> s >> act; ss >> g;   // groups optional (default 1)
    ConvW c; c.stride = s; c.act = act; c.groups = g;
    c.w = from_data({Co, Ci, k, k}, rd(D + "w" + std::to_string(i) + ".bin"));
    c.b = from_data({Co}, rd(D + "b" + std::to_string(i) + ".bin"));
    p.convs.push_back(c);
  }
  return p;
}
inline Tensor conv_apply(const Tensor& x, ConvW& c) {
  int64_t pad = c.w->shape[2] / 2;
  auto y = (c.groups > 1) ? dwconv2d(x, c.w, c.b, c.stride, pad) : conv2d(x, c.w, c.b, c.stride, pad);
  return c.act ? silu(y) : y;
}
inline Tensor bc(const Tensor& x, Provider& p) { return conv_apply(x, p.next()); }   // BaseConv (1x1/3x3)
// 3x3 conv position: depthwise (DWConv = dconv+pconv, 2 convs) if dw, else one conv.
inline Tensor conv3x3(const Tensor& x, Provider& p, bool dw) {
  if (dw) { auto h = conv_apply(x, p.next()); return conv_apply(h, p.next()); }  // dconv(groups=C) + pconv(1x1)
  return conv_apply(x, p.next());
}

// CSPLayer (C3): conv1, conv2, m=n×Bottleneck on conv1-path, conv3(cat).
inline Tensor csp(const Tensor& x, Provider& p, int64_t n, bool shortcut, bool dw = false) {
  auto a = bc(x, p);          // conv1 -> c_ (1x1)
  auto b = bc(x, p);          // conv2 -> c_ (1x1)
  for (int64_t i = 0; i < n; ++i) {
    auto h = bc(a, p);        // bott.conv1 (1x1)
    h = conv3x3(h, p, dw);    // bott.conv2 (3x3, DWConv in nano)
    a = shortcut ? add(h, a) : h;
  }
  return bc(concat_ch({a, b}), p);   // conv3(cat[a,b]) -> c2 (1x1)
}
// SPPBottleneck: conv1, maxpool{5,9,13}, conv2(cat).
inline Tensor spp(const Tensor& x, Provider& p) {
  auto x1 = bc(x, p);
  auto p5 = maxpool2d(x1, 5, 1, 2), p9 = maxpool2d(x1, 9, 1, 4), p13 = maxpool2d(x1, 13, 1, 6);
  return bc(concat_ch({x1, p5, p9, p13}), p);
}
// decoupled head for one level: -> (b, 4+1+nc, h, w) as [reg, obj, cls].
inline Tensor head_level(const Tensor& f, Provider& p, bool dw = false) {
  auto x = bc(f, p);                              // stem (1x1)
  auto cf = conv3x3(conv3x3(x, p, dw), p, dw);    // cls tower (2×3x3, DWConv in nano)
  auto rf = conv3x3(conv3x3(x, p, dw), p, dw);    // reg tower (2×3x3)
  auto cls = conv_apply(cf, p.next());            // cls_pred (plain)
  auto reg = conv_apply(rf, p.next());            // reg_pred (plain)
  auto obj = conv_apply(rf, p.next());            // obj_pred (plain)
  return concat_ch({reg, obj, cls});
}

// Full yolox (non-depthwise: t/s/m/l/x). base_depth bd = CSPLayer repeats (dark2/5=bd,
// dark3/4=3*bd, neck=bd). Widths are data-driven from the loaded conv shapes.
// Full yolox (t/n/s/m/l/x). bd = CSPLayer repeats; dw = depthwise (nano): 3x3 convs
// (downsample, bu, bottleneck.conv2, head towers) become DWConv (dconv+pconv).
inline std::vector<Tensor> yolox_forward(const Tensor& in, Provider& p, int64_t bd = 1, bool dw = false) {
  // backbone (CSPDarknet)
  auto x = bc(focus(in), p);                          // stem: Focus -> BaseConv (regular 3x3)
  x = conv3x3(x, p, dw); x = csp(x, p, bd, true, dw);          // dark2
  x = conv3x3(x, p, dw); auto c3 = csp(x, p, 3*bd, true, dw);  // dark3 -> C3 (s8)
  x = conv3x3(c3, p, dw); auto c4 = csp(x, p, 3*bd, true, dw); // dark4 -> C4 (s16)
  x = conv3x3(c4, p, dw); x = spp(x, p); auto c5 = csp(x, p, bd, false, dw);  // dark5 (shortcut=False)
  // neck (PAFPN) — CSPLayers shortcut=False
  auto fpn0 = bc(c5, p);                              // lateral_conv0 (1x1)
  auto u = upsample_nearest(fpn0, 2);
  auto p4 = csp(concat_ch({u, c4}), p, bd, false, dw); // C3_p4
  auto red = bc(p4, p);                               // reduce_conv1 (1x1)
  auto u2 = upsample_nearest(red, 2);
  auto pan2 = csp(concat_ch({u2, c3}), p, bd, false, dw);   // C3_p3 (s8)
  auto d0 = conv3x3(pan2, p, dw);                     // bu_conv2 (3x3 s2)
  auto pan1 = csp(concat_ch({d0, red}), p, bd, false, dw);  // C3_n3 (s16)
  auto d1 = conv3x3(pan1, p, dw);                     // bu_conv1 (3x3 s2)
  auto pan0 = csp(concat_ch({d1, fpn0}), p, bd, false, dw); // C3_n4 (s32)
  // decoupled head
  std::vector<Tensor> out;
  for (auto& f : {pan2, pan1, pan0}) out.push_back(head_level(f, p, dw));
  return out;
}
