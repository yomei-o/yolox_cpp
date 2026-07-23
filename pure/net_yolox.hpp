// Full yolox-tiny forward on the pure engine. Consumes convs in the exact order
// export_yolox.py emits them. CSPDarknet + PAFPN + decoupled head; Focus stem.
#pragma once
#include "autograd.hpp"
#include <fstream>
#include <string>
#include <cstdio>
#include <cstdlib>

struct ConvW { Tensor w, b; int64_t stride, act; };
struct Provider { std::vector<ConvW> convs; size_t i = 0; ConvW& next() { return convs[i++]; } };

inline std::vector<float> rd(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { printf("cannot open %s\n", p.c_str()); std::exit(1); }
  auto n = f.tellg(); f.seekg(0);
  std::vector<float> v(n / sizeof(float)); f.read((char*)v.data(), n); return v;
}
inline Provider load_net(const std::string& D) {
  std::ifstream f(D + "manifest.txt"); int n; f >> n; Provider p;
  for (int i = 0; i < n; ++i) {
    int64_t Co, Ci, k, s, act; f >> Co >> Ci >> k >> s >> act;
    ConvW c; c.stride = s; c.act = act;
    c.w = from_data({Co, Ci, k, k}, rd(D + "w" + std::to_string(i) + ".bin"));
    c.b = from_data({Co}, rd(D + "b" + std::to_string(i) + ".bin"));
    p.convs.push_back(c);
  }
  return p;
}
inline Tensor conv_apply(const Tensor& x, ConvW& c) {
  auto y = conv2d(x, c.w, c.b, c.stride, c.w->shape[2] / 2);
  return c.act ? silu(y) : y;
}
inline Tensor bc(const Tensor& x, Provider& p) { return conv_apply(x, p.next()); }   // BaseConv

// CSPLayer (C3): conv1, conv2, m=n×Bottleneck on conv1-path, conv3(cat).
inline Tensor csp(const Tensor& x, Provider& p, int64_t n, bool shortcut) {
  auto a = bc(x, p);          // conv1 -> c_
  auto b = bc(x, p);          // conv2 -> c_
  for (int64_t i = 0; i < n; ++i) {
    auto h = bc(a, p);        // bott.conv1 (1x1)
    h = bc(h, p);             // bott.conv2 (3x3)
    a = shortcut ? add(h, a) : h;
  }
  return bc(concat_ch({a, b}), p);   // conv3(cat[a,b]) -> c2
}
// SPPBottleneck: conv1, maxpool{5,9,13}, conv2(cat).
inline Tensor spp(const Tensor& x, Provider& p) {
  auto x1 = bc(x, p);
  auto p5 = maxpool2d(x1, 5, 1, 2), p9 = maxpool2d(x1, 9, 1, 4), p13 = maxpool2d(x1, 13, 1, 6);
  return bc(concat_ch({x1, p5, p9, p13}), p);
}
// decoupled head for one level: -> (b, 4+1+nc, h, w) as [reg, obj, cls].
inline Tensor head_level(const Tensor& f, Provider& p) {
  auto x = bc(f, p);                              // stem
  auto cf = bc(bc(x, p), p);                      // cls tower (2)
  auto rf = bc(bc(x, p), p);                      // reg tower (2)
  auto cls = conv_apply(cf, p.next());            // cls_pred (plain)
  auto reg = conv_apply(rf, p.next());            // reg_pred (plain)
  auto obj = conv_apply(rf, p.next());            // obj_pred (plain)
  return concat_ch({reg, obj, cls});
}

// Full yolox-tiny. Returns 3 per-level raw outputs (b,85,h,w) for strides [8,16,32].
inline std::vector<Tensor> yolox_forward(const Tensor& in, Provider& p) {
  // backbone (CSPDarknet)
  auto x = bc(focus(in), p);                      // stem: Focus -> BaseConv -> 24
  x = bc(x, p); x = csp(x, p, 1, true);           // dark2
  x = bc(x, p); auto c3 = csp(x, p, 3, true);     // dark3 -> C3 (s8, 96)
  x = bc(c3, p); auto c4 = csp(x, p, 3, true);    // dark4 -> C4 (s16, 192)
  x = bc(c4, p); x = spp(x, p); auto c5 = csp(x, p, 1, false);  // dark5 -> C5 (s32, 384) (CSP shortcut=False)
  // neck (PAFPN)
  auto fpn0 = bc(c5, p);                          // lateral_conv0 384->192
  auto u = upsample_nearest(fpn0, 2);
  auto p4 = csp(concat_ch({u, c4}), p, 1, false); // C3_p4 -> 192
  auto red = bc(p4, p);                           // reduce_conv1 192->96
  auto u2 = upsample_nearest(red, 2);
  auto pan2 = csp(concat_ch({u2, c3}), p, 1, false);   // C3_p3 -> 96  (s8)
  auto d0 = bc(pan2, p);                          // bu_conv2 96->96 s2
  auto pan1 = csp(concat_ch({d0, red}), p, 1, false);  // C3_n3 -> 192 (s16)
  auto d1 = bc(pan1, p);                          // bu_conv1 192->192 s2
  auto pan0 = csp(concat_ch({d1, fpn0}), p, 1, false); // C3_n4 -> 384 (s32)
  // decoupled head
  std::vector<Tensor> out;
  for (auto& f : {pan2, pan1, pan0}) out.push_back(head_level(f, p));
  return out;
}
