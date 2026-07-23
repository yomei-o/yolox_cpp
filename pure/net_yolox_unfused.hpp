// yolox-tiny forward with conv + BatchNorm2d + SiLU kept SEPARATE (BN not folded),
// so weights can be trained and written back to a standard YOLOX .pt. Same topology
// as net_yolox.hpp; consumes layers in export_unfused_yolox.py order.
#pragma once
#include "autograd.hpp"
#include "bn.hpp"
#include "net_yolox.hpp"   // reuse rd()
#include <fstream>

struct LayerU {
  int kind;                 // 1 = conv+bn+silu (BaseConv), 0 = plain conv (bias, no act)
  int64_t stride; float eps; int64_t groups = 1;
  Tensor w, b, gamma, beta;
  std::vector<float> rm, rv;
};
struct ProviderU { std::vector<LayerU> layers; size_t i = 0; LayerU& next() { return layers[i++]; } };

inline ProviderU load_unfused(const std::string& D) {
  std::ifstream f(D + "manifest_unfused.txt");
  if (!f) { printf("run: python pure/ref/export_unfused_yolox.py\n"); std::exit(1); }
  std::string h; std::getline(f, h); int n = std::stoi(h); ProviderU p;
  for (int i = 0; i < n; ++i) {
    std::string line; std::getline(f, line); std::istringstream ss(line);
    int kind; int64_t Co, Ci, k, s, g = 1; float eps; ss >> kind >> Co >> Ci >> k >> s >> eps; ss >> g;
    LayerU L; L.kind = kind; L.stride = s; L.eps = eps; L.groups = g;
    L.w = from_data({Co, Ci, k, k}, rd(D + "cw" + std::to_string(i) + ".bin"), true);
    if (kind == 1) {
      L.gamma = from_data({Co}, rd(D + "bg" + std::to_string(i) + ".bin"), true);
      L.beta  = from_data({Co}, rd(D + "bb" + std::to_string(i) + ".bin"), true);
      L.rm = rd(D + "rm" + std::to_string(i) + ".bin");
      L.rv = rd(D + "rv" + std::to_string(i) + ".bin");
    } else {
      L.b = from_data({Co}, rd(D + "cb" + std::to_string(i) + ".bin"), true);
    }
    p.layers.push_back(std::move(L));
  }
  return p;
}

inline Tensor applyU(const Tensor& x, LayerU& L, bool tr) {
  int64_t pad = L.w->shape[2] / 2;
  if (L.kind == 1) {
    auto y = (L.groups > 1) ? dwconv2d(x, L.w, nullptr, L.stride, pad)
                            : conv2d(x, L.w, nullptr, L.stride, pad);
    y = batchnorm2d(y, L.gamma, L.beta, L.rm, L.rv, L.eps, tr, 0.03f);
    return silu(y);
  }
  return conv2d(x, L.w, L.b, L.stride, pad);
}
inline Tensor bcu(const Tensor& x, ProviderU& p, bool tr) { return applyU(x, p.next(), tr); }
inline Tensor conv3x3u(const Tensor& x, ProviderU& p, bool dw, bool tr) {
  if (dw) { auto h = bcu(x, p, tr); return bcu(h, p, tr); }   // dconv + pconv
  return bcu(x, p, tr);
}

inline Tensor csp_u(const Tensor& x, ProviderU& p, int64_t n, bool sc, bool tr, bool dw = false) {
  auto a = bcu(x, p, tr); auto b = bcu(x, p, tr);
  for (int64_t i = 0; i < n; ++i) { auto h = bcu(a, p, tr); h = conv3x3u(h, p, dw, tr); a = sc ? add(h, a) : h; }
  return bcu(concat_ch({a, b}), p, tr);
}
inline Tensor spp_u(const Tensor& x, ProviderU& p, bool tr) {
  auto x1 = bcu(x, p, tr);
  auto p5 = maxpool2d(x1, 5, 1, 2), p9 = maxpool2d(x1, 9, 1, 4), p13 = maxpool2d(x1, 13, 1, 6);
  return bcu(concat_ch({x1, p5, p9, p13}), p, tr);
}
inline Tensor head_level_u(const Tensor& f, ProviderU& p, bool tr, bool dw = false) {
  auto x = bcu(f, p, tr);                                 // stem (1x1)
  auto cf = conv3x3u(conv3x3u(x, p, dw, tr), p, dw, tr);  // cls tower (2×3x3)
  auto rf = conv3x3u(conv3x3u(x, p, dw, tr), p, dw, tr);  // reg tower
  auto cls = applyU(cf, p.next(), tr);   // plain
  auto reg = applyU(rf, p.next(), tr);
  auto obj = applyU(rf, p.next(), tr);
  return concat_ch({reg, obj, cls});
}
inline std::vector<Tensor> yolox_forward_unfused(const Tensor& in, ProviderU& p, bool tr, int64_t bd = 1, bool dw = false) {
  auto x = bcu(focus(in), p, tr);                          // stem (regular 3x3)
  x = conv3x3u(x, p, dw, tr); x = csp_u(x, p, bd, true, tr, dw);
  x = conv3x3u(x, p, dw, tr); auto c3 = csp_u(x, p, 3*bd, true, tr, dw);
  x = conv3x3u(c3, p, dw, tr); auto c4 = csp_u(x, p, 3*bd, true, tr, dw);
  x = conv3x3u(c4, p, dw, tr); x = spp_u(x, p, tr); auto c5 = csp_u(x, p, bd, false, tr, dw);
  auto fpn0 = bcu(c5, p, tr);                              // lateral_conv0 (1x1)
  auto u = upsample_nearest(fpn0, 2);
  auto p4 = csp_u(concat_ch({u, c4}), p, bd, false, tr, dw);
  auto red = bcu(p4, p, tr);                               // reduce_conv1 (1x1)
  auto u2 = upsample_nearest(red, 2);
  auto pan2 = csp_u(concat_ch({u2, c3}), p, bd, false, tr, dw);
  auto d0 = conv3x3u(pan2, p, dw, tr);                     // bu_conv2 (3x3 s2)
  auto pan1 = csp_u(concat_ch({d0, red}), p, bd, false, tr, dw);
  auto d1 = conv3x3u(pan1, p, dw, tr);                     // bu_conv1 (3x3 s2)
  auto pan0 = csp_u(concat_ch({d1, fpn0}), p, bd, false, tr, dw);
  std::vector<Tensor> out;
  for (auto& f : {pan2, pan1, pan0}) out.push_back(head_level_u(f, p, tr, dw));
  return out;
}
