// Minimal reverse-mode autograd over dense float tensors. Standard library only.
// A tensor is a graph node: data + grad + a backward closure + parents (the tape).
#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <unordered_set>
#include "parallel.hpp"
#include "backend.hpp"   // bk::gemm_hosted device seam (CUDA / CPU)

struct Node;
using Tensor = std::shared_ptr<Node>;

struct Node {
  std::vector<int64_t> shape;
  std::vector<float> data;
  std::vector<float> grad;             // same size as data; accumulates
  std::vector<Tensor> parents;
  std::function<void()> backward_fn;   // adds this->grad into parents' grad
  bool requires_grad = false;

  int64_t numel() const {
    return std::accumulate(shape.begin(), shape.end(), (int64_t)1, std::multiplies<>());
  }
};

inline Tensor make_tensor(std::vector<int64_t> shape, bool requires_grad = false) {
  auto t = std::make_shared<Node>();
  t->shape = std::move(shape);
  t->data.assign(t->numel(), 0.f);
  t->grad.assign(t->numel(), 0.f);
  t->requires_grad = requires_grad;
  return t;
}
inline Tensor from_data(std::vector<int64_t> shape, std::vector<float> data, bool rg = false) {
  auto t = make_tensor(std::move(shape), rg);
  assert((int64_t)data.size() == t->numel());
  t->data = std::move(data);
  return t;
}

// ---- topological order + backward ----
inline void build_topo(const Tensor& v, std::unordered_set<Node*>& seen, std::vector<Tensor>& order) {
  if (!v || seen.count(v.get())) return;
  seen.insert(v.get());
  for (auto& p : v->parents) build_topo(p, seen, order);
  order.push_back(v);
}
inline void backward(const Tensor& root) {
  std::vector<Tensor> order; std::unordered_set<Node*> seen;
  build_topo(root, seen, order);
  for (auto& t : order) std::fill(t->grad.begin(), t->grad.end(), 0.f);
  assert(root->numel() == 1 && "backward expects a scalar root");
  root->grad[0] = 1.f;
  for (auto it = order.rbegin(); it != order.rend(); ++it)
    if ((*it)->backward_fn) (*it)->backward_fn();
}

// ---- elementwise binary (same shape) ----
inline Tensor add(const Tensor& a, const Tensor& b) {
  auto o = make_tensor(a->shape, true);
  for (int64_t i = 0; i < o->numel(); ++i) o->data[i] = a->data[i] + b->data[i];
  o->parents = {a, b};
  Node* op = o.get();
  o->backward_fn = [a, b, op] {
    for (int64_t i = 0; i < op->numel(); ++i) { a->grad[i] += op->grad[i]; b->grad[i] += op->grad[i]; }
  };
  return o;
}
inline Tensor mul(const Tensor& a, const Tensor& b) {
  auto o = make_tensor(a->shape, true);
  for (int64_t i = 0; i < o->numel(); ++i) o->data[i] = a->data[i] * b->data[i];
  o->parents = {a, b};
  Node* op = o.get();
  o->backward_fn = [a, b, op] {
    for (int64_t i = 0; i < op->numel(); ++i) {
      a->grad[i] += b->data[i] * op->grad[i];
      b->grad[i] += a->data[i] * op->grad[i];
    }
  };
  return o;
}

// ---- activations ----
inline Tensor silu(const Tensor& a) {
  auto o = make_tensor(a->shape, true);
  for (int64_t i = 0; i < o->numel(); ++i) {
    float s = 1.f / (1.f + std::exp(-a->data[i]));
    o->data[i] = a->data[i] * s;
  }
  o->parents = {a};
  Node* op = o.get();
  o->backward_fn = [a, op] {
    for (int64_t i = 0; i < op->numel(); ++i) {
      float s = 1.f / (1.f + std::exp(-a->data[i]));
      float d = s + a->data[i] * s * (1.f - s);     // d/dx [x*sigmoid(x)]
      a->grad[i] += d * op->grad[i];
    }
  };
  return o;
}
inline Tensor sigmoid(const Tensor& a) {
  auto o = make_tensor(a->shape, true);
  for (int64_t i = 0; i < o->numel(); ++i) o->data[i] = 1.f / (1.f + std::exp(-a->data[i]));
  o->parents = {a};
  Node* op = o.get();
  o->backward_fn = [a, op] {
    for (int64_t i = 0; i < op->numel(); ++i) {
      float y = op->data[i];
      a->grad[i] += y * (1.f - y) * op->grad[i];
    }
  };
  return o;
}

// ---- reductions to scalar ----
inline Tensor sum(const Tensor& a) {
  auto o = make_tensor({1}, true);
  double s = 0; for (int64_t i = 0; i < a->numel(); ++i) s += a->data[i];
  o->data[0] = (float)s;
  o->parents = {a};
  Node* op = o.get();
  o->backward_fn = [a, op] { for (int64_t i = 0; i < a->numel(); ++i) a->grad[i] += op->grad[0]; };
  return o;
}
inline Tensor mean(const Tensor& a) {
  auto o = make_tensor({1}, true);
  double s = 0; int64_t n = a->numel();
  for (int64_t i = 0; i < n; ++i) s += a->data[i];
  o->data[0] = (float)(s / n);
  o->parents = {a};
  Node* op = o.get();
  o->backward_fn = [a, op, n] { for (int64_t i = 0; i < n; ++i) a->grad[i] += op->grad[0] / n; };
  return o;
}

// ============================================================================
//  Spatial ops. Layout is NCHW throughout. OpenMP pragmas activate with -fopenmp
//  and are inert otherwise (CPU<->OpenMP = a compiler flag, no source change).
// ============================================================================

// conv2d: in (N,Cin,H,W), w (Cout,Cin,kh,kw), bias (Cout) or nullptr. groups=1.
// im2col + GEMM: the input patches are gathered into a (K, P) matrix (K=Cin*kh*kw,
// P=OH*OW) and the conv becomes W(Cout,K) @ col(K,P). The inner loops run over the
// contiguous P dimension so the compiler auto-vectorises them, and parallel_for spreads
// the outer dimension across cores — far faster than the naive index-per-output loop.
inline void im2col_(const float* I, int64_t Cin, int64_t H, int64_t W, int64_t kh, int64_t kw,
                    int64_t OH, int64_t OW, int64_t stride, int64_t pad, float* col) {
  int64_t P = OH * OW;
  for (int64_t ci = 0; ci < Cin; ++ci)
    for (int64_t r = 0; r < kh; ++r)
      for (int64_t s = 0; s < kw; ++s) {
        float* crow = col + (((ci * kh + r) * kw + s) * P);
        for (int64_t oh = 0; oh < OH; ++oh) {
          int64_t ih = oh * stride - pad + r;
          if (ih < 0 || ih >= H) { for (int64_t ow = 0; ow < OW; ++ow) crow[oh * OW + ow] = 0.f; continue; }
          const float* irow = I + (ci * H + ih) * W;
          for (int64_t ow = 0; ow < OW; ++ow) {
            int64_t iw = ow * stride - pad + s;
            crow[oh * OW + ow] = (iw < 0 || iw >= W) ? 0.f : irow[iw];
          }
        }
      }
}

inline Tensor conv2d(const Tensor& in, const Tensor& w, const Tensor& bias,
                     int64_t stride, int64_t pad) {
  int64_t N = in->shape[0], Cin = in->shape[1], H = in->shape[2], W = in->shape[3];
  int64_t Cout = w->shape[0], kh = w->shape[2], kw = w->shape[3];
  int64_t OH = (H + 2 * pad - kh) / stride + 1;
  int64_t OW = (W + 2 * pad - kw) / stride + 1;
  int64_t K = Cin * kh * kw, P = OH * OW;
  auto o = make_tensor({N, Cout, OH, OW}, true);
  const float* Wd = w->data.data();
  const float* B = bias ? bias->data.data() : nullptr;
  float* O = o->data.data();

  for (int64_t n = 0; n < N; ++n) {
    thread_local std::vector<float> colf;                 // reused across conv calls
    colf.resize(K * P);
    im2col_(in->data.data() + n * Cin * H * W, Cin, H, W, kh, kw, OH, OW, stride, pad, colf.data());
    float* On = O + n * Cout * P;
    bk::gemm_hosted(Wd, colf.data(), On, Cout, K, P);      // On(Cout,P) = W(Cout,K) @ col(K,P)
    if (B) parallel_for(Cout, [&](int64_t co) {            // add bias per row
      float b = B[co]; float* orow = On + co * P;
      for (int64_t p = 0; p < P; ++p) orow[p] += b;
    });
  }

  o->parents = bias ? std::vector<Tensor>{in, w, bias} : std::vector<Tensor>{in, w};
  Node* op = o.get();
  o->backward_fn = [in, w, bias, op, N, Cin, H, W, Cout, kh, kw, OH, OW, stride, pad, K, P] {
    const float* Wd = w->data.data();
    float* GK = w->grad.data(); float* GI = in->grad.data();
    if (bias) { float* GB = bias->grad.data();
      parallel_for(Cout, [&](int64_t co) { double a = 0; for (int64_t n = 0; n < N; ++n) { const float* g = op->grad.data() + (n * Cout + co) * P; for (int64_t p = 0; p < P; ++p) a += g[p]; } GB[co] += (float)a; });
    }
    for (int64_t n = 0; n < N; ++n) {
      thread_local std::vector<float> colb;
      colb.resize(K * P);
      im2col_(in->data.data() + n * Cin * H * W, Cin, H, W, kh, kw, OH, OW, stride, pad, colb.data());
      const float* colp = colb.data();
      const float* GOn = op->grad.data() + n * Cout * P;
      // dW(Cout,K) += dO(Cout,P) @ col(K,P)^T   (accumulate across batch n)
      bk::gemm_nt_hosted(GOn, colp, GK, Cout, K, P, 1.f);
      // dcol(K,P) = W(Cout,K)^T @ dO(Cout,P), then col2im -> dIn
      thread_local std::vector<float> dcolb;
      dcolb.resize(K * P);
      float* dcol = dcolb.data();
      bk::gemm_tn_hosted(Wd, GOn, dcol, K, P, Cout, 0.f);
      float* GIn = GI + n * Cin * H * W;
      parallel_for(Cin, [&](int64_t ci) {
        for (int64_t r = 0; r < kh; ++r)
          for (int64_t s = 0; s < kw; ++s) {
            const float* dcrow = dcol + (((ci * kh + r) * kw + s) * P);
            for (int64_t oh = 0; oh < OH; ++oh) {
              int64_t ih = oh * stride - pad + r; if (ih < 0 || ih >= H) continue;
              float* girow = GIn + (ci * H + ih) * W;
              for (int64_t ow = 0; ow < OW; ++ow) { int64_t iw = ow * stride - pad + s; if (iw < 0 || iw >= W) continue; girow[iw] += dcrow[oh * OW + ow]; }
            }
          }
      });
    }
  };
  return o;
}

// maxpool2d (stride s, pad p). Ceil not used; yolov8 SPPF uses k=5,s=1,p=2.
inline Tensor maxpool2d(const Tensor& in, int64_t k, int64_t stride, int64_t pad) {
  int64_t N = in->shape[0], C = in->shape[1], H = in->shape[2], W = in->shape[3];
  int64_t OH = (H + 2 * pad - k) / stride + 1, OW = (W + 2 * pad - k) / stride + 1;
  auto o = make_tensor({N, C, OH, OW}, true);
  auto argmax = std::make_shared<std::vector<int64_t>>(o->numel(), -1);
  const float* I = in->data.data(); float* O = o->data.data();
  parallel_for(N * C, [&](int64_t nc) {
    int64_t n = nc / C, c = nc % C;
    for (int64_t oh = 0; oh < OH; ++oh)
      for (int64_t ow = 0; ow < OW; ++ow) {
        float best = -1e30f; int64_t bidx = -1;
        for (int64_t r = 0; r < k; ++r) {
          int64_t ih = oh * stride - pad + r; if (ih < 0 || ih >= H) continue;
          for (int64_t s = 0; s < k; ++s) {
            int64_t iw = ow * stride - pad + s; if (iw < 0 || iw >= W) continue;
            int64_t idx = ((n * C + c) * H + ih) * W + iw;
            if (I[idx] > best) { best = I[idx]; bidx = idx; }
          }
        }
        int64_t oidx = ((n * C + c) * OH + oh) * OW + ow;
        O[oidx] = best; (*argmax)[oidx] = bidx;
      }
  });
  o->parents = {in};
  Node* op = o.get();
  o->backward_fn = [in, op, argmax] {
    for (int64_t i = 0; i < op->numel(); ++i)
      if ((*argmax)[i] >= 0) in->grad[(*argmax)[i]] += op->grad[i];
  };
  return o;
}

// nearest-neighbour upsample by integer factor f.
inline Tensor upsample_nearest(const Tensor& in, int64_t f) {
  int64_t N = in->shape[0], C = in->shape[1], H = in->shape[2], W = in->shape[3];
  int64_t OH = H * f, OW = W * f;
  auto o = make_tensor({N, C, OH, OW}, true);
  const float* I = in->data.data(); float* O = o->data.data();
  parallel_for(N * C, [&](int64_t nc) {
    int64_t n = nc / C, c = nc % C;
    for (int64_t oh = 0; oh < OH; ++oh)
      for (int64_t ow = 0; ow < OW; ++ow)
        O[((n * C + c) * OH + oh) * OW + ow] = I[((n * C + c) * H + oh / f) * W + ow / f];
  });
  o->parents = {in};
  Node* op = o.get();
  o->backward_fn = [in, op, N, C, H, W, OH, OW, f] {
    parallel_for(N * C, [&](int64_t nc) {
      int64_t n = nc / C, c = nc % C;
      for (int64_t oh = 0; oh < OH; ++oh)
        for (int64_t ow = 0; ow < OW; ++ow)
          in->grad[((n * C + c) * H + oh / f) * W + ow / f] +=
              op->grad[((n * C + c) * OH + oh) * OW + ow];
    });
  };
  return o;
}

// concat along channel dim (dim=1). All inputs share N,H,W.
inline Tensor concat_ch(const std::vector<Tensor>& xs) {
  int64_t N = xs[0]->shape[0], H = xs[0]->shape[2], W = xs[0]->shape[3];
  int64_t Ctot = 0; for (auto& x : xs) Ctot += x->shape[1];
  auto o = make_tensor({N, Ctot, H, W}, true);
  int64_t coff = 0;
  for (auto& x : xs) {
    int64_t C = x->shape[1];
    for (int64_t n = 0; n < N; ++n)
      for (int64_t c = 0; c < C; ++c)
        std::copy_n(&x->data[((n * C + c) * H) * W], H * W,
                    &o->data[((n * Ctot + coff + c) * H) * W]);
    coff += C;
  }
  o->parents = xs;
  Node* op = o.get();
  o->backward_fn = [xs, op, N, Ctot, H, W] {
    int64_t coff = 0;
    for (auto& x : xs) {
      int64_t C = x->shape[1];
      for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c)
          for (int64_t i = 0; i < H * W; ++i)
            x->grad[((n * C + c) * H) * W + i] +=
                op->grad[((n * Ctot + coff + c) * H) * W + i];
      coff += C;
    }
  };
  return o;
}

// slice channels [c0, c1) (dim=1).
inline Tensor slice_ch(const Tensor& x, int64_t c0, int64_t c1) {
  int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3], Cs = c1 - c0;
  auto o = make_tensor({N, Cs, H, W}, true);
  for (int64_t n = 0; n < N; ++n)
    for (int64_t c = 0; c < Cs; ++c)
      std::copy_n(&x->data[((n * C + c0 + c) * H) * W], H * W,
                  &o->data[((n * Cs + c) * H) * W]);
  o->parents = {x};
  Node* op = o.get();
  o->backward_fn = [x, op, N, C, H, W, c0, Cs] {
    for (int64_t n = 0; n < N; ++n)
      for (int64_t c = 0; c < Cs; ++c)
        for (int64_t i = 0; i < H * W; ++i)
          x->grad[((n * C + c0 + c) * H) * W + i] += op->grad[((n * Cs + c) * H) * W + i];
  };
  return o;
}

// Focus (space-to-depth): (N,C,H,W) -> (N,4C,H/2,W/2). Channel block order matches
// YOLOX: [top-left, bottom-left, top-right, bottom-right] = [(::2,::2),(1::2,::2),(::2,1::2),(1::2,1::2)].
inline Tensor focus(const Tensor& x) {
  int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
  int64_t OH = H / 2, OW = W / 2;
  auto o = make_tensor({N, 4 * C, OH, OW}, true);
  // per output block b in {0:tl,1:bl,2:tr,3:br} with row/col offsets (dr,dc)
  const int64_t dr[4] = {0, 1, 0, 1}, dc[4] = {0, 0, 1, 1};
  const float* I = x->data.data(); float* O = o->data.data();
  for (int64_t n = 0; n < N; ++n)
    for (int64_t b = 0; b < 4; ++b)
      for (int64_t c = 0; c < C; ++c)
        for (int64_t i = 0; i < OH; ++i)
          for (int64_t j = 0; j < OW; ++j)
            O[(((n * 4 * C + b * C + c) * OH) + i) * OW + j] =
                I[(((n * C + c) * H) + (2 * i + dr[b])) * W + (2 * j + dc[b])];
  o->parents = {x};
  Node* op = o.get();
  o->backward_fn = [x, op, N, C, H, W, OH, OW] {
    const int64_t dr[4] = {0, 1, 0, 1}, dc[4] = {0, 0, 1, 1};
    float* GI = x->grad.data(); const float* GO = op->grad.data();
    for (int64_t n = 0; n < N; ++n)
      for (int64_t b = 0; b < 4; ++b)
        for (int64_t c = 0; c < C; ++c)
          for (int64_t i = 0; i < OH; ++i)
            for (int64_t j = 0; j < OW; ++j)
              GI[(((n * C + c) * H) + (2 * i + dr[b])) * W + (2 * j + dc[b])] +=
                  GO[(((n * 4 * C + b * C + c) * OH) + i) * OW + j];
  };
  return o;
}
