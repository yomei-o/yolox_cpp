// Device-resident autograd core (Phase 1 of the GPU device-residency work).
// A tensor whose data/grad live in a thrust::device_vector, so intermediate results stay on
// the selected backend across ops — no per-op host<->device copies. The SAME source builds
// CPU (Thrust host device system, g++/MSVC) or GPU (nvcc/CUDA); elementwise ops go through
// thrust::transform, reductions through thrust::reduce, and matmul reuses the existing
// CPU/GPU-switchable bk::gemm (backend.hpp). Reverse-mode autograd via a small tape.
#pragma once
#include "backend.hpp"                 // bk::gemm / gemm_nt / gemm_tn (host or device)
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <cmath>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>

#if defined(__CUDACC__)
  #define DHD __host__ __device__
#else
  #define DHD
#endif

// ---- functors (portable: real device functions under nvcc, plain host otherwise) ----
struct SiLUf   { DHD float operator()(float x) const { return x / (1.f + expf(-x)); } };
struct dSiLUf  { DHD float operator()(float x) const { float s = 1.f/(1.f+expf(-x)); return s*(1.f + x*(1.f - s)); } };
struct AddCf   { float g; DHD float operator()(float v) const { return v + g; } };

struct DNode;
using DT = std::shared_ptr<DNode>;
struct DNode {
  std::vector<int64_t> shape;
  thrust::device_vector<float> data, grad;   // device-resident (host under CPP backend)
  std::vector<DT> parents;
  std::function<void()> backward_fn;
  int64_t numel() const { int64_t n = 1; for (auto d : shape) n *= d; return n; }
  float* dp()  { return thrust::raw_pointer_cast(data.data()); }
  float* gp()  { return thrust::raw_pointer_cast(grad.data()); }
};

inline DT dmake(std::vector<int64_t> shape) {
  auto n = std::make_shared<DNode>(); n->shape = std::move(shape);
  int64_t k = n->numel(); n->data.resize(k, 0.f); n->grad.resize(k, 0.f); return n;
}
inline DT dfrom(std::vector<int64_t> shape, const std::vector<float>& host) {
  DT t = dmake(std::move(shape)); thrust::copy(host.begin(), host.end(), t->data.begin()); return t;
}
inline std::vector<float> dto_host(const DT& t) {
  std::vector<float> h(t->numel()); thrust::copy(t->data.begin(), t->data.end(), h.begin()); return h;
}

// ---- elementwise (same-shape) ----
inline DT dadd(DT a, DT b) {
  DT y = dmake(a->shape);
  thrust::transform(a->data.begin(), a->data.end(), b->data.begin(), y->data.begin(), thrust::plus<float>());
  y->parents = {a, b};
  y->backward_fn = [a, b, yo = y.get()]() {                 // yo = raw ptr: no shared_ptr cycle
    thrust::transform(a->grad.begin(), a->grad.end(), yo->grad.begin(), a->grad.begin(), thrust::plus<float>());
    thrust::transform(b->grad.begin(), b->grad.end(), yo->grad.begin(), b->grad.begin(), thrust::plus<float>());
  };
  return y;
}
inline DT dmul(DT a, DT b) {                 // Hadamard
  DT y = dmake(a->shape);
  thrust::transform(a->data.begin(), a->data.end(), b->data.begin(), y->data.begin(), thrust::multiplies<float>());
  y->parents = {a, b};
  y->backward_fn = [a, b, yo = y.get()]() {
    thrust::device_vector<float> t(a->numel());
    thrust::transform(yo->grad.begin(), yo->grad.end(), b->data.begin(), t.begin(), thrust::multiplies<float>());
    thrust::transform(a->grad.begin(), a->grad.end(), t.begin(), a->grad.begin(), thrust::plus<float>());
    thrust::transform(yo->grad.begin(), yo->grad.end(), a->data.begin(), t.begin(), thrust::multiplies<float>());
    thrust::transform(b->grad.begin(), b->grad.end(), t.begin(), b->grad.begin(), thrust::plus<float>());
  };
  return y;
}
inline DT dsilu(DT x) {
  DT y = dmake(x->shape);
  thrust::transform(x->data.begin(), x->data.end(), y->data.begin(), SiLUf());
  y->parents = {x};
  y->backward_fn = [x, yo = y.get()]() {
    thrust::device_vector<float> d(x->numel());
    thrust::transform(x->data.begin(), x->data.end(), d.begin(), dSiLUf());       // silu'(x)
    thrust::transform(yo->grad.begin(), yo->grad.end(), d.begin(), d.begin(), thrust::multiplies<float>());
    thrust::transform(x->grad.begin(), x->grad.end(), d.begin(), x->grad.begin(), thrust::plus<float>());
  };
  return y;
}

// ---- matmul: Y[M,N] = A[M,K] * B[K,N] (reuses bk::gemm, CPU/GPU switchable) ----
inline DT dmatmul(DT A, DT B) {
  int64_t M = A->shape[0], K = A->shape[1], N = B->shape[1];
  DT Y = dmake({M, N});
  bk::gemm(A->dp(), B->dp(), Y->dp(), M, K, N);
  Y->parents = {A, B};
  Y->backward_fn = [A, B, M, K, N, Yo = Y.get()]() {
    bk::gemm_nt(Yo->gp(), B->dp(), A->gp(), M, K, N, 1.f);  // dA[M,K] += dY[M,N] * B[K,N]^T
    bk::gemm_tn(A->dp(), Yo->gp(), B->gp(), K, N, M, 1.f);  // dB[K,N] += A[M,K]^T * dY[M,N]
  };
  return Y;
}

// ---- conv2d (im2col + bk::gemm), device-resident ----
// im2col gather: col[K,P] from I[Cin,H,W]; K=Cin*kh*kw, P=OH*OW. One thread per col element.
inline void dim2col(const float* I, int64_t Cin, int64_t H, int64_t W, int64_t kh, int64_t kw,
                    int64_t OH, int64_t OW, int64_t stride, int64_t pad, float* col) {
  int64_t P = OH * OW, tot = Cin * kh * kw * P;
  bk::parallel_for(tot, [=] BK_HD (int64_t idx) {
    int64_t p = idx % P, row = idx / P;
    int64_t s = row % kw, t = row / kw; int64_t r = t % kh, ci = t / kh;
    int64_t oh = p / OW, ow = p % OW;
    int64_t ih = oh * stride - pad + r, iw = ow * stride - pad + s;
    col[idx] = (ih >= 0 && ih < H && iw >= 0 && iw < W) ? I[(ci * H + ih) * W + iw] : 0.f;
  });
}
// col2im scatter-add: dI[Cin,H,W] += from dcol[K,P]. Parallelised over Cin so each thread
// writes a disjoint channel — race-free on host (threads) AND device (kernel), no atomics.
inline void dcol2im(const float* dcol, int64_t Cin, int64_t H, int64_t W, int64_t kh, int64_t kw,
                    int64_t OH, int64_t OW, int64_t stride, int64_t pad, float* dI) {
  int64_t P = OH * OW;
  bk::parallel_for(Cin, [=] BK_HD (int64_t ci) {
    for (int64_t r = 0; r < kh; ++r) for (int64_t s = 0; s < kw; ++s) {
      const float* dcrow = dcol + (((ci * kh + r) * kw + s) * P);
      for (int64_t oh = 0; oh < OH; ++oh) {
        int64_t ih = oh * stride - pad + r; if (ih < 0 || ih >= H) continue;
        float* girow = dI + (ci * H + ih) * W;
        for (int64_t ow = 0; ow < OW; ++ow) { int64_t iw = ow * stride - pad + s; if (iw < 0 || iw >= W) continue; girow[iw] += dcrow[oh * OW + ow]; }
      }
    }
  });
}
// in (N,Cin,H,W), w (Cout,Cin,kh,kw), bias (Cout) or null. groups=1.
inline DT dconv2d(DT in, DT w, DT bias, int64_t stride, int64_t pad) {
  int64_t N = in->shape[0], Cin = in->shape[1], H = in->shape[2], Wd = in->shape[3];
  int64_t Cout = w->shape[0], kh = w->shape[2], kw = w->shape[3];
  int64_t OH = (H + 2*pad - kh)/stride + 1, OW = (Wd + 2*pad - kw)/stride + 1;
  int64_t K = Cin*kh*kw, P = OH*OW;
  DT o = dmake({N, Cout, OH, OW});
  { thrust::device_vector<float> col(K*P); float* colp = thrust::raw_pointer_cast(col.data());
    for (int64_t n = 0; n < N; ++n) {
      dim2col(in->dp() + n*Cin*H*Wd, Cin, H, Wd, kh, kw, OH, OW, stride, pad, colp);
      bk::gemm(w->dp(), colp, o->dp() + n*Cout*P, Cout, K, P);              // O(Cout,P)=W(Cout,K)*col(K,P)
      if (bias) { float* B = bias->dp(); float* On = o->dp() + n*Cout*P;
        bk::parallel_for(Cout*P, [=] BK_HD (int64_t idx) { On[idx] += B[idx/P]; }); }
    }
  }
  o->parents = bias ? std::vector<DT>{in,w,bias} : std::vector<DT>{in,w};
  o->backward_fn = [in,w,bias,N,Cin,H,Wd,Cout,kh,kw,OH,OW,stride,pad,K,P, oo=o.get()]() {
    if (bias) { float* GB = bias->gp(); float* GO = oo->gp();
      bk::parallel_for(Cout, [=] BK_HD (int64_t co) { float a = 0.f;
        for (int64_t n = 0; n < N; ++n) { const float* g = GO + (n*Cout+co)*P; for (int64_t p = 0; p < P; ++p) a += g[p]; }
        GB[co] += a; }); }
    thrust::device_vector<float> col(K*P), dcol(K*P);
    float* colp = thrust::raw_pointer_cast(col.data()), *dcolp = thrust::raw_pointer_cast(dcol.data());
    for (int64_t n = 0; n < N; ++n) {
      dim2col(in->dp() + n*Cin*H*Wd, Cin, H, Wd, kh, kw, OH, OW, stride, pad, colp);
      bk::gemm_nt(oo->gp() + n*Cout*P, colp, w->gp(), Cout, K, P, 1.f);     // dW += dO(Cout,P)*col(K,P)^T
      bk::gemm_tn(w->dp(), oo->gp() + n*Cout*P, dcolp, K, P, Cout, 0.f);    // dcol = W^T * dO
      dcol2im(dcolp, Cin, H, Wd, kh, kw, OH, OW, stride, pad, in->gp() + n*Cin*H*Wd);
    }
  };
  return o;
}

// ---- reduction: sum -> scalar ----
inline DT dsum(DT x) {
  DT y = dmake({1});
  y->data[0] = thrust::reduce(x->data.begin(), x->data.end(), 0.f);
  y->parents = {x};
  y->backward_fn = [x, yo = y.get()]() {
    float g = yo->grad[0];                                 // scalar broadcast to all inputs
    thrust::transform(x->grad.begin(), x->grad.end(), x->grad.begin(), AddCf{g});
  };
  return y;
}

// ---- structural ops (NCHW): concat / slice / upsample2x / maxpool2d ----
inline DT dconcat(std::vector<DT> xs) {                 // along channel dim
  int64_t N = xs[0]->shape[0], H = xs[0]->shape[2], W = xs[0]->shape[3], C = 0;
  for (auto& t : xs) C += t->shape[1];
  DT y = dmake({N, C, H, W});
  int64_t off = 0;
  for (auto& t : xs) { int64_t Ck = t->shape[1], blk = Ck*H*W;
    for (int64_t n = 0; n < N; ++n)
      thrust::copy(t->data.begin()+n*blk, t->data.begin()+(n+1)*blk, y->data.begin()+n*C*H*W + off*H*W);
    off += Ck; }
  y->parents = xs;
  y->backward_fn = [xs, N, C, H, W, yo = y.get()]() {
    int64_t off = 0;
    for (auto& t : xs) { int64_t Ck = t->shape[1], blk = Ck*H*W;
      for (int64_t n = 0; n < N; ++n) { auto ys = yo->grad.begin()+n*C*H*W + off*H*W;
        thrust::transform(ys, ys+blk, t->grad.begin()+n*blk, t->grad.begin()+n*blk, thrust::plus<float>()); }
      off += Ck; }
  };
  return y;
}
inline DT dslice(DT x, int64_t c0, int64_t c1) {        // channels [c0, c1)
  int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
  int64_t Cs = c1-c0, blk = Cs*H*W, xblk = C*H*W;
  DT y = dmake({N, Cs, H, W});
  for (int64_t n = 0; n < N; ++n)
    thrust::copy(x->data.begin()+n*xblk + c0*H*W, x->data.begin()+n*xblk + c1*H*W, y->data.begin()+n*blk);
  y->parents = {x};
  y->backward_fn = [x, N, C, H, W, c0, Cs, blk, xblk, yo = y.get()]() {
    for (int64_t n = 0; n < N; ++n) { auto ys = yo->grad.begin()+n*blk;
      thrust::transform(ys, ys+blk, x->grad.begin()+n*xblk + c0*H*W, x->grad.begin()+n*xblk + c0*H*W, thrust::plus<float>()); }
  };
  return y;
}
inline DT dupsample2x(DT x) {                            // nearest-neighbour 2x
  int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3], OH = 2*H, OW = 2*W;
  DT y = dmake({N, C, OH, OW});
  const float* X = x->dp(); float* Y = y->dp();
  bk::parallel_for(N*C*OH*OW, [=] BK_HD (int64_t idx) {
    int64_t ow = idx%OW, t = idx/OW, oh = t%OH, t2 = t/OH, c = t2%C, n = t2/C;
    Y[idx] = X[((n*C+c)*H + oh/2)*W + ow/2];
  });
  y->parents = {x};
  y->backward_fn = [x, N, C, H, W, OH, OW, yo = y.get()]() {
    const float* GY = yo->gp(); float* GX = x->gp();
    bk::parallel_for(N*C*H*W, [=] BK_HD (int64_t idx) {   // per input element: sum its 2x2 (race-free)
      int64_t iw = idx%W, t = idx/W, ih = t%H, t2 = t/H, c = t2%C, n = t2/C; float a = 0.f;
      for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) a += GY[((n*C+c)*OH + 2*ih+dy)*OW + 2*iw+dx];
      GX[idx] += a;
    });
  };
  return y;
}
inline DT dmaxpool2d(DT x, int64_t k, int64_t s, int64_t p) {
  int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
  int64_t OH = (H+2*p-k)/s+1, OW = (W+2*p-k)/s+1;
  DT y = dmake({N, C, OH, OW});
  auto argi = std::make_shared<thrust::device_vector<int64_t>>(N*C*OH*OW, (int64_t)-1);
  const float* X = x->dp(); float* Y = y->dp(); int64_t* AI = thrust::raw_pointer_cast(argi->data());
  bk::parallel_for(N*C*OH*OW, [=] BK_HD (int64_t idx) {
    int64_t ow = idx%OW, t = idx/OW, oh = t%OH, t2 = t/OH, c = t2%C, n = t2/C;
    float best = -3.4e38f; int64_t bi = -1;
    for (int64_t r = 0; r < k; ++r) for (int64_t q = 0; q < k; ++q) {
      int64_t ih = oh*s-p+r, iw = ow*s-p+q; if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
      int64_t ii = ((n*C+c)*H+ih)*W+iw; float v = X[ii]; if (v > best) { best = v; bi = ii; }
    }
    Y[idx] = best; AI[idx] = bi;
  });
  y->parents = {x};
  y->backward_fn = [x, argi, N, C, OH, OW, yo = y.get()]() {
    const float* GY = yo->gp(); float* GX = x->gp(); const int64_t* AI = thrust::raw_pointer_cast(argi->data());
    bk::parallel_for(N*C, [=] BK_HD (int64_t nc) {        // per (n,c) plane: scatter to argmax (race-free)
      int64_t base = nc*OH*OW;
      for (int64_t o = 0; o < OH*OW; ++o) { int64_t ii = AI[base+o]; if (ii >= 0) GX[ii] += GY[base+o]; }
    });
  };
  return y;
}

// ---- BatchNorm2d (training, biased batch var), device-resident ----
// Per-channel stats reduced over N*H*W (kernel parallel over C). Matches bn.hpp math.
// rm/rv (optional): running_mean/var device tensors, EMA-updated in training so the model
// can be saved for eval (biased var normalises; unbiased updates running_var, per bn.hpp).
inline DT dbn(DT x, DT gamma, DT beta, float eps = 1e-5f, DT rm = DT(), DT rv = DT(), float mom = 0.03f) {
  int64_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3], HW = H*W, M = N*HW;
  DT y = dmake({N, C, H, W});
  auto mean = std::make_shared<thrust::device_vector<float>>(C, 0.f);
  auto rstd = std::make_shared<thrust::device_vector<float>>(C, 0.f);
  const float* X = x->dp(); float* Y = y->dp();
  float* MEAN = thrust::raw_pointer_cast(mean->data()), *RSTD = thrust::raw_pointer_cast(rstd->data());
  const float* G = gamma->dp(); const float* B = beta->dp();
  bk::parallel_for(C, [=] BK_HD (int64_t c) {                     // batch mean/var per channel
    double s = 0, sq = 0;
    for (int64_t n = 0; n < N; ++n) { int64_t base = n*C*HW + c*HW; for (int64_t j = 0; j < HW; ++j) { float v = X[base+j]; s += v; sq += (double)v*v; } }
    double mu = s/M, var = sq/M - mu*mu; if (var < 0) var = 0;
    MEAN[c] = (float)mu; RSTD[c] = (float)(1.0 / sqrt(var + eps));
  });
  if (rm && rv) { float* RM = rm->dp(); float* RV = rv->dp(); float m_ = mom, e_ = eps; int64_t MM = M;
    bk::parallel_for(C, [=] BK_HD (int64_t c) {                   // EMA running stats
      float var = 1.f/(RSTD[c]*RSTD[c]) - e_; float vunb = MM > 1 ? var*(float)MM/(float)(MM-1) : var;
      RM[c] = (1.f-m_)*RM[c] + m_*MEAN[c]; RV[c] = (1.f-m_)*RV[c] + m_*vunb;
    }); }
  bk::parallel_for(N*C*HW, [=] BK_HD (int64_t idx) {              // normalize + affine
    int64_t c = (idx/HW) % C; Y[idx] = G[c]*(X[idx]-MEAN[c])*RSTD[c] + B[c];
  });
  y->parents = {x, gamma, beta};
  y->backward_fn = [x, gamma, beta, mean, rstd, N, C, HW, M, yo = y.get()]() {
    const float* X = x->dp(); const float* GY = yo->gp();
    float* GX = x->gp(); float* GG = gamma->gp(); float* GB = beta->gp(); const float* Gd = gamma->dp();
    const float* MEAN = thrust::raw_pointer_cast(mean->data()); const float* RSTD = thrust::raw_pointer_cast(rstd->data());
    thrust::device_vector<float> sdy(C, 0.f), sdyx(C, 0.f);
    float* SDY = thrust::raw_pointer_cast(sdy.data()); float* SDYX = thrust::raw_pointer_cast(sdyx.data());
    bk::parallel_for(C, [=] BK_HD (int64_t c) {                   // Σdy, Σ(dy*xhat) per channel; accumulate dgamma/dbeta
      double a = 0, b = 0; float mu = MEAN[c], rs = RSTD[c];
      for (int64_t n = 0; n < N; ++n) { int64_t base = n*C*HW + c*HW; for (int64_t j = 0; j < HW; ++j) { int64_t idx = base+j; float dy = GY[idx], xhat = (X[idx]-mu)*rs; a += dy; b += (double)dy*xhat; } }
      SDY[c] = (float)a; SDYX[c] = (float)b; GB[c] += (float)a; GG[c] += (float)b;
    });
    float invM = 1.f/(float)M;
    bk::parallel_for(N*C*HW, [=] BK_HD (int64_t idx) {            // dx
      int64_t c = (idx/HW) % C; float mu = MEAN[c], rs = RSTD[c], g = Gd[c], xhat = (X[idx]-mu)*rs;
      GX[idx] += g*rs*(GY[idx] - invM*SDY[c] - xhat*invM*SDYX[c]);
    });
  };
  return y;
}

// ---- device-resident Adam / AdamW ----
struct DAdam {
  std::vector<DT> params; float lr, b1, b2, eps, wd; int t = 0;
  std::vector<thrust::device_vector<float>> m, v;
  DAdam(std::vector<DT> p, float lr_ = 1e-3f, float b1_ = 0.9f, float b2_ = 0.999f, float eps_ = 1e-8f, float wd_ = 0.f)
    : params(std::move(p)), lr(lr_), b1(b1_), b2(b2_), eps(eps_), wd(wd_) {
    for (auto& pt : params) { m.emplace_back(pt->numel(), 0.f); v.emplace_back(pt->numel(), 0.f); }
  }
  void zero_grad() { for (auto& p : params) thrust::fill(p->grad.begin(), p->grad.end(), 0.f); }
  void step() {
    ++t; float bc1 = 1.f - std::pow(b1, t), bc2 = 1.f - std::pow(b2, t);
    float lr = this->lr, b1 = this->b1, b2 = this->b2, eps = this->eps, wd = this->wd;
    for (size_t i = 0; i < params.size(); ++i) {
      float* P = params[i]->dp(); const float* G = params[i]->gp();
      float* M = thrust::raw_pointer_cast(m[i].data()); float* V = thrust::raw_pointer_cast(v[i].data());
      int64_t n = params[i]->numel();
      bk::parallel_for(n, [=] BK_HD (int64_t j) {
        float g = G[j] + wd * P[j];
        float mm = b1*M[j] + (1.f-b1)*g, vv = b2*V[j] + (1.f-b2)*g*g;
        M[j] = mm; V[j] = vv;
        P[j] -= lr * (mm/bc1) / (sqrtf(vv/bc2) + eps);
      });
    }
  }
};

// ---- reverse-mode backward over the tape ----
inline void dbackward(DT root) {
  std::vector<DT> topo; std::unordered_set<DNode*> seen;
  std::function<void(const DT&)> build = [&](const DT& n) {
    if (!n || seen.count(n.get())) return; seen.insert(n.get());
    for (auto& p : n->parents) build(p); topo.push_back(n);
  };
  build(root);
  thrust::fill(root->grad.begin(), root->grad.end(), 1.f);  // seed dL/dL = 1
  for (auto it = topo.rbegin(); it != topo.rend(); ++it)
    if ((*it)->backward_fn) (*it)->backward_fn();
}

// Backward from multiple head nodes whose .grad the caller has ALREADY set (e.g. injected
// from an external loss). Does NOT re-seed — just runs the tape in reverse topo order.
inline void dbackward_from(const std::vector<DT>& heads) {
  std::vector<DT> topo; std::unordered_set<DNode*> seen;
  std::function<void(const DT&)> build = [&](const DT& n) {
    if (!n || seen.count(n.get())) return; seen.insert(n.get());
    for (auto& p : n->parents) build(p); topo.push_back(n);
  };
  for (auto& h : heads) build(h);
  for (auto it = topo.rbegin(); it != topo.rend(); ++it)
    if ((*it)->backward_fn) (*it)->backward_fn();
}
