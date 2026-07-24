// Device-resident YOLOX net (any size) + trainable provider + checkpoint save.
// Mirrors net_yolox_unfused.hpp: Focus stem, BaseConv (conv+bn+silu), CSP, SPP(5/9/13),
// PAFPN, decoupled head (cls/reg tower 2×3x3 -> plain cls/reg/obj -> concat[reg,obj,cls]).
// Depthwise (nano) via grouped dconv2d (groups from manifest). One source CPU/GPU.
#pragma once
#include "net_yolox_unfused.hpp"   // ProviderU (LayerU has groups), load_unfused_pt, focus ref
#include "dtensor.hpp"
#include <string>
#include <fstream>

struct DTLX { int kind; int64_t stride; float eps; int64_t groups; DT w, gamma, beta, b, rm, rv; };
struct ProvDX { std::vector<DTLX> L; size_t i = 0; DTLX& next() { return L[i++]; } };

inline ProvDX dnetx_build(const std::string& arch, const std::string& weights) {
  ProviderU pu = load_unfused_pt(arch, weights);
  ProvDX p;
  for (auto& L : pu.layers) {
    DTLX d; d.kind = L.kind; d.stride = L.stride; d.eps = L.eps; d.groups = L.groups;
    int64_t Co = L.w->shape[0], Ci = L.w->shape[1], k = L.w->shape[2];
    d.w = dfrom({Co,Ci,k,k}, L.w->data);
    if (L.kind == 1) { d.gamma = dfrom({Co}, L.gamma->data); d.beta = dfrom({Co}, L.beta->data);
                       d.rm = dfrom({Co}, L.rm); d.rv = dfrom({Co}, L.rv); }
    else d.b = dfrom({Co}, L.b->data);
    p.L.push_back(std::move(d));
  }
  return p;
}
inline std::vector<DT> dnetx_params(ProvDX& p) {
  std::vector<DT> ps; for (auto& L : p.L) { ps.push_back(L.w); if (L.kind==1){ps.push_back(L.gamma);ps.push_back(L.beta);} else ps.push_back(L.b); } return ps;
}

static inline DT dx_apply(DT x, DTLX& L, bool tr) {
  int64_t pad = L.w->shape[2]/2;
  if (L.kind == 1) return dsilu(dbn(dconv2d(x, L.w, DT(), L.stride, pad, L.groups), L.gamma, L.beta, L.eps,
                                    tr ? L.rm : DT(), tr ? L.rv : DT(), 0.03f));
  return dconv2d(x, L.w, L.b, L.stride, pad, L.groups);
}
static inline DT dx_bcu(DT x, ProvDX& p, bool tr) { return dx_apply(x, p.next(), tr); }
static inline DT dx_conv3x3(DT x, ProvDX& p, bool dw, bool tr) {
  if (dw) { DT h = dx_bcu(x, p, tr); return dx_bcu(h, p, tr); }   // depthwise 3x3 + pointwise 1x1
  return dx_bcu(x, p, tr);
}
static inline DT dx_csp(DT x, ProvDX& p, int64_t n, bool sc, bool tr, bool dw) {
  DT a = dx_bcu(x, p, tr), b = dx_bcu(x, p, tr);
  for (int64_t i=0;i<n;++i){ DT h=dx_bcu(a,p,tr); h=dx_conv3x3(h,p,dw,tr); a=sc?dadd(h,a):h; }
  return dx_bcu(dconcat({a,b}), p, tr);
}
static inline DT dx_spp(DT x, ProvDX& p, bool tr) {
  DT x1=dx_bcu(x,p,tr); DT p5=dmaxpool2d(x1,5,1,2),p9=dmaxpool2d(x1,9,1,4),p13=dmaxpool2d(x1,13,1,6);
  return dx_bcu(dconcat({x1,p5,p9,p13}), p, tr);
}
static inline DT dx_head(DT f, ProvDX& p, bool tr, bool dw) {
  DT x=dx_bcu(f,p,tr);
  DT cf=dx_conv3x3(dx_conv3x3(x,p,dw,tr),p,dw,tr);
  DT rf=dx_conv3x3(dx_conv3x3(x,p,dw,tr),p,dw,tr);
  DT cls=dx_apply(cf,p.next(),tr), reg=dx_apply(rf,p.next(),tr), obj=dx_apply(rf,p.next(),tr);
  return dconcat({reg, obj, cls});
}
inline std::vector<DT> dnetx_forward(DT in, ProvDX& p, int64_t bd, bool dw, bool tr) {
  DT x=dx_bcu(dfocus(in),p,tr);
  x=dx_conv3x3(x,p,dw,tr); x=dx_csp(x,p,bd,true,tr,dw);
  x=dx_conv3x3(x,p,dw,tr); DT c3=dx_csp(x,p,3*bd,true,tr,dw);
  x=dx_conv3x3(c3,p,dw,tr); DT c4=dx_csp(x,p,3*bd,true,tr,dw);
  x=dx_conv3x3(c4,p,dw,tr); x=dx_spp(x,p,tr); DT c5=dx_csp(x,p,bd,false,tr,dw);
  DT fpn0=dx_bcu(c5,p,tr); DT u=dupsample2x(fpn0);
  DT p4=dx_csp(dconcat({u,c4}),p,bd,false,tr,dw);
  DT red=dx_bcu(p4,p,tr); DT u2=dupsample2x(red);
  DT pan2=dx_csp(dconcat({u2,c3}),p,bd,false,tr,dw);
  DT d0=dx_conv3x3(pan2,p,dw,tr);
  DT pan1=dx_csp(dconcat({d0,red}),p,bd,false,tr,dw);
  DT d1=dx_conv3x3(pan1,p,dw,tr);
  DT pan0=dx_csp(dconcat({d1,fpn0}),p,bd,false,tr,dw);
  std::vector<DT> out; DT lv[3]={pan2,pan1,pan0}; for (auto& f:lv) out.push_back(dx_head(f,p,tr,dw)); return out;
}
inline void dnetx_save(ProvDX& p, const std::string& arch, const std::string& path) {
  std::vector<std::string> names; { std::ifstream f(arch + "names.txt"); std::string s; while (f >> s) names.push_back(s); }
  std::vector<pt::Tensor> ck; size_t k = 0;
  auto push = [&](DT t){ pt::Tensor o; if (k<names.size()) o.name=names[k]; o.shape.assign(t->shape.begin(),t->shape.end()); o.data=dto_host(t); ck.push_back(std::move(o)); ++k; };
  for (auto& L : p.L) { push(L.w); if (L.kind==1){push(L.gamma);push(L.beta);push(L.rm);push(L.rv);} else push(L.b); }
  pt::save_pt(ck, path);
}
