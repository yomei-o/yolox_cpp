// SimOTA dynamic label assignment (YOLOX), plain C++ / no-grad. Faithful port of
// get_geometry_constraint + cost + simota_matching.
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstdint>

struct Assign {
  std::vector<float> fg;          // (A) 0/1
  std::vector<int64_t> matched_gt;   // (num_fg) gt index per fg anchor (anchor order)
  std::vector<int64_t> matched_cls;  // (num_fg)
  std::vector<float> pred_ious;      // (num_fg)
  int64_t num_fg = 0;
};

// IoU of two cxcywh boxes (YOLOX bboxes_iou, xyxy=False).
static inline float iou_cxcywh(const float* a, const float* b) {
  float al=a[0]-a[2]/2, ar=a[0]+a[2]/2, at=a[1]-a[3]/2, ab=a[1]+a[3]/2;
  float bl=b[0]-b[2]/2, br=b[0]+b[2]/2, bt=b[1]-b[3]/2, bb=b[1]+b[3]/2;
  float iw=std::min(ar,br)-std::max(al,bl), ih=std::min(ab,bb)-std::max(at,bt);
  if (iw<=0||ih<=0) return 0.f;
  float inter=iw*ih, uni=a[2]*a[3]+b[2]*b[3]-inter;
  return inter/uni;
}
static inline float sigm(float x){ return 1.f/(1.f+std::exp(-x)); }

// bbox_preds (A,4 cxcywh), cls/obj (A,nc)/(A,1) raw logits, xs/ys/strides (A),
// gt_boxes (G,4 cxcywh), gt_cls (G).
inline Assign simota_assign(const std::vector<float>& bbox, const std::vector<float>& cls,
                            const std::vector<float>& obj, const std::vector<float>& xs,
                            const std::vector<float>& ys, const std::vector<float>& strides,
                            const std::vector<float>& gtb, const std::vector<int64_t>& gtc,
                            int64_t A, int64_t nc, int64_t G) {
  const float R = 1.5f;
  // geometry: is_in_centers[g*A + a]
  std::vector<char> inc(G * A, 0);
  std::vector<char> anchor_filter(A, 0);
  for (int64_t a = 0; a < A; ++a) {
    float xc = (xs[a] + 0.5f) * strides[a], yc = (ys[a] + 0.5f) * strides[a];
    float cd = strides[a] * R;
    for (int64_t g = 0; g < G; ++g) {
      float gx = gtb[g*4+0], gy = gtb[g*4+1];
      bool in = (xc - (gx-cd) > 0) && ((gx+cd) - xc > 0) && (yc - (gy-cd) > 0) && ((gy+cd) - yc > 0);
      if (in) { inc[g*A+a] = 1; anchor_filter[a] = 1; }
    }
  }
  std::vector<int64_t> cand;
  for (int64_t a = 0; a < A; ++a) if (anchor_filter[a]) cand.push_back(a);
  int64_t M = (int64_t)cand.size();

  // pair_wise ious + cost (G,M)
  std::vector<float> ious(G * M), cost(G * M);
  for (int64_t g = 0; g < G; ++g)
    for (int64_t j = 0; j < M; ++j) {
      int64_t a = cand[j];
      float io = iou_cxcywh(&gtb[g*4], &bbox[a*4]);
      ious[g*M+j] = io;
      // cls cost: BCE(sqrt(sig(cls)*sig(obj)), onehot) summed over classes
      float clsc = 0.f, ob = sigm(obj[a]);
      for (int64_t c = 0; c < nc; ++c) {
        float p = std::sqrt(sigm(cls[a*nc+c]) * ob);
        p = std::min(std::max(p, 1e-12f), 1.f - 1e-12f);
        float z = (gtc[g] == c) ? 1.f : 0.f;
        clsc += -(z*std::log(p) + (1-z)*std::log(1-p));
      }
      float iloss = -std::log(io + 1e-8f);
      float geo = inc[g*A+a] ? 0.f : 1e6f;
      cost[g*M+j] = clsc + 3.f*iloss + geo;
    }

  // dynamic_k_matching
  std::vector<char> matching(G * M, 0);
  int64_t nck = std::min<int64_t>(10, M);
  for (int64_t g = 0; g < G; ++g) {
    std::vector<int64_t> idx(M); std::iota(idx.begin(), idx.end(), 0);
    // top-nck ious for dynamic_k
    std::partial_sort(idx.begin(), idx.begin()+nck, idx.end(),
                      [&](int64_t p,int64_t q){return ious[g*M+p]>ious[g*M+q];});
    float s=0; for (int64_t t=0;t<nck;++t) s+=ious[g*M+idx[t]];
    int64_t dk = std::max<int64_t>(1, (int64_t)s);
    // dk lowest-cost
    std::vector<int64_t> ci(M); std::iota(ci.begin(), ci.end(), 0);
    std::partial_sort(ci.begin(), ci.begin()+dk, ci.end(),
                      [&](int64_t p,int64_t q){return cost[g*M+p]<cost[g*M+q];});
    for (int64_t t=0;t<dk;++t) matching[g*M+ci[t]] = 1;
  }
  // resolve anchors matched to multiple gts -> keep min cost gt
  for (int64_t j=0;j<M;++j) {
    int64_t cnt=0; for (int64_t g=0;g<G;++g) cnt+=matching[g*M+j];
    if (cnt>1) {
      int64_t best=0; float bc=1e30f;
      for (int64_t g=0;g<G;++g) if (cost[g*M+j]<bc){bc=cost[g*M+j];best=g;}
      for (int64_t g=0;g<G;++g) matching[g*M+j] = (g==best)?1:0;
    }
  }
  // build outputs
  Assign r; r.fg.assign(A, 0.f);
  for (int64_t j=0;j<M;++j) {
    int64_t cnt=0, mg=-1; for (int64_t g=0;g<G;++g) if (matching[g*M+j]){cnt++; mg=g;}
    if (cnt>0) {
      int64_t a = cand[j];
      r.fg[a] = 1.f;
      r.matched_gt.push_back(mg);
      r.matched_cls.push_back(gtc[mg]);
      r.pred_ious.push_back(ious[mg*M+j]);
    }
  }
  r.num_fg = (int64_t)r.matched_gt.size();
  return r;
}
