// COCO-style mAP (bbox) in plain C++, matching pycocotools for area=all, maxDets=100:
// per-class, per-IoU greedy matching (max-IoU unmatched GT), then 101-point interpolated
// AP over recall, averaged over the 10 IoU thresholds 0.50:0.05:0.95 and over classes
// that have ground truth. Returns {mAP@0.50, mAP@0.50:0.95}.
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <array>
#include <set>
#include <utility>

namespace mapeval {

struct GT { float x1, y1, x2, y2; int cls; };
struct DT { float x1, y1, x2, y2; int cls; float score; };
struct Image { std::vector<GT> gts; std::vector<DT> dts; };

static inline float iou(const DT& a, const GT& b) {
  float iw = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  float ih = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  float inter = iw * ih;
  float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0.f;
}

struct Scored { float score; std::array<char, 10> tp; };   // matched flag per IoU threshold

inline std::pair<double, double> coco_map(const std::vector<Image>& imgs, int maxDets = 100) {
  std::array<double, 10> thr;
  for (int t = 0; t < 10; ++t) thr[t] = 0.5 + 0.05 * t;

  std::set<int> classes;
  for (auto& im : imgs) { for (auto& g : im.gts) classes.insert(g.cls); for (auto& d : im.dts) classes.insert(d.cls); }

  std::vector<double> ap50, ap5095;                 // per qualifying class
  for (int c : classes) {
    int npig = 0;
    for (auto& im : imgs) for (auto& g : im.gts) if (g.cls == c) ++npig;
    if (npig == 0) continue;                          // pycocotools excludes classes with no GT

    std::vector<Scored> dets;
    for (auto& im : imgs) {
      std::vector<const GT*> g; for (auto& gg : im.gts) if (gg.cls == c) g.push_back(&gg);
      std::vector<const DT*> d; for (auto& dd : im.dts) if (dd.cls == c) d.push_back(&dd);
      std::stable_sort(d.begin(), d.end(), [](const DT* a, const DT* b) { return a->score > b->score; });
      if ((int)d.size() > maxDets) d.resize(maxDets);
      int D = (int)d.size(), G = (int)g.size();
      std::vector<std::vector<float>> ious(D, std::vector<float>(G));
      for (int i = 0; i < D; ++i) for (int j = 0; j < G; ++j) ious[i][j] = iou(*d[i], *g[j]);
      for (int i = 0; i < D; ++i) {
        Scored s; s.score = d[i]->score; s.tp.fill(0);
        dets.push_back(s);
      }
      // match per threshold
      for (int t = 0; t < 10; ++t) {
        std::vector<char> gtm(G, 0);
        for (int i = 0; i < D; ++i) {
          double cur = std::min(thr[t], 1.0 - 1e-10); int m = -1;
          for (int j = 0; j < G; ++j) {
            if (gtm[j]) continue;
            if (ious[i][j] < cur) continue;
            cur = ious[i][j]; m = j;
          }
          if (m >= 0) { gtm[m] = 1; dets[dets.size() - D + i].tp[t] = 1; }
        }
      }
    }
    std::stable_sort(dets.begin(), dets.end(), [](const Scored& a, const Scored& b) { return a.score > b.score; });

    for (int t = 0; t < 10; ++t) {
      int N = (int)dets.size();
      std::vector<double> pr(N), rc(N);
      int tp = 0, fp = 0;
      for (int i = 0; i < N; ++i) { dets[i].tp[t] ? ++tp : ++fp; pr[i] = (double)tp / (tp + fp); rc[i] = (double)tp / npig; }
      for (int i = N - 1; i > 0; --i) if (pr[i] > pr[i - 1]) pr[i - 1] = pr[i];   // monotonic from right
      double ap = 0;
      for (int r = 0; r <= 100; ++r) {
        double rt = r / 100.0;
        int idx = (int)(std::lower_bound(rc.begin(), rc.end(), rt) - rc.begin());  // searchsorted left
        ap += (idx < N) ? pr[idx] : 0.0;
      }
      ap /= 101.0;
      if (t == 0) ap50.push_back(ap);
      ap5095.push_back(ap);
    }
  }
  auto mean = [](const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return v.empty() ? -1.0 : s / v.size(); };
  return {mean(ap50), mean(ap5095)};
}

}  // namespace mapeval
