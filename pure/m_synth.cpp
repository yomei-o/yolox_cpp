// End-to-end (yolox): generate a few dozen 128x128 synthetic images (colored squares),
// train yolox in the pure engine, then run inference and check it detects them.
#include "net_yolox.hpp"
#include "yolox_loss.hpp"
#include "infer_yolox.hpp"
#include <cstdio>
#include <random>

int main() {
  const std::string D = "pure/ref/data_net/";
  const int64_t IMG = 128, NC = 80;
  int64_t BD, DW = 0; { std::ifstream f(D + "io.txt"); int64_t im; f >> im >> BD >> DW; }
  auto prov = load_net(D);
  std::vector<Tensor> params; for (auto& c : prov.convs) { params.push_back(c.w); params.push_back(c.b); }

  // anchors for 128px: strides 8/16/32 -> grids 16/8/4
  struct Lv { int64_t h, w; float s; }; std::vector<Lv> lv = {{16,16,8},{8,8,16},{4,4,32}};
  std::vector<float> xs, ys, st;
  for (auto& L : lv) for (int64_t r=0;r<L.h;++r) for (int64_t c=0;c<L.w;++c){ xs.push_back(c); ys.push_back(r); st.push_back(L.s); }
  int64_t A = (int64_t)xs.size();

  // synthetic dataset: K images, each 1-2 bright squares (class = color 0/1/2) on dark bg
  const int K = 24;
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> pos(16, 96), sz(18, 34), ncls(0, 2), nobj(1, 2);
  std::vector<Tensor> imgs; std::vector<std::vector<float>> gtbs; std::vector<std::vector<int64_t>> gtcs;
  for (int k = 0; k < K; ++k) {
    auto img = make_tensor({1, 3, IMG, IMG});
    for (auto& v : img->data) v = 0.1f;                    // dark bg
    std::vector<float> gtb; std::vector<int64_t> gtc; int no = nobj(rng);
    for (int o = 0; o < no; ++o) {
      int cx = pos(rng), cy = pos(rng), s = sz(rng), cl = ncls(rng);
      for (int c = 0; c < 3; ++c) { float val = (c == cl) ? 0.95f : 0.15f;   // color-coded class
        for (int y = cy - s/2; y < cy + s/2; ++y) for (int x = cx - s/2; x < cx + s/2; ++x)
          if (y>=0&&y<IMG&&x>=0&&x<IMG) img->data[(c*IMG+y)*IMG+x] = val; }
      gtb.insert(gtb.end(), {(float)cx,(float)cy,(float)s,(float)s}); gtc.push_back(cl);
    }
    imgs.push_back(img); gtbs.push_back(gtb); gtcs.push_back(gtc);
  }

  // train
  float lr = 2e-4f; const int EPOCHS = 25;
  printf("training on %d synthetic 128x128 images...\n", K);
  for (int e = 0; e < EPOCHS; ++e) {
    double tot = 0;
    for (int k = 0; k < K; ++k) {
      prov.i = 0;
      auto raw = yolox_forward(imgs[k], prov, BD, (bool)DW);
      auto L = yolox_loss(raw, xs, ys, st, gtbs[k], gtcs[k], A, NC, (int64_t)gtcs[k].size());
      backward(L.total);
      for (auto& p : params) for (int64_t i=0;i<p->numel();++i) p->data[i] -= lr * p->grad[i];
      tot += L.total->data[0];
    }
    printf("epoch %d  mean loss %.4f\n", e, tot / K);
  }

  // inference on image 0
  prov.i = 0;
  auto raw = yolox_forward(imgs[0], prov, BD, (bool)DW);
  auto dets = yolox_detect(raw, {8,16,32}, NC, 0.15f, 0.45f);
  printf("\nimage 0 GT: ");
  for (size_t g=0; g<gtcs[0].size(); ++g) printf("cls%lld @(%.0f,%.0f) ", (long long)gtcs[0][g], gtbs[0][g*4], gtbs[0][g*4+1]);
  printf("\nimage 0 detections (%zu):\n", dets.size());
  for (auto& d : dets) printf("  cls%d conf=%.2f  box=(%.0f,%.0f,%.0f,%.0f)\n", d.cls, d.score, d.x1, d.y1, d.x2, d.y2);
  printf("\n%s\n", dets.size() ? "end-to-end OK: trained yolox detects synthetic objects" : "no detections");
  return 0;
}
