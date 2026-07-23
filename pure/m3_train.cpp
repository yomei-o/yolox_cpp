// M3: train yolox-tiny end-to-end in the pure engine (forward -> SimOTA -> loss ->
// backward -> SGD). Overfits a fixed synthetic batch; loss must drop. No LibTorch.
#include "net_yolox.hpp"
#include "yolox_loss.hpp"
#include <cstdio>
#include <random>

int main() {
  const std::string D = "pure/ref/data_net/";     // fused weights from export_yolox.py
  const int64_t IMG = 64, NC = 80;
  auto prov = load_net(D);
  std::vector<Tensor> params;
  for (auto& c : prov.convs) { params.push_back(c.w); params.push_back(c.b); }

  // anchors (row-major per level), strides 8/16/32
  struct Lv { int64_t h, w; float s; };
  std::vector<Lv> levels = {{8,8,8},{4,4,16},{2,2,32}};
  std::vector<float> xs, ys, st;
  for (auto& L : levels) for (int64_t r=0;r<L.h;++r) for (int64_t c=0;c<L.w;++c){ xs.push_back((float)c); ys.push_back((float)r); st.push_back(L.s); }
  int64_t A = (int64_t)xs.size();

  // fixed synthetic image + gt (cxcywh abs px, class)
  auto img = make_tensor({1,3,IMG,IMG});
  { std::mt19937 rng(0); std::normal_distribution<float> nd(0,1); for (auto& v: img->data) v=nd(rng); }
  std::vector<float> gtb = {20,22,24,20,  44,40,28,30,  12,46,16,24};
  std::vector<int64_t> gtc = {5, 17, 3};
  int64_t G = (int64_t)gtc.size();

  float lr = 1e-3f;
  printf("iter |   total     iou      obj      cls\n");
  for (int it = 0; it <= 50; ++it) {
    prov.i = 0;
    auto raw = yolox_forward(img, prov);
    auto L = yolox_loss(raw, xs, ys, st, gtb, gtc, A, NC, G);
    backward(L.total);
    for (auto& p : params) for (int64_t i=0;i<p->numel();++i) p->data[i] -= lr * p->grad[i];
    if (it % 5 == 0)
      printf("%4d | %8.4f %8.4f %8.4f %8.4f\n", it, L.total->data[0], L.iou->data[0], L.obj->data[0], L.cls->data[0]);
  }
  printf("done — trained yolox-tiny in the pure engine.\n");
  return 0;
}
