// M2b: YOLOX loss forward + backward in the pure engine, vs the PyTorch reference.
#include "yolox_loss.hpp"
#include "net_yolox.hpp"   // rd()
#include <cstdio>
#include <fstream>

int main() {
  const std::string D = "pure/ref/data_loss/";
  int64_t IMG, A, NC, G, NF; { std::ifstream f(D+"meta.txt"); f>>IMG>>A>>NC>>G>>NF; }
  // per-level shapes
  std::vector<std::array<int64_t,3>> lv; { std::ifstream f(D+"levels.txt"); int64_t h,w,s; while (f>>h>>w>>s) lv.push_back({h,w,s}); }
  std::vector<Tensor> raw;
  for (size_t i=0;i<lv.size();++i)
    raw.push_back(from_data({1,4+1+NC,lv[i][0],lv[i][1]}, rd(D+"raw_L"+std::to_string(i)+".bin"), true));
  auto xs=rd(D+"xs.bin"), ys=rd(D+"ys.bin"), st=rd(D+"strides.bin"), gtb=rd(D+"gt_boxes.bin");
  auto gtcf=rd(D+"gt_cls.bin"); std::vector<int64_t> gtc(G); for(int64_t i=0;i<G;++i) gtc[i]=(int64_t)gtcf[i];

  auto L = yolox_loss(raw, xs, ys, st, gtb, gtc, A, NC, G);

  auto rl = rd(D+"loss.bin");
  printf("forward  iou=%.6f obj=%.6f cls=%.6f total=%.6f\n", L.iou->data[0],L.obj->data[0],L.cls->data[0],L.total->data[0]);
  printf("ref      iou=%.6f obj=%.6f cls=%.6f total=%.6f\n", rl[0],rl[1],rl[2],rl[3]);
  double df = std::max({std::abs(L.iou->data[0]-rl[0]),std::abs(L.obj->data[0]-rl[1]),std::abs(L.cls->data[0]-rl[2]),std::abs(L.total->data[0]-rl[3])});
  printf("[M2b forward] max|diff|=%.3e  %s\n\n", df, df<1e-3?"OK":"FAIL");

  backward(L.total);
  double dg=0;
  for (size_t i=0;i<raw.size();++i){ auto g=rd(D+"grad_L"+std::to_string(i)+".bin");
    for (int64_t j=0;j<raw[i]->numel();++j) dg=std::max(dg,(double)std::abs(raw[i]->grad[j]-g[j])); }
  printf("[M2b backward] grad max|diff|=%.3e  %s\n", dg, dg<1e-5?"OK":"FAIL");
  bool ok = df<1e-3 && dg<1e-5;
  printf("\n%s\n", ok ? "M2b: YOLOX loss + autograd == PyTorch" : "MISMATCH");
  return ok?0:1;
}
