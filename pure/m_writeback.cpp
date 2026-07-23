// A-1 (yolox) step 2: train the unfused net in C++, then write all weights to data_wb/
// (canonical order) so writeback_yolox.py can drop them into a standard YOLOX .pt.
#include "net_yolox_unfused.hpp"
#include "yolox_loss.hpp"
#include <cstdio>
#include <random>

static void wr(const std::string& p, const std::vector<float>& v) {
  std::ofstream f(p, std::ios::binary); f.write((const char*)v.data(), v.size()*sizeof(float));
}

int main() {
  const std::string DU = "pure/ref/data_unf/", DW = "pure/ref/data_wb/";
  const int64_t IMG = 64, NC = 80;
  auto prov = load_unfused(DU);
  std::vector<Tensor> params;
  for (auto& L : prov.layers) { params.push_back(L.w);
    if (L.kind==1){ params.push_back(L.gamma); params.push_back(L.beta);} else params.push_back(L.b); }

  struct Lv{int64_t h,w; float s;}; std::vector<Lv> lv={{8,8,8},{4,4,16},{2,2,32}};
  std::vector<float> xs,ys,st; for(auto&L:lv) for(int64_t r=0;r<L.h;++r) for(int64_t c=0;c<L.w;++c){xs.push_back(c);ys.push_back(r);st.push_back(L.s);}
  int64_t A=(int64_t)xs.size();
  auto img = make_tensor({1,3,IMG,IMG});
  { std::mt19937 rng(1); std::normal_distribution<float> nd(0,1); for(auto&v:img->data)v=nd(rng); }
  std::vector<float> gtb={20,22,24,20, 44,40,28,30, 12,46,16,24}; std::vector<int64_t> gtc={5,17,3}; int64_t G=3;

  float lr=1e-4f;
  printf("iter | total\n");
  for (int it=0; it<=40; ++it) {
    prov.i=0;
    auto raw = yolox_forward_unfused(img, prov, /*training=*/false);   // BN eval: freeze running stats
    auto L = yolox_loss(raw, xs, ys, st, gtb, gtc, A, NC, G);
    backward(L.total);
    for (auto& p: params) for (int64_t i=0;i<p->numel();++i) p->data[i]-=lr*p->grad[i];
    if (it%10==0) printf("%4d | %8.4f\n", it, L.total->data[0]);
  }
  // dump weights in layer order
  for (size_t i=0;i<prov.layers.size();++i){ auto& L=prov.layers[i]; std::string s=std::to_string(i);
    wr(DW+"cw"+s+".bin", L.w->data);
    if (L.kind==1){ wr(DW+"bg"+s+".bin",L.gamma->data); wr(DW+"bb"+s+".bin",L.beta->data); wr(DW+"rm"+s+".bin",L.rm); wr(DW+"rv"+s+".bin",L.rv);}
    else wr(DW+"cb"+s+".bin", L.b->data); }
  printf("wrote %zu layers to %s\n", prov.layers.size(), DW.c_str());
  return 0;
}
