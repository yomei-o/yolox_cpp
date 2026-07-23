// yolox training CLI — a real training loop, pure C++ / no Python at run time:
// dataset scan -> shuffled mini-batches (+ hflip/brightness aug) over epochs -> SimOTA ->
// yolox loss (IoU+obj+cls) -> Adam (warmup + cosine) -> per-epoch validation mAP -> save
// last.pt / best.pt via the pure-C++ .pt writer. Initial weights + state_dict key names
// come from data_unf/ (export_unfused_yolox.py: unfused conv+BN, canonical order + names.txt).
//   build: bash /c/prog/claude/cc11.sh -std:c++20 -O2 -EHsc -Ipure/third_party pure/train_cli.cpp -Fe:train_cli.exe -Fo:scratch/
//   run:   train_cli <train_list> <val_list> <epochs> <batch>
#define STB_IMAGE_IMPLEMENTATION
#include "dataset.hpp"
#include "net_yolox_unfused.hpp"
#include "yolox_loss.hpp"
#include "infer_yolox.hpp"
#include "metrics.hpp"
#include "optim.hpp"
#include "ptio.hpp"
#include <cstdio>
#include <numeric>
#include <algorithm>
#include <random>

static const int64_t NC = 80;

int main(int argc, char** argv) {
  std::string trainL = argc>1?argv[1]:"pure/ref/data_synth/list.txt";
  std::string valL   = argc>2?argv[2]:"pure/ref/data_synth/val.txt";
  int EPOCHS = argc>3?atoi(argv[3]):8, BATCH = argc>4?atoi(argv[4]):4;
  const std::string DU = "pure/ref/data_unf/";

  Dataset tr = read_dataset(trainL), va = read_dataset(valL);
  int64_t S = tr.S;
  int64_t BD=1, DWF=0; { std::ifstream f(DU + "io.txt"); int64_t im; f >> im >> BD >> DWF; }
  printf("train=%zu val=%zu imgsz=%lld batch=%d epochs=%d  (BD=%lld DW=%lld)\n",
         tr.items.size(), va.items.size(), (long long)S, BATCH, EPOCHS, (long long)BD, (long long)DWF);

  auto prov = load_unfused(DU);
  std::vector<Tensor> params;
  for (auto& L : prov.layers) { params.push_back(L.w); if (L.kind==1){params.push_back(L.gamma);params.push_back(L.beta);} else params.push_back(L.b); }
  Adam opt(params, 2e-3f, 0.9f, 0.999f, 1e-8f, 5e-4f, false);

  // anchors (grid col/row index per level), strides 8/16/32 for imgsz S
  struct Lv { int64_t h,w; float s; }; std::vector<Lv> lv = {{S/8,S/8,8.f},{S/16,S/16,16.f},{S/32,S/32,32.f}};
  std::vector<float> xs,ys,st; for (auto& L:lv) for (int64_t r=0;r<L.h;++r) for (int64_t c=0;c<L.w;++c){xs.push_back((float)c);ys.push_back((float)r);st.push_back(L.s);}
  int64_t A = (int64_t)xs.size();
  std::vector<int64_t> strides = {8,16,32};

  // state_dict KEY per engine tensor (engine walk order != state_dict order, so pair by
  // name — load_state_dict matches by key). names.txt from export_unfused_yolox.py.
  std::vector<std::string> names; { std::ifstream f(DU + "names.txt"); std::string s; while (f >> s) names.push_back(s); }
  auto save_ckpt = [&](const std::string& path) {
    std::vector<pt::Tensor> ck; size_t k = 0;
    auto push = [&](const std::vector<float>& d, const std::vector<int64_t>& shp){ pt::Tensor t; if (k<names.size()) t.name=names[k]; t.shape=shp; t.data=d; ck.push_back(t); ++k; };
    for (auto& L : prov.layers) { std::vector<int64_t> ws(L.w->shape.begin(),L.w->shape.end());
      push(L.w->data, ws);
      if (L.kind==1){ std::vector<int64_t> c={L.gamma->shape[0]}; push(L.gamma->data,c); push(L.beta->data,c); push(L.rm,c); push(L.rv,c); }
      else push(L.b->data, {L.b->shape[0]}); }
    pt::save_pt(ck, path);
  };

  // load one image into a 1-image (1,3,S,S) tensor + GT (labels are xyxy in S-space)
  auto load_one = [&](const Dataset& d, int i, bool aug, uint32_t seed, std::vector<float>& gtb_cxcywh,
                      std::vector<int64_t>& gtc, std::vector<float>& gt_xyxy) -> Tensor {
    Batch b = load_minibatch(d, {i}, aug, seed);
    int64_t M = b.M; gtb_cxcywh.clear(); gtc.clear(); gt_xyxy.clear();
    for (int64_t m=0;m<M;++m) if (b.mask[m] > 0.5f) {
      float x1=b.gt_boxes[m*4], y1=b.gt_boxes[m*4+1], x2=b.gt_boxes[m*4+2], y2=b.gt_boxes[m*4+3];
      gtb_cxcywh.insert(gtb_cxcywh.end(), {(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1});
      gt_xyxy.insert(gt_xyxy.end(), {x1,y1,x2,y2}); gtc.push_back(b.gt_labels[m]);
    }
    return b.x;
  };

  auto validate = [&]() -> double {
    std::vector<mapeval::Image> imgs;
    for (int i=0;i<(int)va.items.size();++i) {
      std::vector<float> gcx,gxy; std::vector<int64_t> gc;
      auto x = load_one(va, i, false, 0, gcx, gc, gxy);
      prov.i=0; auto raw = yolox_forward_unfused(x, prov, false, BD, (bool)DWF);
      auto dets = yolox_detect(raw, strides, NC, 0.05f, 0.5f);   // yolox score=obj*cls runs low
      mapeval::Image im;
      for (auto& d : dets) im.dts.push_back({d.x1,d.y1,d.x2,d.y2,d.cls,d.score});
      for (size_t g=0;g<gc.size();++g) im.gts.push_back({gxy[g*4],gxy[g*4+1],gxy[g*4+2],gxy[g*4+3],(int)gc[g]});
      imgs.push_back(im);
    }
    return mapeval::coco_map(imgs).first;   // mAP@0.50
  };

  std::vector<int> order(tr.items.size()); std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(0);
  int steps_per_epoch = ((int)tr.items.size() + BATCH - 1) / BATCH, total = EPOCHS * steps_per_epoch, gstep = 0;
  double best = -1;
  for (int ep = 0; ep < EPOCHS; ++ep) {
    std::shuffle(order.begin(), order.end(), rng); double eloss = 0; int nb = 0;
    for (size_t off = 0; off < order.size(); off += BATCH) {
      size_t end = std::min(order.size(), off + (size_t)BATCH);
      // yolox forward/SimOTA/loss are per-image; sum the mini-batch losses into one graph,
      // one backward + Adam step per mini-batch (BN train mode updates running stats).
      Tensor mb; int cnt = 0;
      for (size_t t = off; t < end; ++t) {
        std::vector<float> gcx,gxy; std::vector<int64_t> gc;
        auto x = load_one(tr, order[t], true, rng(), gcx, gc, gxy);
        prov.i=0; auto raw = yolox_forward_unfused(x, prov, true, BD, (bool)DWF);
        auto L = yolox_loss(raw, xs, ys, st, gcx, gc, A, NC, (int64_t)gc.size());
        mb = cnt ? add(mb, L.total) : L.total; ++cnt;
      }
      auto mbmean = mul_scalar(mb, 1.f/std::max(1,cnt));
      backward(mbmean);
      opt.lr = cosine_lr(gstep, total, 2e-3f, std::max(1, total/20)); opt.step(); opt.zero_grad(); ++gstep;
      eloss += mbmean->data[0]; ++nb;
    }
    double m50 = validate();
    printf("epoch %2d/%d  loss %6.3f  val mAP@0.5 %.3f%s\n", ep+1, EPOCHS, eloss/nb, m50, m50>best?"  *best*":"");
    save_ckpt("last.pt"); if (m50 > best) { best = m50; save_ckpt("best.pt"); }
  }
  printf("done. best val mAP@0.5 = %.3f. wrote last.pt / best.pt (pure C++)\n", best);
  return 0;
}
