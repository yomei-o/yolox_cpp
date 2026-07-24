// Device-resident YOLOX training over a standard-YOLO dataset (e.g. COCO128), any size,
// with checkpoint save. Device fwd(train BN, EMA running stats)+bwd+Adam; trusted host
// SimOTA loss (yolox_loss) bridged in per image (YOLOX loss is per-image). Saves last/best.pt.
//   run: dtrain_coco_yolox <images_dir> <imgsz> <batch> <epochs> [model=yolox_tiny]
//   GPU: nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA [-DUSE_CUBLAS -lcublas] -Ipure/third_party pure/dtrain_coco_yolox.cpp -o dtrain_coco_yolox
#define STB_IMAGE_IMPLEMENTATION
#include "dataset.hpp"
#include "dnet_yolox.hpp"       // device YOLOX + build/forward/params/save
#include "yolox_loss.hpp"       // yolox_loss (trusted host, SimOTA)
#include <cstdio>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::string dir = argc>1?argv[1]:"pure/ref/data_yolo/images/train";
  int64_t S = argc>2?atoll(argv[2]):96;
  int BATCH = argc>3?atoi(argv[3]):4, EPOCHS = argc>4?atoi(argv[4]):2;
  std::string model = argc>5?argv[5]:"yolox_tiny";
  const int64_t NC = 80;
  // per-size: base depth (bd) + depthwise (dw). nano is depthwise; bd = round(3*depth_mult).
  int64_t bd = 1; bool dw = false;
  if (model=="yolox_nano"){bd=1;dw=true;} else if (model=="yolox_tiny"||model=="yolox_s"){bd=1;}
  else if (model=="yolox_m"){bd=2;} else if (model=="yolox_l"){bd=3;} else if (model=="yolox_x"){bd=4;}
  std::string arch = (model=="yolox_tiny") ? "pure/ref/data_unf/" : "pure/ref/arch/"+model+"/";
  std::string weights = model + ".pth";

  ProvDX prov = dnetx_build(arch, weights);
  std::vector<DT> params = dnetx_params(prov);
  DAdam opt(params, 1e-3f);
  struct Lv{int64_t h,w; float s;}; std::vector<Lv> lv={{S/8,S/8,8.f},{S/16,S/16,16.f},{S/32,S/32,32.f}};
  std::vector<float> xs,ys,st; for(auto&L:lv)for(int64_t r=0;r<L.h;++r)for(int64_t c=0;c<L.w;++c){xs.push_back((float)c);ys.push_back((float)r);st.push_back(L.s);}
  int64_t A=(int64_t)xs.size();
  Dataset tr = read_yolo_dataset(dir, S);
  printf("%s train=%zu imgsz=%lld batch=%d epochs=%d bd=%lld dw=%d\n", model.c_str(), tr.items.size(), (long long)S, BATCH, EPOCHS, (long long)bd, (int)dw);

  std::vector<int> order(tr.items.size()); std::iota(order.begin(),order.end(),0); std::mt19937 rng(0);
  double best = 1e30;
  for (int ep=0; ep<EPOCHS; ++ep) {
    std::shuffle(order.begin(),order.end(),rng); double eloss=0; int nimg=0;
    auto t0 = std::chrono::steady_clock::now();
    for (size_t off=0; off<order.size(); off+=BATCH) {
      size_t endi = std::min(order.size(), off+(size_t)BATCH);
      float inv = 1.f/(float)(endi-off);                        // minibatch-mean scaling
      opt.zero_grad();
      for (size_t t=off; t<endi; ++t) {
        Batch b = load_minibatch(tr, {order[(int)t]}, false, rng());  // B=1 (YOLOX loss is per-image)
        std::vector<float> gcx; std::vector<int64_t> gc; int64_t M=b.M;
        for (int64_t m=0;m<M;++m) if (b.mask[m]>0.5f) { float x1=b.gt_boxes[m*4],y1=b.gt_boxes[m*4+1],x2=b.gt_boxes[m*4+2],y2=b.gt_boxes[m*4+3];
          gcx.insert(gcx.end(), {(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1}); gc.push_back(b.gt_labels[m]); }
        prov.i=0; auto dev = dnetx_forward(dfrom({1,3,S,S}, b.x->data), prov, bd, dw, true);   // DEVICE
        std::vector<Tensor> heads_cpu;
        for (auto& h : dev) heads_cpu.push_back(from_data({h->shape[0],h->shape[1],h->shape[2],h->shape[3]}, dto_host(h), true));
        auto L = yolox_loss(heads_cpu, xs, ys, st, gcx, gc, A, NC, (int64_t)gc.size());   // TRUSTED host
        backward(L.total);
        for (size_t l=0;l<dev.size();++l) { for (auto& v : heads_cpu[l]->grad) v *= inv;   // mean scale
          thrust::copy(heads_cpu[l]->grad.begin(), heads_cpu[l]->grad.end(), dev[l]->grad.begin()); }
        dbackward_from(dev);                                    // DEVICE backward through the net
        eloss += L.total->data[0]; ++nimg;
      }
      opt.step();
    }
    bk::sync();
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    double avg = eloss/std::max(1,nimg);
    dnetx_save(prov, arch, "last.pt");
    if (avg < best) { best = avg; dnetx_save(prov, arch, "best.pt"); }
    printf("epoch %d/%d  loss %.4f  %.1f s/epoch%s\n", ep+1, EPOCHS, avg, secs, avg<=best?"  *best*":"");
  }
  printf("done. best loss %.4f. wrote last.pt / best.pt (pure C++, %s)\n", best, model.c_str());
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)\n");
#else
  printf("backend: CPU (host)\n");
#endif
  return 0;
}
