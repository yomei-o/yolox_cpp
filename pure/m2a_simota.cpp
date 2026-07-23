// M2a: verify the pure-C++ SimOTA against YOLOX's get_assignments.
#include "simota.hpp"
#include "net_yolox.hpp"   // rd()
#include <cstdio>
#include <fstream>

int main() {
  const std::string D = "pure/ref/data_loss/";
  int64_t IMG, A, NC, G, NF; { std::ifstream f(D+"meta.txt"); f>>IMG>>A>>NC>>G>>NF; }
  auto bbox = rd(D+"bbox_preds.bin"), cls = rd(D+"cls_preds.bin"), obj = rd(D+"obj_preds.bin");
  auto xs = rd(D+"xs.bin"), ys = rd(D+"ys.bin"), st = rd(D+"strides.bin");
  auto gtb = rd(D+"gt_boxes.bin"); auto gtcf = rd(D+"gt_cls.bin");
  std::vector<int64_t> gtc(G); for (int64_t i=0;i<G;++i) gtc[i]=(int64_t)gtcf[i];

  auto r = simota_assign(bbox, cls, obj, xs, ys, st, gtb, gtc, A, NC, G);

  auto rfg = rd(D+"fg_mask.bin"), rmg = rd(D+"matched_gt.bin"), rmc = rd(D+"gt_matched_cls.bin"), rpi = rd(D+"pred_ious.bin");
  int fgdiff=0; for (int64_t a=0;a<A;++a) if ((r.fg[a]>0.5)!=(rfg[a]>0.5)) fgdiff++;
  bool nfok = (r.num_fg==NF);
  int mgd=0, mcd=0; double pid=0;
  if (nfok) for (int64_t k=0;k<NF;++k){ mgd+=(r.matched_gt[k]!=(int64_t)rmg[k]); mcd+=(r.matched_cls[k]!=(int64_t)rmc[k]); pid=std::max(pid,(double)std::abs(r.pred_ious[k]-rpi[k])); }
  printf("num_fg cpp=%lld ref=%lld  %s\n", (long long)r.num_fg,(long long)NF, nfok?"OK":"FAIL");
  printf("fg_mask mismatches: %d\n", fgdiff);
  printf("matched_gt mismatches: %d  matched_cls: %d  pred_ious max|diff|=%.3e\n", mgd, mcd, pid);
  bool ok = nfok && fgdiff==0 && mgd==0 && mcd==0 && pid<1e-4;
  printf("\n%s\n", ok ? "M2a: pure SimOTA == YOLOX get_assignments" : "MISMATCH");
  return ok?0:1;
}
