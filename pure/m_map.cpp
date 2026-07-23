// yolox: verify the pure COCO-mAP against pycocotools on a synthetic detection set.
#include "metrics.hpp"
#include <cstdio>
#include <fstream>

int main() {
  const std::string D = "pure/ref/data_map/";
  std::ifstream f(D + "set.txt");
  if (!f) { printf("run: python pure/ref/map_ref.py\n"); return 1; }
  int NIMG, NCLS; f >> NIMG >> NCLS;
  std::vector<mapeval::Image> imgs(NIMG);
  for (int i = 0; i < NIMG; ++i) {
    int G; f >> G;
    for (int j = 0; j < G; ++j) { mapeval::GT g; f >> g.cls >> g.x1 >> g.y1 >> g.x2 >> g.y2; imgs[i].gts.push_back(g); }
    int Dn; f >> Dn;
    for (int j = 0; j < Dn; ++j) { mapeval::DT d; f >> d.cls >> d.x1 >> d.y1 >> d.x2 >> d.y2 >> d.score; imgs[i].dts.push_back(d); }
  }

  auto [map50, map5095] = mapeval::coco_map(imgs);
  std::ifstream rf(D + "ref.txt"); double r50, r5095; rf >> r50 >> r5095;

  printf("mAP@0.50      pure=%.6f  pycocotools=%.6f  |diff|=%.2e\n", map50, r50, std::abs(map50 - r50));
  printf("mAP@0.50:0.95 pure=%.6f  pycocotools=%.6f  |diff|=%.2e\n", map5095, r5095, std::abs(map5095 - r5095));
  bool ok = std::abs(map50 - r50) < 1e-3 && std::abs(map5095 - r5095) < 1e-3;
  printf("\n%s\n", ok ? "yolox: PURE mAP == pycocotools" : "MISMATCH");
  return ok ? 0 : 1;
}
