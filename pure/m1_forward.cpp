// M1: full yolox-tiny forward in the pure engine, matched to the real net (per level).
#include "net_yolox.hpp"
#include <cstdio>

int main() {
  const std::string D = "pure/ref/data_net/";
  int64_t IMG, BD, DW = 0; { std::ifstream f(D + "io.txt"); f >> IMG >> BD >> DW; }
  auto prov = load_net(D);
  auto x = from_data({1, 3, IMG, IMG}, rd(D + "x.bin"));

  auto outs = yolox_forward(x, prov, BD, (bool)DW);
  printf("consumed %zu/%zu convs, %zu levels\n", prov.i, prov.convs.size(), outs.size());

  double worst = 0;
  for (size_t i = 0; i < outs.size(); ++i) {
    auto ref = rd(D + "ref_L" + std::to_string(i) + ".bin");
    double d = 0; for (int64_t j = 0; j < outs[i]->numel(); ++j)
      d = std::max(d, (double)std::abs(outs[i]->data[j] - ref[j]));
    printf("L%zu %lldx%lldx%lld  max|diff| = %.3e  %s\n", i,
           (long long)outs[i]->shape[1], (long long)outs[i]->shape[2], (long long)outs[i]->shape[3],
           d, d < 1e-3 ? "OK" : "FAIL");
    worst = std::max(worst, d);
  }
  bool ok = worst < 1e-3;
  printf("\n%s\n", ok ? "yolox M1: PURE ENGINE == yolox-tiny forward" : "MISMATCH");
  return ok ? 0 : 1;
}
