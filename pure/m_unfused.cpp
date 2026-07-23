// A-1 (yolox) step 1: verify the unfused (conv+BN separate) forward vs YOLOX.
#include "net_yolox_unfused.hpp"
#include <cstdio>
int main() {
  const std::string D = "pure/ref/data_unf/";
  int64_t IMG; { std::ifstream f(D+"io.txt"); f>>IMG; }
  auto prov = load_unfused(D);
  auto x = from_data({1,3,IMG,IMG}, rd(D+"x.bin"));
  auto outs = yolox_forward_unfused(x, prov, /*training=*/false);
  double worst = 0;
  for (size_t i=0;i<outs.size();++i){ auto ref=rd(D+"ref_L"+std::to_string(i)+".bin");
    double d=0; for(int64_t j=0;j<outs[i]->numel();++j) d=std::max(d,(double)std::abs(outs[i]->data[j]-ref[j]));
    printf("L%zu max|diff|=%.3e %s\n", i, d, d<1e-3?"OK":"FAIL"); worst=std::max(worst,d); }
  printf("\n%s\n", worst<1e-3 ? "yolox unfused == yolox-tiny forward" : "MISMATCH");
  return worst<1e-3?0:1;
}
