// ONNX import (yolox): parse yolox_tiny.onnx and run it graph-driven in the pure
// engine (no hardcoded arch), compare to the reference forward.
#include "onnx_run.hpp"
#include "net_yolox.hpp"   // rd()
#include <cstdio>
using namespace onx;

int main() {
  const std::string D = "pure/ref/data_net/";
  int64_t IMG; { std::ifstream f(D+"io.txt"); f >> IMG; }
  auto g = load_onnx("yolox_tiny.onnx");
  auto x = from_data({1,3,IMG,IMG}, rd(D+"x.bin"));
  auto vals = run_onnx(g, x);
  double worst = 0;
  for (int i=0;i<3;++i) {
    auto y = vals.at("out"+std::to_string(i));
    auto ref = rd(D+"ref_L"+std::to_string(i)+".bin");
    double d=0; for (int64_t j=0;j<y->numel();++j) d=std::max(d,(double)std::abs(y->data[j]-ref[j]));
    printf("out%d %lldx%lldx%lld  max|diff|=%.3e\n", i,(long long)y->shape[1],(long long)y->shape[2],(long long)y->shape[3], d);
    worst=std::max(worst,d);
  }
  printf("\n%s  onnx-import (pure engine runs .onnx) == reference (worst %.3e)\n", worst<1e-3?"OK":"FAIL", worst);
  return worst<1e-3?0:1;
}
