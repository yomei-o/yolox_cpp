// Debug: localize where the pure forward diverges from yolox-tiny (stem, c3, c4, c5).
#include "net_yolox.hpp"
#include <cstdio>
static double cmp(const Tensor& t, const std::string& ref) {
  auto r = rd(ref); double d = 0;
  for (int64_t j = 0; j < t->numel(); ++j) d = std::max(d, (double)std::abs(t->data[j] - r[j]));
  return d;
}
int main() {
  const std::string D = "pure/ref/data_net/";
  int64_t IMG; { std::ifstream f(D + "io.txt"); f >> IMG; }
  auto prov = load_net(D);
  auto x = from_data({1, 3, IMG, IMG}, rd(D + "x.bin"));

  auto stem = bc(focus(x), prov);
  printf("stem  %lldx%lldx%lld  diff=%.3e\n", (long long)stem->shape[1],(long long)stem->shape[2],(long long)stem->shape[3], cmp(stem, D+"dbg_stem.bin"));
  auto d2 = bc(stem, prov); d2 = csp(d2, prov, 1, true);          // dark2
  auto x3 = bc(d2, prov); auto c3 = csp(x3, prov, 3, true);       // dark3
  printf("c3    diff=%.3e\n", cmp(c3, D+"dbg_dark3.bin"));
  auto x4 = bc(c3, prov); auto c4 = csp(x4, prov, 3, true);       // dark4
  printf("c4    diff=%.3e\n", cmp(c4, D+"dbg_dark4.bin"));
  auto d50 = bc(c4, prov);                                        // dark5.0 downsample
  printf("d50   %lldx%lldx%lld  diff=%.3e\n", (long long)d50->shape[1],(long long)d50->shape[2],(long long)d50->shape[3], cmp(d50, D+"dbg_d50.bin"));
  auto d51 = spp(d50, prov);                                      // SPP
  printf("spp   diff=%.3e\n", cmp(d51, D+"dbg_d51.bin"));
  auto c5 = csp(d51, prov, 1, true);                              // dark5 CSP
  printf("c5    diff=%.3e\n", cmp(c5, D+"dbg_dark5.bin"));
  return 0;
}
