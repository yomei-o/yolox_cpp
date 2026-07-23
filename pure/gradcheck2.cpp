// Gradient checks for the spatial ops (conv2d, maxpool, upsample, concat, slice).
#include "autograd.hpp"
#include <random>
#include <cstdio>
#include <string>

static std::mt19937 rng(7);
static void randn(Tensor& t) { std::normal_distribution<float> nd(0, 1); for (auto& v : t->data) v = nd(rng); }
static Tensor R(std::vector<int64_t> s) { auto t = make_tensor(s, true); randn(t); return t; }

static bool gradcheck(const std::string& name, std::function<Tensor()> build,
                      std::vector<Tensor> inputs, float eps = 5e-3f) {
  auto out = build(); backward(out);
  std::vector<std::vector<float>> ana;
  for (auto& x : inputs) ana.push_back(x->grad);
  double maxerr = 0, maxviol = 0;
  for (size_t k = 0; k < inputs.size(); ++k) {
    auto& x = inputs[k];
    for (int64_t i = 0; i < x->numel(); ++i) {
      float o = x->data[i];
      x->data[i] = o + eps; float fp = build()->data[0];
      x->data[i] = o - eps; float fm = build()->data[0];
      x->data[i] = o;
      float num = (fp - fm) / (2 * eps);
      double err = std::abs(num - ana[k][i]);
      double tol = 5e-3 + 1e-2 * std::abs(ana[k][i]);
      maxerr = std::max(maxerr, err); maxviol = std::max(maxviol, err - tol);
    }
  }
  bool pass = maxviol <= 0;
  printf("[%-28s] max |num-ana| = %.3e  %s\n", name.c_str(), maxerr, pass ? "OK" : "FAIL");
  return pass;
}

int main() {
  bool ok = true;
  {   // conv 3x3 s1 p1 with bias
    auto in = R({1, 2, 5, 5}), w = R({3, 2, 3, 3}), b = R({3});
    ok &= gradcheck("conv3x3 s1 p1", [&] { return sum(conv2d(in, w, b, 1, 1)); }, {in, w, b});
  }
  {   // conv 3x3 stride 2, then silu
    auto in = R({1, 2, 6, 6}), w = R({4, 2, 3, 3}), b = R({4});
    ok &= gradcheck("silu(conv s2)", [&] { return sum(silu(conv2d(in, w, b, 2, 1))); }, {in, w, b});
  }
  {   // 1x1 conv
    auto in = R({2, 3, 4, 4}), w = R({5, 3, 1, 1}), b = R({5});
    ok &= gradcheck("conv1x1", [&] { return sum(conv2d(in, w, b, 1, 0)); }, {in, w, b});
  }
  {   // maxpool k5 s1 p2 (SPPF)
    auto in = R({1, 3, 6, 6});
    ok &= gradcheck("maxpool k5 s1 p2", [&] { return sum(maxpool2d(in, 5, 1, 2)); }, {in});
  }
  {   // upsample x2
    auto in = R({1, 2, 3, 3});
    ok &= gradcheck("upsample x2", [&] { return sum(mul(upsample_nearest(in, 2), upsample_nearest(in, 2))); }, {in});
  }
  {   // concat then slice
    auto a = R({1, 2, 4, 4}), b = R({1, 3, 4, 4});
    ok &= gradcheck("concat+slice", [&] {
      auto c = concat_ch({a, b});           // (1,5,4,4)
      return sum(mul(slice_ch(c, 1, 4), slice_ch(c, 1, 4)));
    }, {a, b});
  }
  {   // Focus (space-to-depth), YOLOX stem
    auto a = R({1, 3, 6, 6});
    ok &= gradcheck("focus (space-to-depth)", [&] { auto f = focus(a); return sum(mul(f, f)); }, {a});
  }
  {   // a small residual conv block: silu(conv) + in-ish
    auto in = R({1, 3, 5, 5}), w = R({3, 3, 3, 3}), b = R({3});
    ok &= gradcheck("residual conv block", [&] {
      return sum(add(silu(conv2d(in, w, b, 1, 1)), in));
    }, {in, w, b});
  }
  printf("\n%s\n", ok ? "ALL SPATIAL GRADCHECKS PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
