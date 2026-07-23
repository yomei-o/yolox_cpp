// Make an initial-weights .pt ENTIRELY in C++ — no Python, no libtorch, no YOLOX repo.
// The model architecture comes from two tiny text files (data_unf/manifest_unfused.txt +
// names.txt, a few KB) that describe every layer's shape and state_dict key; this program
// allocates each tensor, initialises it, and writes a state_dict .pt via the pure-C++
// writer. The result loads into the YOLOX reference model (load_state_dict) and is
// trainable by train_cli.
//   build: cl /std:c++20 /O2 /EHsc pure/make_init_pt.cpp   (or via cc11.sh)
//   run:   make_init_pt [out.pt] [rand|from] [pretrained.pth]
//     rand (default): He/Kaiming random init from the manifest — fully self-contained.
//     from : copy values from a pretrained .pth read in C++ by ptio (handles a plain
//            state_dict AND a Megvii checkpoint {'model': OrderedDict[str->Tensor], ...}).
#include "ptio.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <string>
#include <vector>
#include <map>

int main(int argc, char** argv) {
  std::string out  = argc > 1 ? argv[1] : "init.pt";
  std::string mode = argc > 2 ? argv[2] : "rand";
  std::string pre  = argc > 3 ? argv[3] : "yolox_tiny.pth";
  const std::string DU = "pure/ref/data_unf/";

  // architecture: header line = layer count, then "kind Co Ci k s eps g" per layer.
  std::ifstream mf(DU + "manifest_unfused.txt");
  if (!mf) { printf("cannot open %smanifest_unfused.txt (run export_unfused_yolox.py once)\n", DU.c_str()); return 1; }
  std::string h; std::getline(mf, h); int nlayers = std::stoi(h);
  std::vector<std::string> names; { std::ifstream nf(DU + "names.txt"); std::string s; while (nf >> s) names.push_back(s); }

  // "from": read pretrained weights (pure C++). Try a plain state_dict, else the YOLOX
  // {'model': state_dict} checkpoint, else a raw {'model': nn.Module} module.
  std::map<std::string, pt::Tensor> premap;
  if (mode == "from") {
    auto ts = pt::load_pt(pre);
    if (ts.empty()) ts = pt::load_pt_state_under(pre, "model");
    if (ts.empty()) ts = pt::load_pt_module(pre);
    for (auto& t : ts) premap[t.name] = t;
    printf("read %zu pretrained tensors from %s\n", premap.size(), pre.c_str());
  }

  std::mt19937 rng(1234); size_t ni = 0;
  std::vector<pt::Tensor> ck;
  auto emit = [&](const std::vector<int64_t>& shape, char role, double fan_in) {
    pt::Tensor t; t.shape = shape; if (ni < names.size()) t.name = names[ni];
    int64_t n = 1; for (auto d : shape) n *= d; t.data.resize(n);
    if (mode == "from" && premap.count(t.name)) { t.data = premap[t.name].data; }
    else switch (role) {                                   // random / analytic init
      case 'w': { double std = std::sqrt(2.0 / fan_in);    // conv weight: Kaiming (He) normal
                  std::normal_distribution<float> N(0.f, (float)std);
                  for (auto& v : t.data) v = N(rng); break; }
      case 'g': for (auto& v : t.data) v = 1.f; break;     // bn.weight (gamma)
      case 'v': for (auto& v : t.data) v = 1.f; break;     // bn.running_var
      default:  for (auto& v : t.data) v = 0.f; break;     // bn.bias / running_mean / conv bias
    }
    ck.push_back(t); ++ni;
  };

  for (int i = 0; i < nlayers; ++i) {
    std::string line; std::getline(mf, line); std::istringstream ss(line);
    int kind; int64_t Co, Ci, k, s, g = 1; double eps; ss >> kind >> Co >> Ci >> k >> s >> eps; ss >> g;
    double fan_in = (double)Ci * k * k;
    emit({Co, Ci, k, k}, 'w', fan_in);                     // conv weight
    if (kind == 1) { emit({Co}, 'g', 0); emit({Co}, 'b', 0); emit({Co}, 'b', 0); emit({Co}, 'v', 0); } // bn: w,b,rm,rv
    else emit({Co}, 'b', 0);                               // plain conv bias
  }

  pt::save_pt(ck, out);
  printf("wrote %s: %zu tensors (%d layers), mode=%s%s  [pure C++, no Python]\n",
         out.c_str(), ck.size(), nlayers, mode.c_str(), mode=="from"?(" <- "+pre).c_str():"");
  return 0;
}
