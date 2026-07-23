// A-2 (read): a graph-driven interpreter that executes a parsed ONNX Graph using the
// pure engine's ops — no hardcoded architecture. Supports the op subset yolov8 uses:
// Conv, Sigmoid, Mul, Add, MaxPool, Resize (nearest), Concat, Slice, Identity.
#pragma once
#include "onnx.hpp"
#include "autograd.hpp"
#include "ops2d.hpp"        // mul (elementwise)
#include <map>
#include <string>

namespace onx {

inline const Attr* find_attr(const Node& n, const std::string& name) {
  for (auto& a : n.attr) if (a.name == name) return &a; return nullptr;
}
inline int64_t attr_i0(const Node& n, const std::string& name, int64_t def) {
  const Attr* a = find_attr(n, name);
  if (!a) return def; return a->ints.empty() ? a->i : a->ints[0];
}

// Run the graph on input x; returns name -> Tensor for every produced value.
inline std::map<std::string, Tensor> run_onnx(const Graph& g, const Tensor& x) {
  std::map<std::string, Tensor> vals;
  std::map<std::string, const IntsTensor*> imap;
  for (const auto& t : g.init_f) vals[t.name] = from_data(t.dims, t.data);
  for (const auto& t : g.init_i) imap[t.name] = &t;

  // the graph input is the declared input that isn't an initializer
  std::string in_name;
  for (const auto& vi : g.inputs) if (!vals.count(vi.name)) { in_name = vi.name; break; }
  if (in_name.empty() && !g.inputs.empty()) in_name = g.inputs[0].name;
  vals[in_name] = x;

  auto get = [&](const std::string& n) -> Tensor { return vals.at(n); };

  for (const auto& nd : g.nodes) {
    const std::string& op = nd.op_type;
    Tensor y;
    if (op == "Conv") {
      Tensor w = get(nd.input[1]);
      Tensor b = (nd.input.size() >= 3 && !nd.input[2].empty()) ? get(nd.input[2]) : nullptr;
      int64_t stride = attr_i0(nd, "strides", 1), pad = attr_i0(nd, "pads", 0);
      y = conv2d(get(nd.input[0]), w, b, stride, pad);
    } else if (op == "Sigmoid") {
      y = sigmoid(get(nd.input[0]));
    } else if (op == "Mul") {
      y = mul(get(nd.input[0]), get(nd.input[1]));
    } else if (op == "Add") {
      y = add(get(nd.input[0]), get(nd.input[1]));
    } else if (op == "MaxPool") {
      int64_t k = attr_i0(nd, "kernel_shape", 1), stride = attr_i0(nd, "strides", 1), pad = attr_i0(nd, "pads", 0);
      y = maxpool2d(get(nd.input[0]), k, stride, pad);
    } else if (op == "Resize") {
      const IntsTensor* it = nullptr;                       // scales is a float init
      Tensor sc = vals.count(nd.input.back()) ? get(nd.input.back()) : nullptr;
      int64_t f = sc ? (int64_t)sc->data[2] : 2;            // spatial scale factor
      y = upsample_nearest(get(nd.input[0]), f);
      (void)it;
    } else if (op == "Concat") {
      std::vector<Tensor> xs; for (auto& s : nd.input) xs.push_back(get(s));
      y = concat_ch(xs);
    } else if (op == "Slice") {
      auto* st = imap.at(nd.input[1]); auto* en = imap.at(nd.input[2]);
      const IntsTensor* ax = nd.input.size() > 3 ? imap.at(nd.input[3]) : nullptr;
      const IntsTensor* stp = nd.input.size() > 4 ? imap.at(nd.input[4]) : nullptr;
      Tensor xin = get(nd.input[0]);
      if (ax && ax->data.size() == 2 && ax->data[0] == 2 && ax->data[1] == 3) {
        int64_t hstep = stp ? stp->data[0] : 1, wstep = stp ? stp->data[1] : 1;   // Focus strided slice
        y = slice_hw(xin, st->data[0], st->data[1], hstep, wstep);
      } else {
        int64_t c0 = st->data[0], c1 = en->data[0], C = xin->shape[1]; if (c1 > C) c1 = C;
        y = slice_ch(xin, c0, c1);
      }
    } else if (op == "Identity") {
      y = get(nd.input[0]);
    } else {
      printf("unsupported ONNX op: %s\n", op.c_str()); std::exit(1);
    }
    vals[nd.output[0]] = y;
  }
  return vals;
}

}  // namespace onx
