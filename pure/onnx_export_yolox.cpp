// ONNX export (yolox): write the yolox-tiny forward to a standard .onnx (opset 13)
// from fused weights, with a hand-rolled protobuf writer (no deps). Focus is emitted
// as 4 strided Slice + Concat. Outputs the 3 raw per-level tensors [reg,obj,cls].
//   run from repo root: onnx_export_yolox   (needs pure/ref/data_net/)
#include "net_yolox.hpp"    // Provider/ConvW, load_net, rd
#include "onnx.hpp"
using namespace onx;

int main() {
  const std::string D = "pure/ref/data_net/";
  auto prov = load_net(D);
  int64_t IMG, BD; { std::ifstream f(D+"io.txt"); f >> IMG >> BD; }
  const int64_t NC = 80, RM = 16;   // yolox reg=4 raw (no DFL); 85 = 4+1+80

  Graph g; g.opset = 13;
  g.inputs.push_back({"images", {1, 3, IMG, IMG}});
  int uid = 0;
  auto uniq = [&](const char* p){ return std::string(p)+std::to_string(uid++); };

  auto conv = [&](const std::string& in) -> std::string {
    ConvW& c = prov.next();
    int64_t Co=c.w->shape[0], Ci=c.w->shape[1], k=c.w->shape[2], pad=k/2;
    std::string wn=uniq("w"), bn=uniq("b"), yn=uniq("conv");
    g.init_f.push_back({wn,{Co,Ci,k,k},c.w->data}); g.init_f.push_back({bn,{Co},c.b->data});
    onx::Node n{"Conv",yn,{in,wn,bn},{yn},{}};
    n.attr.push_back({"kernel_shape",A_INTS,0,0,"",{k,k},{}});
    n.attr.push_back({"strides",A_INTS,0,0,"",{c.stride,c.stride},{}});
    n.attr.push_back({"pads",A_INTS,0,0,"",{pad,pad,pad,pad},{}});
    n.attr.push_back({"group",A_INT,1,0,"",{},{}});
    g.nodes.push_back(n);
    if (!c.act) return yn;
    std::string sn=uniq("sig"), mn=uniq("silu");
    g.nodes.push_back({"Sigmoid",sn,{yn},{sn},{}});
    g.nodes.push_back({"Mul",mn,{yn,sn},{mn},{}});
    return mn;
  };
  auto concat = [&](const std::vector<std::string>& xs) -> std::string {
    std::string yn=uniq("cat"); onx::Node n{"Concat",yn,xs,{yn},{}};
    n.attr.push_back({"axis",A_INT,1,0,"",{},{}}); g.nodes.push_back(n); return yn;
  };
  auto slice_ch_ = [&](const std::string& in, int64_t c0, int64_t c1) -> std::string {
    std::string s=uniq("st"),e=uniq("en"),a=uniq("ax"),yn=uniq("sl");
    g.init_i.push_back({s,{1},{c0}}); g.init_i.push_back({e,{1},{c1}}); g.init_i.push_back({a,{1},{1}});
    g.nodes.push_back({"Slice",yn,{in,s,e,a},{yn},{}}); return yn;
  };
  auto add = [&](const std::string& x,const std::string& y){ std::string yn=uniq("add"); g.nodes.push_back({"Add",yn,{x,y},{yn},{}}); return yn; };
  auto maxpool = [&](const std::string& in, int64_t k) -> std::string {
    std::string yn=uniq("mp"); onx::Node n{"MaxPool",yn,{in},{yn},{}};
    n.attr.push_back({"kernel_shape",A_INTS,0,0,"",{k,k},{}});
    n.attr.push_back({"strides",A_INTS,0,0,"",{1,1},{}});
    n.attr.push_back({"pads",A_INTS,0,0,"",{k/2,k/2,k/2,k/2},{}});
    g.nodes.push_back(n); return yn;
  };
  auto resize2x = [&](const std::string& in) -> std::string {
    std::string sc=uniq("scales"), yn=uniq("up");
    g.init_f.push_back({sc,{4},{1.f,1.f,2.f,2.f}});
    onx::Node n{"Resize",yn,{in,"",sc},{yn},{}};
    n.attr.push_back({"mode",A_STRING,0,0,"nearest",{},{}});
    n.attr.push_back({"coordinate_transformation_mode",A_STRING,0,0,"asymmetric",{},{}});
    n.attr.push_back({"nearest_mode",A_STRING,0,0,"floor",{},{}});
    g.nodes.push_back(n); return yn;
  };
  // Focus: 4 strided slices (axes 2,3 step 2) + concat [tl,bl,tr,br].
  auto focus = [&](const std::string& in) -> std::string {
    const int64_t BIG = 1<<30;
    auto sl = [&](int64_t hs, int64_t ws) -> std::string {
      std::string s=uniq("fs"),e=uniq("fe"),a=uniq("fa"),st=uniq("fst"),yn=uniq("fsl");
      g.init_i.push_back({s,{2},{hs,ws}}); g.init_i.push_back({e,{2},{BIG,BIG}});
      g.init_i.push_back({a,{2},{2,3}});   g.init_i.push_back({st,{2},{2,2}});
      g.nodes.push_back({"Slice",yn,{in,s,e,a,st},{yn},{}}); return yn;
    };
    return concat({sl(0,0), sl(1,0), sl(0,1), sl(1,1)});
  };
  auto csp = [&](const std::string& x, int64_t n, bool sc) -> std::string {
    std::string a=conv(x), b=conv(x);
    for (int64_t i=0;i<n;++i){ std::string h=conv(a); h=conv(h); a = sc? add(h,a): h; }
    return conv(concat({a,b}));
  };
  auto spp = [&](const std::string& x) -> std::string {
    std::string x1=conv(x), p5=maxpool(x1,5), p9=maxpool(x1,9), p13=maxpool(x1,13);
    return conv(concat({x1,p5,p9,p13}));
  };

  // topology (mirror net_yolox.hpp)
  std::string x = conv(focus("images"));
  x = conv(x); x = csp(x,BD,true);
  x = conv(x); std::string c3 = csp(x,3*BD,true);
  x = conv(c3); std::string c4 = csp(x,3*BD,true);
  x = conv(c4); x = spp(x); std::string c5 = csp(x,BD,false);
  std::string fpn0 = conv(c5);
  std::string u = resize2x(fpn0);
  std::string p4 = csp(concat({u,c4}),BD,false);
  std::string red = conv(p4);
  std::string u2 = resize2x(red);
  std::string pan2 = csp(concat({u2,c3}),BD,false);
  std::string d0 = conv(pan2);
  std::string pan1 = csp(concat({d0,red}),BD,false);
  std::string d1 = conv(pan1);
  std::string pan0 = csp(concat({d1,fpn0}),BD,false);
  std::string pans[3] = {pan2, pan1, pan0};
  for (int i=0;i<3;++i) {
    std::string st=conv(pans[i]);
    std::string cf=conv(conv(st)), rf=conv(conv(st));
    std::string cls=conv(cf), reg=conv(rf), obj=conv(rf);
    std::string on = "out"+std::to_string(i);
    g.nodes.push_back({"Concat",uniq("oc"),{reg,obj,cls},{on},{}});   // [reg,obj,cls]=85
    g.nodes.back().attr.push_back({"axis",A_INT,1,0,"",{},{}});
    int64_t s = 8<<i;
    g.outputs.push_back({on,{1,4+1+NC,IMG/s,IMG/s}});
  }
  save_onnx(g, "yolox_tiny.onnx");
  printf("wrote yolox_tiny.onnx  (%zu nodes, consumed %zu/%zu convs)\n", g.nodes.size(), prov.i, prov.convs.size());
  return 0;
}
