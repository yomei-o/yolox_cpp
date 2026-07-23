#include "ptio.hpp"
#include <cstdio>
int main(){
  std::vector<pt::Tensor> ts;
  { pt::Tensor a; a.name="a.weight"; a.shape={2,3}; for(int i=0;i<6;++i)a.data.push_back(i*1.5f-2); ts.push_back(a); }
  { pt::Tensor b; b.name="a.bias"; b.shape={2}; b.data={0.25f,-7.5f}; ts.push_back(b); }
  { pt::Tensor c; c.name="deep.conv.weight"; c.shape={4,2,1,1}; for(int i=0;i<8;++i)c.data.push_back(i*0.1f); ts.push_back(c); }
  pt::save_pt(ts, "cpp_out.pt");
  auto rd = pt::load_pt("cpp_out.pt");
  printf("wrote+reread %zu tensors:\n", rd.size());
  double m=0; for(size_t i=0;i<rd.size();++i){ double d=0; for(size_t k=0;k<rd[i].data.size();++k) d=std::max(d,(double)std::abs(rd[i].data[k]-ts[i].data[k])); m=std::max(m,d);
    printf("  %-18s shape[%zu] selfdiff %.1e\n", rd[i].name.c_str(), rd[i].shape.size(), d); }
  printf("C++ self round-trip max|diff| = %.1e  %s\n", m, m==0?"OK":"FAIL");
  return 0;
}
