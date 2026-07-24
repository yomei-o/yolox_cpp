// YOLOX device forward parity vs CPU engine (yolox_forward_unfused), train-mode, tiny (bd=1,dw=false).
#include "net_yolox_unfused.hpp"
#include "dnet_yolox.hpp"
#include <cstdio>
#include <cmath>
static float md(const std::vector<float>& a, const Tensor& b){ float m=0; for(size_t i=0;i<a.size();++i) m=std::max(m,std::abs(a[i]-b->data[i])); return m; }
int main(){
  const std::string DU="pure/ref/data_unf/"; const int64_t S=64, bd=1; const bool dw=false;
  ProviderU pu = load_unfused_pt(DU, "yolox_tiny.pth");
  ProvDX pd = dnetx_build(DU, "yolox_tiny.pth");
  std::vector<float> xh(1*3*S*S); for(size_t i=0;i<xh.size();++i) xh[i]=std::sin(0.05f*i)*0.5f+0.1f;
  pu.i=0; auto cpu = yolox_forward_unfused(from_data({1,3,S,S},xh), pu, true, bd, dw);
  pd.i=0; auto dev = dnetx_forward(dfrom({1,3,S,S},xh), pd, bd, dw, true); bk::sync();
  float worst=0;
  for(int l=0;l<3;++l){ float d=md(dto_host(dev[l]), cpu[l]);
    printf("  level %d: head[%lldx%lldx%lld] max|d|=%.3e\n", l,(long long)cpu[l]->shape[1],(long long)cpu[l]->shape[2],(long long)cpu[l]->shape[3], d); worst=std::max(worst,d); }
  printf("YOLOX-tiny forward: worst |device - CPU-engine| = %.3e  %s\n", worst, worst<2e-3f?"MATCH":"MISMATCH");
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)\n");
#else
  printf("backend: CPU (host)\n");
#endif
  return 0;
}
