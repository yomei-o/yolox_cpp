// Depthwise/grouped conv grad-check (the path yolox-nano uses). L=sum(SiLU(dwconv(X,W,groups=C))).
#include "dtensor.hpp"
#include <cstdio><cmath><numeric>
static std::vector<float> ramp(int64_t n,float a,float s){std::vector<float>v(n);for(int64_t i=0;i<n;++i)v[i]=a+s*i;return v;}
static std::vector<int64_t> IS{1,4,6,6}, WS{4,1,3,3};   // depthwise: Co=4, Ci/g=1, groups=4
static std::vector<float> XH=ramp(1*4*6*6,-0.8f,0.05f), WH=ramp(4*1*3*3,-0.4f,0.06f);
static float fwd(const std::vector<float>&xh,const std::vector<float>&wh){ DT x=dfrom(IS,xh),w=dfrom(WS,wh); return dto_host(dsum(dsilu(dconv2d(x,w,DT(),1,1,4))))[0]; }
int main(){
  DT x=dfrom(IS,XH),w=dfrom(WS,WH); DT L=dsum(dsilu(dconv2d(x,w,DT(),1,1,4))); dbackward(L);
  float Lv=dto_host(L)[0]; std::vector<float> wg(w->numel()); thrust::copy(w->grad.begin(),w->grad.end(),wg.begin());
  printf("L=%.6f  (groups=4 depthwise)\n",Lv); const float eps=1e-3f; int bad=0;
  for(int i:{0,5,17,35}){ if(i>=(int)WH.size())continue; auto wp=WH; wp[i]+=eps; float fd=(fwd(XH,wp)-Lv)/eps;
    printf("  dW[%2d] analytic %.5f fd %.5f |d| %.2e\n",i,wg[i],fd,std::abs(wg[i]-fd)); if(std::abs(wg[i]-fd) > 0.02f*std::abs(wg[i])+2e-3f)++bad; }
  printf("depthwise grad-check %s\n", bad?"MISMATCH":"OK");
  return 0;
}
