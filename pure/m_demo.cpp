// YOLOX real-image demo: load an image, YOLOX preproc (letterbox pad 114, BGR, 0-255),
// forward with shipped weights, decode + NMS, draw boxes. No Python, no libs but stb.
//   g++ -std=c++20 -O2 -fopenmp -Ipure/third_party pure/m_demo.cpp -o m_demo
//   ./m_demo assets/bus.jpg out.png 640
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "net_yolox.hpp"
#include "infer_yolox.hpp"
#include <cstdio>

static const char* COCO[80] = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

int main(int argc, char** argv) {
  const char* inp = argc > 1 ? argv[1] : "assets/bus.jpg";
  const char* outp = argc > 2 ? argv[2] : "out.png";
  int S = argc > 3 ? atoi(argv[3]) : 640;

  int W, H, ch; unsigned char* im = stbi_load(inp, &W, &H, &ch, 3);
  if (!im) { printf("cannot load %s\n", inp); return 1; }
  printf("loaded %s  %dx%d\n", inp, W, H);
  float r = std::min((float)S / H, (float)S / W);
  int rw = (int)(W * r), rh = (int)(H * r);

  // preproc: bilinear resize -> pad 114 (top-left) -> BGR, 0-255, CHW
  auto x = make_tensor({1, 3, S, S});
  for (auto& v : x->data) v = 114.f;
  for (int y = 0; y < rh; ++y) for (int xx = 0; xx < rw; ++xx) {
    float sy = (y + 0.5f) / r - 0.5f, sx = (xx + 0.5f) / r - 0.5f;
    int y0 = (int)std::floor(sy), x0 = (int)std::floor(sx);
    float fy = sy - y0, fx = sx - x0;
    auto px = [&](int yy, int xx2, int c) { yy = std::min(std::max(yy,0),H-1); xx2 = std::min(std::max(xx2,0),W-1);
      return (float)im[(yy*W + xx2)*3 + c]; };
    for (int c = 0; c < 3; ++c) {
      float v = (1-fy)*((1-fx)*px(y0,x0,c) + fx*px(y0,x0+1,c)) + fy*((1-fx)*px(y0+1,x0,c) + fx*px(y0+1,x0+1,c));
      int bgr = 2 - c;                                   // RGB -> BGR
      x->data[(bgr*S + y)*S + xx] = v;
    }
  }

  const std::string D = "weights/yolox_tiny/";
  int64_t BD = 1, DW = 0; { std::ifstream f(D + "io.txt"); int64_t im0; f >> im0 >> BD >> DW; }
  auto prov = load_net_blob(D);
  auto raw = yolox_forward(x, prov, BD, (bool)DW);
  auto dets = yolox_detect(raw, {8,16,32}, 80, 0.25f, 0.45f);

  printf("%zu detections:\n", dets.size());
  for (auto& d : dets) {
    printf("  %-12s conf=%.2f  xyxy=(%.0f,%.0f,%.0f,%.0f)\n",
           COCO[d.cls], d.score, d.x1/r, d.y1/r, d.x2/r, d.y2/r);
    // draw on original image (scale box back by 1/r), 2px box
    int X1=(int)(d.x1/r), Y1=(int)(d.y1/r), X2=(int)(d.x2/r), Y2=(int)(d.y2/r);
    for (int t = 0; t < 2; ++t) for (int xx=X1; xx<=X2; ++xx) { for (int yy : {Y1+t, Y2-t}) if (yy>=0&&yy<H&&xx>=0&&xx<W){ im[(yy*W+xx)*3]=0; im[(yy*W+xx)*3+1]=255; im[(yy*W+xx)*3+2]=0; } }
    for (int t = 0; t < 2; ++t) for (int yy=Y1; yy<=Y2; ++yy) { for (int xx : {X1+t, X2-t}) if (yy>=0&&yy<H&&xx>=0&&xx<W){ im[(yy*W+xx)*3]=0; im[(yy*W+xx)*3+1]=255; im[(yy*W+xx)*3+2]=0; } }
  }
  stbi_write_png(outp, W, H, 3, im, W*3);
  printf("wrote %s\n", outp);
  stbi_image_free(im);
  return 0;
}
