// Real data loader for training: load an image (stb_image), letterbox it to SxS, and
// load YOLO labels into ground-truth boxes/labels in that letterboxed image space
// (the units TAL expects). The including TU must define STB_IMAGE_IMPLEMENTATION once.
#pragma once
#include "autograd.hpp"
#include "stb_image.h"                  // add -Ipure/third_party
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>

struct Letterbox { float r; int left, top, w0, h0; };

// Load + letterbox into a TW x TH tile (grey 114, bilinear), RGB, /255 -> (1,3,TH,TW).
// Fills lb (scale r + left/top pad in the tile + original w0,h0). load_image_letterbox is
// the square TW=TH=S case used everywhere; the rectangular form is used by mosaic tiles.
inline Tensor load_image_letterbox_wh(const std::string& path, int64_t TW, int64_t TH, Letterbox& lb) {
  int w0, h0, ch;
  unsigned char* im = stbi_load(path.c_str(), &w0, &h0, &ch, 3);
  if (!im) { printf("cannot load %s\n", path.c_str()); std::exit(1); }
  float r = std::min((float)TW / w0, (float)TH / h0);
  int nw = (int)std::round(w0 * r), nh = (int)std::round(h0 * r);
  int left = (int)((TW - nw) / 2), top = (int)((TH - nh) / 2);
  auto x = make_tensor({1, 3, TH, TW});
  for (auto& v : x->data) v = 114.f / 255.f;
  for (int y = 0; y < nh; ++y)
    for (int xx = 0; xx < nw; ++xx) {
      float sx = (xx + 0.5f) / r - 0.5f, sy = (y + 0.5f) / r - 0.5f;
      int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
      float fx = sx - x0, fy = sy - y0;
      int x1 = std::min(x0 + 1, w0 - 1), y1 = std::min(y0 + 1, h0 - 1);
      x0 = std::clamp(x0, 0, w0 - 1); y0 = std::clamp(y0, 0, h0 - 1);
      for (int c = 0; c < 3; ++c) {
        float p00 = im[(y0 * w0 + x0) * 3 + c], p01 = im[(y0 * w0 + x1) * 3 + c];
        float p10 = im[(y1 * w0 + x0) * 3 + c], p11 = im[(y1 * w0 + x1) * 3 + c];
        float v = (p00 * (1 - fx) + p01 * fx) * (1 - fy) + (p10 * (1 - fx) + p11 * fx) * fy;
        x->data[(c * TH + top + y) * TW + left + xx] = v / 255.f;
      }
    }
  stbi_image_free(im);
  lb = {r, left, top, w0, h0};
  return x;
}
inline Tensor load_image_letterbox(const std::string& path, int64_t S, Letterbox& lb) {
  return load_image_letterbox_wh(path, S, S, lb);
}

// Labels file: one "cls x1 y1 x2 y2" per line, xyxy already in letterboxed image units.
// Returns M; fills gt_boxes (M*4) and gt_labels (M).
inline int load_labels(const std::string& path, std::vector<float>& gt_boxes, std::vector<int64_t>& gt_labels) {
  std::ifstream f(path);
  if (!f) { printf("cannot open %s\n", path.c_str()); std::exit(1); }
  gt_boxes.clear(); gt_labels.clear();
  int cls; float x1, y1, x2, y2; int M = 0;
  while (f >> cls >> x1 >> y1 >> x2 >> y2) {
    gt_labels.push_back(cls);
    gt_boxes.insert(gt_boxes.end(), {x1, y1, x2, y2});
    ++M;
  }
  return M;
}

// A padded batch for training: x (B,3,S,S), and GT padded to M = max labels per image,
// with mask (B*M) = 1 for real boxes and 0 for padding (the units TAL expects).
struct Batch {
  Tensor x; int64_t B = 0, M = 0;
  std::vector<float> gt_boxes;    // (B*M*4)
  std::vector<int64_t> gt_labels; // (B*M)
  std::vector<float> mask;        // (B*M)
};

// list.txt: first line "S N", then N lines "<image_path> <label_path>".
inline Batch load_batch(const std::string& list_path) {
  std::ifstream f(list_path);
  if (!f) { printf("cannot open %s (run pure/ref/make_labels.py)\n", list_path.c_str()); std::exit(1); }
  int64_t S, N; f >> S >> N;
  std::vector<std::string> imgs(N), lbls(N);
  for (int64_t i = 0; i < N; ++i) f >> imgs[i] >> lbls[i];

  std::vector<std::vector<float>> gbs(N); std::vector<std::vector<int64_t>> gls(N);
  int64_t M = 0;
  Batch bt; bt.B = N; bt.x = make_tensor({N, 3, S, S});
  for (int64_t n = 0; n < N; ++n) {
    Letterbox lb;
    auto xi = load_image_letterbox(imgs[n], S, lb);
    std::copy(xi->data.begin(), xi->data.end(), bt.x->data.begin() + n * 3 * S * S);
    int m = load_labels(lbls[n], gbs[n], gls[n]); M = std::max<int64_t>(M, m);
  }
  bt.M = M;
  bt.gt_boxes.assign(N * M * 4, 0.f); bt.gt_labels.assign(N * M, 0); bt.mask.assign(N * M, 0.f);
  for (int64_t n = 0; n < N; ++n)
    for (size_t m = 0; m < gls[n].size(); ++m) {
      bt.gt_labels[n * M + m] = gls[n][m]; bt.mask[n * M + m] = 1.f;
      for (int j = 0; j < 4; ++j) bt.gt_boxes[(n * M + m) * 4 + j] = gbs[n][m * 4 + j];
    }
  return bt;
}

// ------------------------- dataset (multi-epoch, minibatch, augmentation) -------------------------
#include <random>
struct Sample { std::string img, lbl; };
// yolo=true  -> labels are STANDARD YOLO "cls xc yc w h" normalised to [0,1] (Ultralytics
//               format); images may be any size (letterboxed to S, GT mapped accordingly).
// yolo=false -> legacy "cls x1 y1 x2 y2" pixel xyxy, square images (letterbox identity).
struct Dataset { int64_t S = 0; bool yolo = false; std::vector<Sample> items; };

// legacy list.txt: "S N" then N lines "<image> <label>".
inline Dataset read_dataset(const std::string& list_path) {
  std::ifstream f(list_path); if (!f) { printf("cannot open %s\n", list_path.c_str()); std::exit(1); }
  Dataset d; int64_t N; f >> d.S >> N; d.items.resize(N);
  for (auto& s : d.items) f >> s.img >> s.lbl;
  return d;
}

// Standard Ultralytics layout: an images directory whose sibling "labels" dir holds one
// "<stem>.txt" per image ("cls xc yc w h" normalised). Derives label paths by the usual
// /images/->/labels/ + .txt rule. `spec` is that images dir, OR a .txt file listing image
// paths (one per line). imgsz = target square S.
inline Dataset read_yolo_dataset(const std::string& spec, int64_t imgsz) {
  namespace fs = std::filesystem;
  Dataset d; d.S = imgsz; d.yolo = true;
  auto add = [&](const std::string& img) {
    std::string lbl = img; auto p = lbl.rfind("images");
    if (p != std::string::npos) lbl.replace(p, 6, "labels");
    auto dot = lbl.find_last_of('.'); if (dot != std::string::npos) lbl = lbl.substr(0, dot);
    lbl += ".txt"; d.items.push_back({img, lbl});
  };
  if (fs::is_directory(spec)) {
    for (auto& e : fs::directory_iterator(spec)) {
      if (!e.is_regular_file()) continue;
      auto ext = e.path().extension().string(); for (auto& c : ext) c = (char)tolower(c);
      if (ext==".jpg"||ext==".jpeg"||ext==".png"||ext==".bmp") add(e.path().string());
    }
  } else {
    std::ifstream f(spec); if (!f) { printf("cannot open %s\n", spec.c_str()); std::exit(1); }
    std::string line; while (std::getline(f, line)) { if (!line.empty() && line.back()=='\r') line.pop_back(); if (!line.empty()) add(line); }
  }
  std::sort(d.items.begin(), d.items.end(), [](const Sample& a, const Sample& b){ return a.img < b.img; });
  return d;
}

// Read one label file into ORIGINAL-image pixel xyxy (both formats). Missing file => 0
// boxes (a valid background/negative image). yolo needs the original (w0,h0).
inline int load_boxes_orig(const std::string& path, bool yolo, int w0, int h0,
                           std::vector<float>& gb, std::vector<int64_t>& gl) {
  std::ifstream f(path); gb.clear(); gl.clear(); if (!f) return 0; int M = 0;
  if (yolo) { int c; float xc, yc, w, h;
    while (f >> c >> xc >> yc >> w >> h) { gl.push_back(c);
      gb.insert(gb.end(), {(xc-w/2)*w0, (yc-h/2)*h0, (xc+w/2)*w0, (yc+h/2)*h0}); ++M; } }
  else { int c; float x1, y1, x2, y2;
    while (f >> c >> x1 >> y1 >> x2 >> y2) { gl.push_back(c); gb.insert(gb.end(), {x1,y1,x2,y2}); ++M; } }
  return M;
}
// Map original-pixel xyxy into a letterboxed tile: x' = x*r + left (+ tile offset ox,oy).
inline void lb_map(std::vector<float>& gb, const Letterbox& lb, float ox = 0, float oy = 0) {
  for (size_t i = 0; i < gb.size(); i += 4) {
    gb[i]   = gb[i]  *lb.r + lb.left + ox; gb[i+2] = gb[i+2]*lb.r + lb.left + ox;
    gb[i+1] = gb[i+1]*lb.r + lb.top  + oy; gb[i+3] = gb[i+3]*lb.r + lb.top  + oy;
  }
}

// Build one mosaic image: 4 images tiled into SxS at a random split (cx,cy); each tile is
// letterboxed into its quadrant and its GT mapped + clipped to that quadrant. Fills px
// (3*S*S) and the combined boxes/labels (SxS pixel xyxy).
inline void make_mosaic(const Dataset& d, const int four[4], int64_t S, std::mt19937& rng,
                        std::vector<float>& px, std::vector<float>& gb, std::vector<int64_t>& gl) {
  std::uniform_real_distribution<float> U(0.f, 1.f);
  px.assign(3*S*S, 114.f/255.f); gb.clear(); gl.clear();
  int cx = (int)(S*(0.3f + 0.4f*U(rng))), cy = (int)(S*(0.3f + 0.4f*U(rng)));
  int rx[4]={0,cx,0,cx}, ry[4]={0,0,cy,cy};
  int rw[4]={cx,(int)S-cx,cx,(int)S-cx}, rh[4]={cy,cy,(int)S-cy,(int)S-cy};
  for (int q = 0; q < 4; ++q) {
    if (rw[q] <= 0 || rh[q] <= 0) continue;
    Letterbox lb; auto tile = load_image_letterbox_wh(d.items[four[q]].img, rw[q], rh[q], lb);
    for (int c = 0; c < 3; ++c) for (int y = 0; y < rh[q]; ++y) for (int x = 0; x < rw[q]; ++x)
      px[(c*S + ry[q]+y)*S + rx[q]+x] = tile->data[(c*rh[q] + y)*rw[q] + x];
    std::vector<float> tb; std::vector<int64_t> tl;
    load_boxes_orig(d.items[four[q]].lbl, d.yolo, lb.w0, lb.h0, tb, tl);
    lb_map(tb, lb, (float)rx[q], (float)ry[q]);
    for (size_t m = 0; m < tl.size(); ++m) {
      float x1 = std::clamp(tb[m*4],   (float)rx[q], (float)(rx[q]+rw[q]));
      float x2 = std::clamp(tb[m*4+2], (float)rx[q], (float)(rx[q]+rw[q]));
      float y1 = std::clamp(tb[m*4+1], (float)ry[q], (float)(ry[q]+rh[q]));
      float y2 = std::clamp(tb[m*4+3], (float)ry[q], (float)(ry[q]+rh[q]));
      if (x2-x1 > 2 && y2-y1 > 2) { gl.push_back(tl[m]); gb.insert(gb.end(), {x1,y1,x2,y2}); }
    }
  }
}

// Load a mini-batch (the given indices) into (B,3,S,S) + padded GT, both label formats.
// augment: horizontal flip + brightness jitter. mosaic: each output is a 4-image mosaic
// (its first tile = idx[n], the other three random) — off for validation.
inline Batch load_minibatch(const Dataset& d, const std::vector<int>& idx, bool augment,
                            uint32_t seed, bool mosaic = false) {
  std::mt19937 rng(seed); std::uniform_real_distribution<float> U(0.f, 1.f);
  int64_t B = (int64_t)idx.size(), S = d.S, N = (int64_t)d.items.size();
  std::vector<std::vector<float>> gbs(B); std::vector<std::vector<int64_t>> gls(B);
  Batch bt; bt.B = B; bt.x = make_tensor({B, 3, S, S}); int64_t M = 0;
  for (int64_t n = 0; n < B; ++n) {
    std::vector<float> px;
    if (mosaic) {
      int four[4] = { idx[n], (int)(rng()%N), (int)(rng()%N), (int)(rng()%N) };
      make_mosaic(d, four, S, rng, px, gbs[n], gls[n]);
    } else {
      Letterbox lb; auto xi = load_image_letterbox(d.items[idx[n]].img, S, lb);
      px.assign(xi->data.begin(), xi->data.end());
      load_boxes_orig(d.items[idx[n]].lbl, d.yolo, lb.w0, lb.h0, gbs[n], gls[n]);
      lb_map(gbs[n], lb);
    }
    bool flip = augment && U(rng) < 0.5f;
    float bri = augment ? 0.8f + 0.4f*U(rng) : 1.f;         // brightness 0.8..1.2
    float* dst = bt.x->data.data() + n*3*S*S;
    for (int c = 0; c < 3; ++c) for (int64_t y = 0; y < S; ++y) for (int64_t x = 0; x < S; ++x) {
      float v = px[(c*S + y)*S + (flip ? S-1-x : x)] * bri;
      dst[(c*S + y)*S + x] = v < 0 ? 0 : (v > 1 ? 1 : v);
    }
    if (flip) for (size_t m = 0; m < gls[n].size(); ++m) { float x1 = gbs[n][m*4], x2 = gbs[n][m*4+2]; gbs[n][m*4] = S - x2; gbs[n][m*4+2] = S - x1; }
    M = std::max<int64_t>(M, (int64_t)gls[n].size());
  }
  bt.M = M; bt.gt_boxes.assign(B*M*4, 0.f); bt.gt_labels.assign(B*M, 0); bt.mask.assign(B*M, 0.f);
  for (int64_t n = 0; n < B; ++n) for (size_t m = 0; m < gls[n].size(); ++m) {
    bt.gt_labels[n*M+m] = gls[n][m]; bt.mask[n*M+m] = 1.f;
    for (int j = 0; j < 4; ++j) bt.gt_boxes[(n*M+m)*4+j] = gbs[n][m*4+j];
  }
  return bt;
}
