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

struct Letterbox { float r; int left, top, w0, h0; };

// Load + letterbox to SxS (grey 114, bilinear), RGB, /255 -> (1,3,S,S). Fills lb.
inline Tensor load_image_letterbox(const std::string& path, int64_t S, Letterbox& lb) {
  int w0, h0, ch;
  unsigned char* im = stbi_load(path.c_str(), &w0, &h0, &ch, 3);
  if (!im) { printf("cannot load %s\n", path.c_str()); std::exit(1); }
  float r = std::min((float)S / w0, (float)S / h0);
  int nw = (int)std::round(w0 * r), nh = (int)std::round(h0 * r);
  int left = (int)((S - nw) / 2), top = (int)((S - nh) / 2);
  auto x = make_tensor({1, 3, S, S});
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
        x->data[(c * S + top + y) * S + left + xx] = v / 255.f;
      }
    }
  stbi_image_free(im);
  lb = {r, left, top, w0, h0};
  return x;
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
struct Dataset { int64_t S = 0; std::vector<Sample> items; };

// list.txt: "S N" then N lines "<image> <label>".
inline Dataset read_dataset(const std::string& list_path) {
  std::ifstream f(list_path); if (!f) { printf("cannot open %s\n", list_path.c_str()); std::exit(1); }
  Dataset d; int64_t N; f >> d.S >> N; d.items.resize(N);
  for (auto& s : d.items) f >> s.img >> s.lbl;
  return d;
}

// Load a mini-batch (the given indices) into (B,3,S,S) + padded GT. Square images ->
// letterbox is identity. Light augmentation: horizontal flip + brightness/contrast jitter.
inline Batch load_minibatch(const Dataset& d, const std::vector<int>& idx, bool augment, uint32_t seed) {
  std::mt19937 rng(seed); std::uniform_real_distribution<float> U(0.f, 1.f);
  int64_t B = (int64_t)idx.size(), S = d.S;
  std::vector<std::vector<float>> gbs(B); std::vector<std::vector<int64_t>> gls(B);
  Batch bt; bt.B = B; bt.x = make_tensor({B, 3, S, S}); int64_t M = 0;
  for (int64_t n = 0; n < B; ++n) {
    Letterbox lb; auto xi = load_image_letterbox(d.items[idx[n]].img, S, lb);
    load_labels(d.items[idx[n]].lbl, gbs[n], gls[n]);
    bool flip = augment && U(rng) < 0.5f;
    float bri = augment ? 0.8f + 0.4f * U(rng) : 1.f;       // brightness 0.8..1.2
    float* dst = bt.x->data.data() + n * 3 * S * S;
    for (int c = 0; c < 3; ++c) for (int64_t y = 0; y < S; ++y) for (int64_t x = 0; x < S; ++x) {
      float v = xi->data[(c * S + y) * S + (flip ? S - 1 - x : x)] * bri;
      dst[(c * S + y) * S + x] = v < 0 ? 0 : (v > 1 ? 1 : v);
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
