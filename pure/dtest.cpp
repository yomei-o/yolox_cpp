// Phase-1 test for the device-resident autograd core (pure/dtensor.hpp):
//   forward  Y = SiLU(X @ W),  L = sum(Y)   then backward.
// Validates (a) gradient correctness on CPU via finite differences, and (b) CPU/GPU parity
// by printing deterministic checksums (both backends must agree to ~1e-4).
//
//   CPU (MSVC + CUDA-13 CCCL): cl /std:c++17 /O2 /EHsc /Zc:preprocessor /DNOMINMAX
//        /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
//        /I"%CUDA%/include/cccl" /I"%CUDA%/include" pure\dtest.cpp
//   CPU (Colab g++): g++ -O2 -std=c++17 -I/usr/local/cuda/include
//        -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP pure/dtest.cpp -o dtest_cpu
//   GPU (Colab nvcc): nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA pure/dtest.cpp -o dtest_gpu
//     (-DUSE_CUDA is REQUIRED: it switches bk::gemm to the device path so it matches thrust's
//      device_vector memory. Without it bk::gemm runs on the host over device pointers -> segfault.)
#include "dtensor.hpp"
#include <cstdio>
#include <numeric>

static const std::vector<int64_t> XS{2, 3}, WS{3, 4};
static std::vector<float> XH{ 0.5f, -1.0f, 2.0f,  -0.3f, 0.8f, 1.5f };
static std::vector<float> WH{ 0.2f, -0.5f, 0.1f, 0.9f,   0.4f, 0.3f, -0.7f, 0.6f,  -0.2f, 0.8f, 0.5f, -0.4f };

// forward only: returns L = sum(SiLU(X@W))
static float forward_L(const std::vector<float>& xh, const std::vector<float>& wh) {
  DT x = dfrom(XS, xh), W = dfrom(WS, wh);
  DT L = dsum(dsilu(dmatmul(x, W)));
  return dto_host(L)[0];
}

int main() {
  fprintf(stderr, "[0] start\n");
  DT x = dfrom(XS, XH), W = dfrom(WS, WH);           bk::sync(); fprintf(stderr, "[1] inputs on device\n");
  DT mm = dmatmul(x, W);                             bk::sync(); fprintf(stderr, "[2] matmul ok\n");
  DT ys = dsilu(mm);                                 bk::sync(); fprintf(stderr, "[3] silu ok\n");
  DT L  = dsum(ys);                                  bk::sync(); fprintf(stderr, "[4] sum ok  L=%.6f\n", dto_host(L)[0]);
  dbackward(L);                                      bk::sync(); fprintf(stderr, "[5] backward ok\n");

  std::vector<float> xg = dto_host(x), Wg_ignored;      // (data, not grad — need grad below)
  float Lval = dto_host(L)[0];
  // pull grads to host
  std::vector<float> xgrad(x->numel()), Wgrad(W->numel());
  thrust::copy(x->grad.begin(), x->grad.end(), xgrad.begin());
  thrust::copy(W->grad.begin(), W->grad.end(), Wgrad.begin());
  double xsum = std::accumulate(xgrad.begin(), xgrad.end(), 0.0);
  double Wsum = std::accumulate(Wgrad.begin(), Wgrad.end(), 0.0);

  printf("L = %.6f\n", Lval);
  printf("checksum sum(dL/dX) = %.6f   sum(dL/dW) = %.6f\n", xsum, Wsum);

  // finite-difference check on a few W entries (CPU-meaningful; GPU should agree analytically)
  const float eps = 1e-3f; int bad = 0;
  for (int i : {0, 5, 11}) {
    std::vector<float> wp = WH; wp[i] += eps;
    float fd = (forward_L(XH, wp) - Lval) / eps;
    float an = Wgrad[i];
    printf("  dL/dW[%2d]  analytic %.5f  finite-diff %.5f  |d| %.2e\n", i, an, fd, std::abs(an - fd));
    if (std::abs(an - fd) > 5e-2f) ++bad;
  }
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)  %s\n", bad ? "GRAD MISMATCH" : "grad-check OK");
#else
  printf("backend: CPU (host)  %s\n", bad ? "GRAD MISMATCH" : "grad-check OK");
#endif
  return 0;
}
