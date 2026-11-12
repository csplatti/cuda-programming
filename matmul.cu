#include <iostream>
#include <chrono>
#include <vector>

#include "./benchmark_harness/Launchers.cuh"
#include "./benchmark_harness/BenchmarkResult.cuh"

using namespace std;

// ---- Config ------------------------------------------------------------
const int BLOCK_SIZE = 16;

// ---- Verification ------------------------------------------------------
bool verify(const Matrix<float>& expected, const Matrix<float>& actual, unsigned long N, float eps = 1e-3f) {
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      if (expected[row * N + col] - actual[row * N + col] > eps) {
        return false;
      }
    }
  }
  return true;
}

__global__ void MatmulKernel_d(float *Md, float *Nd, float *Pd, unsigned long width) {
  int thread = threadIdx.x;

  // extract indices
  int i = thread / width; // rounds down
  int j = thread - i * width;

  int sum = 0;
  for (size_t t = 0; t < width; t++) {
    sum += Md[t*width + j] * Nd[i*width + t]; 
  }

  Pd[thread] = sum;
}

Timing MatmulKernel(const Matrix<float> &M, const Matrix<float> &N, Matrix<float> &P, unsigned long Width) {
  int size = Width * Width * sizeof(float); // TODO: sizeof should be different. depends on type of matrix
  float *Md, *Nd, *Pd;

  // Allocate device memory for M, N, and P
  // copy M and N to allocated device memory locations
  cudaMalloc((void**)&Md, size);
  cudaMalloc((void**)&Nd, size);
  cudaMalloc((void**)&Pd, size);

  cudaMemcpy(Md, M.data.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(Nd, N.data.data(), size, cudaMemcpyHostToDevice);

  // Kernel invocation code - to have the device to perform

  // first start matmul
  MatmulKernel_d<<<1, Width * Width>>>(Md, Nd, Pd, Width);

  // timing the actual matmul
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  MatmulKernel_d<<<1, Width * Width>>>(Md, Nd, Pd, Width);
 
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  string error = cudaGetErrorString(cudaGetLastError());
  if (error != "no error") {
    cout << error << endl;
  }

  // copy P from the device memory
  cudaMemcpy(P.data.data(), Pd, size, cudaMemcpyDeviceToHost);

  // Free device memories
  cudaFree(Md);
  cudaFree(Nd);
  cudaFree(Pd);

  return Timing{ms};
}

Timing serialMatmul(const Matrix<float> &M, const Matrix<float> &N, Matrix<float> &P, unsigned long n) {
  auto start = chrono::steady_clock::now();
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      int sum = 0;
      for (size_t t = 0; t < n; t++) {
        sum += M[t*n + j] * N[i*n + t]; 
      }
      P[i*n + j] = sum;
    }
  }
  auto end = chrono::steady_clock::now();
  chrono::duration<double, std::milli> elapsed = end - start;
  return Timing{static_cast<float>(elapsed.count())};
}

vector<BenchmarkResult> profile(
    Launcher serial,
    Launcher kernel,
    const vector<int>& sizes
) {
  vector<BenchmarkResult> results;

  for (int N : sizes) {
    const int elements = N * N;

    // Host-side input matrices for testing serial() and kernel().
    Matrix<float> a;
    a.data.resize(elements);
    Matrix<float> b;
    b.data.resize(elements);

    for (int i = 0; i < elements; i++) {
      a[i] = static_cast<float>((i % 5) - 2); // TODO: Understand These
      b[i] = static_cast<float>((i % 3) + 1); // TODO: Understand These
    }

    // SERIAL RESULT
    Matrix<float> serial_result;
    serial_result.data.resize(elements);
    Timing serial_timing = serial(a, b, serial_result, N);

    // KERNEL RESULT
    Matrix<float> kernel_result;
    kernel_result.data.resize(elements);
    Timing kernel_timing = kernel(a, b, kernel_result, N);

    results.push_back({
      static_cast<unsigned long>(N),
      serial_timing,
      kernel_timing,
      verify(serial_result, kernel_result, N)
    });
  } 

  return results;
}

// ---- Host --------------------------------------------------------------
int main() {
  std::vector<int> sizes = {4, 8, 16, 32};
  vector<BenchmarkResult> results = profile(serialMatmul, MatmulKernel, sizes);

  for (const BenchmarkResult& result : results) {
    cout << "N=" << result.size
         << " serial=" << result.serial.milliseconds << "ms"
         << " kernel=" << result.parallel.milliseconds << "ms"
         << " correct=" << result.correct
         << endl;
  }
  // for (int N : sizes) {
  //   printf("N=%d\n", N);
  //   const int elements = N * N;
  //
  //   // Host-side input matrices for testing serial() and kernel().
  //   float *a = new float[elements];
  //   float *b = new float[elements];
  //
  //   for (int i = 0; i < elements; i++) {
  //     a[i] = static_cast<float>((i % 5) - 2); // TODO: Understand These
  //     b[i] = static_cast<float>((i % 3) + 1); // TODO: Understand These
  //   }
  //
  //   // SERIAL RESULT
  //   float* serial_result = new float[elements];
  //   serialMatmul(a, b, serial_result, N);
  //
  //   // KERNEL RESULT
  //   float* kernel_result = new float[elements];
  //   MatmulKernel(a, b, kernel_result, N);
  //
  //   cout << "result: " << verify(serial_result, kernel_result, N) << endl; 
  //
  //   delete[] a;
  //   delete[] b;
  // }
  return 0;
}
