#include <iostream>
#include <chrono>
#include <vector>

#include "./benchmark_harness/Launchers.cuh"

using namespace std;

// ---- Config ------------------------------------------------------------
const int BLOCK_SIZE = 16;

// ---- Kernel ------------------------------------------------------------
__global__ void kernel(const float *a, const float *b, float *out, int N) {
   // TODO:
}

// ---- CPU Reference -----------------------------------------------------
void serial(const float *a, const float *b, float *out, int N) {
  // TODO: implement
}

// ---- Verification ------------------------------------------------------
bool verify(const float *expected, const float *actual, int N, float eps = 1e-3f) {
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      if (expected[row * N + col] - actual[row * N + col] > eps) {
        return false;
      }
    }
  }
  return true;
}

__global__ void MatmulKernel_d(float* Md, float* Nd, float* Pd, size_t width) {
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

void MatmulKernel(float* M, float* N, float* P, int Width) {
  int size = Width * Width * sizeof(float);
  float *Md, *Nd, *Pd;

  // Allocate device memory for M, N, and P
  // copy M and N to allocated device memory locations
  cudaMalloc((void**)&Md, size);
  cudaMalloc((void**)&Nd, size);
  cudaMalloc((void**)&Pd, size);

  cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
  cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);

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

  std::cout << "Kernel Runtime: " << ms << "ms" << endl;

  string error = cudaGetErrorString(cudaGetLastError());
  if (error != "no error") {
    cout << error << endl;
  }

  // copy P from the device memory
  cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);

  // Free device memories
  cudaFree(Md);
  cudaFree(Nd);
  cudaFree(Pd);
}

void serialMatmul(float* M, float* N, float* P, size_t n) {
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
  cout << "serialMatmul Runtime: " << elapsed.count() << "ms" << endl;
}

// ---- Host --------------------------------------------------------------
int main() {
  std::vector<int> sizes = {4, 8, 16, 32};
  profile(serialMatmul, MatmulKernel, sizes);
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

void profile(Launcher serial, Launcher kernel, vector<int> sizes) {
  for (int N : sizes) {
    const int elements = N * N;

    // Host-side input matrices for testing serial() and kernel().
    float *a = new float[elements];
    float *b = new float[elements];

    for (int i = 0; i < elements; i++) {
      a[i] = static_cast<float>((i % 5) - 2); // TODO: Understand These
      b[i] = static_cast<float>((i % 3) + 1); // TODO: Understand These
    }

    // SERIAL RESULT
    float* serial_result = new float[elements];
    serial(a, b, serial_result, N);

    // KERNEL RESULT
    float* kernel_result = new float[elements];
    kernel(a, b, kernel_result, N);

    cout << "result: " << verify(serial_result, kernel_result, N) << endl; 

    delete[] a;
    delete[] b;
  } 
}
