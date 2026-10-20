#include <iostream>
#include <chrono>

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
    // TODO: implement
    return true;
}

__global__ void MatmulKernel(float* Md, float* Nd, float* Pd, size_t width) {
  return;
}

void MatrixMultiplication(float* M, float* N, float* P, int Width) {
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
  // the actual matmul
  MatmulKernel<<<1, Width>>>(M, N, P, Width);

  // copy P from the device memory
  // Free device memories
  cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);

  cudaFree((void**)&Md);
  cudaFree((void**)&Nd);
  cudaFree((void**)&Pd);
}

void serialMatmul(float* M, float* N, float* P, size_t n) {
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      int sum = 0;
      for (size_t t = 0; t < n; t++) {
        sum += M[t*n + j] * N[i*n + t]; 
      }
      P[i*n + j] = sum;
    }
  }

}

// ---- Host --------------------------------------------------------------
int main() {
  const int N = 4;
  const int elements = N * N;

  // Host-side input matrices for testing serial() and kernel().
  float *a = new float[elements];
  float *b = new float[elements];

  for (int i = 0; i < elements; i++) {
    a[i] = static_cast<float>((i % 5) - 2); // TODO: Understand These
    b[i] = static_cast<float>((i % 3) + 1); // TODO: Understand These
  }

  cout << "Matrix A:" << endl;
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      cout << a[row * N + col] << ' ';
    }
    cout << endl;
  }

  cout << "Matrix B:" << endl;
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      cout << b[row * N + col] << ' ';
    }
    cout << endl;
  }

  delete[] a;
  delete[] b;
  return 0;
}
