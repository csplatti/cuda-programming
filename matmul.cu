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

  // printf("i: %d, j: %d, thread: %d\n", i, j, Pd[thread]);

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
  // the actual matmul
  MatmulKernel_d<<<1, Width * Width>>>(Md, Nd, Pd, Width);

  cudaError_t error = cudaGetLastError();
  cout << cudaGetErrorString(error) << endl;

  // copy P from the device memory
  cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);

  // Free device memories
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

  // SERIAL RESULT
  float* serial_result = new float[elements];
  serialMatmul(a, b, serial_result, N);
  cout << "Serial Output:" << endl;
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      cout << serial_result[row * N + col] << ' ';
    }
    cout << endl;
  }

  // // PRINT A
  // cout << "Matrix A:" << endl;
  // for (int row = 0; row < N; row++) {
  //   for (int col = 0; col < N; col++) {
  //     cout << a[row * N + col] << ' ';
  //   }
  //   cout << endl;
  // }
  //
  // // PRINT B
  // cout << "Matrix B:" << endl;
  // for (int row = 0; row < N; row++) {
  //   for (int col = 0; col < N; col++) {
  //     cout << b[row * N + col] << ' ';
  //   }
  //   cout << endl;
  // }

  float* kernel_result = new float[elements];
  MatmulKernel(a, b, kernel_result, N);
  cout << "Kernel Output:" << endl;
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      cout << kernel_result[row * N + col] << ' ';
    }
    cout << endl;
  }
  
  cout << "result: " << verify(serial_result, kernel_result, N) << endl; 

  delete[] a;
  delete[] b;
  return 0;
}
