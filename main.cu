#include <iostream>

using namespace std;

__global__ void manyHelloWorlds() {
  printf("Hello World! (thread: %d)\n", threadIdx.x);
}

int main() {
  manyHelloWorlds<<<1, 20>>>();

  cudaError_t error = cudaDeviceSynchronize();

  if (error != cudaSuccess) {
    cerr << cudaGetErrorString(error) << endl;
  }
}

