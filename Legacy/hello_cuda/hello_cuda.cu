#include <iostream>

__global__ void helloFromGPU() {
    printf("Hello from GPU! Thread ID: %d\n", threadIdx.x);
}

int main() {
    helloFromGPU<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}