# cuda-programming

## Overview

This repository contains various algorithms implemented using CUDA for parallel processing. My goal is to explore and experiment with parallelization techniques in CUDA across different types of algorithms. Each algorithm will be added as an independent module, and the repository will be updated with new implementations over time.  

I am by no means an expert in CUDA (or C++), and do not claim that my solutions are optimal, efficient, or memory safe. If you see any room for improvement feel free to message me or make a pull request implementing the fix with a detailed explanation. Upon receiving either I will fix the issue and give you credit in a comment!

## Table of Contents
1. [Vector Addition](#vector-addition)

## Vector Addition
### Code
The serial algorithm I wrote to compare to a parallel implementation used a simple for loop.

```CUDA
void addVectorsSerial(int *a, int *b, int *c, int vectorLength) {
    for (int i = 0; i < vectorLength; i++) {
        c[i] = a[i] + b[i];
    }
}
```

The CUDA kernel itself was quite simple, requiring only pointers to the input and output arrays and the size of the arrays.
```cuda
__global__ void addVectorsParallel(int *a, int *b, int *c, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}
```
The first line of the kernel calculates its thread index. If the kernel is assigned an index within the array, it performs the addition. I couldnt find a way to avoid this check without giving up the property that the input array could be of arbitrary size. Reach out to me or make a pull request with an explanation if such a solution exists!  

The call to the CUDA kernel was as follows:

```CUDA
addVectorsParallel<<<sizeof(parallelOut) / 1024 + 1, 1024>>>(pa, pb, pOut, N);
```

One block for each multiple of 1024, plus 1 for any remainder, was called with 1024 threads.

### Performance
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/0cfe47d3-f71a-4ac1-8c90-1008e61776d4" />
