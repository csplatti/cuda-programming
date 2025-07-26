# cuda-programming

## Overview

This repository contains various algorithms implemented using CUDA for parallel processing. My goal is to explore and experiment with parallelization techniques in CUDA across different types of algorithms. Each algorithm will be added as an independent module, and the repository will be updated with new implementations over time.  

I am by no means an expert in CUDA (or C++), and do not claim that my solutions are optimal, efficient, or memory safe. If you see any room for improvement feel free to message me or make a pull request implementing the fix with a detailed explanation. Upon receiving either I will fix the issue and give you credit in a comment!

## Prerequisites
- **Hardware & Software Requirements**
  - Latest version of CUDA (as I am writing this it is 12.9)
  - NVIDIA GPU with CUDA capability (I use an RTX 2000 Ada which has CUDA capabililty 8.9)
- **Dependencies**
  - Python Libraries (For runtime plotting notebooks): matplotlib
 
## Installation
1. Fork this repository
2. Clone the repository
   ```bash
   git clone https://github.com/{Your GitHub Username}/cuda-programming
   ```
3. Navigate to the local repository
   ```bash
   cd ./cuda-programming
   ```
4. Navigate to any of the folders, e.g. hello_cuda
   ```bash
   cd ./hello_cuda
   ```
5. Compile with NVCC
   ```bash
   nvcc -o your_cuda your_cuda.cu
   ```
## Usage
Run the binary with input from a file. Currently, input to all of my programs is simply an integer denoting the size of the randomly generated input array to the given algorithm.
```bash
cat input.txt | ./your_cuda
```
Input can also be given via user input by running
```bash
./your_cuda
```
The program will wait for you to input a number on the newline.

## Table of Contents
1. [Vector Addition](#vector-addition)

## Vector Addition
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/0cfe47d3-f71a-4ac1-8c90-1008e61776d4" />
  
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
