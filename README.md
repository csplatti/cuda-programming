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
2. [Parallel Reduction](#parallel-reduction)

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

## Parallel Reduction
### What is it?
Say you have a list of integers
```c++
    int nums[] = {1, 2, 3, 4, 5, 6, 7};
```

For many problems, it can be useful to **reduce** such a list of numbers down to a single value (their sum, product, max, min, etc.). Parallel reduction aims to paralellize this class of problem for better performance.

### My First Implementation

The strategy for my first attempt was as follows where $N$ is the length of the input array:

1) Create $\lceil\frac{N}{2}\rceil$ threads so that the first half of the elements in the list are indexed to
2) Add the number at index $i + \lceil\frac{N}{2}\rceil$ to the number at index $i$ and store it in the array
3) Recursively call the kernel on the first half of the list with threads assigned to half of the indeces, unless the first half of the list only contains one element.

The following results are for adding up all the integers from 1 to 1024:
```bash
serial: 524800 Runtime: 0ms
Parallel: 524800 Runtime: 26ms
```

I was happy to have a solution that worked, but objectively this wasnt a good solution for a number of reasons.

#### 1) Recursion
Recursion is not advised to be run on GPUs due to limited stack space
#### 2) Only works for lists with less than 1024 elements
The benefits of parallel programs are greatest for large lists. Since moving data into GPU memory is expensive, serial algorithms are often faster at performing reduction tasks on smaller arrays.
#### 3) Modifies the array in place
Running this implementation would modify the original ```nums``` array, which makes it useless in many contexts (e.g. sorting, etc.).

### Fixing the problems

The first problem I chose to tackle was that of the maximum list capacity. This wasnt a strategic choice in hindsight because whatever indexing changes I made would be redone again when I inveitably choose to fix the recursion problem, but I wanted to learn more about thread indexing in CUDA.  

I made this fix by first extending the number of blocks in each kernel call to $\lceil N / 2 \rceil$ and fixed the number of threads in each block at 1024 (the maximum possible on my GPU setup). Then, I updated the indexing to account for the use of blocks in the parallel sum kernel. 

```c++
int i = threadIdx.x + blockIdx.x * blockDim.x;
```

Now my kernel was properly enabled to run parallel sums on arrays with more than 1024 elements, but I quickly discovered that I still had an implicit constraint on my array size.

```bash
Serial: -2116934296 Runtime: 0ms
Parallel: -2116934296 Runtime: 16ms
```

I was storing the resulting sum in an ```int```, so with large arrays of positive integers it was inevitable that an overflow would occur.  

I was able to fix this for the serial sum (well, more like kick the can down the road) by simply changing the sum variable to a ```long```. I could not do the same for the parallel algorithm because the sum occurs within the array, and changing the array type to a long caused a segmentation fault, presumably because of the increased size of such a large long array. The fix for the serial sum yeld the following result

```zsh
Serial: 2178033000 Runtime: 0ms
Parallel: -2116934296 Runtime: 17ms
```

Naturally, I wanted to figure out exactly how large of an array I could create without getting an overflow from the parallel sum.  

In C++, the largest value that can be stored in an ```int``` is 2,147,483,647 (```INT_MAX``` in C++). Since I was using arrays of the first $N$ natural numbers, I could represent my final sum as the following summation.

$$\sum_{n=1}^N n$$

Thanks to my recent integral calculus course (thank you MATH 121), I applied the following identity:

$$\sum_{n=1}^N n = \frac{N(N+1)}{2}$$

Next, I wanted to find which $N$ were small enough to keep the sum less than ```INT_MAX```.  

$$ \frac{N(N+1)}{2} \le 2,147,483,647$$

Using the fact that $N+1 \approx N$ for large $N$, I was able to solve for $N$ quickly without the quadratic formula

$$ \frac{N(N+1)}{2} \le 2,147,483,647$$

$$ N^2 \approx N(N+1) \le 2 \cdot 2,147,483,647$$

$$ N \approx \sqrt{N(N+1)} \le \sqrt{2 \cdot 2,147,483,647}$$

$$ N \approx \sqrt{N(N+1)} \le 65535.9999847$$

So more or less, $N \le 65,535$, so the largest $N$ I could use was 65535. Plugging this into my program yeld the following.

```bash
Serial: 2147450880 Runtime: 0ms
Parallel: 2147450880 Runtime: 16ms
```

Increasing $N$ to just 65536 caused an overflow, so the approximation worked quite well!

```bash
Serial: 2147516416 Runtime: 0ms
Parallel: -2147450880 Runtime: 16ms
```

I wrote a python script to plot the performance of the serial and parallel algorithms at this point, and unfortunately it is quite disapointing.

![alt text](./parallel_reduction/runtime_plot_after_block_implementation.png)

This means that either 65,535 elements is too small a list for the benefits of a GPU to take place (unlikely), or my algorithm is the problem (very likely).
