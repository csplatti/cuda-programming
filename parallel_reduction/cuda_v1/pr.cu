#include <iostream>
#include <math.h>
#include <chrono>

using namespace std;

__global__ void parallelSum1(int *nums, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    int roundedJump = ceilf((float) N / 2);

    if (i + roundedJump < N) {
        nums[i] += nums[i+roundedJump];
    }

    float nextSize = (float) N / (float) 2;
    // printf("%f\n", 5 / 2);
    int nextN = ceilf(nextSize);
    int nextNThreads = ceilf(nextN / 2);
    if (nextN > 1 && i == 0) {
        // printf("Nextn: %d\n", nextN);
        // parallelSum1<<<ceil((float) nextN / 1024), 1024>>>(nums, nextN); // this runs in each thread: that is bad
    }
}

// REQUIRES: N == length of nums
void serialSum(int nums[], int numsSize, long long *serialOut) {
    long long sum = 0;
    for (int i = 0; i < numsSize; i++) {
        sum += nums[i];
    }

    *serialOut = sum;
}

void runSerial(int nums[], int numsSize) {
    long long serialSumResult = 0;

    auto startSerial = chrono::high_resolution_clock::now();
    serialSum(nums, numsSize, &serialSumResult);
    auto endSerial = chrono::high_resolution_clock::now();

    auto serialTime = chrono::duration_cast<chrono::milliseconds>(endSerial-startSerial);

    cout << "Serial: " << serialSumResult << " Runtime: " << serialTime.count() << "ms" << endl;
}

__global__ void parallelSum2(int *nums, long long *sums, int numsSize, int additionIndex) {
    int i = threadIdx.x;

    if (additionIndex == 0 && i < numsSize) {
        sums[i] = nums[i];
    }

    if (i + additionIndex * 1024 < numsSize) {
        sums[i] += nums[i + additionIndex * 1024];
    }
}

long long parallel2Host(int nums[], int numsLength) {
    int n = numsLength;
    int numAdds = ceil((float) n / 1024);
    long long sums[1024];
    
    long long *s = 0;
    int *np = 0;

    cudaMalloc(&s, sizeof(sums));
    cudaMalloc(&np, numsLength * sizeof(int));
    cudaMemcpy(s, sums, sizeof(sums), cudaMemcpyDeviceToHost);
    cudaMemcpy(np, nums, numsLength * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numAdds; i++) {
        parallelSum2<<<1, 1024>>>(np, s, numsLength, i);
    }

    cudaMemcpy(sums, s, sizeof(sums), cudaMemcpyHostToDevice);
    cudaMemcpy(nums, np, numsLength * sizeof(int), cudaMemcpyHostToDevice);
    cudaFree(np);
    cudaFree(s);

    long long sum = 0;
    for (int i = 0; i < sizeof(sums) / sizeof(long long); i++) {
        sum += sums[i];
    }

    return sum;
}

void runParallel(int nums[], int numsSize) {
    int parallelSumResult = 0;

    // Allocate Memory and Copy to Device
    // int *pn = 0;
    // cudaMalloc(&pn, numsSize * sizeof(int));
    // cudaMemcpy(pn, nums, numsSize * sizeof(int), cudaMemcpyHostToDevice);

    // Congifure and Run Kernel
    // int numThreads = ceil(numsSize / 2);
    auto startParallel = chrono::high_resolution_clock::now();
    parallelSumResult = parallel2Host(nums, numsSize);
    cudaDeviceSynchronize();
    auto endParallel = chrono::high_resolution_clock::now();

    auto parallelTime = chrono::duration_cast<chrono::milliseconds>(endParallel - startParallel);

    // cudaMemcpy(nums, pn, numsSize * sizeof(int), cudaMemcpyDeviceToHost);
    // parallelSumResult = nums[0];

    cout << "Parallel: " << parallelSumResult << " Runtime: " << parallelTime.count() << "ms" << endl;

    // cout << parallelTime.count() << endl;

    // cudaFree(&pn);
}

int main() {
    int N = 65535;
    // cout << "Enter the number you would like to sum to (0 < N < 65535): ";
    // cin >> N;
    int nums[N];
    int numsSize = sizeof(nums) / sizeof(int);

    for (int i = 0; i < numsSize; i++) {
        nums[i] = i+1;
        // printf("%d\n", i);
    }
    // printf("done with for loop to create array");

    runSerial(nums, numsSize);
    runParallel(nums, numsSize);

}