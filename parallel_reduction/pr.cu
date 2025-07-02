#include <iostream>
#include <math.h>
#include <chrono>

using namespace std;

__global__ void parallelSum(int *nums, int N) {
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
        parallelSum<<<ceil((float) nextN / 1024), 1024>>>(nums, nextN); // this runs in each thread: that is bad
    }
}

// REQUIRES: N == length of nums
void serialSum(int nums[], int numsSize, long *serialOut) {
    long sum = 0;
    for (int i = 0; i < numsSize; i++) {
        sum += nums[i];
    }

    *serialOut = sum;
}

void runSerial(int nums[], int numsSize) {
    long serialSumResult = 0;

    auto startSerial = chrono::high_resolution_clock::now();
    serialSum(nums, numsSize, &serialSumResult);
    auto endSerial = chrono::high_resolution_clock::now();

    auto serialTime = chrono::duration_cast<chrono::milliseconds>(endSerial-startSerial);

    cout << "Serial: " << serialSumResult << " Runtime: " << serialTime.count() << "ms" << endl;
}

void runParallel(int nums[], int numsSize) {
    int parallelSumResult = 0;

    int *pn = 0;

    cudaMalloc(&pn, numsSize * sizeof(int));
    cudaMemcpy(pn, nums, numsSize * sizeof(int), cudaMemcpyHostToDevice);

    int numThreads = ceil(numsSize / 2);

    auto startParallel = chrono::high_resolution_clock::now();
    parallelSum<<<(int) ceil((float) numsSize / 1024), 1024>>>(pn, numsSize);
    cudaDeviceSynchronize();
    auto endParallel = chrono::high_resolution_clock::now();

    auto parallelTime = chrono::duration_cast<chrono::milliseconds>(endParallel - startParallel);

    cudaMemcpy(nums, pn, numsSize * sizeof(int), cudaMemcpyDeviceToHost);
    parallelSumResult = nums[0];

    cout << "Parallel: " << parallelSumResult << " Runtime: " << parallelTime.count() << "ms" << endl;

    // cout << parallelTime.count() << endl;

    cudaFree(&pn);
}

int main() {
    int N = 65535;
    // cout << "Enter the number you would like to sum to (0 < N < 65535): ";
    // cin >> N;
    int nums[N];
    int numsSize = sizeof(nums) / sizeof(int);

    for (int i = 0; i < numsSize; i++) {
        nums[i] = i+1;
    }

    runSerial(nums, numsSize);
    runParallel(nums, numsSize);
}