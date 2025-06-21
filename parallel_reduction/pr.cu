#include <iostream>
#include <math.h>
#include <chrono>

using namespace std;

__global__ void parallelSum(int *nums, int N) {
    int i = threadIdx.x;

    // printf("Thread id: %d\n", i);

    // if (i == 0) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%d ", nums[j]);
    //     }
    //     printf("\n");
    //     printf("N: %d\n", N);
    // }

    float indexJump = (float) N / (float) 2;
    int roundedJump = ceilf(indexJump);

    // if (i + roundedJump < N) {
        nums[i] += nums[i+roundedJump];
        // printf("i: %d; 2i: %d\n", nums[i], nums[2*i]);
    // }

    float nextSize = (float) N / (float) 2;
    // printf("%f\n", 5 / 2);
    int nextN = ceilf(nextSize);
    int nextNThreads = ceilf(nextN / 2);
    if (nextN > 1 && i == 0) {
        // printf("Nextn: %d\n", nextN);
        parallelSum<<<1, nextN>>>(nums, nextN); // this runs in each thread: that is bad
    }
}

// REQUIRES: N == length of nums
void serialSum(int nums[], int numsSize, int *serialOut) {
    int sum = 0;
    for (int i = 0; i < numsSize; i++) {
        sum += nums[i];
    }

    *serialOut = sum;
}

int main() {
    int nums[1024];
    int numsSize = sizeof(nums) / sizeof(int);

    for (int i = 0; i < numsSize; i++) {
        nums[i] = i+1;
        // cout << nums[i] << endl;
    }

    int serialSumResult = 0;

    auto startSerial = chrono::high_resolution_clock::now();
    serialSum(nums, numsSize, &serialSumResult);
    auto endSerial = chrono::high_resolution_clock::now();

    auto serialTime = chrono::duration_cast<chrono::milliseconds>(endSerial-startSerial);

    cout << "Serial: " << serialSumResult << " Runtime: " << serialTime.count() << "ms" << endl;

    int parallelSumResult = 0;

    int *pn = 0;

    cudaMalloc(&pn, sizeof(nums));
    cudaMemcpy(pn, nums, sizeof(nums), cudaMemcpyHostToDevice);

    int numThreads = ceil(numsSize / 2);

    auto startParallel = chrono::high_resolution_clock::now();
    parallelSum<<<1, numThreads>>>(pn, numsSize);
    auto endParallel = chrono::high_resolution_clock::now();

    auto parallelTime = chrono::duration_cast<chrono::milliseconds>(endParallel - startParallel);

    cudaMemcpy(nums, pn, sizeof(nums), cudaMemcpyDeviceToHost);
    parallelSumResult = nums[0];

    cout << "Parallel: " << parallelSumResult << " Runtime: " << parallelTime.count() << "ms" << endl;

    cudaFree(&pn);
}