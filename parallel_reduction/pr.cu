#include <iostream>
#include <math.h>
#include <chrono>

using namespace std;

__global__ void parallelSum(int *nums, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // printf("%d\n", i);

    // printf("Thread id: %d\n", i);

    // if (i == 0) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%d ", nums[j]);
    //     }
    //     printf("\n");
    //     printf("N: %d\n", N);
    // }

    float indexJump = (float) N / 2;
    int roundedJump = ceilf(indexJump);

    if (i + roundedJump < N) {
        nums[i] += nums[i+roundedJump];
        // printf("i: %d; 2i: %d\n", nums[i], nums[2*i]);
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

int main() {
    int N; 
    // cout << "Enter the number you would like to sum to (0 < N < 65535): ";
    cin >> N;
    int nums[N];
    int numsSize = sizeof(nums) / sizeof(int);

    for (int i = 0; i < numsSize; i++) {
        nums[i] = i+1;
        // cout << nums[i] << endl;
    }

    long serialSumResult = 0;

    auto startSerial = chrono::high_resolution_clock::now();
    serialSum(nums, numsSize, &serialSumResult);
    auto endSerial = chrono::high_resolution_clock::now();

    auto serialTime = chrono::duration_cast<chrono::milliseconds>(endSerial-startSerial);

    // cout << "Serial: " << serialSumResult << " Runtime: " << serialTime.count() << "ms" << endl;

    cout << serialTime.count() << endl;

    int parallelSumResult = 0;

    int *pn = 0;

    cudaMalloc(&pn, sizeof(nums));
    cudaMemcpy(pn, nums, sizeof(nums), cudaMemcpyHostToDevice);

    int numThreads = ceil(numsSize / 2);

    // printf("%d\n", (int) ceil((float) numsSize / 1024));

    auto startParallel = chrono::high_resolution_clock::now();
    parallelSum<<<(int) ceil((float) numsSize / 1024), 1024>>>(pn, numsSize);
    auto endParallel = chrono::high_resolution_clock::now();

    auto parallelTime = chrono::duration_cast<chrono::milliseconds>(endParallel - startParallel);

    cudaMemcpy(nums, pn, sizeof(nums), cudaMemcpyDeviceToHost);
    parallelSumResult = nums[0];

    // cout << "Parallel: " << parallelSumResult << " Runtime: " << parallelTime.count() << "ms" << endl;

    cout << parallelTime.count() << endl;

    cudaFree(&pn);
}