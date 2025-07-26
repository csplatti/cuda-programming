#include <iostream>
#include <chrono>

using namespace std;

__global__ void addVectorsParallel(int *a, int *b, int *c, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

void addVectorsSerial(int *a, int *b, int *c, int vectorLength) {
    for (int i = 0; i < vectorLength; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Get Vector Size Input
    int N;
    cout << "Please Enter Desired Vector Size: " << endl; 
    cin >> N;

    // Vector Setup
    int a[N];
    int b[N];
    int serialOut[N];
    int parallelOut[N];

    // Fill Vectors
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 501;
        b[i] = rand() % 501;
    }

    // Run Serial Sum
    auto startSerial = chrono::high_resolution_clock::now();
    addVectorsSerial(a, b, serialOut, N);
    auto endSerial = chrono::high_resolution_clock::now();

    // calculate and Display Serial Sum Runtime
    auto serialTime = chrono::duration_cast<chrono::microseconds>(endSerial - startSerial).count();

    cout << "Serial Runtime: " << serialTime << " microseconds" << endl;
    // cout << serialTime << endl;

    // GPU Memory Management Setup
    int *pa = 0;
    int *pb = 0;
    int *pOut = 0;

    cudaMalloc(&pa, sizeof(a));
    cudaMalloc(&pb, sizeof(b));
    cudaMalloc(&pOut, sizeof(parallelOut));

    cudaMemcpy(pa, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(pb, b, sizeof(b), cudaMemcpyHostToDevice);
    cudaMemcpy(pOut, parallelOut, sizeof(parallelOut), cudaMemcpyHostToDevice);

    // Run Parallel Sum
    auto startParallel = chrono::high_resolution_clock::now();
    addVectorsParallel<<<sizeof(parallelOut) / 1024 + 1, 1024>>>(pa, pb, pOut, N);
    cudaDeviceSynchronize();
    auto endParallel = chrono::high_resolution_clock::now();

    // Calculate and Display Parallel Sum Runtime
    auto parallelTime = chrono::duration_cast<chrono::microseconds>(endParallel - startParallel).count();

    cout << "Parallel Runtime: " << parallelTime << " microseconds" << endl;
    // cout << parallelTime << endl;

    // Copy Parallel Sum Result Back from GPU
    cudaMemcpy(parallelOut, pOut, sizeof(parallelOut), cudaMemcpyDeviceToHost);

    // Error Detection
    for (int i = 0; i < N; i++) {
        if (parallelOut[i] != serialOut[i]) {
            cout << "ERROR AT " << i << endl;
            break;
        }
    }
}