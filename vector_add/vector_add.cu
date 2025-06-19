#include <iostream>

using namespace std;

__global__ void addVectors(int *a, int *b, int *c, int *n) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];

    n[i] = sizeof(c) / sizeof(int); 
    // this is not what you are expecting it to be because you cannot find the length of an array from a pointer!
}

int main() {
    int a[] = {1, 2, 3, 4, 5};
    int b[] = {5, 4, 3, 2, 1};
    int c[] = {0, 0, 0, 0, 0};
    int n[] = {0, 0, 0, 0, 0};

    int *pa = 0;
    int *pb = 0;
    int *pc = 0;
    int *pn = 0;

    cudaMalloc(&pa, sizeof(a));
    cudaMalloc(&pb, sizeof(b));
    cudaMalloc(&pc, sizeof(c));
    cudaMalloc(&pn, sizeof(n));

    cudaMemcpy(pa, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(pb, b, sizeof(b), cudaMemcpyHostToDevice);
    cudaMemcpy(pc, c, sizeof(c), cudaMemcpyHostToDevice);
    cudaMemcpy(pn, n, sizeof(n), cudaMemcpyHostToDevice);

    addVectors<<<1, sizeof(c)>>>(pa, pb, pc, pn);

    cudaMemcpy(c, pc, sizeof(c), cudaMemcpyDeviceToHost);
    cudaMemcpy(n, pn, sizeof(n), cudaMemcpyDeviceToHost);

    for (int i = 0; i < sizeof(c) / sizeof(int); i++) {
        cout << c[i];
    }
    cout << endl;

    for (int i = 0; i < sizeof(n) / sizeof(int); i++) {
        cout << n[i];
    }
    cout << endl;
}