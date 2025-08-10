#include "./Sources/ach.h"
#include <iostream>
#include <chrono>

using namespace std;

long serialSum(const thrust::universal_vector<int>& nums) {
    return thrust::reduce(thrust::host, nums.begin(), nums.end(), 0, thrust::plus<int>{});
}

long parallelSum(const thrust::universal_vector<int>& nums) {
    return thrust::reduce(thrust::device, nums.begin(), nums.end(), 0, thrust::plus<int>{});
}

int main() {
    int N;
    cin >> N;

    // Store Nums in Array
    thrust::universal_vector<int> nums(N);
    thrust::sequence(nums.begin(), nums.end());

    // SERIAL
    auto start_serial = chrono::high_resolution_clock::now();
    cout << serialSum(nums) << endl;
    auto end_serial = chrono::high_resolution_clock::now();
    const double serial_duration = chrono::duration_cast<chrono::microseconds>(end_serial - start_serial).count();
    // cout << "SERIAL RUNTIME: " << serial_duration << endl;
    cout << serial_duration << endl;

    // PARALLEL
    auto start_parallel = chrono::high_resolution_clock::now();
    cout << parallelSum(nums) << endl;
    auto end_parallel = chrono::high_resolution_clock::now();
    const double parallel_duration = chrono::duration_cast<chrono::microseconds>(end_parallel - start_parallel).count();
    // cout << "PARALLEL RUNTIME: " << parallel_duration << endl;
    cout << parallel_duration << endl;
}