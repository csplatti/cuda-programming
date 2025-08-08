#include "./Sources/ach.h"
#include <iostream>
#include <chrono>

using namespace std;

struct count_iterator {
    int operator[](int i) {
        return i + 1;
    }
};


long serialSum(int N) {
    count_iterator it;
    long sum = 0;
    for (int i = 0; i < N; i++) {
        sum += it[i];
    }
    return sum;
}

int main() {
    int N;
    cin >> N;



    // SERIAL
    // TODO: Log Start Time


    auto start_serial = chrono::high_resolution_clock::now();
    cout << serialSum(N) << endl;
    auto end_serial = chrono::high_resolution_clock::now();
    const double serial_duration = chrono::duration_cast<chrono::microseconds>(end_serial - start_serial).count();
    cout << "SERIAL RUNTIME: " << serial_duration << endl;

    auto start_parallel = chrono::high_resolution_clock::now();
    auto end_parallel = chrono::high_resolution_clock::now();
    const double parallel_duration = chrono::duration_cast<chrono::microseconds>(end_parallel - start_parallel).count();
    cout << "PARALLEL RUNTIME: " << parallel_duration << endl;
}