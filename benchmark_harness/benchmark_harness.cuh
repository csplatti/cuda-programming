#pragma once

#include<vector>
#include<utility>

class BenchmarkHarness {
public:
  BenchmarkResult run(
    Launcher fn,
    const vector<Matrix<float>>& testSet
  );
}
