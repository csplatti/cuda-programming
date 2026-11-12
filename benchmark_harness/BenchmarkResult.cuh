#pragma once

#include "../Datatypes/Timing.cuh"

struct BenchmarkResult {
  unsigned long size;
  Timing serial;
  Timing parallel;
  bool correct;
};
