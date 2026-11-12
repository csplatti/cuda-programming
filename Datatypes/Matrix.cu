#include <vector>
#include <cstddef>

template <typename T>
struct Matrix {
  int width;
  std::vector<T> data;

  T& operator()(int row, int col) {
    return data[row*width + col];
  }

  T& operator[](std::size_t i) {
    return data[i];
  }

  const T& operator[](std::size_t i) const {
    return data[i];
  }
};
