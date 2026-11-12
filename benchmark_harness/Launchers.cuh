#pragma once

#include "../Datatypes/Matrix.cu"
#include "../Datatypes/Timing.cuh"

using Launcher = Timing (*)(const Matrix<float>&, const Matrix<float>&, Matrix<float>&, unsigned long N);
