#pragma once

#include "../Datatypes/Matrix.cu"

using Timing = void;
using Launcher = Timing (*)(const Matrix<float>&, const Matrix<float>&, Matrix<float>&, unsigned long N);
