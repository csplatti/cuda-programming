# Setup & Credits for file contents

The contents of install_deps.bash and ./Sources/ach.h were obtained from the Extending Algorithms nodebook from [NVIDIA Accelerated Computing Hub](https://github.com/NVIDIA/accelerated-computing-hub/)'s [CUDA Made Easy Course](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/gpu-cpp-tutorial).

To install the necessary dependencies to run pr_thrust.cu, run the following terminal command in this directory.
```bash
bash install_deps.bash
```

## Compiling with Thrust
The following command may be needed to compile a CUDA program which uses the thrust library
```bash
nvcc --extended-lambda -o pr_thrust pr_thrust.cu
```