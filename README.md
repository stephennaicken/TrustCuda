TrustCuda
=========

Implementation of various trust metrics on the NVidia Cuda platform

Building
--------

Requires CMake 3.18+ and the CUDA Toolkit 11 or newer (for the cuSPARSE generic API).

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
    cmake --build build
    ./build/trustcuda

The Visual Studio 2012 / CUDA 6.0 project (`TrustCuda.sln`) predates the sparse implementation and no longer builds.
