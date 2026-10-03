cmake_minimum_required(VERSION 3.18)
project(TrustCuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit 11.0 REQUIRED)

add_executable(trustcuda
	TrustCuda/Main.cu
	TrustCuda/SimpleEigentrustGPU.cu
	TrustCuda/Eigentrust.cpp
	TrustCuda/Peer.cpp
	TrustCuda/ReputationSystem.cpp
)
target_compile_features(trustcuda PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(trustcuda PRIVATE CUDA::cudart CUDA::cusparse)
