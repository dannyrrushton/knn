cmake_minimum_required(VERSION 3.27)
project(knn LANGUAGES CXX CUDA)

# RT-core K-nearest-neighbour queries: Morrical et al., "Hardware Accelerated KNN Queries using
# Ray-Time Marching", HPG 2023. The same sources are also built directly by the spectral renderer's
# CMakeLists (as knn/...), so keep the target layout in sync with it.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

set(OPTIX_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/optix/include"
    CACHE PATH "Path to the OptiX SDK include directory")

add_library(optix_function_table STATIC src/optix_function_table.cpp)
target_include_directories(optix_function_table PUBLIC ${OPTIX_INCLUDE_DIR})
target_link_libraries(optix_function_table PUBLIC CUDA::cudart ${CMAKE_DL_LIBS})

# Intersection and ray generation programs, compiled to OptiX IR and loaded at runtime.
add_library(knn_device OBJECT src/knn_device.cu)
set_target_properties(knn_device PROPERTIES CUDA_OPTIX_COMPILATION ON)
target_include_directories(knn_device PRIVATE ${OPTIX_INCLUDE_DIR} include src)
target_compile_options(knn_device PRIVATE -lineinfo -diag-suppress 20044)

add_library(knn STATIC src/knn.cpp src/knn_kernels.cu)
add_dependencies(knn knn_device)
target_include_directories(knn PUBLIC include PRIVATE src)
target_link_libraries(knn PUBLIC optix_function_table)
target_compile_definitions(knn PRIVATE KNN_DEVICE_CODE_PATH="$<TARGET_OBJECTS:knn_device>")

# Synthetic benchmark and validation (Section 4.1).
add_executable(knn_bench bench/knn_bench.cpp)
target_link_libraries(knn_bench PRIVATE knn)
