cmake_minimum_required(VERSION 3.10)
project(TSDF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-std=c++11 -fpermissive -pthread -lrt -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(PCL REQUIRED)
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS -arch=sm_60;-std=c++11;-lm;)
ADD_DEFINITIONS(-fpermissive)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/include
)

# solve the problem: nvcc fatal: A single input file is required for a non-link phase when an outputfile is specified
set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "")


# shared library
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
#cuda_add_library(${PROJECT_NAME} SHARED
#                src/Reader.cpp 
#                src/CUDATSDFIntegrator.cpp
#                src/CUDATSDFIntegrator.cu
#)
#target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})


# Example
cuda_add_executable(tsdf_fusion app/example_tsdf_fusion.cpp
                           src/Reader.cpp     
                           src/CUDATSDFIntegrator.cpp
                           src/CUDATSDFIntegrator.cu
)
target_link_libraries(tsdf_fusion ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
