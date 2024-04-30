cmake_minimum_required(VERSION 3.0...3.29)
project(dcgan)

set(Torch_DIR ../../dependencies/libtorch-cxx11-abi-shared-with-deps-2.2.1+cpu/libtorch/share/cmake/Torch)

find_package(Torch REQUIRED)

add_executable(dcgan dcgan.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 17)
