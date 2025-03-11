cmake_minimum_required(VERSION 3.18)
project(cuda_portfolios LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA Architecture (Adjust according to your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)  # Example: Turing (75), Ampere (80/86)

# Enable device linking for atomic functions
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include CUDA headers
include_directories(${CUDA_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/main_cuda.cpp
)

# Create the CUDA executable
add_executable(cuda_stocks ${SOURCES})

# Enable CUDA as a separate language
set_target_properties(cuda_stocks PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link against CUDA runtime
target_link_libraries(cuda_stocks PRIVATE ${CUDA_LIBRARIES} cudart)
