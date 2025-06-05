cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# Configure
set(NAME molpot_op)
project(${NAME} LANGUAGES CXX)

# Force disable CUDA
option(ENABLE_CUDA "Enable CUDA support" OFF)
set(ENABLE_CUDA OFF CACHE BOOL "" FORCE)

# Set PyTorch CPU-only variables
set(USE_CUDA OFF CACHE BOOL "Use CUDA" FORCE)
set(USE_CUDNN OFF CACHE BOOL "Use CUDNN" FORCE)
set(TORCH_CUDA_ARCH_LIST "" CACHE STRING "TORCH_CUDA_ARCH_LIST" FORCE)

# Force CPU-only environment for PyTorch
set(ENV{USE_CUDA} "0")
set(ENV{USE_CUDNN} "0")

# Skip CUDA checks
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND AND CUDAToolkit_NVCC_EXECUTABLE)
  message(STATUS "NVCC found: ${CUDAToolkit_NVCC_EXECUTABLE}")
  # But we're forcing CPU-only anyway
  message(STATUS "CUDA found but disabled by configuration")
else()
  message(STATUS "NVCC not found. CUDA features will be disabled.")
endif()

# Find dependencies
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Special handling for PyTorch with potential CUDA dependency
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
  unset(ENV{TORCH_CUDA_ARCH_LIST})
endif()

# Try to find Torch with CUDA disabled
set(_OLD_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
get_filename_component(_CONDA_PREFIX "$ENV{CONDA_PREFIX}" ABSOLUTE)
set(CMAKE_PREFIX_PATH ${_CONDA_PREFIX} ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)
set(CMAKE_PREFIX_PATH ${_OLD_CMAKE_PREFIX_PATH})

enable_testing()

# Source files of the library
file(GLOB_RECURSE SRC_FILES src/*.cpp)
# Explicitly exclude CUDA files
if(ENABLE_CUDA)
  file(GLOB_RECURSE SRC_FILES ${SRC_FILES} src/*.cu)
  # For segment_coo.cu
  add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
  add_definitions(-DWITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

message(STATUS "Found src files: ${SRC_FILES}")

# Build the library
set(LIBRARY ${NAME}Lib)
add_library(${LIBRARY} SHARED ${SRC_FILES})
set_property(TARGET ${LIBRARY} PROPERTY CXX_STANDARD 17)
target_include_directories(${LIBRARY} PRIVATE ${Python3_INCLUDE_DIRS} include)
target_link_libraries(${LIBRARY} ${TORCH_LIBRARIES} ${Python3_LIBRARIES})

# Add compile definitions for Windows DLL export/import
if(WIN32)
  target_compile_definitions(${LIBRARY} PRIVATE
    torchscatter_EXPORTS
    C10_BUILD_SHARED_LIBS
    TORCH_API=__declspec(dllexport)
    C10_API=__declspec(dllexport)
  )
endif()

# Define suffix for Python extension on Windows (.pyd)
if(WIN32)
  set_target_properties(${LIBRARY} PROPERTIES SUFFIX ".pyd")
endif()

# Installation
install(TARGETS ${LIBRARY} DESTINATION ${Python3_SITEARCH}/${NAME})
file(GLOB_RECURSE PYBIND_SCRIPTS binding/*.py)
foreach(PYBIND_SCRIPT ${PYBIND_SCRIPTS})
file(RELATIVE_PATH REL_PATH ${CMAKE_SOURCE_DIR}/binding ${PYBIND_SCRIPT})
    get_filename_component(REL_DIR ${REL_PATH} DIRECTORY)
    install(FILES ${PYBIND_SCRIPT} DESTINATION ${Python3_SITEARCH}/${NAME}/${REL_DIR})
endforeach()