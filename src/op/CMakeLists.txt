cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# Configure
set(NAME molpot_op)
project(${NAME} LANGUAGES CXX CUDA)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
# Find dependencies
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
enable_testing()

# Source files of the library
file(GLOB_RECURSE SRC_FILES src/*.cpp)
file(GLOB_RECURSE SRC_FILES ${SRC_FILES} src/*.cu)

message(STATUS "Found src files: ${SRC_FILES}")

# Build the library
set(LIBRARY ${NAME}Lib)
add_library(${LIBRARY} SHARED ${SRC_FILES})
set_property(TARGET ${LIBRARY} PROPERTY CXX_STANDARD 17)
target_include_directories(${LIBRARY} PRIVATE ${Python3_INCLUDE_DIRS} include)
target_link_libraries(${LIBRARY} ${TORCH_LIBRARIES} ${Python3_LIBRARIES})

  # For segment_coo.cu
  add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
  add_definitions(-DWITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Installation
install(TARGETS ${LIBRARY} DESTINATION ${Python3_SITEARCH}/${NAME})
file(GLOB_RECURSE PYBIND_SCRIPTS binding/*.py)
foreach(PYBIND_SCRIPT ${PYBIND_SCRIPTS})
file(RELATIVE_PATH REL_PATH ${CMAKE_SOURCE_DIR}/binding ${PYBIND_SCRIPT})
    get_filename_component(REL_DIR ${REL_PATH} DIRECTORY)
    install(FILES ${PYBIND_SCRIPT} DESTINATION ${Python3_SITEARCH}/${NAME}/${REL_DIR})
endforeach()
