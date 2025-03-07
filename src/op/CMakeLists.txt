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

# List of directories containing  test scripts
set(PYTHON_TEST_DIRS
  ${CMAKE_SOURCE_DIR}/tests/python
)

# Move test scripts to a test folder in the build directory, create test folder if necessary
add_custom_target(copy_test ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/test
)

# Add the data directory as well
add_custom_command(
  TARGET copy_test POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_SOURCE_DIR}/tests/molecules
  ${CMAKE_BINARY_DIR}/test/molecules
)

# Copy each test to the binary dir and add it
file(GLOB TEST_SCRIPTS ${PYTHON_TEST_DIRS}/*.py)
foreach(TEST_SCRIPT ${TEST_SCRIPTS})
  get_filename_component(TEST_NAME ${TEST_SCRIPT} NAME_WE)
  add_custom_command(
    TARGET copy_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SCRIPT} ${CMAKE_BINARY_DIR}/test/python/${TEST_SCRIPTS}
  )
  add_test(NAME ${TEST_NAME} COMMAND pytest -v ${CMAKE_BINARY_DIR}/test/python/${TEST_SCRIPT})
endforeach()

# Installation
install(TARGETS ${LIBRARY} DESTINATION ${Python3_SITEARCH}/${NAME})
file(GLOB_RECURSE PYBIND_SCRIPTS binding/*.py)
foreach(PYBIND_SCRIPT ${PYBIND_SCRIPTS})
file(RELATIVE_PATH REL_PATH ${CMAKE_SOURCE_DIR}/binding ${PYBIND_SCRIPT})
    get_filename_component(REL_DIR ${REL_PATH} DIRECTORY)
    install(FILES ${PYBIND_SCRIPT} DESTINATION ${Python3_SITEARCH}/${NAME}/${REL_DIR})
endforeach()
