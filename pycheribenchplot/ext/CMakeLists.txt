cmake_minimum_required(VERSION 3.20)
project(fastdwarf LANGUAGES C CXX)

set(CHERISDK "" CACHE STRING "Path to the CHERI SDK directory, used to find LLVM")

set(CMAKE_SHARED_MODULE_PREFIX "")

if (CHERISDK STREQUAL "")
  message(FATAL_ERROR "Must set the CHERISDK CMake option")
endif ()

set(CMAKE_PREFIX_PATH "${CHERISDK}/sdk")
find_package(LLVM REQUIRED CONFIG NO_CMAKE_SYSTEM_PATH)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs
  DebugInfoDWARF
  AllTargetsDescs
  AllTargetsInfos
  MC
  Object
  Support)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})

# find_package(Boost REQUIRED CONFIG COMPONENTS python)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)
message(STATUS "Found PyBind11 ${pybind11_VERSION}")
message(STATUS "Using PyBind11 headers ${pybind11_INCLUDE_DIRS}")
message(STATUS "Using PyBind11 defs ${pybind11_DEFINITIONS}")
message(STATUS "Found Python ${Python3_VERSION}")
message(STATUS "Using Python headers ${Python3_INCLUDE_DIRS}")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(dwarf_lib STATIC "dwarf.cc")
target_include_directories(dwarf_lib PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(dwarf_lib PUBLIC ${LLVM_DEFINITIONS_LIST})
target_link_directories(dwarf_lib PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(dwarf_lib ${llvm_libs})
target_compile_options(dwarf_lib PRIVATE "-fPIC" "-fno-rtti" "-fno-exceptions")

add_library(pydwarf MODULE "pydwarf.cc")
target_include_directories(pydwarf PUBLIC ${pybind11_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_compile_definitions(pydwarf PUBLIC ${pybind11_DEFINITIONS})
target_link_directories(pydwarf PUBLIC ${Python3_LIBRARIES})
target_link_libraries(pydwarf PRIVATE dwarf_lib)

install(TARGETS pydwarf
  LIBRARY
  DESTINATION "lib")

add_library(pychericap MODULE "pychericap.cc")
target_include_directories(pychericap PUBLIC "${PROJECT_SOURCE_DIR}/cheri-compressed-cap")
target_include_directories(pychericap PUBLIC ${pybind11_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
target_compile_definitions(pychericap PUBLIC ${pybind11_DEFINITIONS})
target_link_directories(pychericap PUBLIC ${Python3_LIBRARIES})
install(TARGETS pychericap
  LIBRARY
  DESTINATION "lib")
