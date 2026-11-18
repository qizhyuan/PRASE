cmake_minimum_required(VERSION 3.18)
project(prase LANGUAGES NONE)

option(PRASE_BUILD_NATIVE "Build the PARIS native engine" ON)
if(NOT PRASE_BUILD_NATIVE)
    return()
endif()
enable_language(CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

pybind11_add_module(_paris NO_EXTRAS cpp/bindings.cpp cpp/paris.cpp)
target_compile_features(_paris PRIVATE cxx_std_17)
target_link_libraries(_paris PRIVATE Threads::Threads)
if(MSVC)
    target_compile_options(_paris PRIVATE /fp:precise)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_paris PRIVATE -fno-fast-math -ffp-contract=off)
endif()

install(TARGETS _paris LIBRARY DESTINATION prase RUNTIME DESTINATION prase)
