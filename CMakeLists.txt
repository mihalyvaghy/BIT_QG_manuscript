cmake_minimum_required(VERSION 3.28)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(QGFEM)

# tester
set(SOURCES
  src/mf_neumann_neumann.cpp
  src/mf_quantum_graph.cpp
  src/mf_diagonal_quantum_graph.cpp
  src/mf_polynomial_quantum_graph.cpp
  src/tester.cpp
)

add_executable(solve_graph ${SOURCES})

target_include_directories(solve_graph
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
