cmake_minimum_required(VERSION 3.26)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(
  PeachyIrAgent
  LANGUAGES CXX)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CMAKE_CXX_STANDARD 17)
endif()

enable_testing()

set(PROJECT_ROOT_DIR ${CMAKE_SOURCE_DIR})

add_subdirectory(lib)
add_subdirectory(bin)
add_subdirectory(test)

configure_file(
  include/peachy_ir_agent/config.h.in
  ${CMAKE_BINARY_DIR}/peachy_ir_agent/config.h)
