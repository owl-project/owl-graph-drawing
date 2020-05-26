project(owlSpringEmbedder)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/cmake"
)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

option(BENCHMARK_MODE "Print performance stats that can be parsed by benchmark scripts" OFF)
if(BENCHMARK_MODE)
    add_definitions(-DBENCHMARK_MODE=1)
endif()

find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")

find_package(GLEW REQUIRED)

set(owl_dir ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/owl)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${owl_dir}/owl/cmake/")
include(configure_owl)

set(VSNRAY_ENABLE_CUDA OFF)
set(VSNRAY_ENABLE_VIEWER OFF)
set(visionaray_dir ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/visionaray)
add_subdirectory(${visionaray_dir})

include_directories(${OWL_INCLUDES})
include_directories(${owl_dir}/owl/include)
include_directories(${visionaray_dir}/include)
include_directories(${visionaray_dir}/src)
include_directories(${visionaray_dir}/src/3rdparty/CmdLine/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/3rdparty/visionaray/config)

add_subdirectory(${owl_dir}
  external_owl
  EXCLUDE_FROM_ALL)

set(SOURCES
    gd.cu
    graph.cpp
    lbvh.cu
    main.cpp
)

cuda_compile_and_embed(embedded_spring_embedder_programs optixSpringEmbedder.cu)

cuda_add_executable(gd ${embedded_spring_embedder_programs} ${SOURCES})


target_link_libraries(gd visionaray)
target_link_libraries(gd visionaray_common)
target_link_libraries(gd ${GLEW_LIBRARY})
target_link_libraries(gd ${OWL_LIBRARIES})
