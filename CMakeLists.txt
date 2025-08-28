cmake_minimum_required(VERSION 3.30)
project(cnn-server LANGUAGES CXX)
 
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
 
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
 
add_library(tannic STATIC IMPORTED)
set_target_properties(tannic PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/external/tannic-nn/external/tannic/build/libtannic.a
)

add_library(tannic-nn STATIC IMPORTED)
set_target_properties(tannic-nn PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/external/tannic-nn/build/libtannic-nn.a
)
 
target_include_directories(tannic INTERFACE
    ${CMAKE_SOURCE_DIR}/external/tannic-nn/external/tannic/include
)

target_include_directories(tannic-nn INTERFACE
    ${CMAKE_SOURCE_DIR}/external/tannic-nn/include
    ${CMAKE_SOURCE_DIR}/external/tannic-nn/external/tannic/include
)
 
find_library(OPENBLAS_LIB openblas)
if(OPENBLAS_LIB)
    message(STATUS "Found OpenBLAS: ${OPENBLAS_LIB}")
endif()
 
add_executable(cnn-server main.cpp)
 
target_link_libraries(cnn-server PRIVATE tannic-nn tannic)
 
if(OPENBLAS_LIB)
    target_link_libraries(cnn-server PRIVATE ${OPENBLAS_LIB})
endif()

target_link_libraries(cnn-server PRIVATE pthread dl)
 
target_include_directories(cnn-server PRIVATE
    include/
    model/
    ${CMAKE_SOURCE_DIR}/external/tannic-nn/include
    ${CMAKE_SOURCE_DIR}/external/tannic-nn/external/tannic/include
)