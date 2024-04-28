cmake_minimum_required(VERSION 3.27)

project(onnx_inference_example C)

set(CMAKE_C_STANDARD 17)

set(ONNX_BUILD_CONFIG RelWithDebInfo)

include(CPM.cmake)

CPMAddPackage("gh:raysan5/raylib#5.0")

CPMAddPackage(
        NAME onnxruntime
        VERSION 1.16.3
        GIT_REPOSITORY "https://github.com/microsoft/onnxruntime"
)

add_custom_target(
        build_onnxruntime
        COMMAND ${onnxruntime_SOURCE_DIR}/build.sh --config ${ONNX_BUILD_CONFIG} --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64
        COMMENT "Building ONNX Runtime"
)

include_directories(${raylib_SOURCE_DIR}/src)

include_directories(${onnxruntime_SOURCE_DIR}/include)

add_executable(onnx_inference_example main.c
        inference.h)

add_dependencies(onnx_inference_example build_onnxruntime)

# Link onnxruntime library
find_library(ONNXRUNTIME_LIB NAMES onnxruntime PATHS ${onnxruntime_SOURCE_DIR}/build/MacOS/${ONNX_BUILD_CONFIG})

if (ONNXRUNTIME_LIB)
    target_link_libraries(onnx_inference_example ${ONNXRUNTIME_LIB})
else()
    message(FATAL_ERROR "onnxruntime library not found!")
endif()

target_link_libraries(onnx_inference_example raylib)
