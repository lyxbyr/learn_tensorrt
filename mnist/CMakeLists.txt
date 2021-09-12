cmake_minimum_required(VERSION 3.20)
project(mnist)

set(CMAKE_CXX_STANDARD 14)


find_package(CUDA)
  if(CUDA_FOUND)
    message("CUDA is available!")
    message("CUDA Libs: ${CUDA_LIBRARIES}")
    message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
  else()
    message("CUDA NOT FOUND")
  endif(CUDA_FOUND)



#if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
    ## For TensorRT 7.0.0.11
    set(TENSORRT_INCLUDE_DIR /home/lyx/Downloads/TensorRT-8.0.1.6/include)
    set(NVINFER /home/lyx/Downloads/TensorRT-8.0.1.6/lib/libnvinfer.so)
    set(NVPARSERS /home/lyx/Downloads/TensorRT-8.0.1.6/lib/libnvparsers.so)
    set(NVONNXPARSERS /home/lyx/Downloads/TensorRT-8.0.1.6/lib/libnvonnxparser.so)

    ## For TensorRT 7.2.1.6
    #set(TENSORRT_INCLUDE_DIR /opt/TensorRT-7.2.1.6/include)
    #set(NVINFER /opt/TensorRT-7.2.1.6/lib/libnvinfer.so)
    #set(NVPARSERS /opt/TensorRT-7.2.1.6/lib/libnvparsers.so)
    #set(NVONNXPARSERS /opt/TensorRT-7.2.1.6/lib/libnvonnxparser.so)
#endif()


#if(BUILD_CUDA)
    option(TRT_AVAIL "TensorRT available" OFF)

    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES include)
    find_library(NVINFER nvinfer
            HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/x64)
    find_library(NVPARSERS nvparsers
            HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/x64)
    find_library(NVONNXPARSERS nvonnxparser
            HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/x64)
    if (NVINFER AND NVPARSERS AND NVONNXPARSERS)
        message("TensorRT is available!")
        message("NVINFER: ${NVINFER}")
        message("NVPARSERS: ${NVPARSERS}")
        message("NVONNXPARSERS: ${NVONNXPARSERS}")
        set(TRT_AVAIL ON)
    else ()
        message("TensorRT is NOT Available")
        set(TRT_AVAIL OFF)
    endif()
#endif()

set(CUDA_GEN_CODE "-gencode=arch=compute_61,code=sm_61")
set(GLOBAL_INCLUDE_PATH
    #${CMAKE_CURRENT_BINARY_DIR}/perception/proto
    #${CMAKE_CURRENT_BINARY_DIR}/perception
    ${CMAKE_CURRENT_SOURCE_DIR}
   # ${catkin_INCLUDE_DIRS}
   # ${OpenCV_INCLUDE_DIRS}
)
message("----------------":${CMAKE_CURRENT_SOURCE_DIR})
#include_directories(
#    ${GLOBAL_INCLUDE_PATH}
#)

add_executable(mnist main.cpp
                    onnxMnist.cpp
                    logger.cpp

        )

target_include_directories(mnist PUBLIC
                     #   ${GLOBAL_INCLUDE_PATH}
                        ${TENSORRT_INCLUDE_DIR}
                             ${CUDA_INCLUDE_DIRS}
)
#target_link_libraries(mnist nvonnxparser)

target_link_libraries(mnist
                ${NVINFER}
                ${NVONNXPARSERS}
                ${CUDA_LIBRARIES}
               # ${CUDA_CUBLAS_LIBRARIES}

)