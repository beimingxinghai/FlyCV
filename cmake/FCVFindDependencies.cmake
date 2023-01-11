option(WITH_LIB_JPEG_TURBO "Turn this ON when enable jpeg file support with libjpeg-turbo" ON)
option(WITH_LIB_PNG "Turn this ON when enable png file support with libpng" ON)

if(BUILD_FCV_MEDIA_IO)
    if(WITH_FCV_IMGCODECS)
        # Build with jpeg
        if(WITH_LIB_JPEG_TURBO)
            include(external/libjpeg-turbo)
            add_definitions(-DWITH_LIB_JPEG_TURBO)
        endif(WITH_LIB_JPEG_TURBO)

        # Build with libpng.
        if(WITH_LIB_PNG)
            include(external/libpng)
            include(external/zlib)
            add_dependencies(fcv_libpng fcv_zlib)
            add_definitions(-DWITH_LIB_PNG)
        endif(WITH_LIB_PNG)
    endif(WITH_FCV_IMGCODECS)
endif(BUILD_FCV_MEDIA_IO)

if (BUILD_FCV_IMG_TRANSFORM)
    if(WITH_FCV_WARP_AFFINE OR WITH_FCV_WARP_PERSPECTIVE)
        set(WITH_FCV_REMAP ON)
    endif()
endif()

if (BUILD_FCV_IMG_CALCULATION)
    if(WITH_FCV_FIND_HOMOGRAPHY)
        set(WITH_FCV_MATRIX_MUL ON)
    endif()
endif()

if (CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
    set(WITH_FCV_OPENCL OFF)
endif()

if (BUILD_FCV_IMG_DRAW)
    set(WITH_FCV_LINE ON)
    if (WITH_FCV_CIRCLE)
      set(WITH_FCV_POLY_LINES ON)
    endif()
endif()

# ==================CUDA=======================
option(WITH_CUDA_SUPPORT "Turn ON CUDA support" OFF)
option(WITH_CUDA_SCP "Turn ON CUDA Separate Compilation" ON)
if(WITH_CUDA_SUPPORT)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)

    include_directories(${CUDA_INCLUDE_DIRS})
    list(APPEND FCV_LINK_DEPS ${CUDA_LIBRARIES})

    # If CMake doesn't support separable compilation, complain
    if(WITH_CUDA_SCP AND CMAKE_VERSION VERSION_LESS "2.8.10.1")
        message(SEND_ERROR "CUDA_SEPARABLE_COMPILATION isn't supported for CMake versions less than 2.8.10.1")
        set(WITH_CUDA_SCP OFF)
    endif()
# --------------------------------------------------------------------------- #
set(ARCH_FLAGS )
if(CUDA_MULTI_ARCH)
    set(ARCH_FLAGS "${ARCH_FLAGS} -gencode arch=compute_35,code=sm_35")
    set(ARCH_FLAGS "${ARCH_FLAGS} -gencode arch=compute_37,code=sm_37")
    set(ARCH_FLAGS "${ARCH_FLAGS} -gencode arch=compute_50,code=sm_50")
    if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "8")
        set(ARCH_FLAGS "${ARCH_FLAGS} -gencode arch=compute_60,code=sm_60")
        set(ARCH_FLAGS "${ARCH_FLAGS} -gencode arch=compute_61,code=sm_61")
    endif()
    if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "9")
        set(ARCH_FLAGS "${ARCH_FLAGS} -gencode arch=compute_70,code=sm_70")
    endif()
    if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "10")
        set(ARCH_FLAGS "${ARCH_FLAGS} -gencode arch=compute_72,code=sm_72")
        set(ARCH_FLAGS "${ARCH_FLAGS} -gencode arch=compute_75,code=sm_75")
    endif()
    if (CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "11")
        set(ARCH_FLAGS "${ARCH_FLAGS} -gencode arch=compute_80,code=sm_80")
        if (CUDA_VERSION_MINOR VERSION_GREATER_EQUAL "1")
            set(ARCH_FLAGS "${ARCH_FLAGS} -gencode arch=compute_86,code=sm_86")
        endif ()
    endif ()
else()
if(${CMAKE_VERSION} VERSION_LESS_EQUAL "3.13.4")
    cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto") # optional argument for arch to add
    message("ARCH_FLAGS = ${ARCH_FLAGS}")
    string(REPLACE "-gencode;" "--generate-code=" ARCH_FLAGS "${ARCH_FLAGS}")
    string(APPEND CMAKE_CUDA_FLAGS "${ARCH_FLAGS}")
else()
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
endif()
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${ARCH_FLAGS}")

if(CUDA_VERSION_MAJOR VERSION_GREATER_EQUAL "7")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --default-stream per-thread")
endif()
# --------------------------------------------------------------------------- #
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
# set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)

# We need to check this variable before starting a CUDA project - otherwise it will appear
# as set, with the default value pointing to the oldest supported architecture (52 as of CUDA 11.8)
set(USE_CMAKE_CUDA_ARCHITECTURES TRUE)
# Make sure the cuda host compiler agrees with what we're using,
# unless user overwrites it (at their own risk).
if(NOT CMAKE_CUDA_HOST_COMPILER)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=-fPIC,-Wall,-fvisibility=hidden")
set(CUDA_NVCC_FLAGS_DEBUG "-g")
set(CUDA_NVCC_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_NVCC_FLAGS}")
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

message("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
endif()
# =============================================
