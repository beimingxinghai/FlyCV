# ==================== internal module dependencies ==================== #
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

# ==================== libjpeg-turbo ==================== #
option(WITH_LIB_JPEG_TURBO "Turn this ON when enable jpeg file support with libjpeg-turbo" ON)

if(BUILD_FCV_MEDIA_IO)
    if(WITH_FCV_IMGCODECS)
        if(WITH_LIB_JPEG_TURBO)
            include(external/libjpeg-turbo)
            add_definitions(-DWITH_LIB_JPEG_TURBO)
        endif(WITH_LIB_JPEG_TURBO)
    endif(WITH_FCV_IMGCODECS)
endif(BUILD_FCV_MEDIA_IO)

# ==================== libpng ==================== #
option(WITH_LIB_PNG "Turn this ON when enable png file support with libpng" ON)

if(BUILD_FCV_MEDIA_IO)
    if(WITH_FCV_IMGCODECS)
        if(WITH_LIB_PNG)
            include(external/libpng)
            include(external/zlib)
            add_dependencies(libpng zlib)
            add_definitions(-DWITH_LIB_PNG)
        endif(WITH_LIB_PNG)
    endif(WITH_FCV_IMGCODECS)
endif(BUILD_FCV_MEDIA_IO)


# ==================== cuda ==================== #
option(WITH_CUDA_SUPPORT "Turn ON CUDA support" OFF)
option(WITH_CUDA_SCP "Turn ON CUDA Separate Compilation" ON)

if (WITH_CUDA_SUPPORT)
    add_definitions(-DUSE_CUDA)
    include(external/cuda)
endif()
