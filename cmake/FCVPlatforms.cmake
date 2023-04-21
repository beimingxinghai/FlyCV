if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    if (BUILD_RV1109)
        fcv_status("current platform: RV1109 ")
        include(cmake/platform/armlinux/rk-rv1109.cmake)
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL arm 
            OR CMAKE_SYSTEM_PROCESSOR STREQUAL armhf
            OR CMAKE_SYSTEM_PROCESSOR STREQUAL armel
            OR CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
        fcv_status("current platform: ArmLinux ")
        include(cmake/platform/armlinux/armlinux.cmake)
    else()
        fcv_status("current platform: Linux ")
        include(cmake/platform/linux/linux.cmake)
    endif()
elseif (CMAKE_SYSTEM_NAME MATCHES "Windows")
    fcv_status("current platform: Windows")
	include(cmake/platform/windows/windows.cmake)
elseif (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    fcv_status("current platform: FreeBSD")
elseif (CMAKE_SYSTEM_NAME MATCHES "Android")
    include(cmake/platform/android/android_arm.cmake)
elseif (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    if(IOS)
        fcv_status("current platform: ios")
        include(cmake/platform/ios/ios_arm.cmake)
    else ()
        fcv_status("current platform: Darwin")
        include(cmake/platform/macos/macos.cmake)
    endif ()
elseif (CMAKE_SYSTEM_NAME MATCHES "iOS")
    fcv_status("current platform: iOS")
    include(cmake/platform/ios/ios_arm.cmake)
elseif (CMAKE_SYSTEM_NAME MATCHES "Emscripten")
    fcv_status("current platform: Emscripten")
    include(cmake/platform/js/javascript.cmake)
else ()
    fcv_error("other platform: ${CMAKE_SYSTEM_NAME}")
endif (CMAKE_SYSTEM_NAME MATCHES "Linux")
