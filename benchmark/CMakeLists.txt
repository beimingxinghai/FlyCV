project(flycv_bench)

# set benchs project
set(TARGET_NAME flycv_bench)
set(bench_sources "")
set(FCV_MODULES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../modules)
set(FCV_CPP_BENCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/modules)

if(WITH_CUDA_SUPPORT)
    set(FCV_CUDA_BENCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda/modules)
endif()

get_subdir_list(BENCH_FCV_BUILD_MODULES ${FCV_MODULES_DIR})

foreach(module_name ${BENCH_FCV_BUILD_MODULES})
    string(TOUPPER ${module_name} _name)
    if(${BUILD_FCV_${_name}})
        if(EXISTS "${FCV_CPP_BENCH_DIR}/${module_name}/${module_name}_bench.cpp")
            list(APPEND bench_sources "${FCV_CPP_BENCH_DIR}/${module_name}/${module_name}_bench.cpp")
        else()
            get_subdir_list(BENCH_FCV_WITH_MODULES ${FCV_MODULES_DIR}/${module_name})
            foreach(submodule_name ${BENCH_FCV_WITH_MODULES})
                string(TOUPPER ${submodule_name} _subname)
                if(${WITH_FCV_${_subname}})
                    if(EXISTS "${FCV_CPP_BENCH_DIR}/${module_name}/${submodule_name}_bench.cpp")
                        list(APPEND bench_sources "${FCV_CPP_BENCH_DIR}/${module_name}/${submodule_name}_bench.cpp")
                        file(GLOB extend_sources "${FCV_CPP_BENCH_DIR}/${module_name}/${submodule_name}_*_bench_cuda.cpp")
                        list(APPEND test_sources ${extend_sources})
                    endif()
                endif()
            endforeach()
        endif()

        if(WITH_CUDA_SUPPORT)
            if(EXISTS "${FCV_CUDA_BENCH_DIR}/${module_name}/cuda_${module_name}_bench.cpp")
                list(APPEND bench_sources "${FCV_CUDA_BENCH_DIR}/${module_name}/${module_name}_bench_cuda.cpp")
            else()
                get_subdir_list(BENCH_FCV_WITH_MODULES ${FCV_MODULES_DIR}/${module_name})
                foreach(submodule_name ${BENCH_FCV_WITH_MODULES})
                    string(TOUPPER ${submodule_name} _subname)
                    if(${WITH_FCV_${_subname}})
                        if(EXISTS "${FCV_CUDA_BENCH_DIR}/${module_name}/${submodule_name}_bench_cuda.cpp")
                            list(APPEND bench_sources "${FCV_CUDA_BENCH_DIR}/${module_name}/${submodule_name}_bench_cuda.cpp")
                            file(GLOB extend_sources "${FCV_CUDA_BENCH_DIR}/${module_name}/${submodule_name}_*_bench_cuda.cpp")
                            list(APPEND test_sources ${extend_sources})
                        endif()
                    endif()
                endforeach()
            endif()
        endif()
    endif()
endforeach()

add_executable(${TARGET_NAME}
    ${bench_sources}
)

target_include_directories(${TARGET_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}/
    ${BENCHMARK_INCLUDE_DIR}
    ${CMAKE_INSTALL_PREFIX}/flycv/include
)

target_link_libraries(${TARGET_NAME}
    ${FCV_TARGET_NAME}
    ${BENCHMARK_LIBRARIES}
    ${FCV_EXPORT_LIBS}
    ${CMAKE_DL_LIBS}
)

install(TARGETS ${TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/)
