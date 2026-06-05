cmake_minimum_required(VERSION 3.18)
project(mvo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MVO_CVLIB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cvlib" CACHE PATH
    "Path to bundled cvlib headers and prebuilt library")
set(MVO_CVLIB_PLATFORM "" CACHE STRING
    "Bundled cvlib platform directory override, e.g. msvc or linux")

find_package(OpenCV REQUIRED COMPONENTS
    core
    imgcodecs
    imgproc
    video
    videoio
    features2d
    calib3d
)

set(MVO_CVLIB_INCLUDE_DIR "${MVO_CVLIB_ROOT}/include")
if(NOT EXISTS "${MVO_CVLIB_INCLUDE_DIR}/types.h")
    message(FATAL_ERROR
        "Bundled cvlib headers not found: ${MVO_CVLIB_INCLUDE_DIR}. "
        "Run bundle_cvlib.ps1 or bundle_cvlib.sh to refresh the package.")
endif()

add_library(cvlib_core UNKNOWN IMPORTED GLOBAL)
set_target_properties(cvlib_core PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MVO_CVLIB_INCLUDE_DIR}"
)

set(_mvo_cvlib_imported_configs)
if(MVO_CVLIB_PLATFORM)
    set(_mvo_cvlib_platform_dirs "${MVO_CVLIB_PLATFORM}")
    if(MVO_CVLIB_PLATFORM MATCHES "^(msvc|windows)$")
        set(_mvo_cvlib_library_names cvlib_core.lib)
    elseif(MVO_CVLIB_PLATFORM MATCHES "^(linux|unix)$")
        set(_mvo_cvlib_library_names libcvlib_core.a libcvlib_core.so)
    elseif(MVO_CVLIB_PLATFORM MATCHES "^(macos|darwin)$")
        set(_mvo_cvlib_library_names libcvlib_core.a libcvlib_core.dylib)
    else()
        set(_mvo_cvlib_library_names
            cvlib_core.lib
            libcvlib_core.a
            libcvlib_core.so
            libcvlib_core.dylib)
    endif()
elseif(MSVC)
    set(_mvo_cvlib_library_names cvlib_core.lib)
    set(_mvo_cvlib_platform_dirs msvc windows)
elseif(WIN32)
    set(_mvo_cvlib_library_names cvlib_core.lib libcvlib_core.a)
    set(_mvo_cvlib_platform_dirs windows)
elseif(APPLE)
    set(_mvo_cvlib_library_names libcvlib_core.a libcvlib_core.dylib)
    set(_mvo_cvlib_platform_dirs macos darwin)
else()
    set(_mvo_cvlib_library_names libcvlib_core.a libcvlib_core.so)
    set(_mvo_cvlib_platform_dirs linux unix)
endif()

foreach(_mvo_config IN ITEMS Release Debug RelWithDebInfo MinSizeRel)
    string(TOLOWER "${_mvo_config}" _mvo_config_lower)
    string(TOUPPER "${_mvo_config}" _mvo_config_upper)
    set(_mvo_cvlib_library)
    foreach(_mvo_platform_dir IN LISTS _mvo_cvlib_platform_dirs)
        foreach(_mvo_library_name IN LISTS _mvo_cvlib_library_names)
            foreach(_mvo_candidate IN ITEMS
                    "${MVO_CVLIB_ROOT}/lib/${_mvo_platform_dir}/${_mvo_config_lower}/${_mvo_library_name}"
                    "${MVO_CVLIB_ROOT}/lib/${_mvo_config_lower}/${_mvo_library_name}"
                    "${MVO_CVLIB_ROOT}/lib/${_mvo_library_name}")
                if(NOT _mvo_cvlib_library AND EXISTS "${_mvo_candidate}")
                    set(_mvo_cvlib_library "${_mvo_candidate}")
                endif()
            endforeach()
        endforeach()
    endforeach()
    if(_mvo_cvlib_library)
        list(APPEND _mvo_cvlib_imported_configs "${_mvo_config}")
        set_property(TARGET cvlib_core APPEND PROPERTY
            IMPORTED_CONFIGURATIONS "${_mvo_config}")
        set_target_properties(cvlib_core PROPERTIES
            "IMPORTED_LOCATION_${_mvo_config_upper}"
            "${_mvo_cvlib_library}")
    endif()
endforeach()

if(NOT _mvo_cvlib_imported_configs)
    message(FATAL_ERROR
        "Bundled cvlib library not found under ${MVO_CVLIB_ROOT}/lib. "
        "Run bundle_cvlib.ps1 or bundle_cvlib.sh to create it.")
endif()
if("Release" IN_LIST _mvo_cvlib_imported_configs)
    if(NOT "Debug" IN_LIST _mvo_cvlib_imported_configs)
        set_target_properties(cvlib_core PROPERTIES
            MAP_IMPORTED_CONFIG_DEBUG Release)
    endif()
    if(NOT "RelWithDebInfo" IN_LIST _mvo_cvlib_imported_configs)
        set_target_properties(cvlib_core PROPERTIES
            MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
    endif()
    if(NOT "MinSizeRel" IN_LIST _mvo_cvlib_imported_configs)
        set_target_properties(cvlib_core PROPERTIES
            MAP_IMPORTED_CONFIG_MINSIZEREL Release)
    endif()
endif()

set(MVO_DBOW2_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/DBoW2" CACHE PATH
    "Path to bundled DBoW2 sources and prebuilt library")
set(_mvo_dbow2_imported_configs)
if(MSVC)
    foreach(_mvo_config IN ITEMS Release Debug RelWithDebInfo MinSizeRel)
        string(TOLOWER "${_mvo_config}" _mvo_config_lower)
        string(TOUPPER "${_mvo_config}" _mvo_config_upper)
        set(_mvo_dbow2_library)
        foreach(_mvo_candidate IN ITEMS
                "${MVO_DBOW2_ROOT}/lib/${_mvo_config}/DBoW2.lib"
                "${MVO_DBOW2_ROOT}/lib/${_mvo_config_lower}/DBoW2.lib"
                "${MVO_DBOW2_ROOT}/lib/DBoW2.lib")
            if(NOT _mvo_dbow2_library AND EXISTS "${_mvo_candidate}")
                set(_mvo_dbow2_library "${_mvo_candidate}")
            endif()
        endforeach()
        if(_mvo_dbow2_library)
            list(APPEND _mvo_dbow2_imported_configs "${_mvo_config}")
            if(NOT TARGET DBoW2)
                add_library(DBoW2 UNKNOWN IMPORTED GLOBAL)
                set_target_properties(DBoW2 PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${MVO_DBOW2_ROOT}")
                target_link_libraries(DBoW2 INTERFACE ${OpenCV_LIBS})
            endif()
            set_property(TARGET DBoW2 APPEND PROPERTY
                IMPORTED_CONFIGURATIONS "${_mvo_config}")
            set_target_properties(DBoW2 PROPERTIES
                "IMPORTED_LOCATION_${_mvo_config_upper}" "${_mvo_dbow2_library}")
        endif()
    endforeach()
    if(TARGET DBoW2 AND "Release" IN_LIST _mvo_dbow2_imported_configs)
        if(NOT "RelWithDebInfo" IN_LIST _mvo_dbow2_imported_configs)
            set_target_properties(DBoW2 PROPERTIES
                MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
        endif()
        if(NOT "MinSizeRel" IN_LIST _mvo_dbow2_imported_configs)
            set_target_properties(DBoW2 PROPERTIES
                MAP_IMPORTED_CONFIG_MINSIZEREL Release)
        endif()
    endif()
else()
    set(_mvo_dbow2_library)
    foreach(_mvo_dbow2_name IN ITEMS libDBoW2.a libDBoW2.so)
        if(NOT _mvo_dbow2_library AND EXISTS "${MVO_DBOW2_ROOT}/lib/${_mvo_dbow2_name}")
            set(_mvo_dbow2_library "${MVO_DBOW2_ROOT}/lib/${_mvo_dbow2_name}")
        endif()
    endforeach()
    if(_mvo_dbow2_library)
        add_library(DBoW2 UNKNOWN IMPORTED GLOBAL)
        set_target_properties(DBoW2 PROPERTIES
            IMPORTED_LOCATION "${_mvo_dbow2_library}"
            INTERFACE_INCLUDE_DIRECTORIES "${MVO_DBOW2_ROOT}")
        target_link_libraries(DBoW2 INTERFACE ${OpenCV_LIBS})
    endif()
endif()

if(NOT TARGET DBoW2)
    add_subdirectory(thirdparty/DBoW2)
endif()

add_executable(mvo_cvlib
    src/app.cpp
    src/bundle_adjustment.cpp
    src/config.cpp
    src/converter.cpp
    src/feature.cpp
    src/frame_source.cpp
    src/init.cpp
    src/loop_closure.cpp
    src/map_data.cpp
    src/parameters.cpp
    src/pose_estimation.cpp
    src/visualization.cpp
    src/cvlib_main.cpp
)

target_include_directories(mvo_cvlib BEFORE PRIVATE
    ${MVO_CVLIB_INCLUDE_DIR}
)
target_include_directories(mvo_cvlib PRIVATE
    include
    thirdparty/DBoW2
)

target_link_libraries(mvo_cvlib PRIVATE
    cvlib_core
    DBoW2
    ${OpenCV_LIBS}
)

if(MSVC)
    target_compile_options(mvo_cvlib PRIVATE /W4 /utf-8)
else()
    target_compile_options(mvo_cvlib PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(rerun_sdk CONFIG QUIET)
if(NOT rerun_sdk_FOUND)
    include(FetchContent)
    set(MVO_RERUN_SDK_URL
        "https://github.com/rerun-io/rerun/releases/download/0.33.0/rerun_cpp_sdk.zip"
        CACHE STRING "Rerun C++ SDK bundle URL")
    FetchContent_Declare(rerun_sdk
        URL "${MVO_RERUN_SDK_URL}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_GetProperties(rerun_sdk)
    if(NOT rerun_sdk_POPULATED)
        if(POLICY CMP0169)
            cmake_policy(PUSH)
            cmake_policy(SET CMP0169 OLD)
        endif()
        FetchContent_Populate(rerun_sdk)
        if(POLICY CMP0169)
            cmake_policy(POP)
        endif()
    endif()
    set(_rerun_arrow_cmake
        "${rerun_sdk_SOURCE_DIR}/download_and_build_arrow.cmake")
    if(EXISTS "${_rerun_arrow_cmake}")
        file(READ "${_rerun_arrow_cmake}" _rerun_arrow_cmake_content)
        string(REPLACE
            "-DARROW_MIMALLOC=ON"
            "-DARROW_MIMALLOC=OFF"
            _rerun_arrow_cmake_content
            "${_rerun_arrow_cmake_content}")
        string(REPLACE
            [[        BUILD_BYPRODUCTS ${ARROW_LIBRARY_FILE} ${ARROW_BUNDLED_DEPENDENCIES_FILE}]]
            [[        BUILD_BYPRODUCTS ${ARROW_LIBRARY_FILE}]]
            _rerun_arrow_cmake_content
            "${_rerun_arrow_cmake_content}")
        string(REPLACE
            [[        # We have to explicitly opt in the arrow bundled dependencies, otherwise we're missing the symbols for mimalloc.
        add_library(arrow_targetBundledDeps STATIC IMPORTED)
        add_dependencies(arrow_targetBundledDeps arrow_cpp)
        set_target_properties(arrow_targetBundledDeps PROPERTIES
            IMPORTED_LOCATION ${ARROW_BUNDLED_DEPENDENCIES_FILE}
        )
        target_link_libraries(rerun_arrow_target INTERFACE arrow_targetBundledDeps)]]
            [[        # MVO disables Arrow mimalloc for CMake 4 compatibility, so
        # there is no bundled mimalloc dependency archive to link.]]
            _rerun_arrow_cmake_content
            "${_rerun_arrow_cmake_content}")
        file(WRITE "${_rerun_arrow_cmake}"
            "${_rerun_arrow_cmake_content}")
    endif()
    set(_rerun_top_cmake "${rerun_sdk_SOURCE_DIR}/CMakeLists.txt")
    if(EXISTS "${_rerun_top_cmake}")
        file(READ "${_rerun_top_cmake}" _rerun_top_cmake_content)
        string(REPLACE
            [[if(RERUN_DOWNLOAD_AND_BUILD_ARROW AND NOT RERUN_ARROW_LINK_SHARED)
    get_target_property(RERUN_ARROW_LIBRARY_FILE rerun_arrow_target LOCATION)
    get_target_property(RERUN_ARROW_BUNDLED_DEPENDENCIES_FILE arrow_targetBundledDeps LOCATION)
    install(FILES
        ${RERUN_ARROW_LIBRARY_FILE}
        ${RERUN_ARROW_BUNDLED_DEPENDENCIES_FILE}
        DESTINATION lib
    )
endif()]]
            [[if(RERUN_DOWNLOAD_AND_BUILD_ARROW AND NOT RERUN_ARROW_LINK_SHARED)
    get_target_property(RERUN_ARROW_LIBRARY_FILE rerun_arrow_target LOCATION)
    install(FILES
        ${RERUN_ARROW_LIBRARY_FILE}
        DESTINATION lib
    )
endif()]]
            _rerun_top_cmake_content
            "${_rerun_top_cmake_content}")
        file(WRITE "${_rerun_top_cmake}" "${_rerun_top_cmake_content}")
    endif()
    add_subdirectory("${rerun_sdk_SOURCE_DIR}" "${rerun_sdk_BINARY_DIR}")
endif()
target_link_libraries(mvo_cvlib PRIVATE rerun_sdk)
target_compile_definitions(mvo_cvlib PRIVATE MVO_HAS_RERUN=1)
