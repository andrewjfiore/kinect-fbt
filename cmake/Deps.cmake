# Third-party dependencies, fetched as source archives (no git history).
# Eigen / doctest / openvr are consumed header-only via INTERFACE targets;
# SOURCE_SUBDIR is pointed at a nonexistent dir so MakeAvailable never runs
# their own (older, CMake-4-incompatible) CMakeLists.
include(FetchContent)

FetchContent_Declare(eigen3
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    SOURCE_SUBDIR cmake_skip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(njson
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip
    SOURCE_SUBDIR cmake_skip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(doctest
    URL https://github.com/doctest/doctest/archive/refs/tags/v2.4.11.zip
    SOURCE_SUBDIR cmake_skip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(openvr
    URL https://github.com/ValveSoftware/openvr/archive/refs/tags/v2.5.1.zip
    SOURCE_SUBDIR cmake_skip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(httplib
    URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.18.3.zip
    SOURCE_SUBDIR cmake_skip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)

FetchContent_MakeAvailable(eigen3 njson doctest openvr httplib)

add_library(mn_eigen INTERFACE)
target_include_directories(mn_eigen SYSTEM INTERFACE "${eigen3_SOURCE_DIR}")
add_library(mn::eigen ALIAS mn_eigen)

add_library(mn_json INTERFACE)
target_include_directories(mn_json SYSTEM INTERFACE "${njson_SOURCE_DIR}/include")
add_library(mn::json ALIAS mn_json)

add_library(mn_doctest INTERFACE)
target_include_directories(mn_doctest SYSTEM INTERFACE "${doctest_SOURCE_DIR}")
add_library(mn::doctest ALIAS mn_doctest)

add_library(mn_httplib INTERFACE)
target_include_directories(mn_httplib SYSTEM INTERFACE "${httplib_SOURCE_DIR}")
if(WIN32)
    target_link_libraries(mn_httplib INTERFACE ws2_32 crypt32)
endif()
add_library(mn::httplib ALIAS mn_httplib)

# Driver side only needs headers (openvr_driver.h).
add_library(mn_openvr_headers INTERFACE)
target_include_directories(mn_openvr_headers SYSTEM INTERFACE "${openvr_SOURCE_DIR}/headers")
add_library(mn::openvr_headers ALIAS mn_openvr_headers)

# Client side (app playspace calibration) needs the prebuilt openvr_api library.
set(MN_OPENVR_CLIENT_OK OFF)
if(MN_WITH_OPENVR_CLIENT)
    if(WIN32 AND EXISTS "${openvr_SOURCE_DIR}/lib/win64/openvr_api.lib")
        add_library(mn_openvr_client SHARED IMPORTED GLOBAL)
        set_target_properties(mn_openvr_client PROPERTIES
            IMPORTED_LOCATION "${openvr_SOURCE_DIR}/bin/win64/openvr_api.dll"
            IMPORTED_IMPLIB "${openvr_SOURCE_DIR}/lib/win64/openvr_api.lib"
            INTERFACE_INCLUDE_DIRECTORIES "${openvr_SOURCE_DIR}/headers")
        set(MN_OPENVR_CLIENT_RUNTIME "${openvr_SOURCE_DIR}/bin/win64/openvr_api.dll"
            CACHE INTERNAL "openvr client runtime library")
        set(MN_OPENVR_CLIENT_OK ON)
    elseif(UNIX AND NOT APPLE AND EXISTS "${openvr_SOURCE_DIR}/bin/linux64/libopenvr_api.so")
        # libopenvr_api.so has no SONAME, so linking it by path embeds that
        # path as DT_NEEDED. Link with -l: instead so the loader resolves it
        # by name through the $ORIGIN rpath (the lib is copied next to the
        # executable).
        add_library(mn_openvr_client INTERFACE)
        target_include_directories(mn_openvr_client INTERFACE "${openvr_SOURCE_DIR}/headers")
        target_link_directories(mn_openvr_client INTERFACE "${openvr_SOURCE_DIR}/bin/linux64")
        target_link_libraries(mn_openvr_client INTERFACE "-l:libopenvr_api.so")
        set(MN_OPENVR_CLIENT_RUNTIME "${openvr_SOURCE_DIR}/bin/linux64/libopenvr_api.so"
            CACHE INTERNAL "openvr client runtime library")
        set(MN_OPENVR_CLIENT_OK ON)
    endif()
endif()
if(MN_OPENVR_CLIENT_OK)
    message(STATUS "OpenVR client library available (playspace calibration enabled)")
else()
    message(STATUS "OpenVR client library NOT available (playspace calibration disabled)")
endif()
