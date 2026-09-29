add_library(mn_core STATIC
    src/log.cpp
    src/events.cpp
    src/net.cpp
    src/skeleton.cpp
    src/filters.cpp
    src/calibration.cpp
    src/fusion.cpp
    src/mapping.cpp
    src/projection.cpp
    src/config.cpp
    src/pipeline.cpp)

target_include_directories(mn_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mn_core PUBLIC mn::eigen mn::json)
if(WIN32)
    target_link_libraries(mn_core PUBLIC ws2_32)
endif()
