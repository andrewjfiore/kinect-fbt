# Kinect for Windows SDK 2.0 backend (Windows only; gated by the parent).
file(TO_CMAKE_PATH "$ENV{KINECTSDK20_DIR}" KINECT2_ROOT)

add_library(mn_capture_kinect_v2 STATIC src/kinect2.cpp)
target_include_directories(mn_capture_kinect_v2
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE "${KINECT2_ROOT}/inc")
target_link_libraries(mn_capture_kinect_v2
    PUBLIC mn_core
    PRIVATE "${KINECT2_ROOT}/Lib/x64/Kinect20.lib")
