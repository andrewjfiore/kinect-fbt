# Kinect for Windows SDK 1.8 backend (Windows only; gated by the parent).
file(TO_CMAKE_PATH "$ENV{KINECTSDK10_DIR}" KINECT1_ROOT)

add_library(mn_capture_kinect_v1 STATIC src/kinect1.cpp)
target_include_directories(mn_capture_kinect_v1
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE "${KINECT1_ROOT}/inc")
target_link_libraries(mn_capture_kinect_v1
    PUBLIC mn_core
    PRIVATE "${KINECT1_ROOT}/lib/amd64/Kinect10.lib")
