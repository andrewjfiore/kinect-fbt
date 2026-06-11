add_library(mn_capture_mock STATIC src/mock.cpp)
target_include_directories(mn_capture_mock PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mn_capture_mock PUBLIC mn_core)
