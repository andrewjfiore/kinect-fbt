add_library(mn_endpoint_osc STATIC src/osc.cpp)
target_include_directories(mn_endpoint_osc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mn_endpoint_osc PUBLIC mn_core)
