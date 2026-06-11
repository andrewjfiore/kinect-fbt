# Web dashboard: embedded single-file UI + JSON API on cpp-httplib.
set(MN_WEB_HTML ${CMAKE_SOURCE_DIR}/web/index.html)
set(MN_WEB_HEADER ${CMAKE_BINARY_DIR}/generated/web_embedded.hpp)

add_custom_command(OUTPUT ${MN_WEB_HEADER}
    COMMAND ${CMAKE_COMMAND} -DINPUT=${MN_WEB_HTML} -DOUTPUT=${MN_WEB_HEADER}
            -DVAR=kIndexHtml -P ${CMAKE_SOURCE_DIR}/cmake/EmbedFile.cmake
    DEPENDS ${MN_WEB_HTML} ${CMAKE_SOURCE_DIR}/cmake/EmbedFile.cmake
    COMMENT "Embedding web/index.html")

add_library(mn_dashboard STATIC src/dashboard.cpp ${MN_WEB_HEADER})
target_include_directories(mn_dashboard
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(mn_dashboard PUBLIC mn_core PRIVATE mn::httplib)
