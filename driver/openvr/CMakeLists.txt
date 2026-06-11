# SteamVR driver: virtual generic trackers fed by the openvr_bridge endpoint.
add_library(driver_marionette SHARED src/driver.cpp)
target_link_libraries(driver_marionette PRIVATE mn_core mn::openvr_headers)
set_target_properties(driver_marionette PROPERTIES PREFIX "")

if(WIN32)
    set(MN_OVR_PLATFORM win64)
else()
    set(MN_OVR_PLATFORM linux64)
endif()

# Assemble an installable SteamVR driver folder at <build>/dist/marionette:
#   driver.vrdrivermanifest, resources/..., bin/<platform>/driver_marionette.{dll,so}
set(MN_DRIVER_DIST ${CMAKE_BINARY_DIR}/dist/marionette)
add_custom_command(TARGET driver_marionette POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${MN_DRIVER_DIST}/bin/${MN_OVR_PLATFORM}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:driver_marionette>
            ${MN_DRIVER_DIST}/bin/${MN_OVR_PLATFORM}/
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/assets
            ${MN_DRIVER_DIST}/
    COMMENT "Assembling SteamVR driver folder: ${MN_DRIVER_DIST}")
