add_executable(marionette main.cpp)
target_link_libraries(marionette PRIVATE
    mn_core mn_capture_mock mn_endpoint_osc mn_endpoint_ovr_bridge)

if(MN_HAS_KINECT_V2)
    target_link_libraries(marionette PRIVATE mn_capture_kinect_v2)
    target_compile_definitions(marionette PRIVATE MN_HAS_KINECT_V2)
endif()
if(MN_HAS_KINECT_V1)
    target_link_libraries(marionette PRIVATE mn_capture_kinect_v1)
    target_compile_definitions(marionette PRIVATE MN_HAS_KINECT_V1)
endif()

if(MN_OPENVR_CLIENT_OK)
    target_link_libraries(marionette PRIVATE mn_openvr_client)
    target_compile_definitions(marionette PRIVATE MN_HAS_OPENVR_CLIENT)
    # Keep the runtime library next to the binary on both platforms.
    add_custom_command(TARGET marionette POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${MN_OPENVR_CLIENT_RUNTIME}"
                $<TARGET_FILE_DIR:marionette>)
    if(UNIX)
        set_target_properties(marionette PROPERTIES BUILD_RPATH "\$ORIGIN")
    endif()
endif()
