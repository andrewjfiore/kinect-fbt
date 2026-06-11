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
    if(WIN32)
        add_custom_command(TARGET marionette POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_PROPERTY:mn_openvr_client,IMPORTED_LOCATION>
                    $<TARGET_FILE_DIR:marionette>)
    endif()
endif()
