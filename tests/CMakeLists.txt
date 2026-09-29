add_executable(mn_tests
    main.cpp
    test_filters.cpp
    test_calibration.cpp
    test_fusion.cpp
    test_mapping.cpp
    test_projection.cpp
    test_config.cpp
    test_pipeline.cpp
    test_mock.cpp
    test_osc.cpp
    test_protocol.cpp
    test_events.cpp
    test_watchdog.cpp
    test_e2e.cpp)

target_link_libraries(mn_tests PRIVATE
    mn_core mn_capture_mock mn_endpoint_osc mn_endpoint_ovr_bridge mn::doctest)

if(MN_BUILD_DASHBOARD)
    target_sources(mn_tests PRIVATE test_dashboard.cpp)
    target_link_libraries(mn_tests PRIVATE mn_dashboard mn::httplib)
endif()

add_test(NAME mn_tests COMMAND mn_tests)

# Hardware smoke tests: run with real sensors attached; exit 77 = skipped.
add_subdirectory(hardware)
