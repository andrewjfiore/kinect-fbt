# Hardware smoke test binary: exercises whatever real sensors are attached.
# Exits 77 ("skipped") when no compiled backend finds hardware, so plain
# `ctest` stays green on machines without Kinects. Run explicitly with:
#   ctest --test-dir build -L hardware -C Release --output-on-failure
add_executable(mn_hwsmoke hw_smoke.cpp)
target_link_libraries(mn_hwsmoke PRIVATE mn_core)

if(MN_HAS_KINECT_V2)
    target_link_libraries(mn_hwsmoke PRIVATE mn_capture_kinect_v2)
    target_compile_definitions(mn_hwsmoke PRIVATE MN_HAS_KINECT_V2)
endif()
if(MN_HAS_KINECT_V1)
    target_link_libraries(mn_hwsmoke PRIVATE mn_capture_kinect_v1)
    target_compile_definitions(mn_hwsmoke PRIVATE MN_HAS_KINECT_V1)
endif()

add_test(NAME hardware_smoke COMMAND mn_hwsmoke)
set_tests_properties(hardware_smoke PROPERTIES
    SKIP_RETURN_CODE 77
    LABELS hardware
    TIMEOUT 180)
