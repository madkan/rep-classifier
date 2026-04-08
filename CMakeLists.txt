# The following lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.16)

# ── Patch espressif/mpu6050 v1.2.1 for ESP-IDF 5.x ──────────────────────────
# MUST run before include(project.cmake) — that's where ESP-IDF reads all
# component CMakeLists files. After project() it is already too late.
# Bug: on IDF>=5.3 the component sets REQUIRES to only esp_driver_i2c, but
# mpu6050.c still #includes "driver/i2c.h" which lives in the `driver`
# umbrella component. Adding `driver` back fixes the missing-header error.
set(_MPU_CMAKE
    "${CMAKE_CURRENT_SOURCE_DIR}/managed_components/espressif__mpu6050/CMakeLists.txt")
if(EXISTS "${_MPU_CMAKE}")
    file(READ "${_MPU_CMAKE}" _content)
    string(REPLACE
        "set(REQ esp_driver_gpio esp_driver_i2c)"
        "set(REQ esp_driver_gpio esp_driver_i2c driver)"
        _content_fixed "${_content}")
    if(NOT "${_content_fixed}" STREQUAL "${_content}")
        file(WRITE "${_MPU_CMAKE}" "${_content_fixed}")
        message(STATUS "mpu6050: applied IDF-5.x legacy-driver patch")
    endif()
endif()

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(IMU)
