cmake_minimum_required(VERSION 3.15)
project(whisper_ct2 VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_SHARED_LIBS "Build shared library" ON)
option(WITH_TESTS "Build tests" OFF)

# Find CTranslate2 (package name is lowercase)
find_package(ctranslate2 REQUIRED)

# Find OpenMP for parallelization
find_package(OpenMP)

# Find audio libraries via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED IMPORTED_TARGET sndfile)
pkg_check_modules(SAMPLERATE REQUIRED IMPORTED_TARGET samplerate)

# PocketFFT (header-only)
add_library(pocketfft INTERFACE)
target_include_directories(pocketfft INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pocketfft)

# Main library sources
set(WHISPER_CT2_SOURCES
    src/whisper_ct2.cpp
    src/audio_processor.cpp
    src/stft.cpp
    src/mel_filters.cpp
)

# Create library
add_library(whisper_ct2 ${WHISPER_CT2_SOURCES})

target_include_directories(whisper_ct2
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(whisper_ct2
    PUBLIC
        CTranslate2::ctranslate2
    PRIVATE
        pocketfft
        PkgConfig::SNDFILE
        PkgConfig::SAMPLERATE
        z
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(whisper_ct2 PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(whisper_ct2 PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:Debug>:-g -O0>
)

# Set library version
set_target_properties(whisper_ct2 PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Install rules
include(GNUInstallDirs)

install(TARGETS whisper_ct2
    EXPORT whisper_ct2-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES include/whisper_ct2.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT whisper_ct2-targets
    FILE whisper_ct2-config.cmake
    NAMESPACE whisper_ct2::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/whisper_ct2
)

# Tests
if(WITH_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "whisper_ct2 configuration:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Shared libs:    ${BUILD_SHARED_LIBS}")
message(STATUS "  Tests:          ${WITH_TESTS}")
message(STATUS "")
