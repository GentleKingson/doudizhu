cmake_minimum_required(VERSION 3.14)
project(nanopokerlib)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(NANOPOKERLIB_BUILD "doudizhu" CACHE STRING "Build profile: doudizhu.")
set_property(CACHE NANOPOKERLIB_BUILD PROPERTY STRINGS "doudizhu")

# pybind11_add_module enables LTO by default in Release builds unless this variable is defined.
# Default to OFF for faster local builds; override with -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON.
if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()

if(DEFINED Python_EXECUTABLE AND Python_EXECUTABLE)
    message(STATUS "Honoring legacy Python_EXECUTABLE=${Python_EXECUTABLE}")
    set(Python3_EXECUTABLE "${Python_EXECUTABLE}")
endif()

set(Python3_FIND_STRATEGY LOCATION)
set(Python3_FIND_REGISTRY LAST)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python: ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})

include(FetchContent)

# Avoid re-contacting remotes when deps are already present
set(FETCHCONTENT_UPDATES_DISCONNECTED TRUE)
set(FETCHCONTENT_QUIET OFF)

# Prefer local pybind11 sources (keeps builds offline and avoids system package mismatches).
set(local_pybind11_dir "${CMAKE_CURRENT_SOURCE_DIR}/../external/pybind11")
if(EXISTS "${local_pybind11_dir}/CMakeLists.txt")
    message(STATUS "Using local pybind11 sources from ${local_pybind11_dir}")
    set(FETCHCONTENT_SOURCE_DIR_PYBIND11 "${local_pybind11_dir}")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
    )
    FetchContent_MakeAvailable(pybind11)
    message(STATUS "Fetched pybind11 from ${pybind11_SOURCE_DIR}")
else()
    # Fall back to an existing pybind11 installation, otherwise fetch from upstream.
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND)
        message(STATUS "Using preinstalled pybind11 from ${pybind11_DIR}")
    else()
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.11.1
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
        )
        FetchContent_MakeAvailable(pybind11)
        message(STATUS "Fetched pybind11 from ${pybind11_SOURCE_DIR}")
    endif()
endif()

# PokerHandEvaluator: prefer a pre-installed package (e.g. in the conda prefix)
# to avoid recompiling the large lookup-table sources on every clean build.
find_package(phevaluator CONFIG QUIET)
if(phevaluator_FOUND)
    message(STATUS "Using pre-installed phevaluator (${phevaluator_DIR})")
    # Create unnamespaced alias targets so the rest of this file can just use
    # 'pheval' / 'phevalplo4' regardless of how the library was obtained.
    # Promote imported targets to GLOBAL so that add_library(ALIAS) works on
    # CMake 3.14–3.17 (non-global imported aliases require CMake ≥ 3.18).
    set_target_properties(phevaluator::pheval PROPERTIES IMPORTED_GLOBAL TRUE)
    if(NOT TARGET pheval)
        add_library(pheval ALIAS phevaluator::pheval)
    endif()
    if(TARGET phevaluator::phevalplo4)
        set_target_properties(phevaluator::phevalplo4 PROPERTIES IMPORTED_GLOBAL TRUE)
        if(NOT TARGET phevalplo4)
            add_library(phevalplo4 ALIAS phevaluator::phevalplo4)
        endif()
    endif()
    # Grab the include dir from the imported target for include_directories()
    get_target_property(_pheval_inc phevaluator::pheval INTERFACE_INCLUDE_DIRECTORIES)

    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp
        ${_pheval_inc}
        ${Python3_INCLUDE_DIRS}
    )
else()
    # Fallback: build from source via add_subdirectory
    set(PHEVAL_DIR "" CACHE PATH "Path to PokerHandEvaluator cpp directory")
    if(NOT PHEVAL_DIR)
        set(local_pheval_dir "${CMAKE_CURRENT_SOURCE_DIR}/../external/PokerHandEvaluator/cpp")
        if(EXISTS "${local_pheval_dir}/include")
            message(STATUS "Using local PokerHandEvaluator at ${local_pheval_dir}")
            set(PHEVAL_DIR "${local_pheval_dir}" CACHE PATH "Path to PokerHandEvaluator cpp directory" FORCE)
        endif()
    endif()
    if(PHEVAL_DIR AND EXISTS "${PHEVAL_DIR}/include")
        message(STATUS "Using provided PokerHandEvaluator at ${PHEVAL_DIR}")
    else()
        FetchContent_Declare(
            pokerhandeval
            GIT_REPOSITORY https://github.com/HenryRLee/PokerHandEvaluator.git
            GIT_TAG        v0.5.3
            GIT_SHALLOW    TRUE
            GIT_PROGRESS   TRUE
        )
        FetchContent_MakeAvailable(pokerhandeval)
        set(PHEVAL_DIR ${pokerhandeval_SOURCE_DIR}/cpp)
        message(STATUS "Fetched PokerHandEvaluator into ${PHEVAL_DIR}")
    endif()

    # Define PHEval paths
    set(BUILD_PLO4 ON CACHE BOOL "Build PLO4 library" FORCE)
    set(BUILD_PLO5 OFF CACHE BOOL "Build PLO5 library" FORCE)
    set(BUILD_PLO6 OFF CACHE BOOL "Build PLO6 library" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "Build test ON/OFF" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "Build examples ON/OFF" FORCE)

    # Build PHEval library
    add_subdirectory(${PHEVAL_DIR} ${CMAKE_BINARY_DIR}/pheval_build)

    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp
        ${PHEVAL_DIR}/include
        ${Python3_INCLUDE_DIRS}
    )
endif()

# Create the Python module — split across multiple translation units for
# parallel compilation.  Each pybindings_*.cpp registers one game family
# so the 8-core build compiles them concurrently instead of sequentially.
set(NANOPOKERLIB_BINDING_SOURCES
    cpp/pybindings.cpp
    cpp/pybindings_tabletop.cpp
)

set(NANOPOKERLIB_ENABLE_DOUDIZHU 1)

if(NANOPOKERLIB_BUILD STREQUAL "doudizhu")
    # Isolated artifact: build only the Dou Dizhu runtime endpoints.
else()
    message(FATAL_ERROR "Unknown NANOPOKERLIB_BUILD='${NANOPOKERLIB_BUILD}'. Expected 'doudizhu'.")
endif()

pybind11_add_module(_nanopokerlib ${NANOPOKERLIB_BINDING_SOURCES})

target_compile_definitions(_nanopokerlib PRIVATE NANOPOKERLIB_ENABLE_DOUDIZHU=${NANOPOKERLIB_ENABLE_DOUDIZHU})

# Link libraries
target_link_libraries(_nanopokerlib PRIVATE pheval pybind11::module)
if(TARGET phevalplo4)
    target_link_libraries(_nanopokerlib PRIVATE phevalplo4)
endif()

# Print additional debug information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Python include directory: ${PYTHON_INCLUDE_DIR}")
message(STATUS "pybind11 include dirs: ${pybind11_INCLUDE_DIRS}")
