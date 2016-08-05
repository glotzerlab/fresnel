# Maintainer: joaander

#################################
## Setup default CXXFLAGS
if(NOT PASSED_FIRST_CONFIGURE)
    message(STATUS "Overriding CMake's default CFLAGS")

    ## Allow GCC_ARCH flag to set the -march field
    if(NOT GCC_ARCH AND "$ENV{GCC_ARCH}" STREQUAL "")
        set(GCC_ARCH "native")
        message(STATUS "GCC_ARCH env var not set, setting -march to ${GCC_ARCH}")
    else()
        set(GCC_ARCH $ENV{GCC_ARCH})
        message(STATUS "Found GCC_ARCH env var, setting -march to ${GCC_ARCH}")
    endif()

    # default build type is Release when compiling make files
    if(NOT CMAKE_BUILD_TYPE)
       if(${CMAKE_GENERATOR} STREQUAL "Xcode")

       else(${CMAKE_GENERATOR} STREQUAL "Xcode")
            set(CMAKE_BUILD_TYPE "Release" CACHE STRING  "Build type: options are None, Release, Debug, RelWithDebInfo" FORCE)
        endif(${CMAKE_GENERATOR} STREQUAL "Xcode")
    endif()

    if(CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        # default flags for g++
        set(CMAKE_CXX_FLAGS_DEBUG "-march=${GCC_ARCH} -g -Wall -std=c++11" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CMAKE_CXX_FLAGS_MINSIZEREL "-march=${GCC_ARCH} -Os -Wall -DNDEBUG -std=c++11" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "-march=${GCC_ARCH} -O3 -funroll-loops -DNDEBUG -Wall -std=c++11" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-march=${GCC_ARCH} -g -O3 -funroll-loops -DNDEBUG -Wall -std=c++11" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)

        set(CMAKE_C_FLAGS_DEBUG "-march=${GCC_ARCH} -g -Wall" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CMAKE_C_FLAGS_MINSIZEREL "-march=${GCC_ARCH} -Os -Wall -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CMAKE_C_FLAGS_RELEASE "-march=${GCC_ARCH} -O3 -funroll-loops -DNDEBUG -Wall" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_C_FLAGS_RELWITHDEBINFO "-march=${GCC_ARCH} -g -O3 -funroll-loops -DNDEBUG -Wall" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)

    elseif(CMAKE_CXX_COMPILER MATCHES "icpc")
        # default flags for intel
        set(CMAKE_CXX_FLAGS_DEBUG "-xHOST -O0 -g -std=c++11" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CMAKE_CXX_FLAGS_MINSIZEREL "-xHOST -Os -std=c++11 -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "-xHOST -O3 -std=c++11 -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-xHOST -g -O3 -std=c++11 -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)

        set(CMAKE_C_FLAGS_DEBUG "-xHOST -O0 -g" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CMAKE_C_FLAGS_MINSIZEREL "-xHOST -Os -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CMAKE_C_FLAGS_RELEASE "-xHOST -O3 -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_C_FLAGS_RELWITHDEBINFO "-xHOST -g -O3 -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)

    else()
        message(STATUS "No default CFLAGS for your compiler, set them manually")
    endif()

SET(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "First configure has run: CXX_FLAGS have had their defaults changed" FORCE)
endif()
