## Initial setup

# Ensure that CMake is up to date
cmake_minimum_required(VERSION 2.8)

# Give our project a nice name
project( tracker )

## Flags
if(CMAKE_COMPILER_IS_GNUCXX)
#    set(CMAKE_CXX_FLAGS "-Ofast")        ## Optimize
#    set(CMAKE_EXE_LINKER_FLAGS "-s")  ## Strip binary
endif()

execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
                OUTPUT_VARIABLE GCC_VERSION)
if (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7)
    message(STATUS  "Gcc version >= 4.7, using c++11")
	add_definitions(-std=c++11)
else()
	message(FATAL_ERROR "Gcc version < 4.7, aborting. We need c++11 support")
endif()


## External files

# We need openCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# We need pthreads
find_package (Threads)

# We also need the mavlink library. Don't worry they are only headders
include_directories( c_library )

# And we need Eigen
include_directories( eigen_library )

# Add all our souce files to the build
file(GLOB SOURCE_FILES "*.cpp")
add_executable( tracker ${SOURCE_FILES} )

# Link with openCV
target_link_libraries( tracker ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} )
