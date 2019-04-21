cmake_minimum_required( VERSION 2.8 )
project( slam )

# mac
set( CMAKE_MACOSX_RPATH 1 )

set( CMAKE_CXX_COMPILER "g++" )
set( CMAEK_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++11" )

set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set( THIRD_PARTY_LIBS ${OpenCV_LIBS} )

include_directories( ${PROJECT_SOURCE_DIR}/include )
add_subdirectory( src )
add_subdirectory( demo )
