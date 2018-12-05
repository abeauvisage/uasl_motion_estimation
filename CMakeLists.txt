cmake_minimum_required (VERSION 3.1)
project (MotionEstimation)

# compiler options
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wno-comment")

# output directory
set( LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib )

# dependencies
find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 REQUIRED)

# documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif(DOXYGEN_FOUND)

# source and header files
file(GLOB SOURCE_FILES src/*.cpp)
file(GLOB HEADER_FILES include/*.h)

# library target
add_library(MotionEstimation SHARED ${SOURCE_FILES} ${HEADER_FILES})
install(TARGETS MotionEstimation LIBRARY DESTINATION lib)
target_link_libraries(MotionEstimation ${OpenCV_LIBS})
include_directories(include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
