cmake_minimum_required (VERSION 2.8)
project (MotionEstimation)
set (ME_VERSION_MAJOR 1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
file(GLOB SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)

find_package(OpenCV 3.1 REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif(DOXYGEN_FOUND)

include_directories(${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
add_library(MotionEstimation SHARED ${SOURCE_FILES})
target_link_libraries(MotionEstimation ${OpenCV_LIBS})

add_executable(stereo_example main_stereo.cpp)
target_link_libraries(stereo_example MotionEstimation ${OpenCV_LIBS})
add_executable(mono_example main_mono.cpp)
target_link_libraries(mono_example MotionEstimation ${OpenCV_LIBS})

install(TARGETS MotionEstimation LIBRARY DESTINATION /usr/local/MotionEstimation/lib)
install(DIRECTORY include/ DESTINATION /usr/local/MotionEstimation/include)
