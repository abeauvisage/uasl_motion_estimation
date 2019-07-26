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
find_package(Ceres QUIET)

# documentation
find_package(Doxygen)
option(ME_BUILD_DOC "Build doc target for uasl_MSVO" OFF)
if(DOXYGEN_FOUND AND ME_BUILD_DOC)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif(DOXYGEN_FOUND AND ME_BUILD_DOC)

# source and header files
set(ME_CORE_HEADERS
					include/core/feature_types.h
					include/core/rotation_utils.h
					include/core/data_utils.h
					include/core/gps_utils.h
					include/core/mutual_information.h
					include/core/file_IO.h
)
set(ME_CORE_SOURCES
					src/core/feature_types.cpp
					src/core/rotation_utils.cpp
					src/core/mutual_information.cpp
					src/core/file_IO.cpp
)

set(ME_GUI_HEADERS
					include/gui/gui_utils.h
					include/gui/Graph2D.h
)
set(ME_GUI_SOURCES
					src/gui/gui_utils.cpp
					src/gui/Graph2D.cpp
)

set(ME_OPTIM_HEADERS
					include/optimisation/optimisation.h
)
set(ME_OPTIM_SOURCES
					src/optimisation/optimisation.cpp
)

set(ME_VO_HEADERS
					include/vo/MonoVisualOdometry.h
					include/vo/StereoVisualOdometry.h
)
set(ME_VO_SOURCES
					src/vo/MonoVisualOdometry.cpp
					src/vo/StereoVisualOdometry.cpp
)

#Ceres for optimisation
if(Ceres_FOUND)
message(STATUS "adding Ceres Optimisation")
list(APPEND ME_OPTIM_HEADERS include/optimisation/CeresBA.h)
list(APPEND ME_OPTIM_SOURCES src/optimisation/CeresBA.cpp)
list(APPEND ME_OPTIM_HEADERS include/optimisation/BundleAdjuster.h)
else(Ceres_FOUND)
message(WARNING "Ceres was not found, CeresBA and BundleAdjuster will not be included.")
endif(Ceres_FOUND)

#library option
option(DYNAMIC_LIB "Build dynamic lib" OFF)
if(DYNAMIC_LIB)
set(SHARED_LIBS SHARED)
endif(DYNAMIC_LIB)

#generating different lib modules
add_library(MotionEstimation_core ${SHARED_LIBS} ${ME_CORE_HEADERS} ${ME_CORE_SOURCES})
target_link_libraries(MotionEstimation_core ${OpenCV_LIBS})
add_library(MotionEstimation_optimisation ${SHARED_LIBS} ${ME_OPTIM_HEADERS} ${ME_OPTIM_SOURCES})
target_link_libraries(MotionEstimation_optimisation ${OpenCV_LIBS} ${CERES_LIBRARIES} MotionEstimation_core)
add_library(MotionEstimation_gui ${SHARED_LIBS} ${ME_GUI_HEADERS} ${ME_GUI_SOURCES})
target_link_libraries(MotionEstimation_gui ${OpenCV_LIBS} pthread)
add_library(MotionEstimation_vo ${SHARED_LIBS} ${ME_VO_HEADERS} ${ME_VO_SOURCES})
target_link_libraries(MotionEstimation_vo ${OpenCV_LIBS} pthread)

#install rules
install(TARGETS MotionEstimation_core MotionEstimation_gui MotionEstimation_vo MotionEstimation_optimisation LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

include_directories(	
						include
						${OpenCV_INCLUDE_DIRS}
						${EIGEN3_INCLUDE_DIRS}
						${CERES_INCLUDE_DIRS}
)
