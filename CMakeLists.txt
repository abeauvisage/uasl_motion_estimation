cmake_minimum_required (VERSION 3.5)
project (MotionEstimation)

# compiler options
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wno-comment")

# dependencies
message(STATUS "OpenCV config: ${OPENCV_CONFIG_PATH}")
find_package(OpenCV 3.4 REQUIRED core videoio imgcodecs flann highgui imgproc calib3d video OPTIONAL_COMPONENTS viz PATHS ${OPENCV_CONFIG_PATH})
find_package(Eigen3 REQUIRED PATHS ${EIGEN_CONFIG_PATH})
find_package(Ceres QUIET PATHS ${CERES_CONFIG_PATH})
message(STATUS "Ceres libs: ${CERES_LIBRARIES}")

# documentation
find_package(Doxygen)
option(ME_BUILD_DOC "Build doc target for uasl_MSVO" OFF)
if(DOXYGEN_FOUND AND ME_BUILD_DOC)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/share/doc)
endif(DOXYGEN_FOUND AND ME_BUILD_DOC)

#useful locations
SET(SRC_DIR src)
SET(HEADER_DIR include/${PROJECT_NAME})
SET(LIB_DIR lib/${PROJECT_NAME})
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${LIB_DIR})

# source and header files
set(ME_CORE_HEADERS
					${HEADER_DIR}/core/feature_types.h
					${HEADER_DIR}/core/rotation_utils.h
					${HEADER_DIR}/core/data_utils.h
					${HEADER_DIR}/core/gps_utils.h
					${HEADER_DIR}/core/mutual_information.h
					${HEADER_DIR}/core/file_IO.h
)
set(ME_CORE_SOURCES
					${SRC_DIR}/core/feature_types.cpp
					${SRC_DIR}/core/rotation_utils.cpp
					${SRC_DIR}/core/mutual_information.cpp
					${SRC_DIR}/core/file_IO.cpp
)

set(ME_GUI_HEADERS
					${HEADER_DIR}/gui/gui_utils.h
					${HEADER_DIR}/gui/Graph2D.h
)
set(ME_GUI_SOURCES
					${SRC_DIR}/gui/gui_utils.cpp
					${SRC_DIR}/gui/Graph2D.cpp
)

set(ME_OPTIM_HEADERS
					${HEADER_DIR}/optimisation/optimisation.h
					${HEADER_DIR}/core/mutual_information.h
)
set(ME_OPTIM_SOURCES
					${SRC_DIR}/optimisation/optimisation.cpp
					${SRC_DIR}/core/mutual_information.cpp
)

set(ME_VO_HEADERS
					${HEADER_DIR}/vo/MonoVisualOdometry.h
					${HEADER_DIR}/vo/StereoVisualOdometry.h
)
set(ME_VO_SOURCES
					${SRC_DIR}/vo/MonoVisualOdometry.cpp
					${SRC_DIR}/vo/StereoVisualOdometry.cpp
)

#Ceres for optimisation
if(Ceres_FOUND)
list(APPEND ME_OPTIM_HEADERS ${HEADER_DIR}/optimisation/BundleAdjuster.h)
else(Ceres_FOUND)
message(WARNING "Ceres was not found, BundleAdjuster will not be included.")
endif(Ceres_FOUND)

if(OPENCV_VIZ_FOUND)
message(STATUS "adding graph3D class")
list(APPEND ME_GUI_HEADERS ${HEADER_DIR}/gui/Graph3D.h)
list(APPEND ME_GUI_SOURCES ${SRC_DIR}/gui/Graph3D.cpp)
else(OPENCV_VIZ_FOUND)
message(WARNING "OpenCV viz module was not found, Graph3D will not be included.")
endif(OPENCV_VIZ_FOUND)


#library option
option(DYNAMIC_LIB "Build dynamic lib" OFF)
if(DYNAMIC_LIB)
set(LIB_TYPE SHARED)
else(DYNAMIC_LIB)
set(LIB_TYPE STATIC)
endif(DYNAMIC_LIB)

#generating different lib modules
add_library(${PROJECT_NAME}_core ${LIB_TYPE} ${ME_CORE_HEADERS} ${ME_CORE_SOURCES})
target_link_libraries(${PROJECT_NAME}_core ${OpenCV_LIBS})
add_library(${PROJECT_NAME}_optimisation ${LIB_TYPE} ${ME_OPTIM_HEADERS} ${ME_OPTIM_SOURCES})
target_link_libraries(${PROJECT_NAME}_optimisation ${OpenCV_LIBS} ${CERES_LIBRARIES})
add_library(${PROJECT_NAME}_gui ${LIB_TYPE} ${ME_GUI_HEADERS} ${ME_GUI_SOURCES})
target_link_libraries(${PROJECT_NAME}_gui ${OpenCV_LIBS} pthread)
add_library(${PROJECT_NAME}_vo ${LIB_TYPE} ${ME_VO_HEADERS} ${ME_VO_SOURCES})
target_link_libraries(${PROJECT_NAME}_vo ${OpenCV_LIBS} pthread)

#install rules
set(${PROJECT_NAME}_TARGETS ${PROJECT_NAME}_core ${PROJECT_NAME}_gui ${PROJECT_NAME}_vo	${PROJECT_NAME}_optimisation)
install(TARGETS ${${PROJECT_NAME}_TARGETS} LIBRARY DESTINATION ${LIB_DIR} ARCHIVE DESTINATION ${LIB_DIR})
		
install(DIRECTORY 	${CMAKE_CURRENT_SOURCE_DIR}/${HEADER_DIR}/core 
					${CMAKE_CURRENT_SOURCE_DIR}/${HEADER_DIR}/gui
					${CMAKE_CURRENT_SOURCE_DIR}/${HEADER_DIR}/optimisation
					${CMAKE_CURRENT_SOURCE_DIR}/${HEADER_DIR}/vo
		DESTINATION ${HEADER_DIR})
		
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/share/${PROJECT_NAME}/${PROJECT_NAME}Config.cmake DESTINATION ${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in ${CMAKE_CURRENT_SOURCE_DIR}/share/${PROJECT_NAME}/${PROJECT_NAME}Config.cmake @ONLY)
set(MotionEstimation_LIBRARIES ${${PROJECT_NAME}_TARGETS} ${CERES_LIBRARIES} ${GLOG_LIBRARY} PARENT_SCOPE)
set(MotionEstimation_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/${HEADER_DIR} ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} PARENT_SCOPE)

#includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/${HEADER_DIR} ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
