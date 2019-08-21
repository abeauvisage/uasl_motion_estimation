
- version 3.0

################
# DEPENDENCIES #
################

- Cmake 3.5 minimum (https://cmake.org/install/)
- Opencv 3.4 minimum (https://opencv.org/releases/)
- Ceres 1.12 minimum (optional: http://ceres-solver.org/installation.html)
- Eigen 3 minimum (https://eigen.tuxfamily.org/dox/GettingStarted.html)

To install from the package manager the following:

deb packages (Ubuntu): 
sudo apt-get install cmake libopencv-dev libceres-dev libeigen3-dev

rpm packages (Fedora):
sudo dnf install cmake opencv-devel ceres-solver-devel eigen3-devel

################
# INSTALLATION #
################

1. download the repository: git clone https://github.com/abeauvisage/uasl_motion_estimation.git
2. navigate to the main directory: cd PATH_TO_LIBRARY_ROOT_DIRECTORY
3. mkdir build
4. cd build
5. cmake ..
	if using custom OpenCV, Eigen or Ceres add options:
- -DOPENCV_CONFIG_FILE=/path/to/OpenCV/config -DEIGEN_CONFIG_FILE=/path/to/Eigen/config -DCERES_CONFIG_FILE=/path/to/Ceres/config
6. make
7. sudo make install (optional)
8. make doc (optional for generating documentation)
