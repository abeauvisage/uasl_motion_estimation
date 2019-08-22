# UASL Motion Estimation library
- version 3.0

# DEPENDENCIES

- Cmake 3.5 minimum (https://cmake.org/install/)
- Opencv 3.4 minimum (https://opencv.org/releases/)
- Ceres 1.12 minimum (optional: http://ceres-solver.org/installation.html)
- Eigen 3 minimum (https://eigen.tuxfamily.org/dox/GettingStarted.html)

To install from the package manager the following:

deb packages (Ubuntu): 
sudo apt-get install cmake libopencv-dev libceres-dev libeigen3-dev

rpm packages (Fedora):
sudo dnf install cmake opencv-devel ceres-solver-devel eigen3-devel

# INSTALLATION

1. download the repository:
- git clone https://github.com/abeauvisage/uasl_motion_estimation.git
2. navigate to the main directory:
- cd PATH_TO_LIBRARY_ROOT_DIRECTORY
3. create a build directory:
- mkdir build
4. move to the build directory:
- cd build
5. run cmake to solve dependencies:
- cmake ..
>	if using custom OpenCV, Eigen or Ceres add options:
> - -DOPENCV_CONFIG_PATH=/path/to/OpenCV/config -DEIGEN_CONFIG_PATH=/path/to/Eigen/config -DCERES_CONFIG_PATH=/path/to/Ceres/config
6. compile the library:
- make
7. install the library on the system (optional):
- sudo make install
8. generate documentation (optional)
- make doc 
