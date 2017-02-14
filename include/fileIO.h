#ifndef FILEIO_H_INCLUDED
#define FILEIO_H_INCLUDED

#include <string>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "stereo_viso.h"
#include "mono_viso.h"
#include "data_utils.h"

namespace me{

enum SetupType{stereo,mono};
enum FilterType{EKF,EKFE,RCEKF,Linear};

extern SetupType st;
extern FilterType ft;
extern int fframe,lframe,skip;
extern MonoVisualOdometry::parameters param_mono;
extern StereoVisualOdometry::parameters param_stereo;
extern cv::Mat imgL,imgR;
static std::string rectification;
static std::ifstream imagefile;
static std::ifstream imufile;
static std::ifstream gpsfile;

int loadYML(std::string filename);
int loadImages(std::string& dir, int nb);
int openImageFile(std::string filename);
int openImuFile(std::string filename);
int openGpsFile(std::string filename);
inline void closeImageFile(){imagefile.close();}
inline void openImuFile(){imufile.close();}
inline void openGpsFile(){gpsfile.close();}
int readImageData(int& nb, double& stamp);
int readImuData(ImuData& data);
int readGpsData(GpsData& data);
int getNextImuData(double stamp, ImuData& data);
int getNextGpsData(double stamp, GpsData& data);

}

#endif // FILEIO_H_INCLUDED
