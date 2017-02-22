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

struct FrameInfo{
    int fframe = 0;
    int lframe = 0;
    int skip = 1;
};

extern SetupType st;
extern FilterType ft;
extern FrameInfo fi;
extern MonoVisualOdometry::parameters param_mono;
extern StereoVisualOdometry::parameters param_stereo;
extern std::string appendix;

static std::ifstream imagefile;
static std::ifstream imufile;
static std::ifstream gpsfile;
static std::ofstream logFile;


int loadYML(std::string filename);
std::pair<cv::Mat,cv::Mat> loadImages(std::string& dir, int nb);
int openImageFile(std::string filename);
int openImuFile(std::string filename);
int openGpsFile(std::string filename);
inline void closeImageFile(){imagefile.close();}
inline void closeImuFile(){imufile.close();}
inline void closeGpsFile(){gpsfile.close();}
int readImageData(int& nb, double& stamp);
int readImuData(ImuData& data);
int readGpsData(GpsData& data);
int getNextImuData(double stamp, ImuData& data);
int getNextGpsData(double stamp, GpsData& data);

inline void openLogFile(std::string filename){logFile.open(filename,std::ofstream::trunc);}
inline void writeLogFile(std::string message){logFile << message;}
inline void closeLogFile(){logFile.close();}
}

#endif // FILEIO_H_INCLUDED
