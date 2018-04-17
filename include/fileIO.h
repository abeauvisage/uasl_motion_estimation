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

enum SetupType{stereo,mono}; //!< Type of setup: stereo or monocular
//!< Type of filter if a filter if used.
/*! the filter can be:
    - EKF: an Extended Kalman Filter (Quaternion)
    - EKFE: an Extended Kalman Filter (Euler angles)
    - MREKF: Multi-rate Kalman Filter (Euler angles)
    - RCEKF: an Extended Kalman Filter with robocentric coordinate frame (Quaternion)
    - Linear: a linear Kalman Filter */
enum FilterType{EKF,EKFE,MREKF,RCEKF,Linear,None};

//! structure describing the number of frame and frequency
/*! - fframe: number of the first frame
    - lframe: number of the last frame
    - skip: number of frame skipped between two consecutive frame (define the processing frequency) */
struct FrameInfo{
    int fframe = 0;
    int lframe = 0;
    int skip = 1;
    int bias_frame = 0;
    int init = 0;
};

// setting variables
extern SetupType st;
extern FilterType ft;
extern FrameInfo fi;
extern double gps_orientation;
extern MonoVisualOdometry::parameters param_mono;
extern StereoVisualOdometry::parameters param_stereo;
extern std::string appendix;

// different I/O files (extern extern so only used by fileIO.cpp)
static std::ifstream imagefile;
static std::ifstream gpsfile;
static std::ofstream logFile;

//! parse cofiguration file
int loadYML(std::string filename);
//! returns the image(s) corresponding to the frame number
/*! if monocular, the second image of the pair is empty. */
std::pair<cv::Mat,cv::Mat> loadImages(std::string& dir, int nb);
void loadImages(std::string& dir, int nb, std::pair<cv::Mat,cv::Mat>& imgs);
cv::Mat loadImage(std::string& dir, int cam_nb, int img_nb);

inline void openLogFile(std::string filename){logFile.open(filename,std::ofstream::trunc);}
//! write a string in the logFile. Useful for displaying data without flooding the standard output.
inline void writeLogFile(std::string message){logFile << message;}
inline void closeLogFile(){logFile.close();}

class IOFile{

public:
    IOFile(std::string filename):m_filename(filename){openFile(m_filename);}
    ~IOFile(){m_file.close();}
    bool is_open(){return m_file.is_open();}
    int openFile(std::string filename);
    void closeFile(){m_file.close();}

protected:
    std::string m_filename;
    std::ifstream m_file;

};

class ImuFile : public IOFile{

public:
    ImuFile(std::string filename):IOFile(filename){openFile(filename);}
    int openFile(std::string filename);
    //! reads the one ImuData (the next one in the file).
    /*! returns an ImuData structure containing the acceleration, angular velocity and the corresponding timestamp.*/
    int readData(ImuData& data);
    //! get the first ImuData after the provided timestamp.
    /*! Corresponds to the average of all the imu data between current position in the file and the first imu data after the timestamp.*/
    int getNextData(double stamp, ImuData& data);

private:
    std::vector<std::string> m_file_desc;

};

class GpsFile : public IOFile{

public:
    GpsFile(std::string filename):IOFile(filename){openFile(filename);}
    int openFile(std::string filename);
    //! reads the one GpsData (the next one in the file).
    /*! returns an GpsData structure containing the acceleration, angular velocity and the corresponding timestamp.*/
    int readData(GpsData& data);
    //! get the first GpsData after the provided timestamp.
    int getNextData(double stamp, GpsData& data);

private:
    std::vector<std::string> m_file_desc;

};

class ImageFile : public IOFile{

public:
    ImageFile(std::string filename):IOFile(filename){}
    //! reads the one ImageData (the next one in the file).
    /*! returns the number of the image and its corresponding timestamp.*/
    int readData(int& nb,double& stamp);

};

}

#endif // FILEIO_H_INCLUDED
