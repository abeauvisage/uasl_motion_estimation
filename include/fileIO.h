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

//! structure containing tracking parameters
struct TrackingInfo{
    int nb_feats=500;
    int window_size=5;
    double parallax=10.0;
};

//! structure containing various information about the dataset to use
struct DatasetInfo{
    std::string dir="";
    double gps_orientation = 0;
    SetupType type=SetupType::mono;
    int cam_ID=0;
};

struct cv_sig_handler{

    int _break_=0;
    bool _quit_=false;

    void wait(){
        switch(cv::waitKey(_break_)){
                case 'p':
                _break_ = (_break_ == 0 ? 10 : 0);break;
                case 'q':
                    _quit_ = true;

            }
    }
};

// setting variables
extern FrameInfo frame_info;
extern DatasetInfo dataset_info;
extern TrackingInfo tracking_info;
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
std::pair<cv::Mat,cv::Mat> loadImages(const std::string& dir, int nb);
void loadImagesKitti(const std::string& dir, int nb, std::pair<cv::Mat,cv::Mat>& imgs, const int padding=8);
cv::Mat loadImageKitti(const std::string& dir, int cam_nb, int img_nb, const int padding=8);
void loadImages(const std::string& dir, int nb, std::pair<cv::Mat,cv::Mat>& imgs, const int padding=5);
cv::Mat loadImage(const std::string& dir, int cam_nb, int img_nb, const int padding=5);
void loadPCImages(const std::string& dir, int nb, std::vector<std::pair<cv::Mat,cv::Mat>>& imgs, const int padding=5);
std::vector<cv::Mat> loadPCImage(const std::string& dir, int cam_nb, int img_nb, const int padding=5);

inline void openLogFile(std::string filename){logFile.open(filename,std::ofstream::trunc);}
//! write a string in the logFile. Useful for displaying data without flooding the standard output.
inline void writeLogFile(std::string message){logFile << message;}
inline void closeLogFile(){logFile.close();}

class IOFile{

public:
    IOFile():m_filename(""){}
    IOFile(std::string filename):m_filename(filename){openFile(m_filename);}
    ~IOFile(){m_file.close();}
    bool is_open(){return m_file.is_open();}
    int openFile(std::string filename);
    void closeFile(){m_file.close();}

private:
    int check_header();

protected:

    std::string m_filename;
    std::ifstream m_file;
    std::vector<std::string> m_file_desc;

};

class ImuFile : public IOFile{

public:
    ImuFile():IOFile(){}
    ImuFile(std::string filename):IOFile(filename){}
    //! reads the one ImuData (the next one in the file).
    /*! returns an ImuData structure containing the acceleration, angular velocity and the corresponding timestamp.*/
    int readData(ImuData& data);
    //! get the first ImuData after the provided timestamp.
    /*! Corresponds to the average of all the imu data between current position in the file and the first imu data after the timestamp.*/
    int getNextData(int64_t stamp, ImuData& data);

};

class GpsFile : public IOFile{

public:
    GpsFile():IOFile(){}
    GpsFile(std::string filename):IOFile(filename){}
    //! reads the one GpsData (the next one in the file).
    /*! returns an GpsData structure containing the acceleration, angular velocity and the corresponding timestamp.*/
    int readData(GpsData& data);
    //! get the first GpsData after the provided timestamp.
    int getNextData(int64_t stamp, GpsData& data);

};

class PoseFile : public IOFile{

public:
    PoseFile():IOFile(){}
    PoseFile(std::string filename):IOFile(filename){}
    //! reads the one PoseData (the next one in the file).
    /*! returns an PoseData structure containing the acceleration, angular velocity and the corresponding timestamp.*/
    int readData(PoseData& data);
    //! get the first PoseData after the provided timestamp.
    int getNextData(int64_t stamp, PoseData& data);

};

class ImageFile : public IOFile{

public:
    ImageFile():IOFile(){}
    ImageFile(std::string filename):IOFile(filename){}
    //! reads the one ImageData (the next one in the file).
    /*! returns the number of the image and its corresponding timestamp.*/
    int readData(int& nb, int64_t& stamp);

};

}

#endif // FILEIO_H_INCLUDED
