#ifndef FILEIO_H_INCLUDED
#define FILEIO_H_INCLUDED

/** \file file_IO.h
*   \brief Defines useful functions to read and write libMotionEstimation data types.
*
*   Defines:  - structs for configuration files
*             - signal handler to pause and resume processing
*             - file readers
*
*    \author Axel Beauvisage (axel.beauvisage@gmail.com)
*/

#include "vo/StereoVisualOdometry.h"
#include "vo/MonoVisualOdometry.h"
#include "core/data_utils.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <fstream>

namespace me{

enum class PoseType{ABSOLUTE,RELATIVE}; //!< Type of pose representation: absolute or relative

enum class SetupType{stereo,mono}; //!< Type of setup: stereo or monocular
//!< Type of filter if a filter if used.
/*! the filter can be:
    - EKF: an Extended Kalman Filter (Quaternion)
    - EKFE: an Extended Kalman Filter (Euler angles)
    - MREKF: Multi-rate Kalman Filter (Euler angles)
    - RCEKF: an Extended Kalman Filter with robocentric coordinate frame (Quaternion)
    - Linear: a linear Kalman Filter */
enum class FilterType{EKF,EKFE,MREKF,RCEKF,Linear,None};

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

    void write(cv::FileStorage& fs) const{
        fs << "{" << "start" << fframe << "stop" << lframe << "rate" << skip << "bframe" << bias_frame << "initframe" << init << "}";
    }
    void read(const cv::FileNode& node){
        fframe = (int)node["start"];
        lframe = (int)node["stop"];
        skip = (int)node["rate"];
        bias_frame = (int)node["bframe"];
        init = (int)node["initframe"];
        if(!skip)
			skip = 1;
    }
};

//! structure containing tracking parameters
struct TrackingInfo{
    int nb_feats=500;
    int window_size=5;
    int ba_rate=0;
    double parallax=10.0;
    double feat_cov = 0.25;

    void write(cv::FileStorage& fs) const{
        fs << "{" << "feats" << nb_feats << "window" << window_size << "ba_rate" << ba_rate << "parallax" << parallax << "feat_cov" << feat_cov << "}";
    }
    void read(const cv::FileNode& node){
        nb_feats = (int)node["feats"];
        window_size = (int)node["window"];
        ba_rate = (int)node["ba_rate"];
        parallax = (double)node["parallax"];
        feat_cov = (double)node["feat_cov"];
        if(!feat_cov)
			feat_cov = 1.0;
    }
};

//! structure containing various information about the dataset to use
struct DatasetInfo{
    std::string dir="";
    std::string image_filename="";
    std::string gt_filename="";
    std::string imu_filename="";
    double gps_orientation = 0;
    SetupType type=SetupType::mono;
    bool scaled_traj=false;
    PoseType poses = PoseType::ABSOLUTE;
    int cam_ID=0;
    Quatd q_init, q_cam_to_base;
    cv::Vec3d p_init = cv::Vec3d(0,0,0), p_cam_to_base = cv::Vec3d(0,0,0);
    void write(cv::FileStorage& fs) const{
		fs	<< "{"
			<< "dir" << dir << "image_file" << image_filename << "gt_file" << gt_filename << "imu_file" << imu_filename
			<< "gps" << gps_orientation << "type" << (type==SetupType::mono?"mono":"stereo")
			<< "scaled" << (scaled_traj?"true":"false") << "poses" << (poses==PoseType::ABSOLUTE?"absolute":"relative")
			<< "camID" << cam_ID << "init_orientation" << q_init.getCoeffs() << "init_position" << p_init
			<< "cam_orientation" << q_cam_to_base.getCoeffs() << "cam_position" << p_cam_to_base
			<< "}";
    }
    void read(const cv::FileNode& node){
        dir = (std::string)node["dir"];
        image_filename = (std::string)node["image_file"];
        gt_filename = (std::string)node["gt_file"];
        imu_filename = (std::string)node["imu_file"];
        gps_orientation = (double)node["gps"];
        type = (node["type"]=="mono"?SetupType::mono:SetupType::stereo);
        scaled_traj = (node["scaled"]=="true"?true:false);
        poses = (node["poses"]=="absolute"?PoseType::ABSOLUTE:PoseType::RELATIVE);
        cam_ID = (int)node["camID"];
        cv::Vec4d q;node["init_orientation"] >> q;
        if(norm(q)>0)
            q_init = Quatd(q(0),q(1),q(2),q(3));
		node["cam_orientation"] >> q;
        if(norm(q)>0)
            q_cam_to_base = Quatd(q(0),q(1),q(2),q(3));
        node["init_position"] >> p_init;
        node["cam_position"] >> p_cam_to_base;
    }
};

template<class T>
void write(cv::FileStorage& fs, const std::string&, const T& x){
    x.write(fs);
}

template<class T>
void read(const cv::FileNode& node, T& x, const T& default_value = T()){
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}

struct cv_sig_handler{

    int _break_=0;
    bool _quit_=false;

    void stop(){
        _break_=0;
        wait();
    }

    char wait(){
        char k = cv::waitKey(_break_);
        switch(k){
                case 'p':
                    _break_ = 0;break;
                case 'r':
                _break_
                = 10;break;
                case 'q':
                    _quit_ = true;

            }
            return k;
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
bool loadYML(std::string filename);
//! returns the image(s) corresponding to the frame number
/*! if monocular, the second image of the pair is empty. */
std::pair<cv::Mat,cv::Mat> loadImages(const std::string& dir, int nb);
void loadImagesKitti(const std::string& dir, int nb, std::pair<cv::Mat,cv::Mat>& imgs, const int padding=8);
cv::Mat loadImageKitti(const std::string& dir, int cam_nb, int img_nb, const int padding=8);
void loadImages(const std::string& dir, int nb, std::pair<cv::Mat,cv::Mat>& imgs, const int padding=5);
cv::Mat loadImage(const std::string& dir, int cam_nb, int img_nb, const int padding=5);
void loadPCImages(const std::string& dir, int nb, std::vector<std::pair<cv::Mat,cv::Mat>>& imgs, const int padding=5);
std::vector<cv::Mat> loadPCImage(const std::string& dir, int cam_nb, int img_nb, const int padding=5);

inline void openLogFile(const std::string& filename){logFile.open(filename);if(!logFile.is_open())std::cerr << "could not create log file" << std::endl;}
//! write a string in the logFile. Useful for displaying data without flooding the standard output.
inline void writeLogFile(const std::string& message){logFile << message;}
inline void closeLogFile(){logFile.close();}

class IOFile{

public:
    IOFile():m_filename(""){}
    IOFile(std::string filename):m_filename(filename){openFile(m_filename);}
    ~IOFile(){m_file.close();}
    bool is_open(){return m_file.is_open();}
    int openFile(std::string filename);
    void closeFile(){m_file.close();}
    std::string show_header(){std::string header;for(auto s : m_file_desc)header+=s+';';return header;}

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

//! structure to read images in a directoty or video
struct ImageReader{

    enum class Type{IMAGES,VIDEO}; //!< type of data

    ImageFile fimage; //!< image file in case of directory
    std::vector<cv::VideoCapture> cap; //!< video file
    Type type;
    int img_nb; //!< current image number
    int64_t img_stamp; //! current_image_stamp


    ImageReader(const std::string& filename="", Type type_=Type::IMAGES): fimage{filename}, cap(dataset_info.type==SetupType::stereo?2:1),type{type_},img_nb{0},img_stamp{0}{

        if(!fimage.is_open())
            std::cerr << "[ImageReader] warning: could not find an image file in " << dataset_info.dir << std::endl;

        if(type == Type::VIDEO){ // if video needs to open video streams
            for(uint i=0;i<cap.size();i++){
                cap[i].open(dataset_info.dir+"/cam"+std::to_string(i)+"_image.mp4");
            }
        }
        do{
          readMono(); // this method can be run  with mono and stereo
        }while(img_nb<frame_info.fframe); // reading images until we reach the first frame
    }

    void openReader(const std::string& filename=dataset_info.dir+"/image_data.csv", Type type_=Type::IMAGES){

        fimage.openFile(filename);
        type = type_;

        if(!fimage.is_open())
            std::cerr << "[ImageReader] warning: could not find an image file in " << dataset_info.dir << std::endl;

        if(type == Type::VIDEO){ // if video needs to open video streams
            for(uint i=0;i<cap.size();i++){
                cap[i].open(dataset_info.dir+"/cam"+std::to_string(i)+"_image.mp4");
            }
        }
        do{
          readMono(); // this method can be run  with mono and stereo
        }while(img_nb<frame_info.fframe); // reading images until we reach the first frame
    }

    bool isValid(){
        bool valid = img_nb; //first img should have been read
        if(type == Type::VIDEO) // if video check that every camera stream is open
          std::for_each(cap.begin(),cap.end(),[&valid](cv::VideoCapture& vc){valid = (valid || vc.isOpened());});
        return valid;
    }

    int get_img_nb(){return img_nb;}

    std::pair<cv::Mat,cv::Mat> readStereo(){

      // updating next frame number
      if(fimage.is_open())
          for(int i=0;i<frame_info.skip;i++){
              if(!fimage.readData(img_nb,img_stamp)){ // trying to read in image file
                std::cerr << "[Error] could not read image" << img_nb << " in " << dataset_info.dir << std::endl;
                return std::pair<cv::Mat,cv::Mat>();
              }
          }
      else
        img_nb += frame_info.skip;

      // loading images
      if(type==Type::IMAGES)
          return loadImages(dataset_info.dir,img_nb);
      else{
        std::pair<cv::Mat,cv::Mat> imgs;
        while(cap[0].get(cv::CAP_PROP_POS_FRAMES) < img_nb)
			cap[0].grab();
		while(cap[1].get(cv::CAP_PROP_POS_FRAMES) < img_nb)
			cap[1].grab();
		cap[0] >> imgs.first; cap[1] >> imgs.second;
        if(imgs.first.type() > 8)
            cv::cvtColor(imgs.first,imgs.first,CV_BGR2GRAY);
        if(imgs.second.type() > 8)
            cv::cvtColor(imgs.second,imgs.second,CV_BGR2GRAY);
        return imgs;
      }
    }

    cv::Mat readMono(){

      // updating next frame number
      if(fimage.is_open())
          for(int i=0;i<frame_info.skip;i++){
              if(!fimage.readData(img_nb,img_stamp)){ // trying to read in image file
                std::cerr << "[Error] could not read image" << img_nb << " in " << dataset_info.dir << std::endl;
                return cv::Mat();
              }
          }
      else
        img_nb += frame_info.skip;

      // loading images
      if(type==Type::IMAGES)
          return loadImage(dataset_info.dir,dataset_info.cam_ID,img_nb);
      else{
        cv::Mat img;
        while(cap[0].get(cv::CAP_PROP_POS_FRAMES) < img_nb)
			cap[0].grab();
         cap[0] >> img;
        if(img.type() > 8)
            cv::cvtColor(img,img,CV_BGR2GRAY);
        return img;
      }
    }
};

//! structure to read ground truth poses from a file of transformations
struct GTReader{

    std::ifstream stream; //!< transformation file

    GTReader(const std::string& filename): stream{filename}{
    if(!stream.is_open())
            std::cerr << "[GTReader] warning: file not open!" << std::endl;
    }

    std::string readHeader(){
        std::string header;
        getline(stream,header);
        return header;
    }
    //!< read one line of the file and returns the corresponding pose and its timestamp
    std::pair<uint64_t,CamPose_qd> readPoseLine(){

        std::pair<uint64_t,CamPose_qd> data;
        if(!stream.is_open()){
            std::cerr << "[GTReader] could not read line file not open" << std::endl;
            return data;
        }

        std::string line;
        std::getline(stream,line);
        std::stringstream linestream(line);
        char coma;
        linestream >> data.first >> coma; // reading timestamp
        std::array<double,4> orientation;
        std::array<double,3> position;
        for(auto& o : orientation)
            linestream >> o >> coma;
        for(auto& p : position)
            linestream >> p >> coma;
        data.second.orientation = Quatd{orientation[3],orientation[0],orientation[1],orientation[2]};
        data.second.position = cv::Vec3d{position[0],position[1],position[2]};

        return data;
    }
};

}// namespace me

#endif // FILEIO_H_INCLUDED
