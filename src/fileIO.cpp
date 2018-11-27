#include "fileIO.h"

#include <iostream>
#include <iomanip>
#include <sstream>

using namespace std;
using namespace cv;

namespace me{

SetupType st;
FilterType ft;
FrameInfo fi;
double gps_orientation;
MonoVisualOdometry::parameters param_mono;
StereoVisualOdometry::parameters param_stereo;
std::string appendix;

int loadYML(string filename){

    FileStorage configFile(filename, FileStorage::READ);
    if(!configFile.isOpened()){
        cerr << "YML file could not be opened!" << endl;
        return 0;
    }

    // defining type (mono / stereo)
    string type; configFile["type"] >> type;
    if(type=="stereo"){
        st = stereo;
        #define STEREO
    }
    else{
        st = mono;
        #define MONO
    }

    // defining frame rate
    FileNode frames = configFile["frames"];
    fi.fframe = frames["start"];
    fi.lframe = frames["stop"];
    fi.skip = frames["rate"];
    fi.bias_frame = frames["bias"];
    fi.init = frames["init"];

    //defining calibration parameters
    FileNode calib = configFile["calib"];
    if(st == stereo){
        calib["f1"] >> param_stereo.fu1;
        calib["f2"] >> param_stereo.fu2;
        calib["f1"] >> param_stereo.fv1;
        calib["f2"] >> param_stereo.fv2;
        if(param_stereo.fu1 == 0){
            calib["fu1"] >> param_stereo.fu1;
            calib["fu2"] >> param_stereo.fu2;
            calib["fv1"] >> param_stereo.fv1;
            calib["fv2"] >> param_stereo.fv2;
        }
        calib["cu"] >> param_stereo.cu1;
        if(param_stereo.cu1 == 0)
            calib["cu1"] >> param_stereo.cu1;
        calib["cu"] >> param_stereo.cu2;
        if(param_stereo.cu2 == 0)
            calib["cu2"] >> param_stereo.cu2;
        calib["cv"] >> param_stereo.cv1;
        if(param_stereo.cv1 == 0)
            calib["cv1"] >> param_stereo.cv1;
        calib["cv"] >> param_stereo.cv2;
        if(param_stereo.cv2 == 0)
            calib["cv2"] >> param_stereo.cv2;
        calib["baseline"] >> param_stereo.baseline;
        if(calib["ransac"] == "true")
            param_stereo.ransac=true;
        else
            param_stereo.ransac=false;
        calib["threshold"] >> param_stereo.inlier_threshold;
        if(calib["method"] == "GN")
            param_stereo.method = StereoVisualOdometry::GN;
        else
            param_stereo.method = StereoVisualOdometry::LM;
    }else{
        calib["fu"] >> param_mono.fu;
        calib["fv"] >> param_mono.fv;
        if(param_mono.fu == 0){
            calib["f"] >> param_mono.fu;
            calib["f"] >> param_mono.fv;
        }
        calib["cu"] >> param_mono.cu;
        calib["cv"] >> param_mono.cv;
        calib["ransac"] >> param_mono.ransac;
        calib["threshold"] >> param_mono.inlier_threshold;
    }

    string filter;
    configFile["filter"] >> filter;
    if(filter == "EKF")
        ft = EKF;
    if(filter == "Linear")
        ft = Linear;
    if(filter == "EKFE")
        ft = EKFE;
    if(filter == "RCEKF")
        ft = RCEKF;
    if(filter == "MREKF")
        ft = MREKF;
    configFile["appendix"] >> appendix;

    configFile["gps"] >> gps_orientation;

    return 1;
}

int IOFile::openFile(std::string filename){
    m_file.open(filename);
    if(!m_file.is_open()){
        cerr << "could not open " << filename << endl;
        return 0;
    }
    return check_header();
}

int IOFile::check_header(){

    if(!m_file.is_open())
        return 0;
    string header;getline(m_file,header);
    int pos = header.find("#");
    if(pos < 0){
        cerr << "could not find header in " << m_filename << endl;
        m_file.close();
        return 0;
    }
    else{
        string h_(header.substr(pos+1,header.length()));
        string buff;
        for(auto n:h_){
            if(n != ',') buff+=n;else
            if(n == ',' && buff != ""){m_file_desc.push_back(buff);buff="";}
        }
        if(buff != "") m_file_desc.push_back(buff);
    }

    return 1;
}

int ImageFile::readData(int& nb, int64_t& stamp){
    if(!m_file.is_open() || m_file.eof())
        return 0;

    char c;
    std::string line;
    getline(m_file,line);
    std::stringstream sline(line);
    if(sline >> nb >> c >> stamp)
        return 1;
    else
        return 0;
}

int ImuFile::readData(ImuData& data){

    if(!m_file.is_open() || m_file.eof())
        return 0;

    char c;
    Vec4d orientation;
    std::string line;
    if(!getline(m_file,line))
        return 0;
    std::stringstream sline(line);

    for(unsigned int i=0;i<m_file_desc.size();i++){
        if(m_file_desc[i] == "timestamp")
            sline >> data.stamp >> c;
        if(m_file_desc[i] == "acc_x")
            sline >> data.acc[0] >> c;
        if(m_file_desc[i] == "acc_y")
            sline >> data.acc[1] >> c;
        if(m_file_desc[i] == "acc_z")
            sline >> data.acc[2] >> c;
        if(m_file_desc[i] == "av_x")
            sline >> data.gyr[0] >> c;
        if(m_file_desc[i] == "av_y")
            sline >> data.gyr[1] >> c;
        if(m_file_desc[i] == "av_z")
            sline >> data.gyr[2] >> c;
        if(m_file_desc[i] == "pos_x")
            sline >> data.pos[0] >> c;
        if(m_file_desc[i] == "pos_y")
            sline >> data.pos[1] >> c;
        if(m_file_desc[i] == "pos_z")
            sline >> data.pos[2] >> c;
        if(m_file_desc[i] == "q_w")
            sline >> orientation[0] >> c;
        if(m_file_desc[i] == "q_x")
            sline >> orientation[1] >> c;
        if(m_file_desc[i] == "q_y")
            sline >> orientation[2] >> c;
        if(m_file_desc[i] == "q_z")
            sline >> orientation[3] >> c;
    }
    data.orientation = Quatd(orientation[0],orientation[1],orientation[2],orientation[3]);
    return 1;
}

int GpsFile::readData(GpsData& data){

    if(!m_file.is_open() || m_file.eof())
        return 0;

    char c;
    std::string line;
    if(!getline(m_file,line))
        return 0;
    std::stringstream sline(line);
    for(unsigned int i=0;i<m_file_desc.size();i++){

        if(m_file_desc[i] == "timestamp")
             sline >> data.stamp >> c;
        if(m_file_desc[i] == "longitude")
             sline >> data.lon >> c;
        if(m_file_desc[i] == "latitude")
             sline >> data.lat >> c;
        if(m_file_desc[i] == "elevation")
             sline >> data.alt >> c;
    }

    return 1;
}

int PoseFile::readData(PoseData& data){

    if(!m_file.is_open() || m_file.eof())
        return 0;

    char c;
    Vec4d orientation;
    std::string line;
    if(!getline(m_file,line))
        return 0;
    std::stringstream sline(line);

    for(unsigned int i=0;i<m_file_desc.size();i++){
        if(m_file_desc[i] == "timestamp")
            sline >> data.stamp >> c;
        if(m_file_desc[i] == "x")
            sline >> data.position[0] >> c;
        if(m_file_desc[i] == "y")
            sline >> data.position[1] >> c;
        if(m_file_desc[i] == "z")
            sline >> data.position[2] >> c;
        if(m_file_desc[i] == "q_w")
            sline >> orientation[0] >> c;
        if(m_file_desc[i] == "q_x")
            sline >> orientation[1] >> c;
        if(m_file_desc[i] == "q_y")
            sline >> orientation[2] >> c;
        if(m_file_desc[i] == "q_z")
            sline >> orientation[3] >> c;
    }
    data.orientation = Quatd(orientation[0],orientation[1],orientation[2],orientation[3]);
    return 1;
}

int ImuFile::getNextData(int64_t stamp, ImuData& data){

    if(!m_file.is_open() || m_file.eof())
        return 0;

    data = ImuData();
    int count=0;
    ImuData rdata;
    while(data.stamp <= stamp && readData(rdata)){
        data+=rdata;
        count++;
    }
    data /=count;

    if(m_file.eof())
        return 0;
    else
        return count;
}

int GpsFile::getNextData(int64_t stamp, GpsData& data){

    if(!m_file.is_open() || m_file.eof())
        return 0;
    bool success=true;
    do{
        success = readData(data);
    }while( success && data.stamp <= stamp);

    return success;
}

int PoseFile::getNextData(int64_t stamp, PoseData& data){

    if(!m_file.is_open() || m_file.eof())
        return 0;
    bool success=true;
    do{
        success = readData(data);
    }while( success && data.stamp <= stamp);

    return success;
}

pair<Mat,Mat> loadImages(const std::string& dir, int nb){

    pair<Mat,Mat> imgs;
    stringstream num;num <<  std::setfill('0') << std::setw(5) << nb;
    imgs.first = imread(dir+"/cam0_image"+num.str()+(appendix.empty()?"":"_"+appendix)+".png",0);
    if(imgs.first.empty())
        cerr << "cannot read " << dir+"/cam0_image"+num.str()+(appendix.empty()?"":"_"+appendix)+".png" << endl;
    if(st == stereo){
        imgs.second = imread(dir+"/cam1_image"+num.str()+(appendix.empty()?"":"_"+appendix)+".png",0);
        if(imgs.second.empty())
            cerr << "cannot read " << dir+"/cam1_image"+num.str()+(appendix.empty()?"":"_"+appendix)+".png" << endl;
    }

    return imgs;
}


void loadImagesKitti(const std::string& dir, int nb, std::pair<cv::Mat,cv::Mat>& imgs, const int padding){

    stringstream num;num <<  std::setfill('0') << std::setw(padding) << nb;
    imgs.first = imread(dir+"/L_"+num.str()+".png",0).rowRange(Range(0,374));
    if(imgs.first.empty())
        cerr << "cannot read " << dir+"/L_"+num.str()+".png" << endl;
    if(st == stereo){
        imgs.second = imread(dir+"/R_"+num.str()+".png",0).rowRange(Range(0,374));
        if(imgs.second.empty())
            cerr << "cannot read " << dir+"/R_"+num.str()+".png" << endl;
    }
}

cv::Mat loadImageKitti(const std::string& dir, int cam_nb, int img_nb, const int padding){

    Mat img;
    stringstream num;num <<  std::setfill('0') << std::setw(padding) << img_nb;
    img= imread(dir+"/L_"+num.str()+".png",0).rowRange(Range(0,374));
    if(img.empty())
        cerr << "cannot read " << dir+"/L_"+num.str()+".png" << endl;
    return img;
}

void loadImages(const std::string& dir, int nb, std::pair<cv::Mat,cv::Mat>& imgs, const int padding){

    stringstream num;num <<  std::setfill('0') << std::setw(padding) << nb;
    imgs.first = imread(dir+"/cam0_image"+num.str()+(appendix.empty()?"":"_"+appendix)+".png",0);
    if(imgs.first.empty())
        cerr << "cannot read " << dir+"/cam0_image"+num.str()+(appendix.empty()?"":"_"+appendix)+".png" << endl;
    if(st == stereo){
        imgs.second = imread(dir+"/cam1_image"+num.str()+(appendix.empty()?"":"_"+appendix)+".png",0);
        if(imgs.second.empty())
            cerr << "cannot read " << dir+"/cam1_image"+num.str()+(appendix.empty()?"":"_"+appendix)+".png" << endl;
    }
}


cv::Mat loadImage(const std::string& dir, int cam_nb, int img_nb, const int padding){

    Mat img;
    stringstream num;num <<  std::setfill('0') << std::setw(padding) << img_nb;
    img= imread(dir+"/cam"+to_string(cam_nb)+"_image"+num.str()+(appendix.empty()?"":"_"+appendix)+".png",0);
    if(img.empty())
        cerr << "cannot read " << dir+"/cam"+to_string(cam_nb)+"_image"+num.str()+(appendix.empty()?"":"_"+appendix)+".png" << endl;
    return img;
}

void loadPCImages(const std::string& dir, int nb, std::vector<std::pair<cv::Mat,cv::Mat>>& imgs, const int padding){

    imgs = std::vector<std::pair<cv::Mat,cv::Mat>>(4);
    stringstream num;num <<  std::setfill('0') << std::setw(padding) << nb;
    imread(dir+"/cam0_image"+num.str()+"_M.png",0).convertTo(imgs[0].first,CV_32F);
    imread(dir+"/cam0_image"+num.str()+"_m.png",0).convertTo(imgs[1].first,CV_32F);
    imread(dir+"/cam0_image"+num.str()+"_PC.png",0).convertTo(imgs[2].first,CV_32F);
    imread(dir+"/cam0_image"+num.str()+"_ft.png",0).convertTo(imgs[3].first,CV_32F);
    if(imgs[0].first.empty())
        cerr << "cannot read " << dir+"/cam0_image"+num.str()+"_M.png" << endl;
    imread(dir+"/cam1_image"+num.str()+"_M.png",0).convertTo(imgs[0].second,CV_32F);
    imread(dir+"/cam1_image"+num.str()+"_m.png",0).convertTo(imgs[1].second,CV_32F);
    imread(dir+"/cam1_image"+num.str()+"_PC.png",0).convertTo(imgs[2].second,CV_32F);
    imread(dir+"/cam1_image"+num.str()+"_ft.png",0).convertTo(imgs[3].second,CV_32F);
    if(imgs[0].second.empty())
        cerr << "cannot read " << dir+"/cam1_image"+num.str()+"_M.png" << endl;
    for_each(begin(imgs),end(imgs),[](pair<Mat,Mat> img){img.first /= 255.0;img.second /= 255.0;});
}


std::vector<cv::Mat> loadPCImage(const std::string& dir, int cam_nb, int img_nb, const int padding){

    vector<Mat> imgs(4);
    stringstream num;num <<  std::setfill('0') << std::setw(padding) << img_nb;
    imgs[0] = imread(dir+"/cam"+to_string(cam_nb)+"_image"+num.str()+"_M.png",0);
    imgs[1] = imread(dir+"/cam"+to_string(cam_nb)+"_image"+num.str()+"_m.png",0);
    imgs[2] = imread(dir+"/cam"+to_string(cam_nb)+"_image"+num.str()+"_PC.png",0);
    imgs[3] = imread(dir+"/cam"+to_string(cam_nb)+"_image"+num.str()+"_ft.png",0);
    if(imgs[0].empty())
        cerr << "cannot read " << dir+"/cam"+to_string(cam_nb)+"_image"+num.str()+"_M.png" << endl;
    for_each(begin(imgs),end(imgs),[](Mat img){img /= 255.0;});
    return imgs;
}


}
