#include "fileIO.h"

#include <iostream>
#include <iomanip>
#include <sstream>

using namespace std;
using namespace cv;

//#define KITTI

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

    configFile ["gps"] >> gps_orientation;

    return 1;
}

int IOFile::openFile(std::string filename){
    cout << "opening " << filename << endl;
    m_file.open(filename);
    if(!m_file.is_open()){
        cerr << "could not open " << filename << endl;
        return 0;
    }
    return 1;
}

int ImuFile::openFile(std::string filename){

    if(!m_file.is_open())
        return 0;
    string header;getline(m_file,header);
    int pos = header.find("#");
    if(pos < 0){
        cerr << "could not find header in " << filename << endl;
        m_file.close();
        return 0;
    }
    else{
        string h_(header.substr(pos+1,header.length()));
        string buff;
        for(auto n:h_){
            if(n != ',') buff+=n;else
            if(n == ',' && buff != ""){m_file_desc.push_back(buff);cout << buff << " ";buff="";}
        }
        if(buff != "") m_file_desc.push_back(buff);
        cout << buff << endl;
    }

    return 1;
}

int GpsFile::openFile(std::string filename){

    if(!m_file.is_open())
        return 0;
    string header;getline(m_file,header);
    int pos = header.find("#");
    if(pos < 0){
        cerr << "could not find header in " << filename << endl;
        m_file.close();
        return 0;
    }
    else{
        string h_(header.substr(pos+1,header.length()));
        string buff;
        for(auto n:h_){
            if(n != ',') buff+=n;else
            if(n == ',' && buff != ""){m_file_desc.push_back(buff);cout << buff << " ";buff="";}
        }
        if(buff != "") m_file_desc.push_back(buff);
    }

    return 1;
}

int ImageFile::readData(int& nb, double& stamp){
    if(!m_file.is_open() || m_file.eof())
        return 0;
    char c;
    m_file >> nb >> c >> stamp >> c;
    return 1;
}

int ImuFile::readData(ImuData& data){

    if(!m_file.is_open() || m_file.eof())
        return 0;

    char c;
    double value;
    Vec4d orientation;
    for(unsigned int i=0;i<m_file_desc.size();i++){
        if(!m_file.is_open() || m_file.eof())
        return 0;
        m_file >> value >> c;
        if(m_file_desc[i] == "timestamp")
            data.stamp = value;
        if(m_file_desc[i] == "acc_x")
            data.acc[0] = value;
        if(m_file_desc[i] == "acc_y")
            data.acc[1] = value;
        if(m_file_desc[i] == "acc_z")
            data.acc[2] = value;
        if(m_file_desc[i] == "av_x")
            data.gyr[0] = value;
        if(m_file_desc[i] == "av_y")
            data.gyr[1] = value;
        if(m_file_desc[i] == "av_z")
            data.gyr[2] = value;
        if(m_file_desc[i] == "pos_x")
            data.pos[0] = value;
        if(m_file_desc[i] == "pos_y")
            data.pos[1] = value;
        if(m_file_desc[i] == "pos_z")
            data.pos[2] = value;
        if(m_file_desc[i] == "q_w")
            orientation[0] = value;
        if(m_file_desc[i] == "q_x")
            orientation[1] = value;
        if(m_file_desc[i] == "q_y")
            orientation[2] = value;
        if(m_file_desc[i] == "q_z")
            orientation[3] = value;
    }
    data.orientation = Quatd(orientation[0],orientation[1],orientation[2],orientation[3]);
    return 1;
}

int GpsFile::readData(GpsData& data){

    if(!m_file.is_open() || m_file.eof())
        return 0;

    char c;
    double value;
    for(unsigned int i=0;i<m_file_desc.size();i++){
        if(!m_file.is_open() || m_file.eof())
        return 0;
        m_file >> value >> c;
        if(m_file_desc[i] == "timestamp")
            data.stamp = value;
        if(m_file_desc[i] == "longitude")
            data.lon = value;
        if(m_file_desc[i] == "latitude")
            data.lat = value;
        if(m_file_desc[i] == "elevation")
            data.alt = value;
    }
    return 1;
}

int ImuFile::getNextData(double stamp, ImuData& data){

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

int GpsFile::getNextData(double stamp, GpsData& data){

    if(!m_file.is_open() || m_file.eof())
        return 0;
    bool success=true;
    do{
        success = readData(data);
    }while( success && data.stamp <= stamp);

    return success;
}

pair<Mat,Mat> loadImages(std::string& dir, int nb){

    pair<Mat,Mat> imgs;
    stringstream num;num <<  std::setfill('0') << std::setw(5) << nb;
    imgs.first = imread(dir+"/cam0_image"+num.str()+"_"+appendix+".png",0);
    if(imgs.first.empty())
        cerr << "cannot read " << dir+"/cam0_image"+num.str()+"_"+appendix+".png" << endl;
    if(st == stereo){
        imgs.second = imread(dir+"/cam1_image"+num.str()+"_"+appendix+".png",0);
        if(imgs.second.empty())
            cerr << "cannot read " << dir+"/cam1_image"+num.str()+"_"+appendix+".png" << endl;
    }

    return imgs;
}

#ifdef KITTI
void loadImages(std::string& dir, int nb, std::pair<cv::Mat,cv::Mat>& imgs){

    stringstream num;num <<  std::setfill('0') << std::setw(8) << nb;
    imgs.first = imread(dir+"/L_"+num.str()+".png",0);
    if(imgs.first.empty())
        cerr << "cannot read " << dir+"/L_"+num.str()+".png" << endl;
    if(st == stereo){
        imgs.second = imread(dir+"/R_"+num.str()+".png",0);
        if(imgs.second.empty())
            cerr << "cannot read " << dir+"/R_"+num.str()+".png" << endl;
    }
}
#else
void loadImages(std::string& dir, int nb, std::pair<cv::Mat,cv::Mat>& imgs){

    stringstream num;num <<  std::setfill('0') << std::setw(5) << nb;
    imgs.first = imread(dir+"/cam0_image"+num.str()+"_"+appendix+".png",0);
    if(imgs.first.empty())
        cerr << "cannot read " << dir+"/cam0_image"+num.str()+"_"+appendix+".png" << endl;
    if(st == stereo){
        imgs.second = imread(dir+"/cam1_image"+num.str()+"_"+appendix+".png",0);
        if(imgs.second.empty())
            cerr << "cannot read " << dir+"/cam1_image"+num.str()+"_"+appendix+".png" << endl;
    }
}
#endif // KITTI

cv::Mat loadImage(std::string& dir, int cam_nb, int img_nb){

    Mat img;
    stringstream num;num <<  std::setfill('0') << std::setw(5) << img_nb;
    img= imread(dir+"/cam"+to_string(cam_nb)+"_image"+num.str()+"_"+appendix+".png",0);
    if(img.empty())
        cerr << "cannot read " << dir+"/cam"+to_string(cam_nb)+"_image"+num.str()+"_"+appendix+".png" << endl;
    return img;
}

}
