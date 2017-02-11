#include "fileIO.h"

#include <iostream>
#include <iomanip>

using namespace std;
using namespace cv;



int loadYML(string filename){

    FileStorage configFile(filename, FileStorage::READ);
    if(!configFile.isOpened())
        return 0;

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
    fframe = frames["start"];
    lframe = frames["stop"];
    skip = frames["rate"];

    //defining calibration parameters
    FileNode calib = configFile["calib"];
    if(st == stereo){
        calib["focal"] >> param_stereo.f1;
        calib["focal"] >> param_stereo.f2;
        calib["cu"] >> param_stereo.cu1;
        calib["cu"] >> param_stereo.cu2;
        calib["cv"] >> param_stereo.cv1;
        calib["cv"] >> param_stereo.cv2;
        calib["baseline"] >> param_stereo.baseline;
        calib["ransac"] >> param_stereo.ransac;
        calib["threshold"] >> param_stereo.inlier_threshold;
        if(calib["focal"] == "GN")
            param_stereo.method = StereoVisualOdometry::GN;
        else
            param_stereo.method = StereoVisualOdometry::LM;
    }else{
        calib["focal"] >> param_mono.f;
        calib["cu"] >> param_mono.cu;
        calib["cv"] >> param_mono.cv;
        calib["ransac"] >> param_mono.ransac;
        calib["treshold"] >> param_mono.inlier_threshold;
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
    configFile["rectification"] >> rectification;

    return 1;
}


int openImageFile(std::string filename){
    imagefile.open(filename);
    if(!imagefile.is_open()){
        cerr << "could not open " << filename << endl;
        return 0;
    }
    return 1;
}

int openImuFile(std::string filename){
    imufile.open(filename);
    if(!imufile.is_open()){
        cerr << "could not open " << filename << endl;
        return 0;
    }
    return 1;
}

int openGpsFile(std::string filename){
    gpsfile.open(filename);
    if(!gpsfile.is_open()){
        cerr << "could not open " << filename << endl;
        return 0;
    }
    return 1;
}

int readImageData(int& nb, double& stamp){
    if(!imagefile.is_open())
        return 0;
    char c;
    imagefile >> nb >> c >> stamp >> c;
    return 1;
}

int readImuData(ImuData& data){
    if(!imufile.is_open() || imufile.eof())
        return 0;
    char c;
    double stamp;
    double ax=0,ay=0,az=0,qw=0,qx=0,qy=0,qz=0,gx=0,gy=0,gz=0;
    imufile >> stamp >> c >> qw >> c >> qx >> c >> qy >> c >> qz >> c >> ax >> c >> ay >> c >> az >> c >> gx >> c >> gy >> c >> gz >> c;
    data.stamp = stamp;
//    cout << setprecision(12) << stamp << endl;
    data.acc = Vec3d(ax,ay,az);
    data.gyr = Vec3d(gx,gy,gz);
    data.orientation = Quat<double>(qw,qx,qy,qz);
    return 1;
}

int readGpsData(GpsData& data){
    if(!gpsfile.is_open())
        return 0;
    char c;
    double stamp,lon,lat,alt;
    int status;
    gpsfile >> stamp >> c >> lon >> c >> lat >> c >> alt>> c >> status >> c;
    data.stamp = stamp;
    data.lon = lon;
    data.lat = lat;
    data.alt = alt;
    data.status = status;
    return 1;
}

int getNextImuData(double stamp, ImuData& data){

    if(!imufile.is_open() || imufile.eof())
        return 0;
    data = ImuData();
//    int ok = 1;
    int count=0;
    ImuData rdata;
    while(data.stamp <= stamp && readImuData(rdata)){

        data+=rdata;
        count++;
        cout << count << endl;
        cout << setprecision(12) << rdata.stamp << endl;
    }
    data /=count;
    if(imufile.eof())
        return 0;
    else
        return 1;
}

int getNextGpsData(double stamp, GpsData& data){

    if(!gpsfile.is_open())
        return 0;
    do{
        readGpsData(data);
    }while(data.stamp <= stamp);
    return 1;
}

int loadImages(std::string& dir, int nb){

    stringstream num;num <<  std::setfill('0') << std::setw(5) << nb;
    imgL = imread(dir+"/cam0_image"+num.str()+"_"+rectification+".png",0);
    if(imgL.empty()){
        cerr << "cannot read " << dir+"/cam0_image"+num.str()+"_"+rectification+".png" << endl;
        return 0;
    }
    if(st == stereo){
        imgR = imread(dir+"/cam1_image"+num.str()+"_"+rectification+".png",0);
        if(imgL.empty()){
            cerr << "cannot read " << dir+"/cam0_image"+num.str()+"_"+rectification+".png" << endl;
            return 0;
        }
    }

    return 1;
}
