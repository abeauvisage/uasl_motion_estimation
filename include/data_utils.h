#ifndef DATA_UTILS_H_INCLUDED
#define DATA_UTILS_H_INCLUDED

#include <opencv2/core.hpp>

#include "utils.h"

struct ImuData{

    //accelerometer
    cv::Vec3d acc;
    //gyroscope
    cv::Vec3d gyr;
    //orientation
    Quat<double> orientation;

    double stamp;

    //constructor
    ImuData(double stamp_=0, double a_x=0, double a_y=0, double a_z=0, double g_x=0,double g_y=0, double g_z=0, const Quat<double>& orient=Quat<double>()){acc=cv::Vec3d(a_x,a_y,a_z);gyr=cv::Vec3d(g_x,g_y,g_z);orientation=orient;stamp=stamp_;}
    ImuData(const ImuData& data){acc=data.acc;gyr=data.gyr;orientation=data.orientation;stamp=data.stamp;}

    void operator+=(const ImuData& imu){
        acc += imu.acc;
        gyr += imu.gyr;
        orientation = imu.orientation;

        stamp = imu.stamp;
    }

    ImuData operator+(const ImuData& imu) const{
        ImuData res;
        res.acc = acc+imu.acc;
        res.gyr = gyr+imu.gyr;
        res.orientation = imu.orientation;

        res.stamp = imu.stamp;
        return res;
    }

    void operator/=(const int nb){
        acc /= (double)nb;
        gyr /= (double)nb;
    }

};

struct GpsData{

    double lon;
    double lat;
    double alt;

    int status;

    double stamp;

    //constructor
    GpsData(double stamp_=0, double longitude=0, double latitude=0, double altitude=0, int status_=0){lon=longitude;lat=latitude;alt=altitude;stamp=stamp_;status=status_;}
    GpsData(const GpsData& data){lon=data.lon;lat=data.lat;alt=data.alt;status=data.status;stamp=data.stamp;}
};

//ImuData operator+(const ImuData& imu){
//
//
//}


#endif // DATA_UTILS_H_INCLUDED
