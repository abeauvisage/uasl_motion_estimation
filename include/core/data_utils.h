#ifndef DATA_UTILS_H_INCLUDED
#define DATA_UTILS_H_INCLUDED

#include <opencv2/core.hpp>

#include "core/rotation_utils.h"

namespace me{

//! enumerate representing unit in which timestamps are expressed
enum class TimeUnit{SEC,MILLI,MICRO,NANO};

struct Data{

    int64_t stamp; //!< acquisition time stamp
    TimeUnit time_unit; //!< unit

    Data(int64_t stamp_, TimeUnit time_unit_): stamp(stamp_), time_unit(time_unit_) {}

};

//! Structure representing inertial data.
/*! Contains acceleration, angular velocity, orientation and timestamp. */
struct ImuData: public Data{

    cv::Vec3d acc; //!< accelerometer
    cv::Vec3d gyr; //!< angular velocity (gyro)
    cv::Vec3d pos; //!< position
    Quatd orientation; //!< orientation (optional)

    //! Main constructor. By default all parameters are equal to 0.
    ImuData(double a_x=0, double a_y=0, double a_z=0, double g_x=0,double g_y=0, double g_z=0, double p_x=0, double p_y=0, double p_z=0, int64_t stamp_=0, TimeUnit time_unit_=TimeUnit::SEC, const Quatd& orient=Quatd()): Data(stamp_,time_unit_),acc(cv::Vec3d(a_x,a_y,a_z)),gyr(cv::Vec3d(g_x,g_y,g_z)),pos(cv::Vec3d(p_x,p_y,p_z)),orientation(orient){}
    //! Copy constructor.
    ImuData(const ImuData& data):Data(data.stamp,data.time_unit), acc(data.acc), gyr(data.gyr), pos(data.pos), orientation(data.orientation){}

    //! concatenation operator. Sum up  acceleration, angular velocity and orientation. Timestamp is replaced by the one of the added ImuData.
    void operator+=(const ImuData& imu){
        acc += imu.acc;
        gyr += imu.gyr;
        pos = imu.pos;
        orientation = imu.orientation;

        stamp = imu.stamp;
    }
    //! Sum up  acceleration, angular velocity and orientation. Timestamp is replaced by the one of the added ImuData. Return a new ImuData.
    ImuData operator+(const ImuData& imu) const{
        ImuData res;
        res.acc = acc+imu.acc;
        res.gyr = gyr+imu.gyr;
        res.pos = imu.pos;
        res.orientation = imu.orientation;

        res.stamp = imu.stamp;
        return res;
    }

    //! Division operator. Divide acceleration and angular velocity by a specific value. Used for computing mean of ImuData.
    void operator/=(const int nb){
        acc /= (double)nb;
        gyr /= (double)nb;
    }

};

//! Structure representing a GPS coordinate.
/*! Contains longitude, latitude, altitude and timestamp. */
struct GpsData: Data{

    double lon; //!< longitude
    double lat; //!< latitude
    double alt; //!< altitude

    int status; //!< Gps status (deprecated)

    //! Main constructor. By defaut all parameters are equal to 0.
    GpsData(double longitude=0, double latitude=0, double altitude=0, int64_t stamp_=0, TimeUnit time_unit_=TimeUnit::SEC, int status_=0):Data(stamp_,time_unit_), lon(longitude), lat(latitude), alt(altitude), status(status_){}
    //! Copy constructor.
    GpsData(const GpsData& data): Data(data.stamp,data.time_unit), lon(data.lon), lat(data.lat), alt(data.alt), status(data.status){}
};

//! Structure representing the pose of a camera/vehicle.
/*! Contains orientation and attitude. */
struct PoseData: Data{

    cv::Vec3d position; //!< position
    Quatd orientation; //!< orientation

    //! Main constructor. By defaut all parameters are equal to 0.
    PoseData(const cv::Vec3d& pos=cv::Vec3d(), const Quatd& ori=Quatd(), const int64_t stamp_=0, const TimeUnit time_unit_=TimeUnit::SEC):Data(stamp_,time_unit_), position(pos), orientation(ori){}
    //! Copy constructor.
    PoseData(const PoseData& data): Data(data.stamp,data.time_unit), position(data.position), orientation(data.orientation){}
};

}

#endif // DATA_UTILS_H_INCLUDED
