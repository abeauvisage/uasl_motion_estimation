#ifndef GPS_UTILS_H_INCLUDED
#define GPS_UTILS_H_INCLUDED

#include "opencv2/core.hpp"


namespace me{

static cv::Point2d m_origin=cv::Point2d(51.604834,-1.636528);
static double m_angle=0;
static const cv::Point2d conv(111259.701456712,69288.4150563286);

inline void setOrigin(const cv::Point2d& origin){m_origin=origin;}
inline cv::Point2d getOrigin(){return m_origin;}
inline void setAngle(const double theta){m_angle=theta;}
inline double getAngle(){return m_angle;}

inline double getDistanceFromOrigin(){return 2;}
inline cv::Point2f getCartesianCoordinate(const cv::Point2d& gps_geodetic){
    double gps_x = (gps_geodetic.x-m_origin.x)*conv.x;
    double gps_y = (gps_geodetic.y-m_origin.y)*conv.y;
    return cv::Point2f(sin(m_angle)*gps_x+cos(m_angle)*gps_y,cos(m_angle)*gps_x-sin(m_angle)*gps_y);
}

}
#endif // GPS_UTILS_H_INCLUDED
