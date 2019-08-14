#ifndef GPS_UTILS_H_INCLUDED
#define GPS_UTILS_H_INCLUDED

/** \file gps_utils.h
*   \brief Helper functions to convert from geodetic GNSS data to cartesian data
*
*    \author Axel Beauvisage (axel.beauvisage@gmail.com)
*/

#include "core/rotation_utils.h"

namespace me{

static cv::Point2d m_origin{0,0};
static double m_angle=0;

static constexpr double m1 = 111132.92;		// latitude calculation term 1
static constexpr double	m2 = -559.82;		// latitude calculation term 2
static constexpr double	m3 = 1.175;			// latitude calculation term 3
static constexpr double	m4 = -0.0023;		// latitude calculation term 4
static constexpr double	p1 = 111412.84;		// longitude calculation term 1
static constexpr double	p2 = -93.5;			// longitude calculation term 2
static constexpr double	p3 = 0.118;			// longitude calculation term 3

// values and conversion  obtained from: http://www.csgnetwork.com/degreelenllavcalc.html

inline void setOrigin(const cv::Point2d& origin){m_origin=origin;}
inline cv::Point2d getOrigin(){return m_origin;}
inline void setAngle(const double theta){m_angle=theta;}
inline double getAngle(){return m_angle;}

inline cv::Point2f getCartesianCoordinate(const cv::Point2d& gps_geodetic){

    double latitude = deg2Rad(gps_geodetic.x);
    cv::Point2d lat_long_meters = {m1+(m2*cos(2*latitude))+(m3*cos(4*latitude))+(m4*cos(6*latitude)), (p1*cos(latitude))+(p2*cos(3*latitude))+(p3*cos(5*latitude))};
    double gps_x = (gps_geodetic.x-m_origin.x)*lat_long_meters.x;
    double gps_y = (gps_geodetic.y-m_origin.y)*lat_long_meters.y;
    return cv::Point2f(sin(m_angle)*gps_x+cos(m_angle)*gps_y,cos(m_angle)*gps_x-sin(m_angle)*gps_y);
}

}// namespace me

#endif // GPS_UTILS_H_INCLUDED
