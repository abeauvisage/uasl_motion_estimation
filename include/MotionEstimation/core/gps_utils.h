#ifndef GPS_UTILS_H_INCLUDED
#define GPS_UTILS_H_INCLUDED

/** \file gps_utils.h
*   \brief Helper functions to convert from geodetic GNSS data to cartesian data
*
*    \author Axel Beauvisage (axel.beauvisage@gmail.com)
*/

#include "opencv2/core/core.hpp"

namespace me{

static cv::Point2d kitti_pt = cv::Point2d(48.997575979523,8.4772921616664);
static cv::Point2d da_pt = cv::Point2d(51.604834,-1.636528);
static cv::Point2d m_origin = da_pt;
static double m_angle=0;

static const cv::Point2d conv_kitti(111209.88256343921,73158.07119382407); // Kitti
static const cv::Point2d conv_da(111259.701456712,69288.4150563286); // DA

// values obtained from: http://www.csgnetwork.com/degreelenllavcalc.html

inline void setOrigin(const cv::Point2d& origin){m_origin=origin;}
inline cv::Point2d getOrigin(){return m_origin;}
inline void setAngle(const double theta){m_angle=theta;}
inline double getAngle(){return m_angle;}

inline double getDistanceFromOrigin(){return 2;}
inline cv::Point2f getCartesianCoordinate(const cv::Point2d& gps_geodetic){
    //if origin vector closer to DA than germany than select conv_da otherwise conv_kitti
    cv::Point2d conv = norm(m_origin-da_pt)<norm(m_origin-kitti_pt)?conv_da:conv_kitti;
    double gps_x = (gps_geodetic.x-m_origin.x)*conv.x;
    double gps_y = (gps_geodetic.y-m_origin.y)*conv.y;
    return cv::Point2f(sin(m_angle)*gps_x+cos(m_angle)*gps_y,cos(m_angle)*gps_x-sin(m_angle)*gps_y);
}

}// namespace me

#endif // GPS_UTILS_H_INCLUDED
