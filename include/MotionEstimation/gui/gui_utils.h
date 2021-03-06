#ifndef GUI_UTILS_H_INCLUDED
#define GUI_UTILS_H_INCLUDED

/** \file gui_utils.h
*   \brief Various graphical interface to show 2D features in the current images
*
*    \author Axel Beauvisage (axel.beauvisage@gmail.com)
*/

#include "core/feature_types.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace me{

/**** display WBA_Ptf ****/

//! display features in the latest image
cv::Mat show(const std::vector<me::WBA_Ptf>& pts, const std::vector<me::CamPose_qd>& poses, const cv::Mat& img);
//! display features in the latest stereo pair with color based on the feature depth
cv::Mat show(const std::vector<me::WBA_Ptf>& pts, const std::pair<std::vector<me::CamPose_qd>,std::vector<me::CamPose_qd>>& poses, const std::pair<cv::Mat,cv::Mat>& img);
//! display features in the latest stereo pair with color based on the feature depth
cv::Mat show(const std::vector<me::WBA_stereo_Ptf>& pts, const std::pair<std::vector<me::CamPose_qd>,std::vector<me::CamPose_qd>>& poses, const std::pair<cv::Mat,cv::Mat>& img);
//! display features in the latest stereo pair with color based on camera number
cv::Mat show_stereo_reproj(const std::vector<me::WBA_Ptf>& pts, const std::pair<std::vector<me::CamPose_qd>,std::vector<me::CamPose_qd>>& poses, const std::pair<cv::Mat,cv::Mat>& img, const cv::Matx33d& K);
cv::Mat show_stereo_reproj_scaled(const std::pair<std::vector<me::WBA_Ptf>,std::vector<me::WBA_Ptf>>& pts, const std::pair<std::vector<me::CamPose_qd>,std::vector<me::CamPose_qd>>& poses, const std::pair<cv::Mat,cv::Mat>& img, const std::pair<cv::Matx33d,cv::Matx33d>& K, double baseline,double scale=1.0);

/**** display Point2f ****/

//! display a set of Point2f pts in image img
cv::Mat show(const cv::Mat& img, const std::vector<cv::Point2f>& pts);

/**** display covariance ****/

//! display a 2D confidence ellipses (95%) of the 3D covariance provided along each axis
void display_cov(const std::string& wname, const cv::Mat& cov, double s=1);

}

#endif // GUI_UTILS_H_INCLUDED
