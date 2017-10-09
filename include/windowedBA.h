#ifndef WINDOWEDBA_H_INCLUDED
#define WINDOWEDBA_H_INCLUDED

#include <vector>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/sfm.hpp>

#include <utils.h>
#include <featureType.h>
#include <stereo_viso.h>
#include <mono_viso.h>

namespace me{

enum StopCondition{NO_STOP,SMALL_GRADIENT,SMALL_INCREMENT,MAX_ITERATIONS,SMALL_DECREASE_FUNCTION,SMALL_REPROJ_ERROR,NO_CONVERGENCE};

void solveWindowedBA(const std::vector<std::vector<cv::Point2f>>& observations, std::vector<ptH3D>& pts3D, const std::vector<double>& weight, const MonoVisualOdometry::parameters& params, const cv::Mat& img, std::vector<cv::Vec3d>& ori, std::vector<cv::Vec3d>& pos);
cv::Vec3d solveWindowedBA(const std::vector<std::vector<std::pair<me::ptH2D,me::ptH2D>>>& observations,const std::vector<ptH3D>& pts3D, const cv::Matx33d& K, const cv::Mat& img);

bool optimize(const std::vector<std::vector<cv::Point2f>>& observations,const std::vector<ptH3D>& pts3D, cv::Mat& state);
bool optimize_stereo(const std::vector<std::vector<std::pair<me::ptH2D,me::ptH2D>>>& observations,const std::vector<ptH3D>& pts3D, cv::Mat& state);

std::vector<cv::Matx34d> computeProjectionMatrices(const cv::Mat& Xa);
std::vector<std::vector<cv::Point2f>> project_pts(const cv::Mat& Xb, const std::vector<cv::Matx34d>& pMat);

cv::Mat compute_residuals(const std::vector<std::vector<cv::Point2f>>& observations, const cv::Mat& Xb, const cv::Mat& Xa);
cv::Mat compute_residuals_stereo(const std::vector<std::vector<std::pair<me::ptH2D,me::ptH2D>>>& observations, const std::vector<ptH3D>& pts3D, const cv::Mat& Xa);

void computeJacobian(const cv::Mat& Xa,const cv::Mat& Xb, const cv::Mat& residuals, cv::Mat& JJ, cv::Mat& e);
void computeJacobian_stereo(const cv::Mat& Xa,const std::vector<ptH3D>& pts3D, const cv::Mat& residuals, cv::Mat& JJ, cv::Mat& e);

void showCameraPoses(const cv::Mat& Xa);
void showCameraPosesAndPoints(const Euld& orientation,const cv::Vec3d& position, const std::vector<ptH3D>& pts);


void showReprojectedPts(const cv::Mat& img, const std::vector<cv::Matx34d>& pMat, const std::vector<std::vector<cv::Point2f>>& observations, const cv::Mat& Xb);


//std::vector<std::vector<cv::Point2f>> project_pts(const std::vector<ptH3D>& pts3D, const std::vector<cv::Matx34d>& pMat);
//cv::Mat compute_residuals(const std::vector<std::vector<cv::Point2f>>& observations, const std::vector<ptH3D>& pts3D, const cv::Mat& Xa);
//void computeJacobian(const cv::Mat& Xa,const std::vector<ptH3D>& pts3D, const cv::Mat& residuals, cv::Mat& JJ, cv::Mat& e);
//void showReprojectedPts(const cv::Mat& img, const std::vector<cv::Matx34d>& pMat, const std::vector<std::vector<cv::Point2f>>& observations, const std::vector<ptH3D>& pts3D);


}

#endif // WINDOWEDBA_H_INCLUDED
