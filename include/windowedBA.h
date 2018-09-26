#ifndef WINDOWEDBA_H_INCLUDED
#define WINDOWEDBA_H_INCLUDED

#include <deque>
#include <vector>
#include <iostream>

#include <opencv2/core.hpp>

#include <featureType.h>
#include <stereo_viso.h>
#include <mono_viso.h>

namespace me{

//! function computing Projection matrix from K, R and t
void compPMat(cv::InputArray _K, cv::InputArray _R, cv::InputArray _t, cv::OutputArray _P);
//! Global BA function using OpenCV Point2f
/*! solve BA problem with observations, 3D points and camera poses. Deprecated.*/
void solveWindowedBA(const std::deque<std::vector<cv::Point2f>>& observations, std::vector<ptH3D>& pts3D, std::vector<double>& weight, const MonoVisualOdometry::parameters& params, const cv::Mat& img, std::vector<me::Euld>& ori, std::vector<cv::Vec3d>& pos);
//! Global BA function using WBA_Points
/*! solve BA problem with observations, 3D points and camera poses*/
void solveWindowedBA(std::vector<WBA_Ptf*>& pts, const cv::Matx33d& K, const cv::Mat& img, std::vector<me::Euld>& ori, std::vector<cv::Vec3d>& pos, int start, int window_size, int fixedFrames=10);
//! Global BA function for stereo using OpenCV Point2f
/*! solve BA problem with observations, 3D points and camera poses*/
cv::Vec3d solveWindowedBA(const std::vector<std::vector<std::pair<me::ptH2D,me::ptH2D>>>& observations,const std::vector<ptH3D>& pts3D, const cv::Matx33d& K, const cv::Mat& img);
void solveWindowedBA(std::vector<WBA_Ptf*>& pts, const cv::Matx33d& K, std::vector<CamPose_qd>& poses, int fixedFrames);
void solveWindowedBAManifold(std::vector<WBA_Ptf*>& pts, const cv::Matx33d& K, std::vector<CamPose_qd>& poses, int fixedFrames, std::vector<int>& outliers);
//! optimization function with OpenCV Point2f
/*! implements LM and GN minimization algo. Deprecated */
bool optimize(const std::deque<std::vector<cv::Point2f>>& observations, cv::Mat& state, const cv::Mat& visibility,int fixedFrames);
//! optimization function with WBA_Points
/*! implements LM and GN minimization algo */
bool optimize(const std::vector<WBA_Ptf*>& pts, cv::Mat& state, const int window_size, const int fixedFrames);
//! optimization function for stereo with OpenCV Point2f
/*! implements LM and GN minimization algo. Deprecated */
bool optimize_stereo(const std::vector<std::vector<std::pair<me::ptH2D,me::ptH2D>>>& observations,const std::vector<ptH3D>& pts3D, cv::Mat& state);

//! function computing Projection matrices for each camera from the state vector
std::vector<cv::Matx34d> computeProjectionMatrices(const cv::Mat& Xa);
//! function projecting 3D points into observations
std::deque<std::vector<cv::Point2f>> project_pts(const cv::Mat& Xb, const std::vector<cv::Matx34d>& pMat);

//! computes residual errors from the provided state vector using OpenCV Point2f
/*! Deprecated */
cv::Mat compute_residuals(const std::deque<std::vector<cv::Point2f>>& observations, const cv::Mat& Xa, const cv::Mat& Xb, const cv::Mat& visibility=cv::Mat());
//! computes residual errors from the provided state vector using WBA_Points
cv::Mat compute_residuals(const std::vector<WBA_Ptf*>& pts, const cv::Mat& Xa, const cv::Mat& Xb);
//! computes residual errors from the provided state vector using OpenCV Point2f
/*! Deprecated */
cv::Mat compute_residuals_stereo(const std::vector<std::vector<std::pair<me::ptH2D,me::ptH2D>>>& observations, const std::vector<ptH3D>& pts3D, const cv::Mat& Xa);

//! computes Jacobian matrix from state vector
void computeJacobian(const cv::Mat& Xa,const cv::Mat& Xb, const cv::Mat& residuals, cv::Mat& JJ, cv::Mat& U, cv::Mat& V, cv::Mat& W, cv::Mat& e, const cv::Mat& visibility=cv::Mat(), int fixedFrames=0);
//! computing Jacobian matrix from pts
void computeJacobian(const std::vector<WBA_Ptf*>& pts, const cv::Mat& Xa,  const cv::Mat& Xb, const cv::Mat& residuals, cv::Mat& JJ, cv::Mat& A, cv::Mat& B, cv::Mat& W, cv::Mat& e, int fixedFrames);
void computeJacobian(const cv::Mat& Xa,  const cv::Mat& Xb, const cv::Mat& residuals, std::vector<cv::Matx66d>& U, std::vector<cv::Matx33d>& V, cv::Mat& W, cv::Mat& ea, cv::Mat& eb, const cv::Mat& visibility, int fixedFrames);

//! computing Jacobian matrix for stereo
/*! Deprecated */
void computeJacobian_stereo(const cv::Mat& Xa,const std::vector<ptH3D>& pts3D, const cv::Mat& residuals, cv::Mat& JJ, cv::Mat& e);

//! Display the different camera poses in a 3D environment
void showCameraPoses(const cv::Mat& Xa);
//! Display the different camera poses and 3D points in a 3D environment
void showCameraPosesAndPoints(const Euld& orientation,const cv::Vec3d& position, const std::vector<ptH3D>& pts);
//! Display the different camera poses and 3D points in a 3D environment
void showCameraPosesAndPoints(const std::vector<Euld>& ori ,const std::vector<cv::Vec3d>& pos, const std::vector<WBA_Ptf>& pts);
//! Display the different camera poses and 3D points in a 3D environment
void showCameraPosesAndPoints(const cv::Mat& P, const std::vector<ptH3D>& pts);
//! Display the different camera poses and 3D points in a 3D environment
void showReprojectedPts(const cv::Mat& img, const std::vector<cv::Matx34d>& pMat, const std::vector<std::vector<cv::Point2f>>& observations, const cv::Mat& Xb);
void showReprojectedPts(const cv::Mat& img, const std::vector<cv::Matx44d>& cam_poses, const std::vector<pt3D>& pts3D, const std::deque<std::vector<cv::Point2f>>& observations, const cv::Mat& visibility=cv::Mat());
void showCameraPosesAndPointsT(const std::vector<CamPose_qd>& poses, const std::vector<WBA_Ptf*>& pts);
void vizLoop();

/** On manifold optim ***/
bool optimize(const std::deque<std::vector<cv::Point2f>>& observations, std::vector<cv::Matx44d>& cam_poses, std::vector<pt3D>& pts3D, const std::vector<std::vector<cv::Matx22d>>& cov, const cv::Mat& visibility=cv::Mat(), const int fixedFrames=0);
void computeJacobian(const std::vector<cv::Matx44d>& camera_poses, const std::vector<pt3D>& pts3D, const cv::Mat& residuals, cv::Mat& JJ, cv::Mat& U, cv::Mat& V, cv::Mat& W, cv::Mat& e, const std::vector<std::vector<cv::Matx22d>>& cov, const cv::Mat& visibility=cv::Mat(), int fixedFrames=0);
void computeJacobian(const std::vector<cv::Matx44d>& camera_poses, const std::vector<pt3D>& pts3D, const cv::Mat& residuals, std::vector<cv::Matx66d>& U, std::vector<cv::Matx33d>& V, cv::Mat& W, cv::Mat& ea, cv::Mat& eb, const std::vector<std::vector<cv::Matx22d>>& cov, const cv::Mat& visibility=cv::Mat(), int fixedFrames=0);
cv::Mat compute_residuals(const std::deque<std::vector<cv::Point2f>>& observations, const std::vector<cv::Matx44d>& poses, const std::vector<pt3D>& pts3D, const cv::Mat& visibility=cv::Mat());
cv::Mat compute_dHdp(const cv::Matx44d& pose, const pt3D& pt);
cv::Mat compute_dHde(const cv::Matx44d& pose, const pt3D& pt);

cv::Matx44d exp_map(const cv::Matx61d& mat);
double MedianAbsoluteDeviation(const cv::Mat& squared_error);

}

#endif // WINDOWEDBA_H_INCLUDED
