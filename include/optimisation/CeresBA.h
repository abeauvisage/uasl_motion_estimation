#ifndef CERESBA_H
#define CERESBA_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <core/mutual_information.h>
#include <core/feature_types.h>

class CeresBA {

public:

struct ReprojectionError {
  ReprojectionError(const double observed_x, const double observed_y, const double sigma) : observed_x(observed_x), observed_y(observed_y),sigma_inv(1.0/sigma){}

  template <typename T>
  bool operator()(const T* const camera, const T* const point, T* residuals) const {

    T p[3];
    ceres::AngleAxisRotatePoint(camera+3, point, p);

    p[0] += camera[0];
    p[1] += camera[1];
    p[2] += camera[2];

    T xp =  p[0] / p[2];
    T yp =  p[1] / p[2];

    T predicted_x = (double)(K_(0,0)) * xp + (double)(K_(0,2));
    T predicted_y = K_(1,1) * yp + K_(1,2);

    residuals[0] = sigma_inv*(predicted_x - observed_x);
    residuals[1] = sigma_inv*(predicted_y - observed_y);
    return true;
  }

  static ceres::CostFunction* Create(const double observed_x,const double observed_y, const double sigma) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(new ReprojectionError(observed_x, observed_y,sigma)));
  }
  double observed_x;
  double observed_y;
  double sigma_inv;
};

struct ReprojectionErrorMonoRight {
  ReprojectionErrorMonoRight(double observed_x, double observed_y, const double sigma) : observed_x(observed_x), observed_y(observed_y),sigma_inv(1.0/sigma) {}

  template <typename T>
  bool operator()(const T* const camera,const T* const point,T* residuals) const {

    T p[3];
    ceres::AngleAxisRotatePoint(camera+3, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[0]-baseline_;
    p[1] += camera[1];
    p[2] += camera[2];

    T xp =  p[0] / p[2];
    T yp =  p[1] / p[2];

    T predicted_x = (double)(K_(0,0)) * xp + (double)(K_(0,2));
    T predicted_y = K_(1,1) * yp + K_(1,2);
    // The error is the difference between the predicted and observed position.
    residuals[0] = sigma_inv*(predicted_x - observed_x);
    residuals[1] = sigma_inv*(predicted_y - observed_y);
    return true;
  }

  static ceres::CostFunction* Create(const double observed_x,const double observed_y, const double sigma) {
    return (new ceres::AutoDiffCostFunction<ReprojectionErrorMonoRight, 2, 6, 3>(new ReprojectionErrorMonoRight(observed_x, observed_y,sigma)));
  }
  double observed_x;
  double observed_y;
  double sigma_inv;
};

struct StereoReprojectionError {
  StereoReprojectionError(double x1, double y1, double x2, double y2,const double sigma2) : x1(x1), y1(y1), x2(x2), y2(y2), sigma2_inv(1.0/sigma2){}

  template <typename T>
  bool operator()(const T* const camera, const T* const point, T* residuals) const {

    T p[3];
    ceres::AngleAxisRotatePoint(camera+3, point, p);
    p[0] += camera[0];
    p[1] += camera[1];
    p[2] += camera[2];

    T xp1 =  p[0] / p[2];
    T xp2 =  (p[0]-baseline_)  / p[2];
    T yp =  p[1] / p[2];

    T predicted_x1 = (double)(K_(0,0)) * xp1 + (double)(K_(0,2));
    T predicted_x2 = (double)(K_(0,0)) * xp2 + (double)(K_(0,2));
    T predicted_y = K_(1,1) * yp + K_(1,2);

    residuals[0] = sigma2_inv*(predicted_x1 - x1);
    residuals[1] = sigma2_inv*(predicted_y - y1);
    residuals[2] = sigma2_inv*(predicted_x2 - x2);
    residuals[3] = sigma2_inv*(predicted_y - y2);
    return true;
  }

  static ceres::CostFunction* Create(const double x1, const double y1, const double x2, const double y2, double sigma2) {
    return (new ceres::AutoDiffCostFunction<StereoReprojectionError, 4, 6, 3>(new StereoReprojectionError(x1,y1,x2,y2,sigma2)));
  }
  double x1,y1,x2,y2;
  double sigma2_inv;
};

  static cv::Matx33d K_;
  static double baseline_;

    CeresBA(int nb_view, int nb_pts, const cv::Matx33d& K, double feat_noise=1.0, double baseline=0);
  ~CeresBA() {
    if(problem)
        delete problem;
    if(point_index_)
        delete[] point_index_;
    if(camera_index_)
        delete[] camera_index_;
    if(observations_)
        delete[] observations_;
    if(parameters_)
        delete[] parameters_;
    if(cam_idx)
        delete[] cam_idx;
  }
  int num_observations()       const { return num_observations_;               }
  const double* observations() const { return observations_;                   }
  double* mutable_cameras()          { return parameters_;                     }
  double* mutable_points()           { return parameters_  + 6 * num_cameras_; }
  double* mutable_camera_for_observation(int i) {
    return mutable_cameras() + camera_index_[i] * 6;
  }
  double* mutable_point_for_observation(int i) {
    return mutable_points() + point_index_[i] * 3;
  }

  void fillData(const std::vector<std::vector<cv::Point2f>>& obs, const std::vector<me::ptH3D>& pts);
  void fillData(const std::vector<me::WBA_Ptf>& pts, const std::vector<me::Euld>& ori, const std::vector<cv::Vec3d> pos, int start);
  void fillData(const std::vector<me::WBA_Ptf*>& pts, const std::vector<me::CamPose_qd>& poses);
  void fillData(const std::vector<me::WBA_Ptf>& pts, const std::vector<me::CamPose_qd>& poses);
  void fillStereoData(const std::pair<std::vector<me::WBA_Ptf>,std::vector<me::WBA_Ptf>>& pts, const std::vector<me::CamPose_qd>& poses);
  void fillStereoData(const std::vector<me::WBA_stereo_Ptf>& pts, const std::vector<me::CamPose_qd>& poses);

  void fillPoints(const std::vector<me::ptH3D>& pts, const std::vector<uchar>& mask=std::vector<uchar>());
  void fillCameras(const std::vector<me::CamPose_md>& poses);
  void fillCameras(const std::vector<me::CamPose_qd>& poses);
  void fillObservations(const std::vector<me::StereoOdoMatchesf>& obs);
  void fillObservations(const std::vector<me::WBA_Ptf>& obs);

  std::vector<me::pt3D> get3DPoints();
  std::vector<me::CamPose_qd> getQuatPoses();
  std::vector<me::CamPose_md> getMatPoses();
  std::vector<int> get_inliers(const double threshold);
  std::vector<int> get_inliers_stereo(const double threshold);
  bool getCovariance(std::vector<cv::Mat>& poseCov, std::vector<cv::Matx33d>* pointCov=nullptr);
  bool getCovarianceQuat(std::vector<cv::Mat>& poseCov, std::vector<cv::Matx33d>& pointCov);

  void runSolver(int fixedFrames=0);
  void runStereoSolver(int fixedFrames=0);
  bool runRansac(int nb_iterations, double inlier_threshold, int fixedFrames=0, bool stereo=false);

  std::vector<cv::Point2f> reproject_features(int j_view){

    std::vector<cv::Point2f> pts;
    for(int i=0;i<num_points_;i++){
        double* point = parameters_ + num_cameras_*6+i*3;
        double* camera = parameters_ + j_view*6;
        double p[3];
        ceres::AngleAxisRotatePoint(camera+3, point, p);
        p[0] += camera[0];
        p[1] += camera[1];
        p[2] += camera[2];
        double xp =  p[0] / p[2];
        double yp =  p[1] / p[2];

        double predicted_x = (double)(K_(0,0)) * xp + (double)(K_(0,2));
        double predicted_y = K_(1,1) * yp + K_(1,2);
        pts.push_back(cv::Point2f(predicted_y,predicted_x));
    }
    return pts;
  }

 private:

  ceres::Problem* problem=nullptr;

  std::vector<uchar> m_mask;

  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;
  int* point_index_;
  int* camera_index_;
  int* cam_idx;
  double feat_noise_;

  std::vector<int>  camera_nbs;

  double* observations_;
  double* parameters_;
};

class ScaleOptimiser{

public:

struct MIError {
  MIError(const me::ptH3D& pt_, const me::CamPose_qd& pose_, const int camNum_, const int window_size_, const std::pair<cv::Mat,cv::Mat>* ptr_) : point(pt_),pose(pose_),camNum(camNum_),window_size(window_size_),ptr(ptr_){}

  bool operator()(const double* const lambda, double* residuals) const {

    assert(!ptr->first.empty() && !ptr->first.empty());

    double scale = *lambda;
    cv::Matx44d Tr = pose.TrMat();
    me::ptH2D feat = K_ * cv::Matx34d::eye() * scale *(Tr * point);
    cv::Point2f feat_left(me::to_euclidean(feat)(0),me::to_euclidean(feat)(1));
    me::ptH2D feat_ =  (K_* cv::Matx34d::eye()) * (scale * (Tr * point) - cv::Matx41d(baseline_,0,0,0));
    cv::Point2f feat_right(me::to_euclidean(feat_)(0),me::to_euclidean(feat_)(1));

    cv::Rect bb(window_size,window_size,ptr->first.cols-2*window_size,ptr->first.rows-2*window_size);

    if(bb.contains(feat_left) && bb.contains(feat_right)){
        cv::Mat ROI_left = ptr->first(cv::Rect(feat_left.x-window_size,feat_left.y-window_size,window_size*2,window_size*2));
        cv::Mat ROI_right = ptr->second(cv::Rect(feat_right.x-window_size,feat_right.y-window_size,window_size*2,window_size*2));
        cv::Mat ROI_right_plus = ptr->second(cv::Rect(feat_right.x+1-window_size,feat_right.y-window_size,window_size*2,window_size*2));
        double MI = me::computeMutualInformation(ROI_left,ROI_right), MI_plus = me::computeMutualInformation(ROI_left,ROI_right_plus);
        *residuals = 1.0/(MI+(MI_plus-MI)*(feat_right.x-trunc(feat_right.x))+1e-3);
    }else{
        *residuals = 1.0/1e-3;
    }

    return true;
  }

  static ceres::CostFunction* Create(const me::ptH3D& pt_, const me::CamPose_qd& pose_, const int camNum_, const int window_size_, const std::pair<cv::Mat,cv::Mat>* ptr_) {
    return (new ceres::NumericDiffCostFunction<MIError,ceres::CENTRAL, 1, 1>(new MIError(pt_,pose_,camNum_,window_size_,ptr_)));
  }
  me::ptH3D point;
  me::CamPose_qd pose;
  int camNum,window_size;
  const std::pair<cv::Mat,cv::Mat>* ptr;
};

ScaleOptimiser(const std::pair<cv::Mat,cv::Mat>& stereoPair_, const cv::Matx33d& K, const double baseline): stereoPair(stereoPair_){K_=K;baseline_=baseline;}
~ScaleOptimiser(){delete problem;delete lambda;}
void findScale(const std::vector<me::WBA_Ptf>& features, const me::CamPose_qd& pose, double init_value=1.0);

static cv::Matx33d K_;
static double baseline_;

private:

ceres::Problem* problem=nullptr;
double* lambda=nullptr;

std::pair<cv::Mat,cv::Mat> stereoPair;

};

#endif // CERESBA_H
