#ifndef BUNDLEAJUSTER_H_INCLUDED
#define BUNDLEAJUSTER_H_INCLUDED

/** \file BundleAdjuster.h
*   \brief A class to run bundle adjustment with the ceres library
*
*   This optimiser is only meant to be used once.
*
*    \author Axel Beauvisage (axel.beauvisage@gmail.com)
*/

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "core/feature_types.h"

namespace me{

namespace optimisation{

//! struct to represent an observation of 2D point(s) in image(s)
/*! data do not need to be allocated dynamically, std::array is safer */
template<int N>
struct Observation{
  std::array<double,N> data; //! 2D features
  int camIdx; //!< camera index in the window
  int ptIdx; //!< 3D pt which has been observed
  int camID; //!< cameraID (in which the observation was made)

  Observation(const std::array<double,N>& data_, const int camIdx_, const int ptIdx_, const int camID_=0): data{data_}, camIdx{camIdx_}, ptIdx{ptIdx_}, camID{camID_}{}
  const double& operator[](int i) const {return data[i];}
  double& operator[](int i){return data[i];}
};

struct CalibrationParameters{

  std::vector<cv::Matx33d> K; //!< calibration parameters (for each camera)
  double feat_var; //!< feature location noise covaraiance in x-y = var * I_{2x2}
  double baseline; //!< baseline for stereo setups
  bool compute_cov; //!< estimating pose and pts covariance matrices or not

  explicit CalibrationParameters(const cv::Matx33d& K_,double var,double baseline_=0.0): K{{K_}}, feat_var{var}, baseline{baseline_}, compute_cov{false}{}
  CalibrationParameters(const std::vector<cv::Matx33d>& K_,double var,double baseline_=0.0): K{K_}, feat_var{var}, baseline{baseline_}, compute_cov{false}{}

};

/***********************/
/* Reprojection Errors */
/***********************/

//! Base class for reprojection Error
/*! represent the distance between observations (features)
    and the reprojection of a poitn p_i in camera c_j.
    Each derived class needs to have a tamplated operator(T*,T*,T*)
    and a creat function */
template<int N>
class ReprojectionError{

protected:

  const std::array<double,N> features; //!< 2D observations
  const double sigma_inv;
  const CalibrationParameters* const calib;

public:

  ReprojectionError(const std::array<double,N>& list_, const CalibrationParameters* const calib_ptr, const double sigma): features{list_},sigma_inv{sigma},calib{calib_ptr}{}
};

//! Specialised class for standard error in one image
class StandardReprojectionError: ReprojectionError<2>{

public:

  StandardReprojectionError(const std::array<double,2>& list_, const CalibrationParameters* const params, const double sigma): ReprojectionError{list_,params,1.0/sigma}{}
  StandardReprojectionError(const double obs_x,const double obs_y, const CalibrationParameters* const params, const double sigma): ReprojectionError{{obs_x,obs_y},params,1.0/sigma}{}

  template <typename T>
  bool operator()(const T* const camera, const T* const point, T* residuals) const {

    T p[3];
    ceres::AngleAxisRotatePoint(camera+3, point, p);

    p[0] += camera[0];
    p[1] += camera[1];
    p[2] += camera[2];

    T predicted_x = (double)(calib->K[0](0,0)) * (p[0]/p[2]) + (double)(calib->K[0](0,2));
    T predicted_y = calib->K[0](1,1) * (p[1]/p[2]) + calib->K[0](1,2);

    residuals[0] = sigma_inv*(predicted_x - features[0]);
    residuals[1] = sigma_inv*(predicted_y - features[1]);
    return true;
  }

  //!< builder. Needed for the object to converted to costFunstion, but also to be stored on the heap and avoid going out of scope
  static ceres::CostFunction* create(const std::array<double,2>& list_, const CalibrationParameters* const calib, const double sigma=1.0) {
    return (new ceres::AutoDiffCostFunction<StandardReprojectionError, 2, 6, 3>(new StandardReprojectionError(list_,calib,sigma)));
  }
  static ceres::CostFunction* create(const double obs_x,const double obs_y, const CalibrationParameters* const calib, const double sigma=1.0) {
    return (new ceres::AutoDiffCostFunction<StandardReprojectionError, 2, 6, 3>(new StandardReprojectionError(obs_x,obs_y,calib,sigma)));
  }
};

//! Specialised class for reprojecting a point in the right image of a stereo pair
class StereoRightError: ReprojectionError<2>{

public:

  StereoRightError(const std::array<double,2>& list_, const CalibrationParameters* const params, const double sigma): ReprojectionError{list_,params,1.0/sigma}{}
  StereoRightError(const double obs_x,const double obs_y, const CalibrationParameters* const params, const double sigma): ReprojectionError{{obs_x,obs_y},params,1.0/sigma}{}

  template <typename T>
  bool operator()(const T* const camera, const T* const point, T* residuals) const {

    T p[3];
    ceres::AngleAxisRotatePoint(camera+3, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[0]-calib->baseline;
    p[1] += camera[1];
    p[2] += camera[2];

    T predicted_x = (double)(calib->K[0](0,0)) * (p[0]/p[2]) + (double)(calib->K[0](0,2));
    T predicted_y = calib->K[0](1,1) * (p[1]/p[2]) + calib->K[0](1,2);

    // The error is the difference between the predicted and observed position.
    residuals[0] = sigma_inv*(predicted_x - features[0]);
    residuals[1] = sigma_inv*(predicted_y - features[1]);
    return true;
  }

  //!< builder. Needed for the object to converted to costFunstion, but also to be stored on the heap and avoid going out of scope
  static ceres::CostFunction* create(const std::array<double,2>& list_, const CalibrationParameters* const calib, const double sigma=1.0) {
    return (new ceres::AutoDiffCostFunction<StereoRightError, 2, 6, 3>(new StereoRightError(list_,calib,sigma)));
  }
  static ceres::CostFunction* create(const double obs_x,const double obs_y, const CalibrationParameters* const calib, const double sigma=1.0) {
    return (new ceres::AutoDiffCostFunction<StereoRightError, 2, 6, 3>(new StereoRightError(obs_x,obs_y,calib,sigma)));
  }
};

//! Specialised class for error when reprojecting in a stereo pair
class StereoReprojectionError: ReprojectionError<4>{

public:

  StereoReprojectionError(const std::array<double,4>& list_, const CalibrationParameters* const params, const double sigma): ReprojectionError{list_,params,1.0/sigma}{
    assert(!calib->K.empty() && calib->baseline != 0.0 && "wrong calibration parameters");
  }
  StereoReprojectionError(const double obs1_x,const double obs1_y,const double obs2_x,const double obs2_y, const CalibrationParameters* const params, const double sigma): ReprojectionError{{obs1_x,obs1_y,obs2_x,obs2_y},params,1.0/sigma}{
    assert(!calib->K.empty() && calib->baseline != 0.0 && "wrong calibration parameters");
  }

  template <typename T>
  bool operator()(const T* const camera, const T* const point, T* residuals) const {

    T p[3];
    ceres::AngleAxisRotatePoint(camera+3, point, p);
    p[0] += camera[0];
    p[1] += camera[1];
    p[2] += camera[2];

    T predicted_x1 = calib->K[0](0,0) * (p[0] / p[2]) + calib->K[0](0,2);
    T predicted_x2 = calib->K[1](0,0) * ((p[0]-calib->baseline) / p[2]) + calib->K[1](0,2);
    T predicted_y = calib->K[0](1,1) * (p[1] / p[2]) + calib->K[0](1,2);

    residuals[0] = sigma_inv*(predicted_x1 - features[0]);
    residuals[1] = sigma_inv*(predicted_y - features[1]);
    residuals[2] = sigma_inv*(predicted_x2 - features[2]);
    residuals[3] = sigma_inv*(predicted_y - features[3]);
    return true;
  }

  //!< builder. Needed for the object to converted to costFunstion, but also to be stored on the heap and avoid going out of scope
  static ceres::CostFunction* create(const std::array<double,4>& list_, const CalibrationParameters* const calib, const double sigma=1.0) {
    return (new ceres::AutoDiffCostFunction<StereoReprojectionError, 4, 6, 3>(new StereoReprojectionError(list_,calib,sigma)));
  }
  static ceres::CostFunction* create(const double obs1_x,const double obs1_y,const double obs2_x,const double obs2_y, const CalibrationParameters* const calib, const double sigma=1.0) {
    return (new ceres::AutoDiffCostFunction<StereoReprojectionError, 4, 6, 3>(new StereoReprojectionError(obs1_x,obs1_y,obs2_x,obs2_y,calib,sigma)));
  }
};

template<int M>
class BundleAdjuster{

public:

  //! enum to define the current state of the optimiser
  enum class Status{UNINITIALISED,INITIALISED,SUCCESSFUL,FAILED};

  /* shoter names for parameter types */
  using VecPts = std::vector<pt3D>;
  using VecCams = std::vector<cv::Matx61d>;
  using VecObs = std::vector<Observation<M>>;

  //! contructor, requires at least to set the parameters to optimise
  explicit BundleAdjuster(const CalibrationParameters& params, const VecObs& cams, const VecPts& pts, const VecObs& obs=VecObs()): calib_params{params}, m_status{Status::UNINITIALISED}, m_camera_params{cams}, m_point_params{pts}{
    if(!glog_init){
      google::InitGoogleLogging("CeresBA");
      glog_init=true;
    }
    initialiseObservations(obs); //needs to call initialiseObservation to change status to initialise
  }

  //! contructor, requires at least to set the parameters to optimise
  /*! for monocular windowed BA */
  template<typename T1,typename T2>
  BundleAdjuster(const CalibrationParameters& params, const std::vector<CamPose<me::Quat<T1>,T1>>& cams, const std::vector<WBA_Point<cv::Point_<T2>>>& obs=std::vector<WBA_Point<cv::Point_<T2>>>()): calib_params{params}, m_status{Status::UNINITIALISED}{
    if(!glog_init){
      google::InitGoogleLogging("CeresBA");
      glog_init=true;
    }

    initialiseParameters(cams);
    initialiseObservations(obs,cams[0].ID); //needs to call initialiseObservation to change status to initialise
  }

  //! contructor, requires at least to set the parameters to optimise
  /*! for stereo windowed BA */
  template<typename T1, typename T2>
  BundleAdjuster(const CalibrationParameters& params, const std::vector<CamPose<me::Quat<T1>,T1>>& cams, const std::vector<WBA_Point<std::pair<cv::Point_<T2>,cv::Point_<T2>>>>& obs=std::vector<WBA_Point<cv::Point_<T2>>>()): calib_params{params}, m_status{Status::UNINITIALISED}{
    if(!glog_init){
      google::InitGoogleLogging("CeresBA");
      glog_init=true;
    }

    initialiseParameters(cams);
    initialiseObservations(obs,cams[0].ID); //needs to call initialiseObservation to change status to initialise
  }

  //getters
  std::vector<pt3D> getPoints() const {return m_point_params;}
  std::vector<CamPose_qd> getCameraPoses() const {
    std::vector<CamPose_qd> cams;int idx=0;
    for(auto& cam_pose : m_camera_params)
      cams.push_back(CamPose_qd{idx++,Quatd{exp_map_Quat(cv::Vec3d{cam_pose(3),cam_pose(4),cam_pose(5)})},cv::Vec3d{cam_pose(0),cam_pose(1),cam_pose(2)}});
    return cams;
  }
  std::vector<cv::Mat> getPosesCovariance(){return m_camera_covs;}
  std::vector<cv::Mat> getPointsCovariance(){return m_point_covs;}
  int getNbPoints() const {return m_point_params.size();}
  int getNbCameras() const {return m_camera_params.size();}
  int getNbObservations() const {return m_observations.size();}
  Status getStatus(){return m_status;}

  //! intialise point and camera parameters with optimisation types
  void initialiseParameters(const VecCams& cams, const VecPts& pts);
  //! intialise point and camera parameters with types from feature_types.hpp
  template<typename T>
  void initialiseParameters(const std::vector<CamPose<me::Quat<T>,T>>& cams, const std::vector<pt3D>& pts=std::vector<pt3D>());
  //! initialise observations with optimisation types
  void initialiseObservations(const VecObs& observations);
  //! initialise observations with windowed BA points from feature_types.hpp
  /*! implemented for monocular BA with single 2D features */
  template<typename T>
  void initialiseObservations(const std::vector<WBA_Point<cv::Point_<T>>>& observations,const int first_frame);
  //! initialise observations with windowed BA stereo points from feature_types.hpp
  /*! implemented for stereo BA with stereo 2D features */
  template<typename T>
  void initialiseObservations(const std::vector<WBA_Point<std::pair<cv::Point_<T>,cv::Point_<T>>>>& observations,const int first_frame);

  //! runs the optimisation process when all parameters have been initialised
  Status optimise(int fixedFrames);

private:

    void extract_covariance(ceres::Problem* pb);

    static bool glog_init; //! make sure glog is initilialised only once

    CalibrationParameters calib_params; //!< intrinsic params and baseline if stereo
    Status m_status; //!< current state of the optimiser

    VecCams m_camera_params; //!< camera parameters to be optimised
    VecPts m_point_params; //!< point parameters to be optimised
    VecObs m_observations; //!< observations
    std::vector<cv::Mat> m_camera_covs;//! camera pose covariance matrices after optimisation (6x6)
    std::vector<cv::Mat> m_point_covs;//! point location covariance after optimisation (3x3)
};


/*****************************/
/* Functions implementations */
/*****************************/


template<int M>
void BundleAdjuster<M>::initialiseParameters(const VecCams& cams, const VecPts& pts){
  if(m_status != Status::UNINITIALISED){ // if state other than unitialised it means the object has already been used
    std::cerr << "[Bundle Adjuster] system should be uninitialised!" << std::endl;
    return;
  }

  m_camera_params = cams;
  m_point_params = pts;
}

template<int M>
template<typename T>
void BundleAdjuster<M>::initialiseParameters(const std::vector<CamPose<me::Quat<T>,T>>& cams, const std::vector<pt3D>& pts){
  if(m_status != Status::UNINITIALISED){ // if state other than unitialised it means the object has already been used
    std::cerr << "[Bundle Adjuster] system should be uninitialised!" << std::endl;
    return;
  }
  m_point_params = pts;
  m_camera_params.clear();
  for(auto& pose : cams){
    cv::Vec3d rot_vec = log_map_Quat(pose.orientation);
    m_camera_params.push_back(cv::Matx61d{pose.position(0),pose.position(1),pose.position(2),rot_vec(0),rot_vec(1),rot_vec(2)});
  }
}

template<int M>
void BundleAdjuster<M>::initialiseObservations(const VecObs& observations){
  if(m_status != Status::UNINITIALISED || m_camera_params.empty() || m_point_params.empty()){ // if state other than unitialised it means the object has already been used
    std::cerr << "[Bundle Adjuster] system should be uninitialised and both cameras and points not empty!" << std::endl;
    return;
  }

  assert(observations.size() == m_point_params.size() && "nb of pts and observations do not match");
  m_observations = observations;

  m_status = Status::INITIALISED;// if we arrive here, everything is initialised
}

template<>
template<typename T>
void BundleAdjuster<2>::initialiseObservations(const std::vector<WBA_Point<cv::Point_<T>>>& observations,const int first_frame){
	if(m_status != Status::UNINITIALISED || m_camera_params.empty()){ // if state other than unitialised it means the object has already been used
		std::cerr << "[Bundle Adjuster] system should be uninitialised and cameras not empty!" << std::endl;
		return;
	}

  bool init_points = m_point_params.empty();
  int pt_idx=0;
  if(!init_points)
    assert(observations.size() == m_point_params.size() && "nb of pts and observations do not match");
  m_observations.clear();
  for(auto obs : observations){
    if(init_points)
      m_point_params.push_back(to_euclidean(obs.get3DLocation()));
    for(uint i=0;i<obs.getNbFeatures();i++){
        int frame_idx = obs.getFrameIdx(i);
        if(frame_idx-first_frame>=0)
          m_observations.push_back(Observation<2>{{obs.getFeat(i).x,obs.getFeat(i).y},frame_idx-first_frame,pt_idx,obs.getCameraID()});
    }
    pt_idx++;
  }
  m_status = Status::INITIALISED;
}

template<>
template<typename T>
void BundleAdjuster<4>::initialiseObservations(const std::vector<WBA_Point<std::pair<cv::Point_<T>,cv::Point_<T>>>>& observations,const int first_frame){
  if(m_status != Status::UNINITIALISED || m_camera_params.empty()){ // if state other than unitialised it means the object has already been used
    std::cerr << "[Bundle Adjuster] system should be uninitialised and cameras not empty!" << std::endl;
    return;
  }

  bool init_points = m_point_params.empty();
  int pt_idx=0;
  if(!init_points)
    assert(observations.size() == m_point_params.size() && "nb of pts and observations do not match");
  m_observations.clear();
  for(auto obs : observations){
    if(init_points){
     m_point_params.push_back(to_euclidean(obs.get3DLocation()));
    }
    for(uint i=0;i<obs.getNbFeatures();i++){
      int frame_idx = obs.getFrameIdx(i);
        if(frame_idx-first_frame>=0)
          m_observations.push_back(Observation<4>{{obs.getFeat(i).first.x,obs.getFeat(i).first.y,obs.getFeat(i).second.x,obs.getFeat(i).second.y},frame_idx-first_frame,pt_idx,obs.getCameraID()});
    }
    pt_idx++;
  }
  m_status = Status::INITIALISED;
}

template<>
BundleAdjuster<2>::Status BundleAdjuster<2>::optimise(int fixedFrames){

  if(m_status != BundleAdjuster<2>::Status::INITIALISED){ // if different than initialise there is no point in running the optimisation
    std::cerr << "[Bundle Adjuster] system should be initiliased to perform optimisation!" << std::endl;
    return m_status;
  }

  std::cout << "[Bundle Adjuster] optimising (" << m_camera_params.size() << " cam poses and " << m_point_params.size() << " pts with " << m_observations.size() << " observations." << std::endl;
  ceres::Problem m_problem{}; //!< ceres problem which optimises points and camera parameters

  if(calib_params.baseline==0)
	calib_params.baseline = 0.5;
  double Zmax = calib_params.K[0](0,0)*calib_params.baseline/0.1;
  double Zmin = calib_params.K[0](0,0)*calib_params.baseline/(2*calib_params.K[0](0,2));

  for(auto& obs : m_observations){

    ceres::CostFunction* cstFunc;
    ceres::LossFunction* lossFunc = new ceres::HuberLoss(1.0);//new ceres::CauchyLoss(1.0);

    if(obs.camID == 0)
        cstFunc = StandardReprojectionError::create(obs.data[0],obs.data[1],&calib_params,sqrt(calib_params.feat_var)); //error is squared so should use feat. standard deviation
    else
        cstFunc = StereoRightError::create(obs.data[0],obs.data[1],&calib_params,sqrt(calib_params.feat_var));

    double* obs_cam_params = m_camera_params[obs.camIdx].val, *obs_pt_params = m_point_params[obs.ptIdx].val;
    m_problem.AddResidualBlock(cstFunc,lossFunc,obs_cam_params,obs_pt_params);
    if(obs.camIdx < fixedFrames)
        m_problem.SetParameterBlockConstant(obs_cam_params);
    m_problem.SetParameterUpperBound(obs_pt_params,0,Zmax/calib_params.K[0](0,0)*calib_params.K[0](0,2));
    m_problem.SetParameterUpperBound(obs_pt_params,1,Zmax/calib_params.K[0](1,1)*calib_params.K[0](1,2));
    m_problem.SetParameterUpperBound(obs_pt_params,2,Zmax);
    m_problem.SetParameterLowerBound(obs_pt_params,0,-Zmax/calib_params.K[0](0,0)*calib_params.K[0](0,2));
    m_problem.SetParameterLowerBound(obs_pt_params,1,-Zmax/calib_params.K[0](1,1)*calib_params.K[0](1,2));
    m_problem.SetParameterLowerBound(obs_pt_params,2,Zmin);
  }

  ceres::Solver::Options options;
  options.max_solver_time_in_seconds = 1.0;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.function_tolerance = 1e-3;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options,&m_problem,&summary);

  std::cout << summary.BriefReport() << std::endl;
  std::cout << "is usable? " << std::boolalpha << summary.IsSolutionUsable() << std::endl;

  if(calib_params.compute_cov)
    extract_covariance(&m_problem);

  m_status = (summary.IsSolutionUsable()?Status::SUCCESSFUL:Status::FAILED);
  return m_status;
}

template<>
BundleAdjuster<4>::Status BundleAdjuster<4>::optimise(int fixedFrames){

  if(m_status != Status::INITIALISED){ // if different than initialise there is no point in running the optimisation
    std::cerr << "[Bundle Adjuster] system should be initiliased to perform optimisation!" << std::endl;
    return m_status;
  }

  std::cout << "[Bundle Adjuster] optimising (" << m_camera_params.size() << " cam poses and " << m_point_params.size() << " pts with " << m_observations.size() << " observations." << std::endl;
  ceres::Problem m_problem{}; //!< ceres problem which optimises points and camera parameters

  double Zmax = calib_params.K[0](0,0)*calib_params.baseline/0.1;
  double Zmin = calib_params.K[0](0,0)*calib_params.baseline/(2*calib_params.K[0](0,2));

  for(auto& obs : m_observations){

    ceres::LossFunction* lossFunc = new ceres::HuberLoss(1.0);//new ceres::CauchyLoss(1.0);
    ceres::CostFunction* cstFunc = StereoReprojectionError::create(obs.data[0],obs.data[1],obs.data[2],obs.data[3],&calib_params,sqrt(calib_params.feat_var)); //error is squared so should use feat. standard deviation
    double* obs_cam_params = m_camera_params[obs.camIdx].val, *obs_pt_params = m_point_params[obs.ptIdx].val;

    m_problem.AddResidualBlock(cstFunc,lossFunc,obs_cam_params,obs_pt_params);
    if(obs.camIdx < fixedFrames){
        m_problem.SetParameterBlockConstant(obs_cam_params);
    }
    m_problem.SetParameterUpperBound(obs_pt_params,0,Zmax/calib_params.K[0](0,0)*calib_params.K[0](0,2));
    m_problem.SetParameterUpperBound(obs_pt_params,1,Zmax/calib_params.K[0](1,1)*calib_params.K[0](1,2));
    m_problem.SetParameterUpperBound(obs_pt_params,2,Zmax);
    m_problem.SetParameterLowerBound(obs_pt_params,0,-Zmax/calib_params.K[0](0,0)*calib_params.K[0](0,2));
    m_problem.SetParameterLowerBound(obs_pt_params,1,-Zmax/calib_params.K[0](1,1)*calib_params.K[0](1,2));
    m_problem.SetParameterLowerBound(obs_pt_params,2,Zmin);
  }

  ceres::Solver::Options options;
  options.max_solver_time_in_seconds = 1.0;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.function_tolerance = 1e-3;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options,&m_problem,&summary);

  if(calib_params.compute_cov)
    extract_covariance(&m_problem);

  m_status = (summary.IsSolutionUsable()?Status::SUCCESSFUL:Status::FAILED);
  return m_status;
}

template<int M>
void BundleAdjuster<M>::extract_covariance(ceres::Problem* pb){

    if(!pb){
        std::cerr << "[Bundle Adjuster] wrong ceres problem given for covariance estimation" << std::endl;
        return;
    }

    std::vector<std::pair<const double*,const double*>> cov_blocks;
    std::vector<double*> param_blocks;
    //retrieving param blocks
    pb->GetParameterBlocks(&param_blocks);
    if((int)param_blocks.size() != getNbCameras()+getNbPoints()){
        std::cerr << "[BundleAdjuster] Error retrieving covariance blocks size and number of parameters do not correspond:" << std::endl;
        std::cerr << "[BundleAdjuster] blocks size: " << param_blocks.size() << ", nb parameters: " << getNbCameras()+getNbPoints() << std::endl;
        return;
    }

    for(int i=0;i<getNbCameras();i++)
        cov_blocks.push_back(std::make_pair(m_camera_params[i].val,m_camera_params[i].val));
//	for(int i=0;i<getNbPoints();i++)
//        cov_blocks.push_back(std::make_pair(m_point_params[i].val,m_point_params[i].val));

    //creating Covariance structure
    ceres::Covariance::Options cov_opts;
    ceres::Covariance covariance(cov_opts);


    if(covariance.Compute(cov_blocks, pb)){
		m_camera_covs = std::vector<cv::Mat>(getNbCameras());
		double cov_pose[6*6];//double cov_point[3*3];
		for(int i=0;i<getNbCameras();i++)
			if(covariance.GetCovarianceBlock(m_camera_params[i].val,m_camera_params[i].val,cov_pose)){
				cv::Mat(6,6,CV_64F,cov_pose).copyTo(m_camera_covs[i]);
			}
			else{
				std::cerr << "cov failed for cam " << i << std::endl;
			}
//        for(int i=getNbCameras();i<getNbCameras()+getNbPoints();i++){
//            if(covariance.GetCovarianceBlock(m_point_params[i].val,m_point_params[i].val,cov_point))
//                m_point_covs.push_back(cv::Mat(3,3,CV_64F,cov_point));
//            else{
//                m_camera_covs.push_back(cv::Mat());
//                std::cerr << "cov failed for pt " << i-getNbCameras() << std::endl;
//            }
//        }
    }else{
        std::cerr << "[Bundle Adjuster] error computing the covariance matrix" << std::endl;
        return;
    }
}


template<int M>
bool BundleAdjuster<M>::glog_init = false;

}// namespace optimisation
}// namespace me

#endif // BUNDLEAJUSTER_H_INCLUDED
