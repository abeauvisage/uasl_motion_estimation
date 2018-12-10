#include "CeresBA.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace me;

Matx33d CeresBA::K_;
double CeresBA::baseline_;

vector<int> selectRandomIndices(int nb_samples, int nb_tot) {

    assert(nb_samples < nb_tot);
    vector<int> samples_idx;

    int i=0;
    while(i<nb_samples){
        int idx = rand()% nb_tot; //select random number between 0 and nb_tot
        bool exists = false;
        for(unsigned int j=0;j<samples_idx.size();j++) // ckeck if the index is alredy included
            if(idx == samples_idx[j])
                exists = true;
        if(!exists){
            i++;
            samples_idx.push_back(idx);
        }
    }

  return samples_idx;
}

CeresBA::CeresBA(int nb_view, int nb_pts, const cv::Matx33d& K, double baseline):num_cameras_(nb_view),num_points_(nb_pts),num_observations_(nb_view*nb_pts){
        num_parameters_ = 6 * num_cameras_ + 3 * num_points_;
        parameters_ = new double[num_parameters_];
        K_=K;
        baseline_=baseline;
}

void CeresBA::fillData(const std::vector<std::vector<Point2f>>& obs, const std::vector<ptH3D>& pts){
    assert((int) obs.size() == num_cameras_ && (int) pts.size() == num_points_);

    for(uint i=0;i<pts.size();i++){
        ptH3D pt = pts[i];
        normalize(pt);
        parameters_[num_cameras_*6+i*3+0] = pt(0);
        parameters_[num_cameras_*6+i*3+1] = pt(1);
        parameters_[num_cameras_*6+i*3+2] = pt(2);
    }

    for(uint j=0;j<obs.size();j++){
        for(uint k=0;k<6;k++){
            parameters_[j*6+k] = 0;
        }
    }
    for(int j=0;j<num_cameras_;j++){
        assert((int)obs[j].size() == num_points_);
        for(int i=0;i<num_points_;i++){
            camera_index_[j*num_points_+i] = j;
            point_index_[j*num_points_+i] = i;
            observations_[2*(j*num_points_+i)+0] = obs[j][i].x;
            observations_[2*(j*num_points_+i)+1] = obs[j][i].y;
        }
    }
}


void CeresBA::fillData(const std::vector<me::WBA_Ptf>& pts, const std::vector<me::Euld>& ori, const std::vector<Vec3d> pos, int start){

    assert(num_cameras_ == (int) ori.size() && num_points_ == (int) pts.size());

    for(int i=0;i<num_points_;i++){
        ptH3D pt = pts[i].get3DLocation();
        normalize(pt);
        parameters_[num_cameras_*6+i*3+0] = pt(0);
        parameters_[num_cameras_*6+i*3+1] = pt(1);
        parameters_[num_cameras_*6+i*3+2] = pt(2);
    }

    for(uint j=0;j<ori.size();j++){
        parameters_[j*6] = ori[j].roll();
        parameters_[j*6+1] = ori[j].pitch();
        parameters_[j*6+2] = ori[j].yaw();
        for(uint k=3;k<6;k++){
            parameters_[j*6+k] = pos[j](k-3);
        }
    }

    int nb =0;
    for(int i=0;i<num_points_;i++)
        nb += pts[i].getNbFeatures();

    num_observations_ = nb;

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    nb=0;
    for(int i=0;i<num_points_;i++){
        for(uint j=0;j<pts[i].getNbFeatures();j++){
            camera_index_[nb] = pts[i].getFrameIdx(j)-start;
            point_index_[nb] = i;
            observations_[2*nb+0] = pts[i].getFeat(j).x;
            observations_[2*nb+1] = pts[i].getFeat(j).y;
            nb++;
        }
    }
}

void CeresBA::fillData(const std::vector<me::WBA_Ptf*>& pts, const std::vector<me::CamPose_qd>& poses){

    assert(num_cameras_ == (int) poses.size() && num_points_ == (int) pts.size());
    int start = poses[0].ID;

    for(uint i=0;i<pts.size();i++){
        pt3D pt = to_euclidean(pts[i]->get3DLocation());
        parameters_[num_cameras_*6+i*3+0] = pt(0);
        parameters_[num_cameras_*6+i*3+1] = pt(1);
        parameters_[num_cameras_*6+i*3+2] = pt(2);
    }

    cam_idx = new int[poses.size()];
    for(uint j=0;j<poses.size();j++){
        cam_idx[j] = poses[j].ID;
        Vec3d rot_vec = log_map_Quat<double>(poses[j].orientation);
        parameters_[j*6] = rot_vec[0];
        parameters_[j*6+1] = rot_vec[1];
        parameters_[j*6+2] = rot_vec[2];
        Vec3d new_t = poses[j].position;
        for(uint k=3;k<6;k++){
            parameters_[j*6+k] = new_t[k-3];
        }
    }

    int nb =0;
    for(int i=0;i<num_points_;i++)
        nb += pts[i]->getNbFeatures();

    num_observations_ = nb;

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    nb=0;
    for(int i=0;i<num_points_;i++){
        for(uint j=0;j<pts[i]->getNbFeatures();j++){
            camera_index_[nb] = pts[i]->getFrameIdx(j)-start;
            point_index_[nb] = i;
            observations_[2*nb+0] = pts[i]->getFeat(j).x;
            observations_[2*nb+1] = pts[i]->getFeat(j).y;
            nb++;
        }
    }
}

void CeresBA::fillData(const std::vector<me::WBA_Ptf>& pts, const std::vector<me::CamPose_qd>& poses){

    assert(num_cameras_ == (int) poses.size() && num_points_ == (int) pts.size());
    int start = poses[0].ID;
    camera_nbs.clear();
    for(uint i=0;i<pts.size();i++){
        pt3D pt = to_euclidean(pts[i].get3DLocation());
        parameters_[num_cameras_*6+i*3+0] = pt(0);
        parameters_[num_cameras_*6+i*3+1] = pt(1);
        parameters_[num_cameras_*6+i*3+2] = pt(2);
        camera_nbs.push_back(pts[i].getCameraNum());
    }

    cam_idx = new int[poses.size()];
    for(uint j=0;j<poses.size();j++){
        cam_idx[j] = poses[j].ID;
        Vec3d rot_vec = log_map_Quat<double>(poses[j].orientation);
        parameters_[j*6] = rot_vec[0];
        parameters_[j*6+1] = rot_vec[1];
        parameters_[j*6+2] = rot_vec[2];
        Vec3d new_t = poses[j].position;
        for(uint k=3;k<6;k++){
            parameters_[j*6+k] = new_t[k-3];
        }
    }

    int nb =0;
    for(int i=0;i<num_points_;i++)
        nb += pts[i].getNbFeatures();

    num_observations_ = nb;

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    nb=0;
    for(int i=0;i<num_points_;i++){
        for(uint j=0;j<pts[i].getNbFeatures();j++){
            camera_index_[nb] = pts[i].getFrameIdx(j)-start;
            point_index_[nb] = i;
            observations_[2*nb+0] = pts[i].getFeat(j).x;
            observations_[2*nb+1] = pts[i].getFeat(j).y;
            nb++;
        }
    }
}

void CeresBA::fillStereoData(const std::vector<me::WBA_stereo_Ptf>& pts, const std::vector<me::CamPose_qd>& poses){

    assert(num_cameras_ == (int) poses.size() && num_points_ == (int) pts.size());
    int start = poses[0].ID;
    camera_nbs.clear();
    for(uint i=0;i<pts.size();i++){
        pt3D pt = to_euclidean(pts[i].get3DLocation());
        parameters_[num_cameras_*6+i*3+0] = pt(0);
        parameters_[num_cameras_*6+i*3+1] = pt(1);
        parameters_[num_cameras_*6+i*3+2] = pt(2);
        camera_nbs.push_back(pts[i].getCameraNum());
    }

    cam_idx = new int[poses.size()];
    for(uint j=0;j<poses.size();j++){
        cam_idx[j] = poses[j].ID;
        Vec3d rot_vec = log_map_Quat<double>(poses[j].orientation);
        parameters_[j*6] = rot_vec[0];
        parameters_[j*6+1] = rot_vec[1];
        parameters_[j*6+2] = rot_vec[2];
        Vec3d new_t = poses[j].position;
        for(uint k=3;k<6;k++){
            parameters_[j*6+k] = new_t[k-3];
        }
    }

    int nb =0;
    for(int i=0;i<num_points_;i++)
        nb += pts[i].getNbFeatures();

    num_observations_ = nb;

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[4 * num_observations_];

    nb=0;
    for(int i=0;i<num_points_;i++){
        for(uint j=0;j<pts[i].getNbFeatures();j++){
            camera_index_[nb] = pts[i].getFrameIdx(j)-start;
            point_index_[nb] = i;
            observations_[4*nb+0] = pts[i].getFeat(j).first.x;
            observations_[4*nb+1] = pts[i].getFeat(j).first.y;
            observations_[4*nb+2] = pts[i].getFeat(j).second.x;
            observations_[4*nb+3] = pts[i].getFeat(j).second.y;
            nb++;
        }
    }
}

void CeresBA::fillStereoData(const std::pair<std::vector<me::WBA_Ptf>,std::vector<me::WBA_Ptf>>& pts, const std::vector<me::CamPose_qd>& poses){

    assert(num_cameras_ == (int) poses.size() && num_points_ == (int)(pts.first.size()+pts.second.size()));
    int start = poses[0].ID;
	camera_nbs.clear();

    for(uint i=0;i<pts.first.size();i++){
        pt3D pt = to_euclidean(pts.first[i].get3DLocation());
        parameters_[num_cameras_*6+i*3+0] = pt(0);
        parameters_[num_cameras_*6+i*3+1] = pt(1);
        parameters_[num_cameras_*6+i*3+2] = pt(2);
        camera_nbs.push_back(pts.first[i].getCameraNum());
    }

    for(uint i=0;i<pts.second.size();i++){
        pt3D pt = to_euclidean(pts.second[i].get3DLocation());
        parameters_[num_cameras_*6+(pts.first.size()+i)*3+0] = pt(0);
        parameters_[num_cameras_*6+(pts.first.size()+i)*3+1] = pt(1);
        parameters_[num_cameras_*6+(pts.first.size()+i)*3+2] = pt(2);
        camera_nbs.push_back(pts.second[i].getCameraNum());
    }

    cam_idx = new int[poses.size()];
    for(uint j=0;j<poses.size();j++){
        cam_idx[j] = poses[j].ID;
        Vec3d rot_vec = log_map_Quat<double>(poses[j].orientation);
        parameters_[j*6] = rot_vec[0];
        parameters_[j*6+1] = rot_vec[1];
        parameters_[j*6+2] = rot_vec[2];
        Vec3d new_t = poses[j].position;
        for(uint k=3;k<6;k++){
            parameters_[j*6+k] = new_t[k-3];
        }
    }

    int nb =0;
    for(uint i=0;i<pts.first.size();i++)
        nb += pts.first[i].getNbFeatures();
    for(uint i=0;i<pts.second.size();i++)
        nb += pts.second[i].getNbFeatures();

    num_observations_ = nb;

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    nb=0;
    for(uint i=0;i<pts.first.size();i++){
        for(uint j=0;j<pts.first[i].getNbFeatures();j++){
            camera_index_[nb] = pts.first[i].getFrameIdx(j)-start;
            point_index_[nb] = i;
            observations_[2*nb+0] = pts.first[i].getFeat(j).x;
            observations_[2*nb+1] = pts.first[i].getFeat(j).y;
            nb++;
        }
    }
    for(uint i=0;i<pts.second.size();i++){
        for(uint j=0;j<pts.second[i].getNbFeatures();j++){
            camera_index_[nb] = pts.second[i].getFrameIdx(j)-start;
            point_index_[nb] = pts.first.size() + i;
            observations_[2*nb+0] = pts.second[i].getFeat(j).x;
            observations_[2*nb+1] = pts.second[i].getFeat(j).y;
            nb++;
        }
    }
}


void CeresBA::fillPoints(const std::vector<me::ptH3D>& pts, const std::vector<uchar>& mask){

    assert((int) pts.size() == num_points_);
    assert(mask.empty() || (int) mask.size() == num_points_);

    m_mask = mask;

    for(uint i=0;i<pts.size();i++){
        pt3D pt = to_euclidean(pts[i]);
        parameters_[num_cameras_*6+i*3+0] = pt(0);
        parameters_[num_cameras_*6+i*3+1] = pt(1);
        parameters_[num_cameras_*6+i*3+2] = pt(2);
    }
}

void CeresBA::fillCameras(const std::vector<me::CamPose_md>& poses){

    assert((int) poses.size() == num_cameras_);
    cam_idx = new int[poses.size()];
    for(uint j=0;j<poses.size();j++){
        cam_idx[j] = poses[j].ID;
        //orientation
        Vec3d rot_vec = log_map_Mat<double>(poses[j].orientation);
        parameters_[j*6] = rot_vec[0];
        parameters_[j*6+1] = rot_vec[1];
        parameters_[j*6+2] = rot_vec[2];
        // position
        Vec3d new_t = poses[j].position;
        for(uint k=3;k<6;k++){
            parameters_[j*6+k] = new_t[k-3];
        }
    }
}

void CeresBA::fillCameras(const std::vector<me::CamPose_qd>& poses){

    assert((int)poses.size() == num_cameras_);
    cam_idx = new int[poses.size()];
    for(uint j=0;j<poses.size();j++){
        cam_idx[j] = poses[j].ID;
        //orientation
        Vec3d rot_vec = log_map_Quat<double>(poses[j].orientation);
        parameters_[j*6] = rot_vec[0];
        parameters_[j*6+1] = rot_vec[1];
        parameters_[j*6+2] = rot_vec[2];
        // position
        Vec3d new_t = poses[j].position;
        for(uint k=3;k<6;k++){
            parameters_[j*6+k] = new_t[k-3];
        }
    }
}

void CeresBA::fillObservations(const std::vector<me::WBA_Ptf>& obs){}

void CeresBA::fillObservations(const std::vector<me::StereoOdoMatchesf>& obs){

    assert((int) obs.size() == num_points_);
    num_observations_ = num_points_*2;

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[4 * num_observations_];

    for(uint i=0;i<obs.size();i++){
        for(uint j=0;j<2;j++){
            int nb = 2*i+j;
            camera_index_[nb] = j;
            point_index_[nb] = i;
            if(j==0){
                observations_[4*nb+0] = obs[i].f1.x;
                observations_[4*nb+1] = obs[i].f1.y;
                observations_[4*nb+2] = obs[i].f2.x;
                observations_[4*nb+3] = obs[i].f2.y;
            }else{
                observations_[4*nb+0] = obs[i].f3.x;
                observations_[4*nb+1] = obs[i].f3.y;
                observations_[4*nb+2] = obs[i].f4.x;
                observations_[4*nb+3] = obs[i].f4.y;
            }
        }
    }
}

bool CeresBA::runRansac(int nb_iterations,double inlier_threshold,int fixedFrames, bool stereo){

    vector<int> inliers_idx;
    std::vector<double> init_parameters(parameters_,parameters_+num_parameters_);
    std::vector<me::CamPose_qd> best_poses;

    for (int i=0;i<nb_iterations;i++) {
        vector<int> selection = selectRandomIndices(5,num_points_-1);
        vector<uchar> mask(num_points_,0);
            for(uint j=0;j<selection.size();j++)
                mask[selection[j]] = 1;

        m_mask = mask;
        std::copy(init_parameters.begin(),init_parameters.end(),parameters_);

        vector<int> inliers_tmp;
        if(stereo){
            runStereoSolver(fixedFrames);
            inliers_tmp = get_inliers_stereo(inlier_threshold);
        }
        else{
            runSolver(fixedFrames);
            inliers_tmp = get_inliers(inlier_threshold);
        }

        if(inliers_tmp.size() > inliers_idx.size()){
            inliers_idx = inliers_tmp;
            best_poses = getQuatPoses();
        }
    }

    cout << "final optim (" << inliers_idx.size() << " inliers) " << inlier_threshold << endl;

    if (inliers_idx.size()>=6){ // check that more than 6 inliers have been obtained

        vector<uchar> mask(num_points_,0);
        for(uint j=0;j<inliers_idx.size();j++)
                mask[inliers_idx[j]] = 1;

        m_mask = mask;
        std::copy(init_parameters.begin(),init_parameters.end(),parameters_);

        if(stereo)
            runStereoSolver(fixedFrames);
        else
            runSolver(fixedFrames);

        best_poses = getQuatPoses();
        return true;
    }else{
        std::copy(init_parameters.begin(),init_parameters.end(),parameters_);
        return false;
    }
}

void CeresBA::runSolver(int fixedFrames){

    problem = new ceres::Problem();

    const double* obs = observations();
    cout << "[BA] " << num_cameras_ << " cams | " << num_points_ << " points | " << num_observations() << " observations" << endl;

	assert((int) camera_nbs.size() == num_points_);

    for(int i=0;i<num_observations();++i){

       ceres::CostFunction* cstFunc;
       if(camera_nbs.empty() || camera_nbs[point_index_[i]] == 0 )
            cstFunc = CeresBA::ReprojectionError::Create(obs[2*i+0],obs[2*i+1]);
        else
            cstFunc = CeresBA::ReprojectionErrorMonoRight::Create(obs[2*i+0],obs[2*i+1]);

        ceres::LossFunction* lossFunc = new ceres::CauchyLoss(1.0);
        problem->AddResidualBlock(cstFunc,lossFunc,mutable_camera_for_observation(i),mutable_point_for_observation(i));
        if(camera_index_[i] < fixedFrames)
            problem->SetParameterBlockConstant(mutable_camera_for_observation(i));
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.function_tolerance = 1e-6;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options,problem,&summary);
}

void CeresBA::runStereoSolver(int fixedFrames){

    problem = new ceres::Problem();

    const double* obs = observations();

    double initial_cost=0,final_cost=0;

    for(int i=0;i<num_observations();++i){

        if(!m_mask.empty() && !m_mask[point_index_[i]])
                continue;

        ceres::CostFunction* cstFunc = CeresBA::StereoReprojectionError::Create(obs[4*i+0],obs[4*i+1],obs[4*i+2],obs[4*i+3]);
    //        ceres::LossFunction* lossFunc = new ceres::CauchyLoss(1.0);
        problem->AddResidualBlock(cstFunc,nullptr,mutable_camera_for_observation(i),mutable_point_for_observation(i));
        problem->SetParameterBlockConstant(mutable_point_for_observation(i));
        if(camera_index_[i] < fixedFrames)
            problem->SetParameterBlockConstant(mutable_camera_for_observation(i));

        double residual[4];
        StereoReprojectionError rep(obs[4*i+0],obs[4*i+1],obs[4*i+2],obs[4*i+3]);
        rep(mutable_camera_for_observation(i),mutable_point_for_observation(i),residual);
        initial_cost += sqrt(residual[0]*residual[0]+residual[1]*residual[1])+sqrt(residual[2]*residual[2]+residual[3]*residual[3]);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.function_tolerance = 1e-6;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options,problem,&summary);

    for(int i=0;i<num_observations();++i){
        if(!m_mask.empty() && !m_mask[point_index_[i]])
                continue;
        double residual[4];
        StereoReprojectionError rep(obs[4*i+0],obs[4*i+1],obs[4*i+2],obs[4*i+3]);
        rep(mutable_camera_for_observation(i),mutable_point_for_observation(i),residual);
        final_cost += sqrt(residual[0]*residual[0]+residual[1]*residual[1])+sqrt(residual[2]*residual[2]+residual[3]*residual[3]);
    }
}

std::vector<int> CeresBA::get_inliers_stereo(const double threshold){
    std::vector<int> inliers;
    for(int i=0;i<num_observations()-1;i+=4){
        double residual[4], error=0;
        StereoReprojectionError rep1(observations_[4*i+0],observations_[4*i+1],observations_[4*i+2],observations_[4*i+3]);
        StereoReprojectionError rep2(observations_[4*i+4],observations_[4*i+5],observations_[4*i+6],observations_[4*i+7]);
        rep1(mutable_camera_for_observation(i),mutable_point_for_observation(i),residual);
        error += sqrt(residual[0]*residual[0]+residual[1]*residual[1])+sqrt(residual[2]*residual[2]+residual[3]*residual[3]);
        rep2(mutable_camera_for_observation(i+1),mutable_point_for_observation(i+1),residual);
        error += sqrt(residual[0]*residual[0]+residual[1]*residual[1])+sqrt(residual[2]*residual[2]+residual[3]*residual[3]);
        if(error/4.0 < threshold)
            inliers.push_back(point_index_[i]);
    }
    return inliers;
}


std::vector<int> CeresBA::get_inliers(const double threshold){
    std::vector<int> inliers;
    vector<Point2f> reproj_feats = reproject_features(1);
    for(int i=0;i<num_observations()-1;i+=2){
        double residual[2], error=0;
        ReprojectionError rep1(observations_[2*i+0],observations_[2*i+1]);
        ReprojectionError rep2(observations_[2*i+2],observations_[2*i+3]);
        rep1(mutable_camera_for_observation(i),mutable_point_for_observation(i),residual);
        error += sqrt(residual[0]*residual[0]+residual[1]*residual[1]);
        rep2(mutable_camera_for_observation(i+1),mutable_point_for_observation(i+1),residual);
        error += sqrt(residual[0]*residual[0]+residual[1]*residual[1]);
        if(error/2.0 < threshold)
            inliers.push_back(point_index_[i]);
    }
    return inliers;
}

std::vector<me::pt3D> CeresBA::get3DPoints(){
    vector<me::pt3D> pts;
    double * pts_ptr = mutable_points();
    for(int i=0;i<num_points_;i++){
        pt3D pt = Mat(3,1,CV_64F,pts_ptr);
        pts.push_back(pt);
        pts_ptr +=3;
    }
    return pts;
}

bool CeresBA::getCovariance(std::vector<cv::Matx66d>& poseCov, std::vector<cv::Matx33d>& pointCov){

    if(!problem)
        return false;

    poseCov.clear();pointCov.clear();
    std::vector<std::pair<const double*,const double*>> cov_blocks;
    std::vector<double*> param_blocks;
    //retrieving param blocks
    problem->GetParameterBlocks(&param_blocks);
    assert((int)param_blocks.size() == num_cameras_+num_points_);

    for(uint i=0;i<param_blocks.size();i++){
        cov_blocks.push_back(make_pair(param_blocks[i],param_blocks[i]));
    }
    //creating Covariance structure
    ceres::Covariance::Options cov_opts;
    ceres::Covariance covariance(cov_opts);

    //computing covariance
    if(!covariance.Compute(cov_blocks, problem))
        return false;
    else{
        double cov_pose[6*6];double cov_point[3*3];
        for(int i=0;i<num_cameras_;i++){
            covariance.GetCovarianceBlock(param_blocks[i],param_blocks[i],cov_pose);
            poseCov.push_back(Mat(6,6,CV_64F,cov_pose));
        }
        for(int i=num_cameras_;i<num_cameras_+num_points_;i++){
            covariance.GetCovarianceBlock(param_blocks[i],param_blocks[i],cov_point);
            pointCov.push_back(Mat(3,3,CV_64F,cov_point));
        }
    }
    return true;
}

std::vector<me::CamPose_qd> CeresBA::getQuatPoses(){

    vector<me::CamPose_qd> poses;
    double * cam_ptr = mutable_cameras();
    for(int i=0;i<num_cameras_;i++){
        Mat rot_vec(1,3,CV_64F,cam_ptr),pos_vec(3,1,CV_64F,cam_ptr+3);
        Quatd orientation = exp_map_Quat<double>(rot_vec);
        poses.push_back(CamPose_qd(cam_idx[i],orientation,(Vec3d)pos_vec));
        cam_ptr +=6;
    }
    return poses;
}

std::vector<me::CamPose_md> CeresBA::getMatPoses(){

    vector<me::CamPose_md> poses;
    double * cam_ptr = mutable_cameras();
    for(int i=0;i<num_cameras_;i++){
        Mat rot_vec(1,3,CV_64F,cam_ptr),pos_vec(3,1,CV_64F,cam_ptr+3);
        Matx33d orientation = exp_map_Mat<double>(rot_vec);
        poses.push_back(CamPose_md(cam_idx[i],orientation,(Vec3d)pos_vec));
        cam_ptr +=6;
    }
    return poses;
}
