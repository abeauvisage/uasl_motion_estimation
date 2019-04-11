#include "optimisation.h"
#include "mutualInformation.h"
#include "utils.h"

#include <Eigen/Cholesky>

#include <fstream>
#include <iomanip>


using namespace Eigen;
using namespace cv;
using namespace std;

ofstream log_mi;
ofstream log_scale;

namespace me{

template<class S, class T>
StopCondition Optimiser<S,T>::optimise(S& state, const bool test, const Eigen::VectorXi& mask){

    m_state = state;
    m_mask = mask;
    m_stop = StopCondition::NO_STOP;
    if(test){
        m_params.type = OptimType::GN;
        m_params.MAX_NB_ITER = 300;
        m_params.abs_tol = 1e-30;
        m_params.incr_tol = 1e-30;
        m_params.grad_tol = 1e-30;
        m_params.rel_tol = 1e-30;
       log_mi.open("log/log_mi_test.csv", ofstream::out | ofstream::app);
       cout << boolalpha << "log mi" << log_mi.is_open() << endl;
       log_scale.open("log/log_scale_test.csv", ofstream::out | ofstream::app);
       cout << boolalpha << "log scale " << log_mi.is_open() << endl;
    }

    int k=0;
    do{
        //computing residuals and mean reprojection errors
        MatrixXd residuals = compute_residuals(m_state);
        double e1 = (residuals * residuals.transpose()).diagonal().sum();
        double meanReprojError = e1 / (double)(residuals.rows()*residuals.cols());

        if(test){
            ScaleState* scalestate = dynamic_cast<ScaleState*>(&m_state);
            if(scalestate){
                cout << "s " << scalestate->scale << " e " << e1 << endl;
            }
        }

        // absolute tolerance condition
        if(meanReprojError < m_params.abs_tol)
            m_stop = StopCondition::SMALL_REPROJ_ERROR;

        //normal equations
        MatrixXd JJ(1,1);
        VectorXd e(1);
        if(test){
            JJ(0,0) = 100;
            e(0) = 1;
        }else{
            compute_normal_equations(residuals,JJ,e);
        }

        if(k == 0){
            m_params.mu = JJ.diagonal().maxCoeff();
//            m_params.alpha = (m_state.scale/2.0) * JJ(0)/e(0);
        }
        //gradient tolerance
        if(e.norm() < m_params.grad_tol)
            m_stop = StopCondition::SMALL_GRADIENT;

        //computing update
        VectorXd dX;
        switch(m_params.type){
            case OptimType::GN :
                run_GN_step(JJ,e,dX);break;
            case OptimType::LM :
                run_LM_step(JJ,e,dX,e1);break;
            default:
                std::cerr << "[Optim] optimisation type not recognized." << std::endl;
                break;
        }

        if(!m_stop && dX.norm() <= m_params.incr_tol )
            m_stop = StopCondition::SMALL_INCREMENT;

        //new residual after update
        MatrixXd tmp_residuals = compute_residuals(m_state);
        double e2 = (tmp_residuals * tmp_residuals.transpose()).diagonal().sum();

        if(test){
            ScaleState* scalestate = dynamic_cast<ScaleState*>(&m_state);
            if(scalestate){
                log_mi << e1 << ",";
                log_scale << scalestate->scale << ",";
            }
        }

        //relative tolerance
        if(m_params.type==OptimType::GN && pow(e2-e1,2) < m_params.rel_tol)
            m_stop = StopCondition::SMALL_DECREASE_FUNCTION;

    }while(!m_stop && k++ < m_params.MAX_NB_ITER);

    if(k == m_params.MAX_NB_ITER)
        m_stop = StopCondition::MAX_ITERATIONS;

    switch(m_stop){
        case SMALL_GRADIENT:
            std::cout << "stop: SMALL_GRADIENT" << std::endl;break;
        case SMALL_INCREMENT:
            std::cout << "stop: SMALL_INCREMENT" << std::endl;break;
        case MAX_ITERATIONS:
            std::cout << "stop: MAX_ITERATIONS" << std::endl;break;
        case SMALL_DECREASE_FUNCTION:
            std::cout << "stop: SMALL_DECREASE_FUNCTION" << std::endl;break;
        case SMALL_REPROJ_ERROR:
            std::cout << "stop: SMALL_REPROJ_ERROR" << std::endl;break;
        case NO_CONVERGENCE:
            std::cout << "stop: NO_CONVERGENCE" << std::endl;break;
        default:
            std::cout << "stop: DID NOT STOP" << std::endl;break;
    }

    if(test){
        log_mi << endl;
        log_scale << endl;
        log_mi.close();
        log_scale.close();
    }

    state = m_state;
    return m_stop;
}

template<>
MatrixXd Optimiser<ScaleState, std::vector<std::pair<cv::Mat,cv::Mat>> >::compute_residuals(const ScaleState& state){

    assert(m_mask.size() == 0 || m_mask.rows() == (int) (state.pts.first.size()+state.pts.second.size()));
    Rect bb(state.window_size,state.window_size,m_obs[0].first.cols-2*state.window_size,m_obs[1].first.rows-2*state.window_size);

    int tot_nb_elements;
    if(m_mask.size()==0)
        tot_nb_elements = state.pts.first.size()+state.pts.second.size();
    else
        tot_nb_elements = m_mask.sum();
    MatrixXd res = MatrixXd::Zero(tot_nb_elements,1);

    //features are reprojected in the last frame only
    uint lframe = state.poses.first[0].ID + state.poses.first.size()-1;

    int k=0;
    //for all features extracted from the left camera
    for(uint i=0;i<state.pts.first.size();i++){
        if(!m_mask.size() == 0 && !m_mask(i))
            continue;
        if(state.pts.first[i].isTriangulated()){
            ptH3D pt = state.pts.first[i].get3DLocation();
            if(state.pts.first[i].getLastFrameIdx() == lframe){ // if has been observed in the last keyframe
                int f_idx = state.poses.first.size()-1;
                Mat Tr = (Mat) state.poses.first[f_idx].orientation.getR4();
                ((Mat)state.poses.first[f_idx].position).copyTo(Tr(Range(0,3),Range(3,4)));
                Matx44d Tr_ = Tr;
                ptH2D feat = state.K.first * Matx34d::eye() * state.scale *(Tr_ * pt);
                Point2f feat_left(to_euclidean(feat)(0),to_euclidean(feat)(1)); // reprojection in the left image
                ptH2D feat_ =  (state.K.second * Matx34d::eye()) * (state.scale * (Tr_ * pt) - Matx41d(state.baseline,0,0,0));
                Point2f feat_right(to_euclidean(feat_)(0),to_euclidean(feat_)(1)); // reprojection in the right image
                if(bb.contains(feat_left) && bb.contains(feat_right)){
                    Mat ROI_left = m_obs[f_idx].first(Rect(feat_left.x-state.window_size,feat_left.y-state.window_size,state.window_size*2,state.window_size*2));
                    Mat ROI_right = m_obs[f_idx].second(Rect(feat_right.x-state.window_size,feat_right.y-state.window_size,state.window_size*2,state.window_size*2));
                    ROI_left.convertTo(ROI_left,CV_32F);
                    ROI_right.convertTo(ROI_right,CV_32F);

                    res(k,0) = computeMutualInformation(ROI_left,ROI_right);
                }
            }k++;
        }
    }

    // for all features extracted from the right image
    for(uint i=0;i<state.pts.second.size();i++){
        if(!m_mask.size() == 0 && !m_mask(state.pts.second.size()+i))
            continue;
        if(state.pts.second[i].isTriangulated()){
            ptH3D pt = state.pts.second[i].get3DLocation();
            if(state.pts.second[i].getLastFrameIdx() == lframe){ // if has been observed in the last keyframe
                int f_idx = state.poses.second.size()-1;
                Mat Tr = (Mat) state.poses.second[f_idx].orientation.getR4();
                ((Mat)state.poses.second[f_idx].position).copyTo(Tr(Range(0,3),Range(3,4)));
                Matx44d Tr_ = Tr;
                ptH2D feat = state.K.second * Matx34d::eye() * state.scale *(Tr_ * pt);
                Point2f feat_right(to_euclidean(feat)(0),to_euclidean(feat)(1));
                ptH2D feat_ =  (state.K.second * Matx34d::eye()) * (state.scale * (Tr_ * pt) + Matx41d(state.baseline,0,0,0));
                Point2f feat_left(to_euclidean(feat_)(0),to_euclidean(feat_)(1));
                if(bb.contains(feat_right) && bb.contains(feat_left)){
                    Mat ROI_right = m_obs[f_idx].second(Rect(feat_right.x-state.window_size,feat_right.y-state.window_size,state.window_size*2,state.window_size*2))*255;
                    Mat ROI_left = m_obs[f_idx].first(Rect(feat_left.x-state.window_size,feat_left.y-state.window_size,state.window_size*2,state.window_size*2))*255;
                    ROI_right.convertTo(ROI_right,CV_32F);
                    ROI_left.convertTo(ROI_left,CV_32F);
                    res(k,0) = computeMutualInformation(ROI_right,ROI_left);
                }
            }k++;
        }
    }

    return res;
}

double ScaleState::compute_residuals(std::vector<std::pair<cv::Mat,cv::Mat>>& m_obs){

    Rect bb(window_size,window_size,m_obs[0].first.cols-2*window_size,m_obs[1].first.rows-2*window_size);

    std::vector<std::pair<cv::Mat,cv::Mat>> imgs;

    int tot_nb_elements;
    tot_nb_elements = pts.first.size()+pts.second.size();
    MatrixXd res = MatrixXd::Zero(tot_nb_elements,1);

    //features are reprojected in the last frame only
    uint lframe = poses.first[0].ID + poses.first.size()-1;

    int k=0;
    //for all features extracted from the left camera
    for(uint i=0;i<pts.first.size();i++){
        if(pts.first[i].isTriangulated()){
            ptH3D pt = pts.first[i].get3DLocation();
            if(pts.first[i].getLastFrameIdx() == lframe){ // if has been observed in the last keyframe
                int f_idx = poses.first.size()-1;
                Mat Tr = (Mat) poses.first[f_idx].orientation.getR4();
                ((Mat)poses.first[f_idx].position).copyTo(Tr(Range(0,3),Range(3,4)));
                Matx44d Tr_ = Tr;
                ptH2D feat = K.first * Matx34d::eye() * scale *(Tr_ * pt);
                Point2f feat_left(to_euclidean(feat)(0),to_euclidean(feat)(1)); // reprojection in the left image
                ptH2D feat_ =  (K.second * Matx34d::eye()) * (scale * (Tr_ * pt) - Matx41d(baseline,0,0,0));
                Point2f feat_right(to_euclidean(feat_)(0),to_euclidean(feat_)(1)); // reprojection in the right image
                if(bb.contains(feat_left) && bb.contains(feat_right)){
                    Mat ROI_left = m_obs[f_idx].first(Rect(feat_left.x-window_size,feat_left.y-window_size,window_size*2,window_size*2));
                    Mat ROI_right = m_obs[f_idx].second(Rect(feat_right.x-window_size,feat_right.y-window_size,window_size*2,window_size*2));
                    ROI_left.convertTo(ROI_left,CV_32F);
                    ROI_right.convertTo(ROI_right,CV_32F);

                    imgs.push_back(make_pair(ROI_left,ROI_right));
                }
            }k++;
        }
    }

    cv::Mat left_img(imgs.size()*window_size,window_size,CV_32F);
    cv::Mat right_img(imgs.size()*window_size,window_size,CV_32F);

    for(uint i=0;i<imgs.size();i++){
        left_img(Range(i*window_size,(i+1)*window_size),Range(0,window_size)) = imgs[i].first;
        right_img(Range(i*window_size,(i+1)*window_size),Range(0,window_size)) = imgs[i].second;
    }

    return computeMutualInformation(left_img,right_img);
}

template<>
MatrixXd Optimiser<ScaleState, std::vector<std::vector<cv::Mat>> >::compute_residuals(const ScaleState& state){

    Rect bb(state.window_size,state.window_size,m_obs[0][0].cols-2*state.window_size,m_obs[1][0].rows-2*state.window_size);

    int nb_pix = pow(state.window_size*2,2);
    MatrixXd res = MatrixXd::Zero(state.pts.first.size()+state.pts.second.size(),state.poses.first.size()*nb_pix*2);

//    #define SHOW

    #ifdef SHOW
    cv::RNG rng;
    std::vector<std::pair<cv::Mat,cv::Mat>> images_(m_obs.size(),std::pair<cv::Mat,cv::Mat>());
        for(uint j=0;j<m_obs.size();j++){
            cvtColor(m_obs[j][0],images_[j].first,CV_GRAY2BGR);
            cvtColor(m_obs[j][1],images_[j].second,CV_GRAY2BGR);
        }
    #endif

    std::vector<double> errors;

    for(uint i=0;i<state.pts.first.size();i++){
        if(state.pts.first[i].isTriangulated()){
            #ifdef SHOW
//            int icolor = (unsigned) rng;
//            Scalar color((icolor&255), ((icolor>>8)&255),((icolor>>16)&255));
            Scalar color(255,0,0);
            #endif
            ptH3D pt = state.pts.first[i].get3DLocation();
            for(uint j=0;j<state.pts.first[i].getNbFeatures();j++){
                int f_idx = state.pts.first[i].getFrameIdx(j);
                Mat Tr = (Mat) state.poses.first[f_idx].orientation.getR4();
                ((Mat)state.poses.first[f_idx].position).copyTo(Tr(Range(0,3),Range(3,4)));
                Matx44d Tr_ = Tr;
                ptH2D feat = state.K.first * Matx34d::eye() * state.scale *(Tr_ * pt);
                Point2f feat_ij(to_euclidean(feat)(0),to_euclidean(feat)(1));
                ptH2D feat_ =  (state.K.second * Matx34d::eye()) * (state.scale * (Tr_ * pt) - Matx41d(state.baseline,0,0,0));
                Point2f feat_ij_(to_euclidean(feat_)(0),to_euclidean(feat_)(1));
                if(bb.contains(feat_ij_) && bb.contains(feat_ij)){
                    Mat ROI_ij = m_obs[f_idx][0](Rect(feat_ij.x-state.window_size,feat_ij.y-state.window_size,state.window_size*2,state.window_size*2));
                    Mat ROI_ij_ = m_obs[f_idx][1](Rect(feat_ij_.x-state.window_size,feat_ij_.y-state.window_size,state.window_size*2,state.window_size*2));
                    Mat ROI_ij_2 = m_obs[f_idx][2](Rect(feat_ij.x-state.window_size,feat_ij.y-state.window_size,state.window_size*2,state.window_size*2));
                    Mat ROI_ij_3 = m_obs[f_idx][3](Rect(feat_ij_.x-state.window_size,feat_ij_.y-state.window_size,state.window_size*2,state.window_size*2));
                    ROI_ij.convertTo(ROI_ij,CV_32F);
                    ROI_ij_.convertTo(ROI_ij_,CV_32F);
                    Mat diff = ROI_ij_-ROI_ij;
                    Mat diff2 = ROI_ij_3-ROI_ij_2;
                    float* diff_ptr = diff.ptr<float>();
                    for(int k=0;k<nb_pix;k++){
                        res(i,j*nb_pix+k) = (diff_ptr[k]);
                    }
                    float* diff2_ptr = diff2.ptr<float>();
                    for(int k=0;k<nb_pix;k++){
                        res(i,j*nb_pix+2*nb_pix+k) = (diff2_ptr[k]);
                    }
                    #ifdef SHOW
                    circle(images_[f_idx].first,feat_ij,2,color);
                    circle(images_[f_idx].second,feat_ij_,2,color);
                    circle(images_[f_idx].second,feat_ij,2,Scalar(0,255,0));
                    cout << "here" << endl;
                    #endif
                    errors.push_back(sum((ROI_ij_-ROI_ij).mul(ROI_ij_-ROI_ij))[0]);
//                    sum_errs += sum((ROI_ij_-ROI_ij).mul(ROI_ij_-ROI_ij))[0];
                }
                #ifdef SHOW
                else{
                    sum_errs++;
                    circle(images_[f_idx].first,feat_ij,2,Scalar(0,0,255));
                    circle(images_[f_idx].second,feat_ij_,2,Scalar(0,0,255));
                }
                #endif
            }
        }
    }
    cout << endl;
        for(uint i=0;i<state.pts.second.size();i++){
        if(state.pts.second[i].isTriangulated()){
            #ifdef SHOW
//            int icolor = (unsigned) rng;
//            Scalar color((icolor&255), ((icolor>>8)&255),((icolor>>16)&255));
            Scalar color(0,255,0);
            #endif
            ptH3D pt = state.pts.second[i].get3DLocation();
            for(uint j=0;j<state.pts.second[i].getNbFeatures();j++){
                int f_idx = state.pts.second[i].getFrameIdx(j);
                Mat Tr = (Mat) state.poses.second[f_idx].orientation.getR4();
                ((Mat)state.poses.second[f_idx].position).copyTo(Tr(Range(0,3),Range(3,4)));
                Matx44d Tr_ = Tr;
                ptH2D feat = state.K.second * Matx34d::eye() * state.scale *(Tr_ * pt);
                Point2f feat_right(to_euclidean(feat)(0),to_euclidean(feat)(1));
                ptH2D feat_ =  (state.K.second * Matx34d::eye()) * (state.scale * (Tr_ * pt) + Matx41d(state.baseline,0,0,0));
                Point2f feat_left(to_euclidean(feat_)(0),to_euclidean(feat_)(1));
                if(bb.contains(feat_right) && bb.contains(feat_left)){
                    Mat ROI_right = m_obs[f_idx][1](Rect(feat_right.x-state.window_size,feat_right.y-state.window_size,state.window_size*2,state.window_size*2));
                    Mat ROI_left = m_obs[f_idx][0](Rect(feat_left.x-state.window_size,feat_left.y-state.window_size,state.window_size*2,state.window_size*2));
                    Mat ROI_2 = m_obs[f_idx][2](Rect(feat_right.x-state.window_size,feat_right.y-state.window_size,state.window_size*2,state.window_size*2));
                    Mat ROI_3 = m_obs[f_idx][3](Rect(feat_left.x-state.window_size,feat_left.y-state.window_size,state.window_size*2,state.window_size*2));
                    ROI_right.convertTo(ROI_right,CV_32F);
                    ROI_left.convertTo(ROI_left,CV_32F);
                    Mat diff = ROI_right-ROI_left;
                    Mat diff2 = ROI_3-ROI_2;
                    float* diff_ptr = diff.ptr<float>();
                    for(int k=0;k<nb_pix;k++){
                        res(i+state.pts.first.size(),j*nb_pix+k) = (diff_ptr[k]);
                    }
                    float* diff2_ptr = diff2.ptr<float>();
                    for(int k=0;k<nb_pix;k++){
                        res(i+state.pts.first.size(),j*nb_pix+2*nb_pix+k) = (diff2_ptr[k]);
                    }
                    #ifdef SHOW
                    circle(images_[f_idx].first,feat_left,2,color);
                    circle(images_[f_idx].second,feat_right,2,color);
                    circle(images_[f_idx].first,feat_right,2,Scalar(0,255,0));
                    #endif
//                    errors.push_back(sum((ROI_ij_-ROI_ij).mul(ROI_ij_-ROI_ij))[0]);
//                    sum_errs += sum((ROI_ij_-ROI_ij).mul(ROI_ij_-ROI_ij))[0];
                }
                #ifdef SHOW
                else{
                    sum_errs++;
                    circle(images_[f_idx].first,feat_left,2,Scalar(0,0,255));
                    circle(images_[f_idx].second,feat_right,2,Scalar(0,0,255));
                }
                #endif
            }
        }
    }

    #ifdef SHOW
    for(uint j=0;j<m_obs.size();j++){
            imshow("img "+to_string(j),images_[j].first);
            imshow("img "+to_string(j)+"_",images_[j].second);
        }
        waitKey(5);
    #endif

    return res;
}

template<>
MatrixXd Optimiser<StereoState, std::vector<StereoOdoMatchesf> >::compute_residuals(const StereoState& state){

    Eigen::MatrixXd residuals((m_mask.rows()==0?state.pts.size():m_mask.sum()),4);

    Matx34d P = Matx34d::eye();
    Matx44d Tr = state.pose.TrMat();
    int k=0;
    for (unsigned int i=0; i<state.pts.size(); i++)
        if(m_mask.rows() == 0 || m_mask(i)){
            ptH3D pt = Tr*state.pts[i];
            ptH3D pt_ = Tr * (pt - ptH3D(state.baseline,0,0,0));
            pt2D lpt=to_euclidean(state.K.first*P*pt),rpt=to_euclidean(state.K.second*P*pt_);
            residuals.row(k++) << lpt(0)-m_obs[i].f3.x, lpt(1)-m_obs[i].f3.y, rpt(0)-m_obs[i].f4.x, rpt(1)-m_obs[i].f4.y;
        }

    return residuals;
}

template<class S, class T>
void Optimiser<S,T>::compute_normal_equations(const Eigen::MatrixXd& residuals, Eigen::MatrixXd& JJ, Eigen::VectorXd& e){

}

template<>
void Optimiser<ScaleState,std::vector<std::pair<cv::Mat,cv::Mat>>>::compute_normal_equations(const Eigen::MatrixXd& residuals, Eigen::MatrixXd& JJ, Eigen::VectorXd& e){

//    double ds =0.1;
    double dp = 1;
    Rect bb(2*m_state.window_size,2*m_state.window_size,m_obs[0].first.cols-4*m_state.window_size-2,m_obs[1].first.rows-4*m_state.window_size-2);

    JJ = MatrixXd::Zero(m_state.nb_params,m_state.nb_params);
    e = VectorXd::Zero(m_state.nb_params);

    uint lframe = m_state.poses.first[0].ID+m_state.poses.first.size()-1;
    int k=0;

    //loop for points extracted from the left camera
    for(uint i=0;i<m_state.pts.first.size();i++){ // for each point
        if(!m_mask.size() == 0 && !m_mask(i)) // if is an inlier
            continue;
        if(m_state.pts.first[i].isTriangulated()){ // if it has been triangulated
            ptH3D pt = m_state.pts.first[i].get3DLocation();
            if(m_state.pts.first[i].getLastFrameIdx() == lframe){ // has been observed in the last keyframe
                uint f_idx = m_state.poses.first.size()-1;
                //camera pose
                Matx33d R = m_state.poses.first[f_idx].orientation.getR3();
                Matx31d t = m_state.poses.first[f_idx].position;
                Matx44d Tr = Matx44d::eye();
                ((Mat)R).copyTo(((Mat)Tr)(Range(0,3),Range(0,3)));
                ((Mat)t).copyTo(((Mat)Tr)(Range(0,3),Range(3,4)));

                double duds = m_state.K.second(0,0)*m_state.baseline/(m_state.scale*m_state.scale*(R*to_euclidean(pt)+t)(2));

                ptH2D feat = m_state.K.first * Matx34d::eye() * m_state.scale *(Tr * pt);
                Point2f feat_left(to_euclidean(feat)(0),to_euclidean(feat)(1));

                ptH2D feat_ =  (m_state.K.second * Matx34d::eye()) * ((m_state.scale) * (Tr * pt) - Matx41d(m_state.baseline,0,0,0));
                //gradient estimation by differenciation
//                Point2f feat_right_minus(to_euclidean(feat_)(0),to_euclidean(feat_)(1));
//                ptH2D feat__ =  (m_state.K.second * Matx34d::eye()) * ((m_state.scale+ds) * (Tr * pt) - Matx41d(m_state.baseline,0,0,0));
//                Point2f feat_right_plus(to_euclidean(feat__)(0),to_euclidean(feat__)(1));
                Point2f feat_right_minus(to_euclidean(feat_)(0)-dp,to_euclidean(feat_)(1));
                Point2f feat_right_plus(to_euclidean(feat_)(0)+dp,to_euclidean(feat_)(1)+dp);

                if(bb.contains(feat_left) && bb.contains(feat_right_minus) && bb.contains(feat_right_plus)){ //feature is reprojected in the image and MI can be computed
                    Mat ROIx0 = m_obs[f_idx].first(Rect(feat_left.x-m_state.window_size,feat_left.y-m_state.window_size,m_state.window_size*2,m_state.window_size*2));
                    Mat ROIx1 = m_obs[f_idx].second(Rect(feat_right_minus.x-m_state.window_size,feat_right_minus.y-m_state.window_size,m_state.window_size*2,m_state.window_size*2));
                    Mat ROIx2 = m_obs[f_idx].second(Rect(feat_right_plus.x-m_state.window_size,feat_right_plus.y-m_state.window_size,m_state.window_size*2,m_state.window_size*2));
                    ROIx0.convertTo(ROIx0,CV_32F);
                    ROIx1.convertTo(ROIx1,CV_32F);
                    ROIx2.convertTo(ROIx2,CV_32F);

//                    cout << "dmidu " <<(computeMutualInformation(ROIx2,ROIx0)-computeMutualInformation(ROIx1,ROIx0))/dp << " duds " << duds << endl;

//                    double J = (computeMutualInformation(ROIx2,ROIx0)-computeMutualInformation(ROIx1,ROIx0))/ds;
                    double J = (computeMutualInformation(ROIx2,ROIx0)-computeMutualInformation(ROIx1,ROIx0))/dp * duds;
//                    cout << "J " << J << endl;
                    JJ(0,0) += pow(J,2);
                    e(0) += J * residuals(k,0);
                }
            }k++;
        }
    }
//
//    //loop for points extracted from the right camera
//    for(uint i=0;i<m_state.pts.second.size();i++){
//        if(!m_mask.size() == 0 && !m_mask(m_state.pts.first.size()+i))
//            continue;
//        if(m_state.pts.second[i].isTriangulated()){
//            ptH3D pt = m_state.pts.second[i].get3DLocation();
//            if(m_state.pts.second[i].getLastFrameIdx() == lframe){
//                uint f_idx = m_state.poses.second.size()-1;
//                Matx33d R = m_state.poses.second[f_idx].orientation.getR3();
//                pt3D base(m_state.baseline,0,0);
//                Matx31d t = m_state.poses.second[f_idx].position + R * base;
//                Matx44d Tr = Matx44d::eye();
//                ((Mat)R).copyTo(((Mat)Tr)(Range(0,3),Range(0,3)));
//                ((Mat)t).copyTo(((Mat)Tr)(Range(0,3),Range(3,4)));
//
//                ptH2D feat = m_state.K.first * Matx34d::eye() * m_state.scale *(Tr * pt);
//                Point2f feat_right(to_euclidean(feat)(0),to_euclidean(feat)(1));
//                ptH2D feat_ =  (m_state.K.first * Matx34d::eye()) * ((m_state.scale-ds) * (Tr * pt) + Matx41d(m_state.baseline,0,0,0));
//                Point2f feat_left_minus(to_euclidean(feat_)(0),to_euclidean(feat_)(1));
//                ptH2D feat__ =  (m_state.K.first * Matx34d::eye()) * ((m_state.scale+ds) * (Tr * pt) + Matx41d(m_state.baseline,0,0,0));
//                Point2f feat_left_plus(to_euclidean(feat_)(0),to_euclidean(feat__)(1));
//                if(bb.contains(feat_right) && bb.contains(feat_left_minus) && bb.contains(feat_left_plus)){
//                    Mat ROIx0 = m_obs[f_idx].second(Rect(feat_right.x-m_state.window_size,feat_right.y-m_state.window_size,m_state.window_size*2,m_state.window_size*2))*255;
//                    Mat ROIx1 = m_obs[f_idx].first(Rect(feat_left_minus.x-m_state.window_size,feat_left_minus.y-m_state.window_size,m_state.window_size*2,m_state.window_size*2))*255;
//                    Mat ROIx2 = m_obs[f_idx].first(Rect(feat_left_plus.x-m_state.window_size,feat_left_plus.y-m_state.window_size,m_state.window_size*2,m_state.window_size*2))*255;
//                    ROIx0.convertTo(ROIx0,CV_32F);
//                    ROIx1.convertTo(ROIx1,CV_32F);
//                    ROIx2.convertTo(ROIx2,CV_32F);
//
//                    double J = computeMutualInformation(ROIx2,ROIx0)-computeMutualInformation(ROIx1,ROIx0);
//                    JJ(0,0) += pow(J,2);
//                    e(0) += J * residuals(k,0);
//                }
//            }k++;
//        }
//    }
}

template<>
MatrixXd Optimiser<ScaleState,std::vector<std::pair<cv::Mat,cv::Mat>>>::compute_jacobian(){

    double dp = 1;
    Rect bb(2*m_state.window_size,2*m_state.window_size,m_obs[0].first.cols-4*m_state.window_size-2,m_obs[1].first.rows-4*m_state.window_size-2);

    MatrixXd JJ = MatrixXd::Zero(1,1);

    uint lframe = m_state.poses.first[0].ID+m_state.poses.first.size()-1;
    int k=0;

    //loop for points extracted from the left camera
    for(uint i=0;i<m_state.pts.first.size();i++){ // for each point
        if(!m_mask.size() == 0 && !m_mask(i)) // if is an inlier
            continue;
        if(m_state.pts.first[i].isTriangulated()){ // if it has been triangulated
            ptH3D pt = m_state.pts.first[i].get3DLocation();
            if(m_state.pts.first[i].getLastFrameIdx() == lframe){ // has been observed in the last keyframe
                uint f_idx = m_state.poses.first.size()-1;
                //camera pose
                Matx33d R = m_state.poses.first[f_idx].orientation.getR3();
                Matx31d t = m_state.poses.first[f_idx].position;
                Matx44d Tr = Matx44d::eye();
                ((Mat)R).copyTo(((Mat)Tr)(Range(0,3),Range(0,3)));
                ((Mat)t).copyTo(((Mat)Tr)(Range(0,3),Range(3,4)));

                double duds = m_state.K.second(0,0)*m_state.baseline/(m_state.scale*m_state.scale*(R*to_euclidean(pt)+t)(2));


                ptH2D feat = m_state.K.first * Matx34d::eye() * m_state.scale *(Tr * pt);
                Point2f feat_left(to_euclidean(feat)(0),to_euclidean(feat)(1));
                //gradient estimation by differenciation
                ptH2D feat_ =  (m_state.K.second * Matx34d::eye()) * ((m_state.scale) * (Tr * pt) - Matx41d(m_state.baseline,0,0,0));
//                Point2f feat_right_minus(to_euclidean(feat_)(0),to_euclidean(feat_)(1));
//                ptH2D feat__ =  (m_state.K.second * Matx34d::eye()) * ((m_state.scale+ds) * (Tr * pt) - Matx41d(m_state.baseline,0,0,0));
//                Point2f feat_right_plus(to_euclidean(feat__)(0),to_euclidean(feat__)(1));
                Point2f feat_right_minus(to_euclidean(feat_)(0)-dp,to_euclidean(feat_)(1));
                Point2f feat_right_plus(to_euclidean(feat_)(0)+dp,to_euclidean(feat_)(1)+dp);

                if(bb.contains(feat_left) && bb.contains(feat_right_minus) && bb.contains(feat_right_plus)){ //feature is reprojected in the image and MI can be computed
                    Mat ROIx0 = m_obs[f_idx].first(Rect(feat_left.x-m_state.window_size,feat_left.y-m_state.window_size,m_state.window_size*2,m_state.window_size*2));
                    Mat ROIx1 = m_obs[f_idx].second(Rect(feat_right_minus.x-m_state.window_size,feat_right_minus.y-m_state.window_size,m_state.window_size*2,m_state.window_size*2));
                    Mat ROIx2 = m_obs[f_idx].second(Rect(feat_right_plus.x-m_state.window_size,feat_right_plus.y-m_state.window_size,m_state.window_size*2,m_state.window_size*2));
                    ROIx0.convertTo(ROIx0,CV_32F);
                    ROIx1.convertTo(ROIx1,CV_32F);
                    ROIx2.convertTo(ROIx2,CV_32F);

//                    double J = (computeMutualInformation(ROIx2,ROIx0) - computeMutualInformation(ROIx1,ROIx0))/ds;
                    double J = (computeMutualInformation(ROIx2,ROIx0)-computeMutualInformation(ROIx1,ROIx0))/dp * duds;
                    JJ(0,0) += pow(J,2);
                }
            }k++;
        }
    }
    return  JJ;
}

template<class S, class T>
MatrixXd Optimiser<S,T>::compute_jacobian(){
     return MatrixXd::Zero(1,1);
}

template<>
void Optimiser<ScaleState,std::vector<std::vector<cv::Mat>>>::compute_normal_equations(const Eigen::MatrixXd& residuals, Eigen::MatrixXd& JJ, Eigen::VectorXd& e){

}

template<>
void Optimiser<StereoState,std::vector<StereoOdoMatchesf>>::compute_normal_equations(const Eigen::MatrixXd& residuals, Eigen::MatrixXd& JJ, Eigen::VectorXd& e){

    assert(residuals.cols() == 4);
    JJ = MatrixXd::Zero(m_state.nb_params,m_state.nb_params);
    e = VectorXd::Zero(m_state.nb_params);

    Matx44d Tr = m_state.pose.TrMat();

    int k=0;
    for(uint i=0;i<m_state.pts.size();i++)
        if(m_mask.rows() == 0 || m_mask(i)){
            pt3D p1 = to_euclidean(Tr*m_state.pts[i]);
            pt3D p2 = to_euclidean(Tr*m_state.pts[i])-pt3D(m_state.baseline,0,0);
            MatrixXd J_i(residuals.cols(),m_state.nb_params);
            J_i << m_state.K.first(0,0)/p1(2), 0, -m_state.K.first(0,0)*p1(0)/pow(p1(2),2), -m_state.K.first(0,0)*(p1(0)*p1(1))/pow(p1(2),2), m_state.K.first(0,0)*(1+pow(p1(0),2)/pow(p1(2),2)), -m_state.K.first(0,0)*p1(1)/p1(2),
                    0, m_state.K.first(1,1)/p1(2), -m_state.K.first(1,1)*p1(1)/pow(p1(2),2), -m_state.K.first(1,1)*(1+pow(p1(1),2)/pow(p1(2),2)), m_state.K.first(1,1)*(p1(0)*p1(1))/pow(p1(2),2),  m_state.K.first(1,1)*p1(0)/p1(2),
                    m_state.K.first(0,0)/p2(2), 0, -m_state.K.first(0,0)*p2(0)/pow(p2(2),2), -m_state.K.first(0,0)*(p2(0)*p2(1))/pow(p2(2),2), m_state.K.first(0,0)*(1+pow(p2(0),2)/pow(p2(2),2)), -m_state.K.first(0,0)*p2(1)/p2(2),
                    0, m_state.K.first(1,1)/p2(2), -m_state.K.first(1,1)*p2(1)/pow(p2(2),2), -m_state.K.first(1,1)*(1+pow(p2(1),2)/pow(p2(2),2)), m_state.K.first(1,1)*(p2(0)*p2(1))/pow(p2(2),2),  m_state.K.first(1,1)*p2(0)/p2(2);


            JJ += J_i.transpose() * J_i;
            e += (m_params.minim?-1.0:1.0) * J_i.transpose() * residuals.row(k++).transpose();
    }
}

/**** Steps ****/

template<class S, class T>
void Optimiser<S,T>::run_GN_step(MatrixXd& JJ, Eigen::VectorXd& e, Eigen::VectorXd& dX){

    //adding constant to avoid ill-formed jacobian
    JJ.diagonal() += VectorXd::Ones(JJ.rows()) * m_params.mu;

    dX = JJ.ldlt().solve(e); // solving normal equations

    m_state.update(m_params.alpha * dX);
}

template<class S, class T>
void Optimiser<S,T>::run_LM_step(MatrixXd& JJ, Eigen::VectorXd& e, Eigen::VectorXd& dX, const double e1){

    for(;;){

        JJ.diagonal() += VectorXd::Ones(JJ.rows()) * m_params.mu;

        dX = JJ.ldlt().solve(e); // solving normal equations

        if(dX.norm() <= m_params.incr_tol){
            m_stop = StopCondition::SMALL_INCREMENT;
            break;
        }

        S tmp_state = m_state;
        tmp_state.update(m_params.alpha*dX);
        MatrixXd tmp_residuals = compute_residuals(tmp_state);
        double e2 = (tmp_residuals * tmp_residuals.transpose()).diagonal().sum();

        MatrixXd dL = (dX.transpose()*(m_params.mu*dX+e));
        double rho = (m_params.minim?-1.0:1.0) * (e2-e1);//dL(0);

        if(rho > 0){ //threshold

            m_params.mu *= max(1.0/3.0,1-pow(2*rho-1,3));
            m_params.v = 2;
            if(pow(sqrt(e1) - sqrt(e2),2) < m_params.rel_tol * sqrt(e1)){
                m_stop = StopCondition::SMALL_DECREASE_FUNCTION;
            }

            m_state = tmp_state;
            break;
        }
        else{
            m_params.mu *= m_params.v;
            double v2 = 2*m_params.v;
            if(v2 <= m_params.v){
                m_stop = StopCondition::NO_CONVERGENCE;
                break;
            }
            m_params.v = v2;
        }
    }

}

template<class S, class T>
std::vector<int> Optimiser<S,T>::compute_inliers(const double threshold){

    VectorXi tmp_mask = m_mask;
    m_mask = VectorXi();
    MatrixXd residuals = compute_residuals(m_state);
    vector<int> inliers;

    for(uint i=0;i<residuals.rows();i++)
        if(sqrt(residuals.row(i)*residuals.row(i).transpose()) < threshold)
            inliers.push_back(i);

    m_mask = tmp_mask;

    return inliers;
}

template class Optimiser<ScaleState,std::vector<std::pair<cv::Mat,cv::Mat>>>;
template class Optimiser<ScaleState,std::vector<std::vector<cv::Mat>>>;
template class Optimiser<StereoState,std::vector<StereoOdoMatchesf>>;
}
