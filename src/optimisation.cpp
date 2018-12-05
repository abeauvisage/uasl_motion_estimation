#include "optimisation.h"

#include "mutualInformation.h"
#include "utils.h"

#include <Eigen/Cholesky>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <iomanip>

#define TEST

using namespace Eigen;
using namespace cv;
using namespace std;

namespace me{

    static ofstream fscale,fmi,ftri;

template<class S, class T>
StopCondition Optimiser<S,T>::optimise(S& state, const bool test, const Eigen::VectorXi& mask){

    m_state = state;
    m_mask = mask;
    m_stop = StopCondition::NO_STOP;



    if(test){
        cout << "** testing **" << endl;
        fscale.open("log/log_scale_test.csv",std::ofstream::app);
        fmi.open("log/log_mi_test.csv",std::ofstream::app);
        ftri.open("log/log_tri_test.csv",std::ofstream::app);
        fscale << 0;fmi << 0;ftri << 0;
        ScaleState* state_ptr = dynamic_cast<ScaleState*>(&m_state);
        if(state_ptr)
            state_ptr->scale = 0.0001;
        m_params.type=GN;
        m_params.alpha=1;
        m_params.rel_tol = 0.0;
        m_params.abs_tol = 0.0;
        m_params.MAX_NB_ITER = 300;
        m_params.grad_tol = 0.0;
        m_params.incr_tol = 0.0;
    }else{
        cout << "** optim **" << endl;
        cout << std::boolalpha << (m_params.type == OptimType::LM) << endl;
        cout << " here opening files" << endl;
        fscale.open("log/log_scale_optim.csv",std::ofstream::app);
        fmi.open("log/log_mi_optim.csv",std::ofstream::app);
        ftri.open("log/log_tri_optim.csv",std::ofstream::app);
        fscale << 0;fmi << 0;ftri << 0;
    }


//    double optim_scale=0.0;
    double max_mi=0.0,max_scale=1.0;
    int k=0;
    do{
        MatrixXd residuals = compute_residuals(m_state);
        double e1 = (residuals * residuals.transpose()).diagonal().sum();//sum((residuals.transpose()*residuals).diagonal())[0];
        double meanReprojError = e1 / (double)(residuals.rows()*residuals.cols());
        std::cout << "mean Rep error " << meanReprojError << std::endl;
        if(e1 > max_mi){
            max_mi = e1;
            max_scale = dynamic_cast<ScaleState*>(&m_state)->scale;
//            optim_scale = dynamic_cast<ScaleState*>(&m_state)->scale;
        }

        if(meanReprojError < m_params.abs_tol)
            m_stop = StopCondition::SMALL_REPROJ_ERROR;

        fscale << "," << dynamic_cast<ScaleState*>(&m_state)->scale;
        fmi << "," << e1;

//        Eigen::MatrixXd J = computeJacobian(Xa,Xb,residuals,Uj,Vi,W,ea,eb,visibility,fixedFrames);

        MatrixXd JJ(1,1);
        VectorXd e(1);
        compute_normal_equations(residuals,JJ,e);

//        cout << "e/JJ = " << e(0)/JJ(0) << endl;

        if(test){
            JJ(0) = 5;
            e(0) = .05;
        }
        if(k == 0){
            m_params.mu = 1.0;//JJ.diagonal().maxCoeff();
//            m_params.alpha = (m_state.scale/2.0) * JJ(0)/e(0);
        }
//        cout << "JJ " << JJ << endl;
//        cout << "e " << e <<endl;
////
        if(e.norm() < m_params.grad_tol){
            m_stop = StopCondition::SMALL_GRADIENT;
            cout << "norm e " << e.norm() << endl;
        }



    //if first it, mu = max diag J

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

//        std::cout << "dX: " << dX.transpose() << " norm " << dX.norm() << std::endl;
////


        if(!m_stop && dX.norm() <= m_params.incr_tol )
            m_stop = StopCondition::SMALL_INCREMENT;


//        std::cout << "alpha: " << m_params.alpha << std::endl;
//        std::cout << "new scale: " << m_state.scale << std::endl;

//        m_state.update(dX);
        MatrixXd tmp_residuals = compute_residuals(m_state);
        double e2 = (tmp_residuals * tmp_residuals.transpose()).diagonal().sum();

        fmi << "," << e2;

        std::cout << "state" << m_state.show_params() << " -> " << e2 << endl;
        fscale << "," << m_state.show_params();

//        if(e2-e1 < 0)
//            m_params.alpha *= 1.5;
//        else
//            m_params.alpha /=2;

//        std::cout << "change cost: " << e2-e1 << endl;

    if(m_params.type==GN && pow(e2-e1,2) < m_params.rel_tol){
        m_stop = StopCondition::SMALL_DECREASE_FUNCTION;
    }

//        cout << "scale: " << m_state.scale << " " << e2 << endl;

//        waitKey();

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

    fscale << endl;
    fmi << endl;
    ftri << endl;
    fscale.close();fmi.close();ftri.close();
    dynamic_cast<ScaleState*>(&m_state)->scale = max_scale;
    cout << "final scale: " << max_scale << endl;
    state = m_state;
    return m_stop;
}

template<>
MatrixXd Optimiser<ScaleState, std::vector<std::pair<cv::Mat,cv::Mat>> >::compute_residuals(const ScaleState& state){

    assert(m_mask.size() == 0 || m_mask.rows() == (int) (state.pts.first.size()+state.pts.second.size()));
    Rect bb(state.window_size,state.window_size,m_obs[0].first.cols-2*state.window_size,m_obs[1].first.rows-2*state.window_size);

    double pyr_factor = 1.0;

    int tot_nb_elements;
    if(m_mask.size()==0)
        tot_nb_elements = state.pts.first.size()+state.pts.second.size();
    else
        tot_nb_elements = m_mask.sum();
//    int nb_pix = pow(state.window_size*2,2);
//    MatrixXd res = MatrixXd::Zero(state.pts.first.size()+state.pts.second.size(),state.poses.first.size()*nb_pix);
    MatrixXd res = MatrixXd::Zero(tot_nb_elements,m_obs.size());

//    #define SHOW

    #ifdef SHOW
    cv::RNG rng;
    std::vector<std::pair<cv::Mat,cv::Mat>> images_ = m_obs;
        for(uint j=0;j<m_obs.size();j++){
            cvtColor(m_obs[j].first,images_[j].first,CV_GRAY2BGR);
            cvtColor(m_obs[j].second,images_[j].second,CV_GRAY2BGR);
        }
    #endif

//    std::vector<double> errors;
//    double sum_errs=0;
//    double err_mi=0;

    cout << state.poses.first.size() << " " << m_obs.size() << endl;
    int fframe = state.poses.first[0].ID;
    cout << "first frame: " << fframe << endl;

    int k=0,nk=0;
    for(uint i=0;i<state.pts.first.size();i++){
        if(!m_mask.size() == 0 && !m_mask(i))
            continue;
        if(state.pts.first[i].isTriangulated()){
            #ifdef SHOW
//            int icolor = (unsigned) rng;
//            Scalar color((icolor&255), ((icolor>>8)&255),((icolor>>16)&255));
            Scalar color(255,0,0);
            Scalar color2(0,255,255);
            #endif
            ptH3D pt = state.pts.first[i].get3DLocation();
            for(uint j=0;j<state.pts.first[i].getNbFeatures();j++){
                int f_idx = state.pts.first[i].getFrameIdx(j)-fframe;
//                cout << "(" << f_idx << " " << state.pts.first[i].getLastFrameIdx() << "," << state.pts.first[i].getNbFeatures() <<")";
                Mat Tr = (Mat) state.poses.first[f_idx].orientation.getR4();
                ((Mat)state.poses.first[f_idx].position).copyTo(Tr(Range(0,3),Range(3,4)));
                Matx44d Tr_ = Tr;
//                cout << Tr << endl << Tr_ << endl;
                ptH2D feat = state.K.first * Matx34d::eye() * state.scale *(Tr_ * pt);
                Point2f feat_ij(to_euclidean(feat)(0),to_euclidean(feat)(1));feat_ij /= pyr_factor;
                ptH2D feat_ =  (state.K.second * Matx34d::eye()) * (state.scale * (Tr_ * pt) - Matx41d(state.baseline,0,0,0));
                Point2f feat_ij_(to_euclidean(feat_)(0),to_euclidean(feat_)(1)); feat_ij_ /= pyr_factor;
                if(bb.contains(feat_ij_) && bb.contains(feat_ij)){
                    Mat ROI_ij = m_obs[f_idx].first(Rect(feat_ij.x-state.window_size,feat_ij.y-state.window_size,state.window_size*2,state.window_size*2))*255;
                    Mat ROI_ij_ = m_obs[f_idx].second(Rect(feat_ij_.x-state.window_size,feat_ij_.y-state.window_size,state.window_size*2,state.window_size*2))*255;
                    ROI_ij.convertTo(ROI_ij,CV_32F);
                    ROI_ij_.convertTo(ROI_ij_,CV_32F);
//                    Mat diff = ROI_ij_-ROI_ij;
//                    float* diff_ptr = diff.ptr<float>();
//                    for(int k=0;k<nb_pix;k++){
//                        res(i,j*nb_pix+k) = (diff_ptr[k]);
//                    }
                    #ifdef SHOW
                    circle(images_[f_idx].second,feat_ij,2,color2);
                    circle(images_[f_idx].first,feat_ij,2,color);
                    circle(images_[f_idx].second,feat_ij_,2,color);
                    #endif
//                    errors.push_back(sum((ROI_ij_-ROI_ij).mul(ROI_ij_-ROI_ij))[0]);
//                    sum_errs += sum((ROI_ij_-ROI_ij).mul(ROI_ij_-ROI_ij))[0];
                    res(k,j) = computeMutualInformation(ROI_ij,ROI_ij_);
                }
                else{
                        nk++;
//                    res(i,0) = 0.0;
                    #ifdef SHOW
                    circle(images_[f_idx].first,feat_ij,2,Scalar(0,0,255));
                    circle(images_[f_idx].second,feat_ij_,2,Scalar(0,0,255));
                    #endif
                }
            }k++;
        }
    }

    for(uint i=0;i<state.pts.second.size();i++){
        if(!m_mask.size() == 0 && !m_mask(state.pts.first.size()+i))
            continue;
        if(state.pts.second[i].isTriangulated()){
            #ifdef SHOW
//            int icolor = (unsigned) rng;
//            Scalar color((icolor&255), ((icolor>>8)&255),((icolor>>16)&255));
            Scalar color(0,255,0);
            Scalar color2(255,0,255);
            #endif
            ptH3D pt = state.pts.second[i].get3DLocation();
            for(uint j=0;j<state.pts.second[i].getNbFeatures();j++){
                int f_idx = state.pts.second[i].getFrameIdx(j)-fframe;
                Mat Tr = (Mat) state.poses.second[f_idx].orientation.getR4();
                ((Mat)state.poses.second[f_idx].position).copyTo(Tr(Range(0,3),Range(3,4)));
                Matx44d Tr_ = Tr;
                ptH2D feat = state.K.second * Matx34d::eye() * state.scale *(Tr_ * pt);
                Point2f feat_right(to_euclidean(feat)(0),to_euclidean(feat)(1));feat_right /= pyr_factor;
                ptH2D feat_ =  (state.K.second * Matx34d::eye()) * (state.scale * (Tr_ * pt) + Matx41d(state.baseline,0,0,0));
                Point2f feat_left(to_euclidean(feat_)(0),to_euclidean(feat_)(1));feat_left /= pyr_factor;
                if(bb.contains(feat_right) && bb.contains(feat_left)){
                    Mat ROI_right = m_obs[f_idx].second(Rect(feat_right.x-state.window_size,feat_right.y-state.window_size,state.window_size*2,state.window_size*2))*255;
                    Mat ROI_left = m_obs[f_idx].first(Rect(feat_left.x-state.window_size,feat_left.y-state.window_size,state.window_size*2,state.window_size*2))*255;
                    ROI_right.convertTo(ROI_right,CV_32F);
                    ROI_left.convertTo(ROI_left,CV_32F);
//                    Mat diff = ROI_right-ROI_left;
//                    float* diff_ptr = diff.ptr<float>();
//                    for(int k=0;k<nb_pix;k++){
//                        res(i+state.pts.first.size(),j*nb_pix+k) = (diff_ptr[k]);
//                    }
                    #ifdef SHOW
                    circle(images_[f_idx].first,feat_right,2,color2);
                    circle(images_[f_idx].first,feat_left,2,color);
                    circle(images_[f_idx].second,feat_right,2,color);
                    #endif
//                    errors.push_back(sum((ROI_ij_-ROI_ij).mul(ROI_ij_-ROI_ij))[0]);
//                    sum_errs += sum((ROI_ij_-ROI_ij).mul(ROI_ij_-ROI_ij))[0];
                    res(k,j) = computeMutualInformation(ROI_right,ROI_left);
                }
                else{
//                    res(k++,0) = 0.0;
                    nk++;
                    #ifdef SHOW
                    circle(images_[f_idx].first,feat_left,2,Scalar(0,0,255));
                    circle(images_[f_idx].second,feat_right,2,Scalar(0,0,255));
                    #endif
                }
            }k++;
        }
    }


    #ifdef SHOW
    for(uint j=0;j<m_obs.size();j++){
            imshow("img "+to_string(j),images_[j].first);
            imshow("img "+to_string(j)+"_",images_[j].second);
        }
        waitKey(5);
    #endif

    cout << "non reproj errors: " << nk << endl;
    ftri << "," << nk;

    return res;
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
    double sum_errs=0;

    for(uint i=0;i<state.pts.first.size();i++){
            cout << i << ": ";
        if(state.pts.first[i].isTriangulated()){
            #ifdef SHOW
//            int icolor = (unsigned) rng;
//            Scalar color((icolor&255), ((icolor>>8)&255),((icolor>>16)&255));
            Scalar color(255,0,0);
            #endif
            ptH3D pt = state.pts.first[i].get3DLocation();
            for(uint j=0;j<state.pts.first[i].getNbFeatures();j++){
                int f_idx = state.pts.first[i].getFrameIdx(j);
                cout << f_idx << " ";
                Mat Tr = (Mat) state.poses.first[f_idx].orientation.getR4();
                ((Mat)state.poses.first[f_idx].position).copyTo(Tr(Range(0,3),Range(3,4)));
                Matx44d Tr_ = Tr;
//                cout << Tr << endl << Tr_ << endl;
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

    cout << "non reproj feats: " << sum_errs << endl;
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

    cv::Mat test1 = Mat::zeros(480,640,CV_8UC3);
    cv::Mat test2 = Mat::zeros(480,640,CV_8UC3);
    int k=0;
    for (unsigned int i=0; i<state.pts.size(); i++)
        if(m_mask.rows() == 0 || m_mask(i)){
            ptH3D pt = Tr*state.pts[i];
            ptH3D pt_ = Tr * (pt - ptH3D(state.baseline,0,0,0));
            pt2D lpt=to_euclidean(state.K.first*P*pt),rpt=to_euclidean(state.K.second*P*pt_);
            residuals.row(k++) << lpt(0)-m_obs[i].f3.x, lpt(1)-m_obs[i].f3.y, rpt(0)-m_obs[i].f4.x, rpt(1)-m_obs[i].f4.y;
            circle(test1,Point2f(lpt(0),lpt(1)),3,Scalar(255,0,0));
            circle(test1,Point2f(m_obs[i].f3.x,m_obs[i].f3.y),3,Scalar(0,0,255));
            circle(test2,Point2f(rpt(0),rpt(1)),3,Scalar(255,0,0));
            circle(test2,m_obs[i].f4,3,Scalar(0,0,255));
        }

//    imshow("leftimg",test1);
//    imshow("rightimg",test2);
//    waitKey();

    return residuals;
}

template<class S, class T>
void Optimiser<S,T>::compute_normal_equations(const Eigen::MatrixXd& residuals, Eigen::MatrixXd& JJ, Eigen::VectorXd& e){

}

template<>
void Optimiser<ScaleState,std::vector<std::pair<cv::Mat,cv::Mat>>>::compute_normal_equations(const Eigen::MatrixXd& residuals, Eigen::MatrixXd& JJ, Eigen::VectorXd& e){

    int grad_int = 1;
    Rect bb(2*m_state.window_size+grad_int,2*m_state.window_size+grad_int,m_obs[0].first.cols-4*m_state.window_size-2*grad_int,m_obs[1].first.rows-4*m_state.window_size-2*grad_int);

//    int nb_pix = pow(m_state.window_size*2,2);
    double pyr_factor = 1.0;

    JJ = MatrixXd::Zero(m_state.nb_params,m_state.nb_params);
    e = VectorXd::Zero(m_state.nb_params);

    int fframe = m_state.poses.first[0].ID;
    int k=0;
    for(uint i=0;i<m_state.pts.first.size();i++){
        if(!m_mask.size() == 0 && !m_mask(i))
            continue;
        if(m_state.pts.first[i].isTriangulated()){
            ptH3D pt = m_state.pts.first[i].get3DLocation();
            for(uint j=0;j<m_state.pts.first[i].getNbFeatures();j++){
                int f_idx = m_state.pts.first[i].getFrameIdx(j)-fframe;

                Matx33d R = m_state.poses.first[f_idx].orientation.getR3();
                Matx31d t = m_state.poses.first[f_idx].position;
                Matx44d Tr = Matx44d::eye();
                ((Mat)R).copyTo(((Mat)Tr)(Range(0,3),Range(0,3)));
                ((Mat)t).copyTo(((Mat)Tr)(Range(0,3),Range(3,4)));

                pt3D X = to_euclidean(Tr * pt);
                double dpds = m_state.K.second(0,0) * X(2) * m_state.baseline / pow(m_state.scale * X(2),2);

                ptH2D feat = m_state.K.first * Matx34d::eye() * m_state.scale *(Tr * pt);
                Point2f feat_ij(to_euclidean(feat)(0),to_euclidean(feat)(1)); feat_ij /= pyr_factor;
                ptH2D feat_ =  (m_state.K.second * Matx34d::eye()) * (m_state.scale * (Tr * pt) - Matx41d(m_state.baseline,0,0,0));
                Point2f feat_ij_(to_euclidean(feat_)(0),to_euclidean(feat_)(1)); feat_ij_ /= pyr_factor;
                if(bb.contains(feat_ij_) && bb.contains(feat_ij)){
                    Mat ROIx0 = m_obs[f_idx].first(Rect(feat_ij.x-m_state.window_size,feat_ij.y-m_state.window_size,m_state.window_size*2,m_state.window_size*2))*255;
                    Mat ROIx1 = m_obs[f_idx].second(Rect(feat_ij_.x-m_state.window_size,feat_ij_.y-m_state.window_size,m_state.window_size*2,m_state.window_size*2))*255;
                    Mat ROIx2 = m_obs[f_idx].second(Rect(feat_ij_.x+grad_int-m_state.window_size,feat_ij_.y-m_state.window_size,m_state.window_size*2,m_state.window_size*2))*255;
                    ROIx0.convertTo(ROIx0,CV_32F);
                    ROIx1.convertTo(ROIx1,CV_32F);
                    ROIx2.convertTo(ROIx2,CV_32F);
//                    Mat ROIx = ROIx2-ROIx1;
                    double ROIx = computeMutualInformation(ROIx2,ROIx0)-computeMutualInformation(ROIx1,ROIx0);
//                    float* roi_ptr = ROIx.ptr<float>();
//                    for(int k=0;k<nb_pix;k++){
                        double J = ROIx * dpds;
//                        cout << J << endl;
//                        JJ(0,0) += J * m_obs[f_idx].first.at<float>(i,j) * J;
                        JJ(0,0) += pow(J,2);
                        e(0) += J * residuals(k,j);
//                        cout << J * residuals(i,0) << endl;
//                        e(0) -= J * residuals(i,j*nb_pix+k);
//                    }
                }
            }//cout << endl;k++;
        }
        //cout << endl;
//        cout << "JJ here " << JJ(0,0) << " e " << e(0) << endl;
    }


    for(uint i=0;i<m_state.pts.second.size();i++){
        if(!m_mask.size() == 0 && !m_mask(m_state.pts.first.size()+i))
            continue;
        if(m_state.pts.second[i].isTriangulated()){
            ptH3D pt = m_state.pts.second[i].get3DLocation();
            for(uint j=0;j<m_state.pts.second[i].getNbFeatures();j++){
                int f_idx = m_state.pts.second[i].getFrameIdx(j)-fframe;
                Matx33d R = m_state.poses.second[f_idx].orientation.getR3();
                pt3D base(m_state.baseline,0,0);
                Matx31d t = m_state.poses.second[f_idx].position + R * base;
                Matx44d Tr = Matx44d::eye();
                ((Mat)R).copyTo(((Mat)Tr)(Range(0,3),Range(0,3)));
                ((Mat)t).copyTo(((Mat)Tr)(Range(0,3),Range(3,4)));

                pt3D X = to_euclidean(Tr * pt);
                double dpds = - m_state.K.second(0,0) * X(2) * m_state.baseline / pow(m_state.scale * X(2),2);

                ptH2D feat = m_state.K.first * Matx34d::eye() * m_state.scale *(Tr * pt);
                Point2f feat_ij(to_euclidean(feat)(0),to_euclidean(feat)(1)); feat_ij /= pyr_factor;
                ptH2D feat_ =  (m_state.K.first * Matx34d::eye()) * (m_state.scale * (Tr * pt) + Matx41d(m_state.baseline,0,0,0));
                Point2f feat_ij_(to_euclidean(feat_)(0),to_euclidean(feat_)(1)); feat_ij_ /= pyr_factor;
                if(bb.contains(feat_ij_) && bb.contains(feat_ij)){
                    Mat ROIx0 = m_obs[f_idx].second(Rect(feat_ij.x-m_state.window_size,feat_ij.y-m_state.window_size,m_state.window_size*2,m_state.window_size*2))*255;
                    Mat ROIx1 = m_obs[f_idx].first(Rect(feat_ij_.x-grad_int-m_state.window_size,feat_ij_.y-m_state.window_size,m_state.window_size*2,m_state.window_size*2))*255;
                    Mat ROIx2 = m_obs[f_idx].first(Rect(feat_ij_.x-m_state.window_size,feat_ij_.y-m_state.window_size,m_state.window_size*2,m_state.window_size*2))*255;
                    ROIx0.convertTo(ROIx0,CV_32F);
                    ROIx1.convertTo(ROIx1,CV_32F);
                    ROIx2.convertTo(ROIx2,CV_32F);
//                    Mat ROIx = ROIx2-ROIx1;
                    double ROIx = computeMutualInformation(ROIx2,ROIx0)-computeMutualInformation(ROIx1,ROIx0);
//                    float* roi_ptr = ROIx.ptr<float>();
//                    for(int k=0;k<nb_pix;k++){
                        double J = ROIx * dpds;
//                        JJ(0,0) += J * m_obs[f_idx].second.at<float>(i,j) * J;
                        JJ(0,0) += pow(J,2);
//                        e(0) -= J * residuals(i+m_state.pts.first.size(),j*nb_pix+k);
                        e(0) += J * residuals(k,j);
//                    }
                }
            }k++;
        }
//         cout << "JJ here " << JJ(0,0) << " e " << e(0) << endl;
    }

//    cout << JJ << " " << e <<endl;

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

        cout << "mu " << m_params.mu << endl;

        JJ.diagonal() += VectorXd::Ones(JJ.rows()) * m_params.mu;

        dX = JJ.ldlt().solve(e); // solving normal equations
        cout << "LM " <<  dX << endl;

        if(dX.norm() <= m_params.incr_tol){
            m_stop = StopCondition::SMALL_INCREMENT;
            cout << "LM SMALL INC" << endl;
            break;
        }
//        cout << "dX: " << m_params.alpha*dX << endl;
        S tmp_state = m_state;
        tmp_state.update(m_params.alpha*dX);
        MatrixXd tmp_residuals = compute_residuals(tmp_state);
        double e2 = (tmp_residuals * tmp_residuals.transpose()).diagonal().sum();

//        cout << "e1/e2 " << e1 << " " << e2 << endl;

        MatrixXd dL = (dX.transpose()*(m_params.mu*dX+e));
        double rho = (m_params.minim?-1.0:1.0) * (e2-e1);//dL(0);

        if(rho > 0){ //threshold

            cout << "accepting: " << e1 << " " << e2 << endl << m_state.show_params() << " " << tmp_state.show_params() << endl;

            m_params.mu *= max(1.0/3.0,1-pow(2*rho-1,3));
            m_params.v = 2;
            if(pow(sqrt(e1) - sqrt(e2),2) < m_params.rel_tol * sqrt(e1)){
                m_stop = StopCondition::SMALL_DECREASE_FUNCTION;
//                cout << "decrease: " << dX.norm() << " " << pow(sqrt(e1) - sqrt(e2),2) << endl;
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

//    cout << residuals << endl;
//    waitKey();

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
