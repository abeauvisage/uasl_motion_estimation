#include "stereo_viso.h"

#include "fileIO.h"
#include "utils.h"

#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

namespace me{

void StereoVisualOdometry::project3D(const vector<StereoMatchf>& features){
    m_pts3D.clear();
    m_disparities.clear();

    for(unsigned int i=0;i<features.size();i++){
        double d = (features[i].f1.x-m_param.cu1) - (features[i].f2.x-m_param.cu2);
        ptH3D pt3D((features[i].f1.x-m_param.cu1)*m_param.baseline,(features[i].f1.y-m_param.cv1)*m_param.baseline,m_param.f1*m_param.baseline, d > 0 ? d : 0.00001);
        normalize(pt3D);
        m_pts3D.push_back(pt3D);
    }
}

void StereoVisualOdometry::project3D(const vector<StereoOdoMatchesf>& features){
    m_pts3D.clear();
    m_disparities.clear();

    for(unsigned int i=0;i<features.size();i++){
        double d = (features[i].f1.x-m_param.cu1) - (features[i].f2.x-m_param.cu2);
        ptH3D pt3D((features[i].f1.x-m_param.cu1)*m_param.baseline,(features[i].f1.y-m_param.cv1)*m_param.baseline,m_param.f1*m_param.baseline, d > 0 ? d : 0.00001);
        normalize(pt3D);
        m_pts3D.push_back(pt3D);
    }
}

bool StereoVisualOdometry::process (const vector<StereoOdoMatchesf>& matches, cv::Mat init) {

    //if init not correct, initialize every state parameter to 0
    if (init.rows!=6 || init.cols!=1 || init.type() != CV_64F)
        init = Mat::zeros(6,1,CV_64F);

    // need at least 6 matches
    if(matches.size()<6)
        return false;

    m_state = init;
    m_inliers_idx.clear();

    project3D(matches);
    updateObservations(matches);

    /**** selecting matches ****/
    // if ransac, selecting 3 random matches until nb iterations has been reached
    if(m_param.ransac)
        for (int i=0;i<m_param.n_ransac;i++) {
            vector<int> selection;
            selection = selectRandomIndices(3,matches.size());
            //selecting random matches scattered in the image
            if((matches[selection[0]].f3.x*(matches[selection[1]].f3.y-matches[selection[2]].f3.y)+matches[selection[1]].f3.x*(matches[selection[2]].f3.y-matches[selection[0]].f3.y)+matches[selection[2]].f3.x*(matches[selection[0]].f3.y-matches[selection[1]].f3.y))/2 > 1000){
                m_state = init;
                if (optimize(selection,false)) { // if optimization succeeded and more inliers obtained, inliers are saved
                    vector<int> inliers_tmp = computeInliers();
                    if (inliers_tmp.size()>m_inliers_idx.size())
                        m_inliers_idx = inliers_tmp;
                }
            }
        }
    else{ // if not ransac, selecting every matches
        std::vector<int> selection(matches.size());
        std::iota (std::begin(selection), std::end(selection), 0);
        m_inliers_idx = selection;
    }

    m_state = init;

    /** final optimization **/

    if (m_inliers_idx.size()>=6){ // check that more than 6 inliers have been obtained
        if (optimize(m_inliers_idx,false)) // optimize using inliers
            return true;
        else
            return false;
    }
    else
        return false;
}

vector<int> StereoVisualOdometry::computeInliers() {

    //selecting all matches
    vector<int> selection;
    for (unsigned int i=0;i<m_obs.size();i++)
        selection.push_back(i);

    // predictions
    vector<pair<ptH2D,ptH2D>> pred = reproject(m_state,selection);
    assert(pred.size() == m_obs.size());

    //compute residual and get inlier indexes
    vector<int> inliers_idx;
    for (unsigned int i=0; i<m_obs.size(); i++){
        double score = pow(pred[i].first(0)-m_obs[i].first(0),2)+pow(pred[i].first(1)-m_obs[i].first(1),2)+pow(pred[i].second(0)-m_obs[i].second(0),2)+pow(pred[i].second(1)-m_obs[i].second(1),2);
        if (score < pow(m_param.inlier_threshold,2))
            inliers_idx.push_back(i);
    }

    return inliers_idx;
}

std::vector<std::pair<ptH2D,ptH2D>> StereoVisualOdometry::reproject(cv::Matx61d& state,  const vector<int>& selection){

    std::vector<std::pair<ptH2D,ptH2D>> reproj_pts;

    Euld r(state(0),state(1),state(2));

    /*** Tr matrix ***/

    Matx44d Tr = r.getR4();
    Tr(0,3) = state(3);
    Tr(1,3) = state(4);
    Tr(2,3) = state(5);

    //computing projection matrices
    Matx34d P1(m_param.f1,0,m_param.cu1,0,0,m_param.f1,m_param.cv1,0,0,0,1,0);
    Matx34d P2(m_param.f2,0,m_param.cu2,-m_param.baseline*m_param.f2,0,m_param.f2,m_param.cv2,0,0,0,1,0);


    for (unsigned int i=0; i<selection.size(); i++) {
        ptH3D pt = Tr*m_pts3D[selection[i]];
        ptH2D lpt=P1*pt,rpt=P2*pt;
        normalize(lpt);normalize(rpt);
        reproj_pts.push_back(pair<ptH2D,ptH2D>(lpt,rpt));
    }

    return reproj_pts;
}

vector<int> StereoVisualOdometry::selectRandomIndices(int nb_samples, int nb_tot) {

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

bool StereoVisualOdometry::optimize(const std::vector<int>& selection, bool weight){

    if (selection.size()<3) // if less than 3 points triangulation impossible
        return false;

    int k=0;
    int result=0;
    double lambda=1e-2;

    do {
        // computing residuals (error between predicted and observed features)
        vector<pair<ptH2D,ptH2D>> pred = reproject(m_state,selection);
        Mat residuals(4*selection.size(),1,CV_64F);
        for(unsigned int i=0;i<selection.size();i++){
            residuals.at<double>(i*4) = m_obs[selection[i]].first(0)-pred[i].first(0);
            residuals.at<double>(i*4+1) = m_obs[selection[i]].first(1)-pred[i].first(1);
            residuals.at<double>(i*4+2) = m_obs[selection[i]].second(0)-pred[i].second(0);
            residuals.at<double>(i*4+3) = m_obs[selection[i]].second(1)-pred[i].second(1);
        }

        //computing the jacobian
        updateJacobian(selection);

        // init
        cv::Mat A = m_J * m_J.t();
        cv::Mat B(6,1,CV_64F);
        cv::Mat X(6,1,CV_64F);

        for(int i=0;i<6;i++){
            Mat b = m_J.row(i) * residuals;
            B.at<double>(i) = b.at<double>(0);
        }

        if(m_param.method == LM)
            A += lambda * Mat::diag(A.diag());

        //solve A X = B (X = inv(J^2) * J * residual, for GN with step = 1)
        if(solve(A,B,X,DECOMP_QR)){
            if(m_param.method == GN){
                // Gauss-Newton
                m_state += (Matx61d)(X);
                double min, max;
                cv::minMaxLoc(X,&min,&max);
                if(max < m_param.eps){
                    result = 1;
                }
            }
            else{
                // Levenberg-Marquart
                Matx61d x_test = m_state + (Matx61d)(X);

                Mat r(4*selection.size(),1,CV_64F);
                vector<pair<ptH2D,ptH2D>> pred_ = reproject(x_test,selection);
                for(unsigned int i=0;i<selection.size();i++){
                    r.at<double>(i*4) = m_obs[selection[i]].first(0)-pred_[i].first(0);
                    r.at<double>(i*4+1) = m_obs[selection[i]].first(1)-pred_[i].first(1);
                    r.at<double>(i*4+2) = m_obs[selection[i]].second(0)-pred_[i].second(0);
                    r.at<double>(i*4+3) = m_obs[selection[i]].second(1)-pred_[i].second(1);
                }
                Mat rho = (residuals.t()*residuals - r.t()*r)/(X.t()*(lambda*X+B));
                if(fabs(rho.at<double>(0)) > m_param.e4){ //threshold
                    lambda = max(lambda/9,1.e-7);
                    m_state = x_test;
                }
                else
                    lambda = min(lambda*11,1.e7);

                double min, max, m1,m2;
                cv::minMaxLoc(X,&min,&max);
                cv::minMaxLoc(B,&min,&m1);
                cv::minMaxLoc((Mat)x_test/(Mat)m_state,&min,&m2);
                if(max < m_param.e1 || m1 < m_param.e2 || m2 < m_param.e3)
                    result = 1;

            }
        }else
            result = -1;
    }while(k++ < m_param.max_iter && result==0);

    if(result == -1 || k== m_param.max_iter) // if failed or reached max iterations, return false (didn't work)
        return false;
    else
        return true;
}

void StereoVisualOdometry::updateObservations(const std::vector<StereoOdoMatchesf>& matches){
    m_obs.clear();
    for(unsigned int i=0;i<matches.size();i++)
        m_obs.push_back(pair<ptH2D,ptH2D>(ptH2D(matches[i].f3.x,matches[i].f3.y,1),ptH2D(matches[i].f4.x,matches[i].f4.y,1)));
}

void StereoVisualOdometry::updateJacobian(const vector<int>& selection){

    Euld r(m_state(0),m_state(1),m_state(2));
    Matx44d Tr = r.getR4();
    Matx33d dRdx = r.getdRdr();
    Matx33d dRdy = r.getdRdp();
    Matx33d dRdz = r.getdRdy();

    Tr(0,3) = m_state(3);
    Tr(1,3) = m_state(4);
    Tr(2,3) = m_state(5);

    m_J.create(6,selection.size()*4,CV_64FC1);

    for (unsigned int i=0; i<selection.size(); i++) {

        ptH3D pt = m_pts3D[selection[i]];
        ptH3D pt_next = Tr*pt;
        normalize(pt_next);

        pt3D pt_(pt(0),pt(1),pt(2));
        pt3D dpt_next;

        for(unsigned j=0;j<6;j++){ // derivation depending on element of the state (euler angles and translation)
            switch(j){
                case 0: {dpt_next = dRdx*pt_;break;}
                case 1: {dpt_next = dRdy*pt_;break;}
                case 2: {dpt_next = dRdz*pt_;break;}
                case 3: {dpt_next = pt3D(1,0,0);break;}
                case 4: {dpt_next = pt3D(0,1,0);break;}
                case 5: {dpt_next = pt3D(0,0,1);break;}
            }
            m_J.at<double>(j,i*4) = m_param.f1*(dpt_next(0)*pt_next(2)-pt_next(0)*dpt_next(2))/(pt_next(2)*pt_next(2));
            m_J.at<double>(j,i*4+1) = m_param.f1*(dpt_next(1)*pt_next(2)-pt_next(1)*dpt_next(2))/(pt_next(2)*pt_next(2));
            m_J.at<double>(j,i*4+2) = m_param.f2*(dpt_next(0)*pt_next(2)-(pt_next(0)-m_param.baseline)*dpt_next(2))/(pt_next(2)*pt_next(2));
            m_J.at<double>(j,i*4+3) = m_param.f2*(dpt_next(1)*pt_next(2)-pt_next(1)*dpt_next(2))/(pt_next(2)*pt_next(2));
       }
    }
}

cv::Mat StereoVisualOdometry::getMotion(){

    Euld r(m_state(0),m_state(1),m_state(2));

    Matx44d Rt = r.getR4();
    Rt(0,3) = m_state(3);
    Rt(1,3) = m_state(4);
    Rt(2,3) = m_state(5);

    return (Mat)Rt;
}

}
