#include "stereo_viso.h"

#include "fileIO.h"
#include "utils.h"

#include <numeric>

using namespace std;
using namespace cv;

namespace me{

void StereoVisualOdometry::project3D(const vector<StereoMatchf>& features){
    m_pts3D.clear();
    m_disparities.clear();

    for(unsigned int i=0;i<features.size();i++){
        double d = (features[i].f1.x-m_param.cu1) - (features[i].f2.x-m_param.cu2);
        ptH3D pt3D((features[i].f1.x-m_param.cu1)*m_param.baseline,(features[i].f1.y-m_param.cv1)*m_param.baseline,m_param.fu1*m_param.baseline, d > 0 ? d : 0.00001);
        normalize(pt3D);
        m_pts3D.push_back(pt3D);
    }
}

void StereoVisualOdometry::project3D(const vector<StereoOdoMatchesf>& features){
    m_pts3D.clear();
    m_disparities.clear();

    for(unsigned int i=0;i<features.size();i++){
        double d = (features[i].f1.x-m_param.cu1) - (features[i].f2.x-m_param.cu2);
        ptH3D pt3D((features[i].f1.x-m_param.cu1)*m_param.baseline,(features[i].f1.y-m_param.cv1)*m_param.baseline,m_param.fu1*m_param.baseline, d > 0 ? d : 0.00001);
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
    vector<vector<Point2f>> observations(1);
    for(uint i=0;i<m_obs.size();i++){
        ptH2D pt = m_obs[i].first;
        normalize(pt);
        observations[0].push_back(Point2f(pt(0),pt(1)));
    }

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

    cout << "[Motion Estimation] " << m_inliers_idx.size() << " inliers" << endl;

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

    Matx44d Tr = r.getR4().t();
    Tr(0,3) = state(3);
    Tr(1,3) = state(4);
    Tr(2,3) = state(5);

    //computing projection matrices
    Matx34d P1(m_param.fu1,0,m_param.cu1,0,0,m_param.fv1,m_param.cv1,0,0,0,1,0);
    Matx34d P2(m_param.fu2,0,m_param.cu2,-m_param.baseline*m_param.fu2,0,m_param.fv2,m_param.cv2,0,0,0,1,0);

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
    double v=2,tau=1e-5,mu=1e-20;
    double abs_tol=m_param.e1,grad_tol=m_param.e2,incr_tol=m_param.e3,rel_tol=m_param.e4;
    StopCondition stop=StopCondition::NO_STOP;

    do {

        // computing residuals (error between predicted and observed features)
        vector<pair<ptH2D,ptH2D>> pred = reproject(m_state,selection);
        Mat residuals(4*selection.size(),1,CV_64F);;
        for(unsigned int i=0;i<selection.size();i++){
            residuals.at<double>(i*4) = m_obs[selection[i]].first(0)-pred[i].first(0);
            residuals.at<double>(i*4+1) = m_obs[selection[i]].first(1)-pred[i].first(1);
            residuals.at<double>(i*4+2) = m_obs[selection[i]].second(0)-pred[i].second(0);
            residuals.at<double>(i*4+3) = m_obs[selection[i]].second(1)-pred[i].second(1);
        }

        double meanReprojError = sum((residuals.t()*residuals).diag())[0] / residuals.rows;

        // if mean reprojection error small enough, solution found.
        if(meanReprojError < abs_tol)
            stop = StopCondition::SMALL_REPROJ_ERROR;

        //computing the jacobian
        updateJacobian(selection);

        // init
        cv::Mat A = m_J * m_J.t();
        cv::Mat B(6,1,CV_64F);
        cv::Mat X(6,1,CV_64F);

        B = m_J * residuals;

        //if the gradient is small enough, solution found
        if(norm(B,NORM_INF) < grad_tol)
            stop =StopCondition::SMALL_GRADIENT;

        //if LM initialize mu
        if(m_param.method == Method::LM && k==0){
            double min_,max_;
            cv::minMaxLoc(A.diag(),&min_,&max_);
            mu = max(mu,max_);
            mu = tau * mu;
        }

        for(;;){

            if(m_param.method == Method::LM)
                A += mu * Mat::eye(A.size(),A.type());

            //solve A X = B (X = inv(J^2) * J * residual, for GN with step = 1)
            if(solve(A,B,X,DECOMP_QR)){

                //if solution significantly smaller than the state, minimum found
                if(norm(X) <= incr_tol * norm(m_state)){
                    stop = StopCondition::SMALL_INCREMENT;
                    break;
                }

                // magnitude of X small enough, solution found

                if(m_param.method == Method::GN){
                    // Gauss-Newton
                    m_state += (Matx61d)(X);
                    break;
                }
                else{
                    // Levenberg-Marquart
                    Matx61d x_test = m_state + (Matx61d)(X);

                    vector<pair<ptH2D,ptH2D>> pred_test = reproject(x_test,selection);
                    Mat res_test(4*selection.size(),1,CV_64F);
                    for(unsigned int i=0;i<selection.size();i++){
                        res_test.at<double>(i*4) = m_obs[selection[i]].first(0)-pred_test[i].first(0);
                        res_test.at<double>(i*4+1) = m_obs[selection[i]].first(1)-pred_test[i].first(1);
                        res_test.at<double>(i*4+2) = m_obs[selection[i]].second(0)-pred_test[i].second(0);
                        res_test.at<double>(i*4+3) = m_obs[selection[i]].second(1)-pred_test[i].second(1);
                    }

                    double rho = sum(((residuals.t()*residuals - res_test.t()*res_test)/(X.t()*(mu*X+B))).diag())[0];
                    if(rho > 0){ //threshold

                        mu *= max(0.333,1-pow(2*rho-1,3));
                        v = 2;
                        //if difference in resisuals small enough, minimum found
                        if(pow(sum((residuals.t()*residuals - res_test.t()*res_test).diag())[0],2) < rel_tol * sum((residuals.t()*residuals).diag())[0])
                            stop = StopCondition::SMALL_DECREASE_FUNCTION;

                        m_state = x_test;
                        break;
                    }
                    else{
                        mu *= v;
                        double v2 = 2*v;
                        if(v2 <= v){
                            stop = StopCondition::NO_CONVERGENCE;
                            break;
                        }
                        v = v2;
                    }
                }
            }else{
                stop = NO_CONVERGENCE;
                cout << "solve function failed" << endl;
                break;
            }
        }
    }while(!(k++ < m_param.max_iter?stop:stop=StopCondition::MAX_ITERATIONS));

    if(stop == StopCondition::NO_CONVERGENCE || stop == StopCondition::MAX_ITERATIONS) // if failed or reached max iterations, return false (didn't work)
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
    Matx44d Tr = r.getR4().t();
    Matx33d dRdx = r.getdRdr().t();
    Matx33d dRdy = r.getdRdp().t();
    Matx33d dRdz = r.getdRdy().t();

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
            m_J.at<double>(j,i*4) = m_param.fu1*(dpt_next(0)*pt_next(2)-pt_next(0)*dpt_next(2))/(pt_next(2)*pt_next(2));
            m_J.at<double>(j,i*4+1) = m_param.fv1*(dpt_next(1)*pt_next(2)-pt_next(1)*dpt_next(2))/(pt_next(2)*pt_next(2));
            m_J.at<double>(j,i*4+2) = m_param.fu2*(dpt_next(0)*pt_next(2)-(pt_next(0)-m_param.baseline)*dpt_next(2))/(pt_next(2)*pt_next(2));
            m_J.at<double>(j,i*4+3) = m_param.fv2*(dpt_next(1)*pt_next(2)-pt_next(1)*dpt_next(2))/(pt_next(2)*pt_next(2));
       }
    }
}

cv::Mat StereoVisualOdometry::getMotion(){

    Euld r(m_state(0),m_state(1),m_state(2));

    Matx44d Rt = r.getR4().t();
    Rt(0,3) = m_state(3);
    Rt(1,3) = m_state(4);
    Rt(2,3) = m_state(5);


    return (Mat)Rt;
}

}
