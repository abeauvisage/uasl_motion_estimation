#include "windowedBA.h"
#include "utils.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/viz.hpp"
#include "opencv2/calib3d.hpp"

#include <fstream>
#include <iomanip>
#include <chrono>


using namespace cv;
using namespace std;

//#define QUATERNIONS

namespace me{

static double e1=1e-5,e2=1e-3,e3=1e-1,e4=1e-8;
static int max_iter=500;
static Mat img_ = Mat::zeros(480,640,CV_8U);
static vector<double> weights_;
static int start_;
static Matx33d K_;
static int fixedF_=0;
cv::viz::Viz3d viz("Camera poses and points");

#ifdef QUATERNIONS
static vector<Quatd> quaternion_bases;
vector<Euld> euler_bis;
#endif

static StereoVisualOdometry::parameters params_stereo;
static MonoVisualOdometry::parameters params_mono;

void tukey(const cv::Mat& residuals, cv::Mat& new_residuals, const double &k, cv::Mat& W);
void tukeyMahalanobis(const cv::Mat& residuals, cv::Mat& new_residuals, const double &k, const std::vector<std::vector<cv::Matx22d>>& cov, std::vector<std::vector<cv::Matx22d>>& new_cov);
void least_squares(const cv::Mat& residuals, const double &sigma, const double &m_est_thresh, cv::Mat& W);


void compPMat(InputArray _K, InputArray _R, InputArray _t, OutputArray _P)
{
  const Mat K = _K.getMat(), R = _R.getMat(), t = _t.getMat();
  const int depth = K.depth();
  CV_Assert((K.cols == 3 && K.rows == 3) && (t.cols == 1 && t.rows == 3) && (K.size() == R.size()));
  CV_Assert((depth == CV_32F || depth == CV_64F) && depth == R.depth() && depth == t.depth());

  _P.create(3, 4, depth);

  Mat P = _P.getMat();

  // type
  if( depth == CV_32F )
  {
    hconcat(K*R, K*t, P);
  }
  else
  {
    hconcat(K*R, K*t, P);
  }

}

/* Deprecated */
void solveWindowedBA(const std::deque<std::vector<cv::Point2f>>& observations, vector<ptH3D>& pts3D, vector<double>& weights, const MonoVisualOdometry::parameters& params, const Mat& img, std::vector<Euld>& ori, std::vector<Vec3d>& pos){

    vector<ptH3D> wba_pts(pts3D.begin(),pts3D.begin()+observations[0].size());

    cout << ori.size() << " " << pos.size() << " " << observations.size() << endl;
    assert(observations.size() > 0 && wba_pts.size() == observations[0].size());
    assert(ori.size() == pos.size() && pos.size() == observations.size());
    std::cout << "[Windowed BA] " << observations.size() << " views" << std::endl;
    std::cout << "[Windowed BA] " << wba_pts.size() << " pts" << std::endl;

    params_mono = params;
    img_ = img.clone();
    weights_ = weights;

    Mat state;
    //initialize state
    state = Mat::zeros(6*observations.size()+3*wba_pts.size(),1,CV_64F);
    for(int j=0;j<ori.size();j++){
        ((Mat) ori[j].getVector()).copyTo(state.rowRange(j*6,j*6+3));
        ((Mat)(- ori[j].getR3() * pos[j])).copyTo(state.rowRange(j*6+3,j*6+6));
    }

//
//        state.at<double>(j*6+4,1) = (j)*dist(2)/(observations.size()-1);
//    }

    for(uint i=0;i<wba_pts.size();i++){
        pt3D pt = to_euclidean(wba_pts[i]);
        state.at<double>(observations.size()*6+i*3) = pt(0);
        state.at<double>(observations.size()*6+i*3+1) = pt(1);
        state.at<double>(observations.size()*6+i*3+2) = pt(2);
    }

    bool success = optimize(observations,state,Mat(),0);
    Mat residuals = compute_residuals(observations,state.rowRange(6*observations.size(),state.rows),state.rowRange(0,6*observations.size()));
    if(success){
        cout << "worked"<< endl;
        for(uint i=0;i<wba_pts.size();i++){
            weights[i] = norm(residuals.row(i));
            pts3D[i] = ptH3D(state.at<double>(observations.size()*6+i*3),state.at<double>(observations.size()*6+i*3+1),state.at<double>(observations.size()*6+i*3+2),1);
        }
        ori.clear();pos.clear();
        for(uint j=0;j<observations.size();j++){
            Vec3d vec = state.rowRange(j*6,j*6+3);
            ori.push_back(Euld(vec(0),vec(1),vec(2)));
            vec = - ori[j].getR3().t() * state.rowRange(j*6+3,j*6+6);
            pos.push_back(vec);
        }
    }
    else{
        cout << "didn't work" << endl;
        ori.clear();pos.clear();
        for(uint j=0;j<observations.size();j++){
            Vec3d vec(-1,-1,-1);
            ori.push_back(Euld());
            pos.push_back(vec);
        }
    }

//    return Vec3d(-1,-1,-1);
}

bool optimize(const std::deque<std::vector<cv::Point2f>>& obs, cv::Mat& state, const cv::Mat& visibility,int fixedFrames){


    assert(obs.size() == visibility.rows);
    assert(visibility.rows*3+visibility.cols*6 == state.rows);

    int k=0;
    int result=0;
    double v=2,tau=1e-3,mu=1e-20;
    double abs_tol=1e-6,grad_tol=1e-9,incr_tol=1e-9,rel_tol=1e-6;
    StopCondition stop=StopCondition::NO_STOP;

    Mat Xa = state.rowRange(0,6*visibility.cols);
    Mat Xb = state.rowRange(6*visibility.cols,state.rows);

    do {

        cout << "it: " << k << "\r";
        cout.flush();

        auto tp1 = chrono::steady_clock::now();
        Mat residuals = compute_residuals(obs,Xa,Xb,visibility);

//        cout << "compute residuals " << chrono::duration<double,milli>(tp2-tp1).count() << endl;

        double e1 = sum((residuals.t()*residuals).diag())[0];
        double meanReprojError = e1 / (double)(residuals.rows);
        cout << "MRE " << meanReprojError << endl;
        cout << "diff " << sum(abs(residuals))[0] << endl;

        if(meanReprojError < abs_tol)
            stop = StopCondition::SMALL_REPROJ_ERROR;

        auto tp2 = chrono::steady_clock::now();

        Mat e,ea,eb,JJ,U,V,W;vector<Matx66d>Uj;vector<Matx33d>Vi;
//        computeJacobian(Xa,Xb,residuals,JJ,U,V,W,e,visibility,fixedFrames);
        computeJacobian(Xa,Xb,residuals,Uj,Vi,W,ea,eb,visibility,fixedFrames);

//        if(max(norm(ea,NORM_INF),norm(eb,NORM_INF)) < grad_tol)
//            stop = StopCondition::SMALL_GRADIENT;

//        if(norm(e,NORM_INF) < grad_tol)
//            stop = StopCondition::SMALL_GRADIENT;

//        cv::Mat X = Mat::zeros(Xa.rows+Xb.rows,1,CV_64F);


        if(k==0){
            double min_,max_=0;
            for(uint j=0;j<Uj.size();j++){
                double _min_,_max_;
                cv::minMaxLoc(Uj[j].diag(),&_min_,&_max_);
                if(_max_ > max_)
                    max_ = _max_;
            }
            for(uint i=0;i<Vi.size();i++){
                double _min_,_max_;
                cv::minMaxLoc(Vi[i].diag(),&_min_,&_max_);
                if(_max_ > max_)
                    max_ = _max_;
            }
//            cv::minMaxLoc(JJ.diag(),&min_,&max_);
            mu = max(mu,max_);
            mu = tau * mu;
        }

        auto tp3 = chrono::steady_clock::now();

        for(;;){
//            JJ += mu * Mat::eye(JJ.size(),JJ.type());

            /*** compute Schur complement ****/

            Mat WV_inv = W.clone();
            Mat V_inv = Mat::zeros(Vi.size()*3,Vi.size()*3,CV_64F);
            for(uint i=0;i<Vi.size();i++){
                Vi[i] += mu*Matx33d::eye();
                Matx33d Vinv = Vi[i].inv();
                V_inv(Range(i*3,i*3+3),Range(i*3,i*3+3)) += (Mat)Vinv;
                for(uint j=0;j<Xa.rows/3;j++)
                    WV_inv(Range(j*3,j*3+3),Range(i*3,i*3+3)) *= (Mat) Vinv;
            }

            Mat U_=Mat::zeros(Xa.rows,Xa.rows,CV_64F);
            for(uint j=0;j<Uj.size();j++){
                Uj[j] += mu * Matx66d::eye();
                ((Mat)Uj[j]).copyTo(U_(Range(j*6,j*6+6),Range(j*6,j*6+6)));
            }
//            Mat WV_inv = W * V.inv();
            Mat schur = U_ - ( WV_inv * W.t());
            Mat schur2 = ea - WV_inv * eb;
            Mat X_a;
            if(solve(schur,schur2,X_a,DECOMP_CHOLESKY)){
            Mat X_b = V_inv *(eb-W.t()*X_a);

//            /*********************************/
//            if(solve(JJ,e,X,DECOMP_CHOLESKY)){
//                Mat X_a(Xa.rows,1,CV_64F);
//                Mat X_b(Xb.rows,1,CV_64F);
//                X(Range(0,Xa.rows),Range(0,1)).copyTo(X_a.rowRange(0,X_a.rows));
//                X(Range(Xa.rows,Xa.rows+Xb.rows),Range(0,1)).copyTo(X_b);

                if(norm(X_a)+norm(X_b) <= incr_tol * (norm(Xa)+norm(Xb))){
                    stop = StopCondition::SMALL_INCREMENT;
                    break;
                }

                Mat xa_test = Xa + X_a;
                Mat xb_test = Xb + X_b;

                Mat res_ = compute_residuals(obs,xa_test,xb_test,visibility);
                double e2 = sum((res_.t()*res_).diag())[0];
                Mat Xab;vconcat(X_a,X_b,Xab);
                Mat dL = (Xab.t()*(mu*Xab+e));
                double rho = (e1-e2)/dL.at<double>(0,0);

                if(rho > 0){ //threshold

                    mu *= max(1.0/3.0,1-pow(2*rho-1,3));
                    v = 2;
                    if(pow(sqrt(sum((residuals.t()*residuals).diag())[0]) - sqrt(sum((res_.t()*res_).diag())[0]),2) < rel_tol * sum((residuals.t()*residuals).diag())[0])
                        stop = StopCondition::SMALL_DECREASE_FUNCTION;

                    Xa = xa_test;
                    Xb = xb_test;
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
            }else{
                stop = NO_CONVERGENCE;
                cout << endl << "solve function failed (" << k << ")" << endl;
                break;
            }
        }
        auto tp4 = chrono::steady_clock::now();
        cout << "comp error: " << chrono::duration<double,milli>(tp2-tp1).count() << endl;
        cout << "comp jacob: " << chrono::duration<double,milli>(tp3-tp2).count() << endl;
        cout << "for loop: " << chrono::duration<double,milli>(tp4-tp3).count() << endl;
    }while(!(k++ < max_iter?stop:stop=MAX_ITERATIONS));

    cout << "stop condition " << stop << "(" << k << ")"<< endl;
    cout << "1: small gradient" << endl;
    cout << "2: max nb iterations reached" << endl;
    cout << "3: small decrease function" << endl;
    cout << "4: small reproj error" << endl;
    cout << "5: didn't converge" << endl;
    Xa.copyTo(state.rowRange(0,6*visibility.cols));
    Xb.copyTo(state.rowRange(6*visibility.cols,state.rows));
//     cout << "final state " << Xa.t() << endl << Xb.t() << endl;
//     showCameraPoses(Xa);
    if(stop == NO_CONVERGENCE || stop == MAX_ITERATIONS) // if failed or reached max iterations, return false (didn't work)
        return false;
    else{
        return true;
    }
}

cv::Mat compute_residuals(const std::deque<std::vector<cv::Point2f>>& observations, const cv::Mat& Xa, const cv::Mat& Xb, const cv::Mat& visibility){

    assert(observations.size() == Xb.rows/3);
    assert(visibility.rows == Xb.rows/3 && visibility.cols == Xa.rows/6);
    Mat residuals = Mat::zeros(Xa.rows/6*Xb.rows/3,2,CV_64F);

    /** computing transformation matrices **/
    vector<Mat> Trs;
    for(unsigned int j=0;j<Xa.rows/6;j++){
        Euld orientation(Xa.at<double>(j*6),Xa.at<double>(j*6+1),Xa.at<double>(j*6+2));
        Mat Tr = (Mat) orientation.getR4().t();
        Xa.rowRange(j*6+3,j*6+6).copyTo(Tr(Range(0,3),Range(3,4)));
        Trs.push_back(Tr);
    }

    for(unsigned int i=0;i<Xb.rows/3;i++){
        ptH3D pt = ptH3D(Xb.at<double>(i*3),Xb.at<double>(i*3+1),Xb.at<double>(i*3+2),1);
        int k=0;
        for(unsigned int j=0;j<Xa.rows/6;j++)
            if(visibility.at<uchar>(i,j)){
                ptH3D pt_next = (Matx44d)Trs[j]*pt;
                normalize(pt_next);
                residuals.at<double>(j*Xb.rows/3+i,0) = (observations[i][k].x-(K_(0,0) * pt_next(0)/pt_next(2) + K_(0,2)));
                residuals.at<double>(j*Xb.rows/3+i,1) = (observations[i][k].y-(K_(1,1) * pt_next(1)/pt_next(2) + K_(1,2)));
                k++;
            }
    }
    return residuals;
}

void computeJacobian(const cv::Mat& Xa,  const Mat& Xb, const Mat& residuals, Mat& JJ, Mat& U, Mat& V, Mat& W, Mat& e, const Mat& visibility, int fixedFrames){

    int m_views = Xa.rows/6, n_pts = Xb.rows/3;
    U = Mat::zeros(6*m_views,6*m_views,CV_64F);
    V = Mat::zeros(3*n_pts,3*n_pts,CV_64F);
    W = Mat::zeros(6*m_views,3*n_pts,CV_64F);
    JJ = Mat::zeros(U.rows+V.rows,U.cols+V.cols,CV_64F);
    e = Mat::zeros(JJ.rows,1,CV_64F);
    Mat cov_A=Mat::eye(2,2,CV_64F),cov_B=Mat::eye(2,2,CV_64F);

    vector<vector<Mat>> A_;vector<vector<Mat>> B_;

    for(unsigned int j=0;j<m_views;j++){

        Euld eul(Xa.at<double>(j*6,0),Xa.at<double>(j*6+1,0),Xa.at<double>(j*6+2,0));
        Mat Tr = (Mat) eul.getR4().t();
        Matx33d dRdx = (j<fixedFrames? Matx33d::eye():eul.getdRdr().t());
        Matx33d dRdy = (j<fixedFrames? Matx33d::eye():eul.getdRdp().t());
        Matx33d dRdz = (j<fixedFrames? Matx33d::eye():eul.getdRdy().t());

        Xa.rowRange(j*6+3,j*6+6).copyTo(Tr(Range(0,3),Range(3,4)));

        vector<Mat> A_j;vector<Mat> B_j;
        for(unsigned int i=0;i<n_pts;i++){

            Mat A_ij=Mat::zeros(2,6,CV_64F), B_ij=Mat::zeros(2,3,CV_64F);
            if(!visibility.at<uchar>(i,j)){
                A_j.push_back(A_ij);
                B_j.push_back(B_ij);
                continue;
            }

//            cov_B  = Mat::eye(2,2,CV_64F); //* weights_[i];
            if(j<fixedFrames)
                cov_A = Mat::zeros(2,2,CV_64F);
            else
                cov_A = Mat::eye(2,2,CV_64F);// * weights_[i];


            ptH3D pt = ptH3D(Xb.at<double>(i*3),Xb.at<double>(i*3+1),Xb.at<double>(i*3+2),1);
            ptH3D pt_next = (Matx44d)Tr*pt;
            normalize(pt_next);

            pt3D pt_(pt(0),pt(1),pt(2));
            pt3D dpt_next,dpt_b;

            for(unsigned k=0;k<6;k++){ // derivation depending on element of the state (euler angles and translation)
                switch(k){
                    case 0: {dpt_next = dRdx*pt_;dpt_b=eul.getR3().t()*pt3D(1,0,0);break;}
                    case 1: {dpt_next = dRdy*pt_;dpt_b=eul.getR3().t()*pt3D(0,1,0);break;}
                    case 2: {dpt_next = dRdz*pt_;dpt_b=eul.getR3().t()*pt3D(0,0,1);break;}
                    case 3: {dpt_next = pt3D(1,0,0);break;}
                    case 4: {dpt_next = pt3D(0,1,0);break;}
                    case 5: {dpt_next = pt3D(0,0,1);break;}
                }
                A_ij.at<double>(0,k) = params_mono.fu*(dpt_next(0)*pt_next(2)-pt_next(0)*dpt_next(2))/(pt_next(2)*pt_next(2));
                A_ij.at<double>(1,k) = params_mono.fv*(dpt_next(1)*pt_next(2)-pt_next(1)*dpt_next(2))/(pt_next(2)*pt_next(2));
                if(k<3){
                    B_ij.at<double>(0,k) = params_mono.fu*(dpt_b(0)*pt_next(2)-pt_next(0)*dpt_b(2))/(pt_next(2)*pt_next(2));
                    B_ij.at<double>(1,k) = params_mono.fv*(dpt_b(1)*pt_next(2)-pt_next(1)*dpt_b(2))/(pt_next(2)*pt_next(2));
                }
           }
           U(Range(j*6,j*6+6),Range(j*6,j*6+6)) += A_ij.t() * cov_A * A_ij;
           W(Range(j*6,j*6+6),Range(i*3,i*3+3)) += A_ij.t() * cov_A * B_ij;
           e(Range(j*6,j*6+6),Range(0,1)) += A_ij.t() * cov_A * residuals.row(j*n_pts+i).t();
//           cout << A_ij /*<< B_ij*/ << endl;
           A_j.push_back(A_ij);
           B_j.push_back(B_ij);
        }
        A_.push_back(A_j);
        B_.push_back(B_j);
    }
    for(unsigned int i=0;i<n_pts;i++)
        for(unsigned int j=0;j<m_views;j++){
            V(Range(i*3,i*3+3),Range(i*3,i*3+3)) += B_[j][i].t() * cov_B * B_[j][i];
            e(Range(6*m_views+i*3,6*m_views+i*3+3),Range(0,1)) += B_[j][i].t() * cov_B * residuals.row(j*n_pts+i).t();
        }
    U.copyTo(JJ(Range(0,6*m_views),Range(0,6*m_views)));
    V.copyTo(JJ(Range(6*m_views,6*m_views+3*n_pts),Range(6*m_views,6*m_views+3*n_pts)));
    W.copyTo(JJ(Range(0,6*m_views),Range(6*m_views,6*m_views+3*n_pts)));
    Mat Wt = W.t();
    Wt.copyTo(JJ(Range(6*m_views,6*m_views+3*n_pts),Range(0,6*m_views)));
}

void computeJacobian(const cv::Mat& Xa,  const cv::Mat& Xb, const cv::Mat& residuals, std::vector<Matx66d>& U, std::vector<Matx33d>& V, cv::Mat& W, cv::Mat& ea, cv::Mat& eb, const Mat& visibility, int fixedFrames){

    int m_views = Xa.rows/6, n_pts = Xb.rows/3;
    U = vector<Matx66d>(m_views,Matx66d::zeros());
    V = vector<Matx33d>(n_pts,Matx33d::zeros());
    W = Mat::zeros(6*m_views,3*n_pts,CV_64F);
    ea = Mat::zeros(6*m_views,1,CV_64F);
    eb = Mat::zeros(3*n_pts,1,CV_64F);
    Mat cov_A=Mat::eye(2,2,CV_64F),cov_B=Mat::eye(2,2,CV_64F);

    vector<vector<Mat>> A_;vector<vector<Mat>> B_;

    for(unsigned int j=0;j<m_views;j++){

        vector<Mat> A_j;vector<Mat> B_j;

        Euld eul(Xa.at<double>(j*6,0),Xa.at<double>(j*6+1,0),Xa.at<double>(j*6+2,0));
        Mat Tr = (Mat) eul.getR4().t();
        Matx33d dRdx = eul.getdRdr().t();
        Matx33d dRdy = eul.getdRdp().t();
        Matx33d dRdz = eul.getdRdy().t();

        Xa.rowRange(j*6+3,j*6+6).copyTo(Tr(Range(0,3),Range(3,4)));

        for(unsigned int i=0;i<n_pts;i++){

            Mat A_ij=Mat::zeros(2,6,CV_64F), B_ij=Mat::zeros(2,3,CV_64F);
            if(!visibility.at<uchar>(i,j)){
                A_j.push_back(A_ij);
                B_j.push_back(B_ij);
                continue;
            }

            ptH3D pt = ptH3D(Xb.at<double>(i*3),Xb.at<double>(i*3+1),Xb.at<double>(i*3+2),1);
            ptH3D pt_next = (Matx44d)Tr*pt;
            normalize(pt_next);

            pt3D pt_(pt(0),pt(1),pt(2));
            pt3D dpt_next,dpt_b;

            for(unsigned k=0;k<6;k++){ // derivation depending on element of the state (euler angles and translation)
                switch(k){
                    case 0: {dpt_next = dRdx*pt_;dpt_b=eul.getR3().t()*pt3D(1,0,0);break;}
                    case 1: {dpt_next = dRdy*pt_;dpt_b=eul.getR3().t()*pt3D(0,1,0);break;}
                    case 2: {dpt_next = dRdz*pt_;dpt_b=eul.getR3().t()*pt3D(0,0,1);break;}
                    case 3: {dpt_next = pt3D(1,0,0);break;}
                    case 4: {dpt_next = pt3D(0,1,0);break;}
                    case 5: {dpt_next = pt3D(0,0,1);break;}
                }
                A_ij.at<double>(0,k) = params_mono.fu*(dpt_next(0)*pt_next(2)-pt_next(0)*dpt_next(2))/(pt_next(2)*pt_next(2));
                A_ij.at<double>(1,k) = params_mono.fv*(dpt_next(1)*pt_next(2)-pt_next(1)*dpt_next(2))/(pt_next(2)*pt_next(2));
                if(k<3){
                    B_ij.at<double>(0,k) = params_mono.fu*(dpt_b(0)*pt_next(2)-pt_next(0)*dpt_b(2))/(pt_next(2)*pt_next(2));
                    B_ij.at<double>(1,k) = params_mono.fv*(dpt_b(1)*pt_next(2)-pt_next(1)*dpt_b(2))/(pt_next(2)*pt_next(2));
                }
           }
           A_j.push_back(A_ij);
           B_j.push_back(B_ij);

           if(j<fixedFrames)
             continue;

           U[j] += (Matx66d) (Mat)(A_ij.t() * A_ij);
           W(Range(j*6,j*6+6),Range(i*3,i*3+3)) += A_ij.t() * B_ij;
           ea(Range(j*6,j*6+6),Range(0,1)) += A_ij.t() * residuals.row(j*n_pts+i).t();
        }
        A_.push_back(A_j);
        B_.push_back(B_j);
    }
    for(unsigned int i=0;i<n_pts;i++)
        for(unsigned int j=0;j<m_views;j++)
            if(visibility.at<uchar>(i,j)){
                V[i] += (Matx33d) (Mat)(B_[j][i].t() * B_[j][i]);
                eb(Range(i*3,i*3+3),Range(0,1)) += B_[j][i].t() * residuals.row(j*n_pts+i).t();
            }
}

/**** WBA Points ****/

void solveWindowedBA(std::vector<WBA_Ptf*>& pts, const cv::Matx33d& K, std::vector<CamPose_qd>& poses, int fixedFrames){

    assert(fixedFrames <= poses.size());

    std::cout << "[Windowed BA] " << pts.size() << " pts" << std::endl;
    std::cout << "[Windowed BA] " << poses.size() << " views" << std::endl;
    std::cout << "[Windowed BA] " << fixedFrames << " fixed frames" << std::endl;

    int window_size = poses.size();
    K_=K;
    fixedF_ = fixedFrames;
    start_ = poses[0].ID;
    weights_.clear();

    #ifdef QUATERNIONS
    quaternion_bases.clear();
    euler_bis.clear();
    #endif

    Mat state;
    //initialize state
    state = Mat::zeros(6*window_size+3*pts.size(),1,CV_64F);

    for(int j=0;j<poses.size();j++){
        #ifdef QUATERNIONS
        quaternion_bases.push_back(poses[j].orientation.getQuat().conj());
        euler_bis.push_back(Euld(-poses[j].orientation.roll(),-poses[j].orientation.pitch(),-poses[j].orientation.yaw()));
        cout << quaternion_bases[quaternion_bases.size()-1] << " " << poses[j].orientation << endl;
        #else
        ((Mat) poses[j].orientation.conj().getEuler().getVector()).copyTo(state.rowRange(j*6,j*6+3));
        #endif
        ((Mat) (- poses[j].orientation.getR3() * poses[j].position)).copyTo(state.rowRange(j*6+3,j*6+6));
    }

    cout << "init state: " << state.rowRange(0,6*window_size).t() << endl;

    deque<vector<Point2f>> obs(pts.size());
    Mat visibility = Mat::zeros(pts.size(),window_size,CV_8U);
    for(uint i=0;i<pts.size();i++){
//        cout << pts[i]->get3DLocation().t() << endl;
        if(!pts[i]->isTriangulated())
            continue;
        ptH3D ptH = pts[i]->get3DLocation();
        pt3D pt = to_euclidean(ptH);
        state.at<double>(window_size*6+i*3) = pt(0);
        state.at<double>(window_size*6+i*3+1) = pt(1);
        state.at<double>(window_size*6+i*3+2) = pt(2);
        for(uint k=0;k<pts[i]->getNbFeatures();k++){
            obs[i].push_back(pts[i]->getFeat(k));
            visibility.at<uchar>(i,pts[i]->getFrameIdx(k)-start_) = 255;
        }
    }

    Mat Xa = state.rowRange(0,6*window_size);
    Mat Xb = state.rowRange(6*window_size,state.rows);
    cout << "initial residual " << sum(abs(compute_residuals(pts,Xa,Xb))) << endl;
    Mat residuals = compute_residuals(obs,Xa,Xb,visibility);
//    for(uint i=0;i<pts.size();i++)
//        cout << residuals.row(i+(Xa.rows/6-1)*Xb.rows/3) << " " << pts[i]->get3DLocation().t() << endl;
    cout << "init MRE " << sum((residuals.t()*residuals).diag())[0] / (double)(residuals.rows) << endl;
    cout << "bis " << sum(abs(residuals)) << endl;

    auto tp1 = chrono::steady_clock::now();
//    bool success = optimize(pts,state,window_size,fixedFrames);
    bool success = optimize(obs,state,visibility,fixedFrames);
    auto tp2 = chrono::steady_clock::now();
    cout << chrono::duration<double,milli>(tp2-tp1).count() << endl;

    if(success){

        Xa = state.rowRange(0,6*window_size);
        cout << Xa.t() << endl;
        Xb = state.rowRange(6*window_size,state.rows);
        cout << "final residual " << sum(abs(compute_residuals(pts,Xa,Xb))) << endl;
        Mat residuals = compute_residuals(obs,Xa,Xb,visibility);
//        cout << "bis " << sum(abs(residuals.col(0)))+sum(abs(residuals.col(2))) << endl;
        cout << "bis " << residuals.size() << " " << abs(residuals).size() << endl;
        cout << "final MRE " << sum((residuals.t()*residuals).diag())[0] / (double)(residuals.rows) << endl;

        for(uint i=0;i<pts.size();i++){
            Vec3d pt = Vec3d(state.at<double>(window_size*6+i*3),state.at<double>(window_size*6+i*3+1),state.at<double>(window_size*6+i*3+2));
            pts[i]->set3DLocation(ptH3D(pt(0),pt(1),pt(2),1));
        }

        for(uint j=0;j<window_size;j++){
            Vec3d vec = Vec3d(state.rowRange(j*6,j*6+3));
            #ifdef QUATERNIONS
            cout << "pose: " << state.rowRange(j*6+3,j*6+6).t() << endl;
            Euld e(vec(0),vec(1),vec(2));
            cout << "eul " << e << " Quat " << euler_bis[j] << endl;
            Mat new_rot = (Mat)(e.getR3().t() * euler_bis[j].getR3().t());
            cout << new_rot << endl;
            if(j<fixedFrames)
                cout << "new eul "<< Euld((Mat)(euler_bis[j].getR3().t())) << endl;
            else
                cout << "new eul "<< Euld(new_rot) << endl;
            poses[j].orientation = Euld(new_rot);
            cout << "ori " << poses[j].orientation <<endl;
            Matx31d t_ = (Mat)-( new_rot.t() * state.rowRange(j*6+3,j*6+6));
            poses[j].position = Vec3d(t_(0),t_(1),t_(2));
            #else
            poses[j].orientation = Euld(vec(0),vec(1),vec(2)).getQuat().conj();
            Vec3d new_pose = - (Euld(vec(0),vec(1),vec(2)).getR3() * state.rowRange(j*6+3,j*6+6));
            poses[j].position = new_pose;
            #endif
        }
    }
    else{
        cerr << "didn't work" << endl;
    }
}

bool optimize(const std::vector<WBA_Ptf*>& pts, cv::Mat& state, const int window_size, const int fixedFrames){

    assert(window_size*6+pts.size()*3 == state.rows);
    int k=0;
    int result=0;
    double v=2,tau=1e-3,mu=1e-20;
    double abs_tol=1e-6,grad_tol=1e-12,incr_tol=1e-12,rel_tol=1e-12;
    StopCondition stop=StopCondition::NO_STOP;

    Mat Xa = state.rowRange(0,6*window_size);
    Mat Xb = state.rowRange(6*window_size,state.rows);
    cout << Xa.rows << " | " << Xb.rows << endl;

    do {
//        state.rowRange(0,6*window_size);
//        vector<Matx34d> pMat = computeProjectionMatrices(Xa);
        Mat residuals = compute_residuals(pts,Xa,Xb);
//        cout << residuals.t() << endl;
//        showReprojectedPts(img_,pMat,observations,Xb);
        double e1 = sum((residuals.t()*residuals).diag())[0];
        double meanReprojError = e1 / (double)(residuals.rows);
        cout << "error " << sum((residuals.t()*residuals).diag())[0] << " " << sum(abs(residuals)) << endl;

        if(meanReprojError < abs_tol)
            stop = StopCondition::SMALL_REPROJ_ERROR;

//        cout << "Xa " << Xb.t() << endl;
        Mat e,JJ,U,V,W;
        computeJacobian(pts,Xa,Xb,residuals,JJ,U,V,W,e,fixedFrames);
//            deque<vector<Point2f>> obs(pts.size());
//        Mat visibility = Mat::zeros(pts.size(),window_size,CV_8U);
//        for(uint i=0;i<pts.size();i++){
//            for(uint k=0;k<pts[i]->getNbFeatures();k++){
//                obs[i].push_back(pts[i]->getFeat(k));
//                visibility.at<uchar>(i,pts[i]->getFrameIdx(k)-start_) = 255;
//            }
//        }
//        cout << JJ(Range(0,12),Range(0,12)) << endl;
//        cout << V(Range(0,6),Range(0,6)) << endl;
//        cout << e(Range(0,12),Range(0,1)) << endl << endl;
//        computeJacobian(Xa,Xb,residuals,JJ,U,V,W,e,visibility,fixedFrames);
//        cout << JJ(Range(0,12),Range(0,12)) << endl;
//        cout << V(Range(0,6),Range(0,6)) << endl;
//        cout << e(Range(0,12),Range(0,1)) << endl << endl;
//
//        cout << residuals.size() << endl;
//        cout << residuals(Range(158,157*2+1),Range(0,2)).t() << endl;
//        cout << e(Range(Xa.rows,Xa.rows+12),Range(0,1)) << endl;
//        waitKey();

        if(norm(e,NORM_INF) < grad_tol)
            stop = StopCondition::SMALL_GRADIENT;

        cv::Mat X = Mat::zeros(Xa.rows+Xb.rows,1,CV_64F);


        if(k==0){
            double min_,max_;
            cv::minMaxLoc(JJ.diag(),&min_,&max_);
            mu = max(mu,max_);
            mu = tau * mu;
        }

//        cout << "it: " << k << endl;
        for(;;){
//            cout << "mu: " << mu << endl;

//            U += mu * Mat::eye(U.size(),CV_64F);
//            V += mu * Mat::eye(V.size(),CV_64F);
            JJ += mu * Mat::eye(JJ.size(),JJ.type());


//
//            Mat Y = W * V.inv();
//            cv::Mat da = Mat::zeros(Xa.rows,1,CV_64F);
//            Mat eps_a = e.rowRange(Range(0,Xa.rows)) - Y * e.rowRange(Range(Xa.rows,e.rows));
////            cout << "S " << U - Y * W.t()(Range(0,12),Range(0,12)) << endl;
//            if(solve(U - Y * W.t(),eps_a,da,DECOMP_CHOLESKY)){
//
////            }
////            else
////                cout << "schur failed!" << endl;
//            Mat db = V.inv() * (e.rowRange(Range(Xa.rows,e.rows)) - W.t() * da);
//            cout << da.t() << endl << db.t() << endl;
            auto tp1 = chrono::steady_clock::now();
            if(solve(JJ,e,X,DECOMP_CHOLESKY)){

                auto tp2 = chrono::steady_clock::now();
                cout << "solve " << chrono::duration<double,milli>(tp2-tp1).count() << endl;
                Mat X_a(Xa.rows,1,CV_64F);
                Mat X_b(Xb.rows,1,CV_64F);
//                cout << X.rowRange(100,150).t() << endl;
//                Mat X_a = da;
//                ((Mat)Mat::zeros(6*fixedFrames,1,CV_64F)).copyTo(X_a.rowRange(0,6*fixedFrames));

//                Mat X_b = db;
                X(Range(0,Xa.rows),Range(0,1)).copyTo(X_a.rowRange(0,X_a.rows));
                X(Range(Xa.rows,Xa.rows+Xb.rows),Range(0,1)).copyTo(X_b);
//                da.copyTo(X(Range(0,da.rows),Range(0,1)));
//                db.copyTo(X(Range(da.rows,da.rows+db.rows),Range(0,1)));
//                cout << norm(da)+ norm(db) << " " << norm(Xa)+norm(Xb) << endl;
                if(norm(X_a)+norm(X_b) <= incr_tol * (norm(Xa)+norm(Xb))){
                    stop = StopCondition::SMALL_INCREMENT;
                    break;
                }

                Mat xa_test = Xa + X_a;
                Mat xb_test = Xb + X_b;

                Mat res_ = compute_residuals(pts,xa_test,xb_test);
                        double e2 = sum((res_.t()*res_).diag())[0];
                        Mat Xab;vconcat(X_a,X_b,Xab);
//                        cout << Xab.t() << endl;
                        Mat dL = (X.t()*(mu*X+e));
                        double rho = (e1-e2)/dL.at<double>(0,0);
//                        cout << "rho " << rho << " " << e1 << " " << e2 << endl;
//                        waitKey();
//                        cout << (residuals.t()*residuals - res_.t()*res_) << sum((residuals.t()*residuals - res_.t()*res_).diag())[0] <<  endl;
                        if(rho > 0){ //threshold

                            mu *= max(1.0/3.0,1-pow(2*rho-1,3));
                            v = 2;
//                            cout <<  << " / " << rel_tol << " - " << sum((residuals.t()*residuals).diag())[0] << " " << rel_tol * sum((residuals.t()*residuals).diag())[0] << endl;
                            if(pow(sqrt(sum((residuals.t()*residuals).diag())[0]) - sqrt(sum((res_.t()*res_).diag())[0]),2) < rel_tol * sum((residuals.t()*residuals).diag())[0])
                                stop = StopCondition::SMALL_DECREASE_FUNCTION;

                            Xa = xa_test;
                            Xb = xb_test;
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
            }else{
                stop = NO_CONVERGENCE;
                cout << endl << "solve function failed (" << k << ")" << endl;
//                cout << JJ << endl;
                break;
            }
        }
        cout << "it: " << k << "\r";
        cout.flush();
    }while(!(k++ < max_iter?stop:stop=MAX_ITERATIONS));

    cout << "stop condition " << stop << "(" << k << ")"<< endl;
    Xa.copyTo(state.rowRange(0,6*window_size));
    Xb.copyTo(state.rowRange(6*window_size,state.rows));
//     cout << "final state " << Xa.t() << endl << Xb.t() << endl;
//     showCameraPoses(Xa);
    if(stop == NO_CONVERGENCE || stop == MAX_ITERATIONS) // if failed or reached max iterations, return false (didn't work)
        return false;
    else{
        return true;
    }
}

cv::Mat compute_residuals(const std::vector<WBA_Ptf*>& pts, const cv::Mat& Xa, const cv::Mat& Xb){

    Mat residuals = Mat::zeros(Xa.rows/6*Xb.rows/3,2,CV_64F);
    vector<Matx33d> Rs; vector<Vec3d> ts;
    for(unsigned int j=0;j<Xa.rows/6;j++){
        Euld orientation(Xa.at<double>(j*6),Xa.at<double>(j*6+1),Xa.at<double>(j*6+2));
        Vec3d t(Xa.at<double>(j*6+3),Xa.at<double>(j*6+4),Xa.at<double>(j*6+5));
        #ifdef QUATERNIONS
        Matx33d R = (Mat) (orientation.getR3().t() * euler_bis[j].getR3().t());
        Rs.push_back(R);ts.push_back(t);
        #else
        Matx33d R = (Mat) orientation.getR3().t();
        Rs.push_back(R);ts.push_back(t);
        #endif
    }
//    cout << Xa.t() << endl << Xb.t() << endl;
    for(unsigned int i=0;i<pts.size();i++){
        pt3D pt = pt3D(Xb.at<double>(i*3),Xb.at<double>(i*3+1),Xb.at<double>(i*3+2));
        for(unsigned int k=0;k<pts[i]->getNbFeatures();k++){
            int j = pts[i]->getFrameIdx(k) - start_;
            if(j >= 0 && j< Xa.rows/6){
                pt3D pt_next = (Matx33d)Rs[j]*pt + ts[j];
                residuals.at<double>(j*Xb.rows/3+i,0) = (pts[i]->getFeat(k).x-(K_(0,0) * pt_next(0)/pt_next(2) + K_(0,2)));
                residuals.at<double>(j*Xb.rows/3+i,1) = (pts[i]->getFeat(k).y-(K_(1,1) * pt_next(1)/pt_next(2) + K_(1,2)));
//                pt3D pt_next = Rs[j]*pt - (Rs[j] * ts[j]);
//                pt2D res ( pts[i].getFeat(k).x -(params_mono.f * pt_next(0)/pt_next(2) + params_mono.cu),pts[i].getFeat(k).y-(params_mono.f* pt_next(1)/pt_next(2) + params_mono.cv));
//                ((Mat)res.t()).copyTo(residuals.row(i*Xa.rows/6+j));
            }
        }
    }
    return residuals;
}

void computeJacobian(const std::vector<WBA_Ptf*>& pts, const cv::Mat& Xa,  const Mat& Xb, const Mat& residuals, Mat& JJ, Mat& U, cv::Mat& V, cv::Mat& W, Mat& e,int fixedFrames){

    int m_views = Xa.rows/6, n_pts = Xb.rows/3;
    U = Mat::zeros(6*m_views,6*m_views,CV_64F);
    V = Mat::zeros(3*n_pts,3*n_pts,CV_64F);
    W = Mat::zeros(6*m_views,3*n_pts,CV_64F);
    JJ = Mat::zeros(U.rows+V.rows,U.cols+V.cols,CV_64F);
    e = Mat::zeros(JJ.rows,1,CV_64F);

    vector<vector<Mat>> A_;vector<vector<Mat>> B_;
    vector<vector<Matx33d>> Trss;vector<Vec3d> ts;

    for(unsigned int j=0;j<m_views;j++){

        vector<Matx33d> Trs;
        Euld eul(Xa.at<double>(j*6,0),Xa.at<double>(j*6+1,0),Xa.at<double>(j*6+2,0));
        Vec3d t(Xa.at<double>(j*6+3),Xa.at<double>(j*6+4),Xa.at<double>(j*6+5));
        Matx33d dRdx = eul.getdRdr();
        Matx33d dRdy = eul.getdRdp();
        Matx33d dRdz = eul.getdRdy();

//        Trs.push_back(eul.getR3());
//        Trs.push_back(dRdx);
//        Trs.push_back(dRdy);
//        Trs.push_back(dRdz);
        Trs.push_back(eul.getR3().t());
        if(j<fixedFrames){
            Trs.push_back(Matx33d::zeros());
            Trs.push_back(Matx33d::zeros());
            Trs.push_back(Matx33d::zeros());
            ts.push_back(Vec3d(0,0,0));
        }else{
        Trs.push_back(dRdx.t());
        Trs.push_back(dRdy.t());
        Trs.push_back(dRdz.t());
        ts.push_back(t);
        }
        Trss.push_back(Trs);
    }

    for(unsigned int i=0;i<n_pts;i++){
        vector<Mat> A_i;vector<Mat> B_i;
        Mat cov = Mat::eye(2,2,CV_64F); /* pts[i].getReprojError()/* pts[i].get3DLocation()(2) / (pts[i].getCount());*/
        for(unsigned int k_pt=0;k_pt<pts[i]->getNbFeatures();k_pt++){
            int j = pts[i]->getFrameIdx(k_pt) - start_;
            Mat A_ij = Mat::zeros(2,6,CV_64F), B_ij = Mat::zeros(2,3,CV_64F);
            if(j >= 0 && j<m_views && pts[i]->isTriangulated()){
                pt3D pt = pt3D(Xb.at<double>(i*3),Xb.at<double>(i*3+1),Xb.at<double>(i*3+2));
//                pt3D pt_next = Trss[j][0]*pt-(Trss[j][0]*ts[j]); // reverse
                pt3D pt_next = Trss[j][0]*pt+ts[j];   // normal

                pt3D pt_(pt(0),pt(1),pt(2));
                pt3D dpt_next,dpt_b;

                for(unsigned k=0;k<6;k++){ // derivation depending on element of the state (euler angles and translation)
                    switch(k){
//                        case 0: {dpt_next = (Matx33d)Trss[j][1]*pt_-(Trss[j][1]*ts[j]);dpt_b=(Matx33d)Trss[j][0]*pt3D(1,0,0);break;}
//                        case 1: {dpt_next = (Matx33d)Trss[j][2]*pt_-(Trss[j][2]*ts[j]);dpt_b=(Matx33d)Trss[j][0]*pt3D(0,1,0);break;}
//                        case 2: {dpt_next = (Matx33d)Trss[j][3]*pt_-(Trss[j][3]*ts[j]);dpt_b=(Matx33d)Trss[j][0]*pt3D(0,0,1);break;}
//                        case 3: {dpt_next = -Trss[j][0]*pt3D(1,0,0);break;}
//                        case 4: {dpt_next = -Trss[j][0]*pt3D(0,1,0);break;}
//                        case 5: {dpt_next = -Trss[j][0]*pt3D(0,0,1);break;}
                        #ifdef QUATERNIONS
                        case 0: {dpt_next = (Matx33d)Trss[j][1]*euler_bis[j].getR3().t()*pt_/*+((Matx33d)Trss[j][1]*ts[j])*/;dpt_b=(Matx33d)Trss[j][0]/*euler_bis[j].getR3().t()*/*pt3D(1,0,0);break;} //normal
                        case 1: {dpt_next = (Matx33d)Trss[j][2]*euler_bis[j].getR3().t()*pt_/*+((Matx33d)Trss[j][2]*ts[j])*/;dpt_b=(Matx33d)Trss[j][0]/*euler_bis[j].getR3().t()*/*pt3D(0,1,0);break;}
                        case 2: {dpt_next = (Matx33d)Trss[j][3]*euler_bis[j].getR3().t()*pt_/*+((Matx33d)Trss[j][3]*ts[j])*/;dpt_b=(Matx33d)Trss[j][0]/*euler_bis[j].getR3().t()*/*pt3D(0,0,1);break;}
                        case 3: {dpt_next = /*(Matx33d)Trss[j][0]*/pt3D(1,0,0);break;}
                        case 4: {dpt_next = /*(Matx33d)Trss[j][0]*/pt3D(0,1,0);break;}
                        case 5: {dpt_next = /*(Matx33d)Trss[j][0]*/pt3D(0,0,1);break;}
                        #else
                        case 0: {dpt_next = (Matx33d)Trss[j][1]*pt_;dpt_b=(Matx33d)Trss[j][0]*pt3D(1,0,0);break;} //normal
                        case 1: {dpt_next = (Matx33d)Trss[j][2]*pt_;dpt_b=(Matx33d)Trss[j][0]*pt3D(0,1,0);break;}
                        case 2: {dpt_next = (Matx33d)Trss[j][3]*pt_;dpt_b=(Matx33d)Trss[j][0]*pt3D(0,0,1);break;}
                        case 3: {dpt_next = pt3D(1,0,0);break;}
                        case 4: {dpt_next = pt3D(0,1,0);break;}
                        case 5: {dpt_next = pt3D(0,0,1);break;}
                        #endif
                    }
                    A_ij.at<double>(0,k) = K_(0,0)*(dpt_next(0)*pt_next(2)-pt_next(0)*dpt_next(2))/(pt_next(2)*pt_next(2));
                    A_ij.at<double>(1,k) = K_(1,1)*(dpt_next(1)*pt_next(2)-pt_next(1)*dpt_next(2))/(pt_next(2)*pt_next(2));
                    if(k<3){
                        B_ij.at<double>(0,k) = K_(0,0)*(dpt_b(0)*pt_next(2)-pt_next(0)*dpt_b(2))/(pt_next(2)*pt_next(2));
                        B_ij.at<double>(1,k) = K_(1,1)*(dpt_b(1)*pt_next(2)-pt_next(1)*dpt_b(2))/(pt_next(2)*pt_next(2));
                    }
                }
                V(Range(i*3,i*3+3),Range(i*3,i*3+3)) += B_ij.t() * cov * B_ij;
                e(Range(6*m_views+i*3,6*m_views+i*3+3),Range(0,1)) += B_ij.t() * cov * residuals.row(j*n_pts+i).t();

                A_i.push_back(A_ij.clone());
                B_i.push_back(B_ij.clone());
            }
        }
        A_.push_back(A_i);
        B_.push_back(B_i);

    }

    for(unsigned int i=0;i<n_pts;i++){
        Mat cov = Mat::eye(2,2,CV_64F); /* pts[i].getReprojError()/* pts[i].get3DLocation()(2) / (pts[i].getCount());*/
        for(unsigned int k_pt=0;k_pt<pts[i]->getNbFeatures();k_pt++){
            int j = pts[i]->getFrameIdx(k_pt) - start_;
            if(j >= fixedF_ && j<m_views && pts[i]->isTriangulated()){
                U(Range(j*6,j*6+6),Range(j*6,j*6+6)) += A_[i][k_pt].t() * cov * A_[i][k_pt];
                W(Range(j*6,j*6+6),Range(i*3,i*3+3)) += A_[i][k_pt].t() * cov * B_[i][k_pt];
                e(Range(j*6,j*6+6),Range(0,1)) += A_[i][k_pt].t() * cov * residuals.row(j*n_pts+i).t();
            }
        }
    }
    U.copyTo(JJ(Range(0,6*m_views),Range(0,6*m_views)));
    V.copyTo(JJ(Range(6*m_views,6*m_views+3*n_pts),Range(6*m_views,6*m_views+3*n_pts)));
    W.copyTo(JJ(Range(0,6*m_views),Range(6*m_views,6*m_views+3*n_pts)));
    Mat Wt = W.t();
    Wt.copyTo(JJ(Range(6*m_views,6*m_views+3*n_pts),Range(0,6*m_views)));
}

/*** On-Manifold Optim ***/

void solveWindowedBAManifold(std::vector<WBA_Ptf*>& pts, const cv::Matx33d& K, std::vector<CamPose_qd>& poses, int fixedFrames){

    assert(fixedFrames <= poses.size());

    std::cout << "[Windowed BA] " << pts.size() << " pts" << std::endl;
    std::cout << "[Windowed BA] " << poses.size() << " views" << std::endl;
    std::cout << "[Windowed BA] " << fixedFrames << " fixed frames" << std::endl;

    int window_size = poses.size();
    K_=K;
    fixedF_ = fixedFrames;
    start_ = poses[0].ID;
    weights_.clear();

    //initialize state
    vector<Matx44d> cam_poses;
    vector<pt3D> pts3D;

    for(int j=0;j<poses.size();j++){
        Mat pose = (Mat) poses[j].orientation.getR4();
        ((Mat) (- poses[j].orientation.getR3() * poses[j].position)).copyTo(pose(Range(0,3),Range(3,4)));
        cam_poses.push_back((Matx44d)pose);
    }

    deque<vector<Point2f>> obs;
    vector<vector<Matx22d>> cov(pts.size(),vector<Matx22d>(window_size,Matx22d::eye()));

    Mat visibility = Mat::zeros(pts.size(),window_size,CV_8U);
    for(uint i=0;i<pts.size();i++){
        if(!pts[i]->isTriangulated())
            continue;
//        cout << i << " | " << pts[i]->get3DLocation() << " feats" << endl;
        vector<Point2f> obs_i;vector<Matx22d> cov_i;
        ptH3D ptH = pts[i]->get3DLocation();
        pts3D.push_back(to_euclidean(ptH));
        for(uint k=0;k<pts[i]->getNbFeatures();k++){
            obs_i.push_back(pts[i]->getFeat(k));
//            cov[i][pts[i]->getFrameIdx(k)-start_] = pts[i]->getCov(k);
//            cout << "cov " << pts[i]->getCov(k) << endl;
            visibility.at<uchar>(obs.size(),pts[i]->getFrameIdx(k)-start_) = k+1;
        }
        obs.push_back(obs_i);
    }

    visibility = visibility.rowRange(0,obs.size());
    cout << "[BA] initial residual: " << sum(abs(compute_residuals(obs,cam_poses,pts3D,visibility))) << endl;
    Mat residuals = compute_residuals(obs,cam_poses,pts3D,visibility);

    cout << "[BA] initial MRE: " << sum((residuals.t()*residuals).diag())[0] / (double)(residuals.rows) << endl;
//    showReprojectedPts(img_,cam_poses,pts3D,obs,visibility);

    auto tp1 = chrono::steady_clock::now();
    bool success = optimize(obs,cam_poses,pts3D,cov,visibility,fixedFrames);
    auto tp2 = chrono::steady_clock::now();
    cout << "[BA] optim time: " << chrono::duration<double,milli>(tp2-tp1).count() << endl;

    if(success){

        cout << "[BA] final residual: " << sum(abs(compute_residuals(obs,cam_poses,pts3D,visibility))) << endl;
        Mat residuals = compute_residuals(obs,cam_poses,pts3D,visibility);
        cout << "[BA] final MRE: " << sum((residuals.t()*residuals).diag())[0] / (double)(residuals.rows) << endl;

        for(uint i=0;i<pts.size();i++){
            pts[i]->set3DLocation(to_homogeneous(pts3D[i]));
        }

        for(uint j=0;j<window_size;j++){
            poses[j].orientation = Quatd((Mat)cam_poses[j]);
            Mat new_pose = - ((Mat)cam_poses[j])(Range(0,3),Range(0,3)).t() * ((Mat)cam_poses[j])(Range(0,3),Range(3,4));
            poses[j].position = new_pose;
        }
        showReprojectedPts(img_,cam_poses,pts3D,obs,visibility);
        if( sum((residuals.t()*residuals).diag())[0] / (double)(residuals.rows) > 50.0){
            cerr << "[BA] big reproj error!" << endl;
            waitKey();
        }
    }
    else{
        cerr << "didn't work" << endl;
    }
}


bool optimize(const std::deque<std::vector<cv::Point2f>>& obs, std::vector<cv::Matx44d>& cam_poses, std::vector<pt3D>& pts3D, const std::vector<std::vector<cv::Matx22d>>& cov, const cv::Mat& visibility, const int fixedFrames){

    assert(obs.size() == pts3D.size());
    assert(visibility.rows == pts3D.size() && visibility.cols == cam_poses.size());
    int k=0;
    int result=0;
    double v=2,tau=1e-3,mu=1e-20;
    double abs_tol=1e-6,grad_tol=1e-9,incr_tol=1e-9,rel_tol=1e-9;
    StopCondition stop=StopCondition::NO_STOP;

    vector<pt3D> pts3D_ = pts3D;
    vector<Matx44d> cam_poses_ = cam_poses;

    Mat schur,WV_inv,V_inv,W;
    vector<Matx66d>Uj;vector<Matx33d>Vi;

    do {

        Mat residuals = compute_residuals(obs,cam_poses_,pts3D_,visibility),new_res;
//        Mat squared_residuals = residuals.clone();squared_residuals.mul(residuals);vconcat(squared_residuals.col(0),squared_residuals.col(1),squared_residuals);
//        cout << squared_residuals.depth() << " " << squared_residuals.size << endl;
//        double sigma = MedianAbsoluteDeviation(squared_residuals);
//        cout << "sigma " << sigma << endl;
//        Mat Weights;
//        vector<vector<Matx22d>> new_cov;
//        tukeyMahalanobis(residuals,new_res,4.685/0.6745*sigma,cov,new_cov);

        double e1 = sum((residuals.t()*residuals).diag())[0];
        double meanReprojError = e1 / (double)(residuals.rows);
//        cout << cam_poses_[cam_poses_.size()-1] << endl;
        cout << "[BA] error e1: " << meanReprojError << endl;

        if(meanReprojError < abs_tol)
            stop = StopCondition::SMALL_REPROJ_ERROR;

        Mat ea,eb;
        Mat JJ,U,V,e;

        computeJacobian(cam_poses_,pts3D_,residuals,Uj,Vi,W,ea,eb,1.0,cov,visibility,fixedFrames);
//        computeJacobian(cam_poses_,pts3D_,residuals,JJ,U,V,W,e,cov,visibility,fixedFrames);

        if(norm(ea,NORM_INF)+norm(eb,NORM_INF) < grad_tol)
            stop = StopCondition::SMALL_GRADIENT;

        cv::Mat X = Mat::zeros(cam_poses.size()*6+pts3D.size()*3,1,CV_64F);

        if(k==0){
            double min_,max_=0;
            for(uint j=0;j<Uj.size();j++){
                double _min_,_max_;
                cv::minMaxLoc(Uj[j].diag(),&_min_,&_max_);
                if(_max_ > max_)
                    max_ = _max_;
            }
            for(uint i=0;i<Vi.size();i++){
                double _min_,_max_;
                cv::minMaxLoc(Vi[i].diag(),&_min_,&_max_);
                if(_max_ > max_)
                    max_ = _max_;
            }
            mu = max(mu,max_);
            mu = tau * mu;
        }
        for(;;){
//            cout << mu << endl;
    /*** compute Schur complement ****/
//
            WV_inv = W.clone();
            V_inv = Mat::zeros(Vi.size()*3,Vi.size()*3,CV_64F);
            for(uint i=0;i<Vi.size();i++){
                Vi[i] += mu*Matx33d::eye();
                Matx33d Vinv = Vi[i].inv();
                V_inv(Range(i*3,i*3+3),Range(i*3,i*3+3)) += (Mat)Vinv;
                WV_inv.colRange(i*3,i*3+3) *= (Mat) Vinv;
            }

            Mat U_=Mat::zeros(cam_poses_.size()*6,cam_poses_.size()*6,CV_64F);
            for(uint j=0;j<Uj.size();j++){
                Uj[j] += mu * Matx66d::eye();
                ((Mat)Uj[j]).copyTo(U_(Range(j*6,j*6+6),Range(j*6,j*6+6)));
            }
            schur = U_ - ( WV_inv * W.t());
            Mat schur2 = ea - WV_inv * eb;
            Mat X_a;
            if(solve(schur,schur2,X_a,DECOMP_CHOLESKY)){

    /****  JJ ****/
//            JJ += mu * Mat::eye(JJ.size(),CV_64F);
//            if(solve(JJ,e,X,DECOMP_CHOLESKY)){
//
//                Mat X_a,X_b;
//                X_a  = X.rowRange(0,cam_poses_.size()*6).clone();
//                X_b  = X.rowRange(cam_poses_.size()*6,X.rows).clone();

                Mat X_b = V_inv *(eb-W.t()*X_a);
                if(norm(X_a)+norm(X_b) <= incr_tol ){
                    stop = StopCondition::SMALL_INCREMENT;
                    break;
                }
                vector<Matx44d> cam_poses_test; vector<pt3D> pts3D_test;
                for(int j=0;j<cam_poses_.size();j++)
                    cam_poses_test.push_back(exp_map((Matx61d)(X_a.rowRange(j*6,j*6+6))) * cam_poses_[j]);
                for(int i=0;i<pts3D_.size();i++){
                    pts3D_test.push_back(pts3D_[i]+(Matx31d) X_b.rowRange(i*3,i*3+3));
                }

                Mat res_ = compute_residuals(obs,cam_poses_test,pts3D_test,visibility),new_res_;
//                tukey(res_,new_res_,4.685/0.6745*sigma,3.0,Weights);
//                tukeyMahalanobis(res_,new_res_,4.685/0.6745*sigma,cov,new_cov);
                double e2 = sum((res_.t()*res_).diag())[0];
//                Mat dL = (X.t()*(mu*X+e));
                double rho = (e1-e2)/*dL.at<double>(0,0)*/;

//                cout << " : " << rho << endl;
//                cout << cam_poses_test[cam_poses_test.size()-1].col(3).t() << endl;

                if(rho > 0){ //threshold

                    mu *= 0.1;//max(1.0/3.0,1-pow(2*rho-1,3));
                    v = 2;
//                            cout <<  << " / " << rel_tol << " - " << sum((residuals.t()*residuals).diag())[0] << " " << rel_tol * sum((residuals.t()*residuals).diag())[0] << endl;
                    if(pow(sqrt(sum((residuals.t()*residuals).diag())[0]) - sqrt(sum((res_.t()*res_).diag())[0]),2) < rel_tol * sum((residuals.t()*residuals).diag())[0])
                        stop = StopCondition::SMALL_DECREASE_FUNCTION;

                    pts3D_ = pts3D_test;
                    cam_poses_ = cam_poses_test;
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
            }else{
                stop = NO_CONVERGENCE;
                cout << endl << "solve function failed (" << k << ")" << endl;
                break;
            }
        }
        cout << "it: " << k << "\r";
        cout.flush();
    }while(!(k++ < max_iter?stop:stop=MAX_ITERATIONS));

    cout << "stop condition " << stop << "( it " << k << ")"<< endl;
    cout << "1: small gradient" << endl;
    cout << "2: small increment" << endl;
    cout << "3: max nb iterations reached" << endl;
    cout << "4: small decrease function" << endl;
    cout << "5: small reproj error" << endl;
    cout << "6: didn't converge" << endl;

    cam_poses = cam_poses_;
    pts3D = pts3D_;

    Mat residuals = compute_residuals(obs,cam_poses,pts3D,visibility);

    Mat ea,eb;
    computeJacobian(cam_poses_,pts3D_,residuals,Uj,Vi,W,ea,eb,1.0,cov,visibility,fixedFrames);
//    computeJacobian(cam_poses_,pts3D_,residuals,JJ,U,V,W,e,cov,visibility,fixedFrames);
    residuals.mul(residuals);
    cout << "sum " << sum(residuals) << endl;
    double meanReprojError = sqrt(sum(residuals)[0]) / (double)(residuals.rows*2);
    cout << "final reproj error " << meanReprojError << endl;

    Mat WV_inv_,V_inv_,U_;
    WV_inv_ = W.clone();
    V_inv_ = Mat::zeros(Vi.size()*3,Vi.size()*3,CV_64F);
    for(uint i=0;i<Vi.size();i++){
        Matx33d Vinv = Vi[i].inv();
        ((Mat)Vinv).copyTo(V_inv_(Range(i*3,i*3+3),Range(i*3,i*3+3)));
        WV_inv_.colRange(i*3,i*3+3) *= (Mat) Vinv;
    }

    U_=Mat::zeros(cam_poses_.size()*6,cam_poses_.size()*6,CV_64F);
    for(uint j=0;j<Uj.size();j++){
        ((Mat)Uj[j]).copyTo(U_(Range(j*6,j*6+6),Range(j*6,j*6+6)));
    }

    Mat schur_ = (U_ - WV_inv_ * W.t());
    Mat cova = (schur_.t() * schur_).inv() * schur_.t(),covb;
    covb = WV_inv_.t() * cova * WV_inv_ + V_inv_;

    if(stop == NO_CONVERGENCE || stop == MAX_ITERATIONS) // if failed or reached max iterations, return false (didn't work)
        return false;
    else{
        return true;
    }
}

cv::Mat compute_residuals(const std::deque<std::vector<cv::Point2f>>& obs, const std::vector<cv::Matx44d>& poses, const std::vector<pt3D>& pts3D, const cv::Mat& visibility){

    assert(obs.size() == pts3D.size());
    assert(visibility.rows == pts3D.size() && visibility.cols == poses.size());
    Mat residuals = Mat::zeros(pts3D.size()*poses.size(),2,CV_64F);
    for(unsigned int i=0;i<pts3D.size();i++){
        ptH3D pt = to_homogeneous(pts3D[i]);
        int k=0;
        for(unsigned int j=0;j<poses.size();j++)
            if(visibility.at<uchar>(i,j)){
                ptH3D pt_next = poses[j] * pt;
                normalize(pt_next);
                Mat res = (Mat) Matx12d(obs[i][k].x-(K_(0,0) * pt_next(0)/pt_next(2) + K_(0,2)),obs[i][k].y-(K_(1,1) * pt_next(1)/pt_next(2) + K_(1,2)));
                res.copyTo(residuals.rowRange(j*pts3D.size()+i,j*pts3D.size()+i+1));
                k++;
            }
    }
    return residuals;
}
cv::Mat compute_dHdp(const Matx44d& pose, const pt3D& pt){

    Mat J_ = (Mat_<double>(2,3) <<  K_(0,0)/pt(2),      0,              -K_(0,0)*pt(0)/pow(pt(2),2),
                                    0,                  K_(1,1)/pt(2),   -K_(1,1)*pt(1)/pow(pt(2),2));

    J_ *= ((Mat)pose)(Range(0,3),Range(0,3));
    return J_;
}
cv::Mat compute_dHde(const Matx44d& pose, const pt3D& pt){

    Mat J_ = (Mat_<double>(2,6) <<  K_(0,0)/pt(2),      0,              -K_(0,0)*pt(0)/pow(pt(2),2),     -K_(0,0)*pt(0)*pt(1)/pow(pt(2),2),       K_(0,0)*(1+pow(pt(0),2)/pow(pt(2),2)),        -K_(0,0)*pt(1)/pt(2),
                                    0,                  K_(1,1)/pt(2),   -K_(1,1)*pt(1)/pow(pt(2),2),     -K_(1,1)*(1+pow(pt(1),2)/pow(pt(2),2)),  K_(1,1)*pt(0)*pt(1)/pow(pt(2),2),            K_(1,1)*pt(0)/pt(2));

    return J_;
}

void computeJacobian(const std::vector<cv::Matx44d>& cam_poses, const std::vector<pt3D>& pts3D, const cv::Mat& residuals, Mat& JJ, Mat& U, cv::Mat& V, cv::Mat& W, Mat& e,  const std::vector<std::vector<Matx22d>>& cov, const cv::Mat& visibility, int fixedFrames){

    assert(visibility.rows == pts3D.size() && visibility.cols == cam_poses.size());
    int m_views = cam_poses.size(), n_pts = pts3D.size();
    U = Mat::zeros(6*m_views,6*m_views,CV_64F);
    V = Mat::zeros(3*n_pts,3*n_pts,CV_64F);
    W = Mat::zeros(6*m_views,3*n_pts,CV_64F);
    JJ = Mat::zeros(U.rows+V.rows,U.cols+V.cols,CV_64F);
    e = Mat::zeros(JJ.rows,1,CV_64F);

    vector<vector<Mat>> A_;vector<vector<Mat>> B_;

    for(unsigned int i=0;i<n_pts;i++){
        vector<Mat> A_i;vector<Mat> B_i;
        pt3D pt = pts3D[i];
        for(unsigned int j=0;j<m_views;j++){
            Mat A_ij = Mat::zeros(2,6,CV_64F), B_ij = Mat::zeros(2,3,CV_64F);
            if(visibility.at<uchar>(i,j)){

                A_ij = compute_dHde(cam_poses[j],pt);
                B_ij = compute_dHdp(cam_poses[j],pt);

                U(Range(j*6,j*6+6),Range(j*6,j*6+6)) += A_ij.t() * (Mat) cov[i][j] * A_ij;
                W(Range(j*6,j*6+6),Range(i*3,i*3+3)) += A_ij.t() * (Mat) cov[i][j] * B_ij;
                e(Range(j*6,j*6+6),Range(0,1)) += A_ij.t() * (Mat) cov[i][j] * residuals.row(j*n_pts+i).t();



                A_i.push_back(A_ij.clone());
                B_i.push_back(B_ij.clone());
            }
        }
        A_.push_back(A_i);
        B_.push_back(B_i);

    }

    for(unsigned int i=0;i<n_pts;i++){
        int k=0;
        for(unsigned int j=0;j<m_views;j++){
            if(visibility.at<uchar>(i,j)){
                V(Range(i*3,i*3+3),Range(i*3,i*3+3)) += B_[i][k].t() * (Mat) cov[i][j] * B_[i][k];
                e(Range(6*m_views+i*3,6*m_views+i*3+3),Range(0,1)) += B_[i][k].t() * (Mat) cov[i][j] * residuals.row(j*n_pts+i).t();
                k++;
            }
        }
    }
    U.copyTo(JJ(Range(0,6*m_views),Range(0,6*m_views)));
    V.copyTo(JJ(Range(6*m_views,6*m_views+3*n_pts),Range(6*m_views,6*m_views+3*n_pts)));
    W.copyTo(JJ(Range(0,6*m_views),Range(6*m_views,6*m_views+3*n_pts)));
    Mat Wt = W.t();
    Wt.copyTo(JJ(Range(6*m_views,6*m_views+3*n_pts),Range(0,6*m_views)));

}

void computeJacobian(const std::vector<cv::Matx44d>& cam_poses, const std::vector<pt3D>& pts3D, const cv::Mat& residuals, std::vector<cv::Matx66d>& U, std::vector<cv::Matx33d>& V, cv::Mat& W, cv::Mat& ea, cv::Mat& eb, const double& sigma, const std::vector<std::vector<Matx22d>>& cov, const cv::Mat& visibility, int fixedFrames){

    int m_views = cam_poses.size(), n_pts = pts3D.size();

    assert(visibility.rows == n_pts && visibility.cols == m_views);
    assert(cov.size() == n_pts);

    U = vector<Matx66d>(m_views,Matx66d::zeros());
    V = vector<Matx33d>(n_pts,Matx33d::zeros());
    W = Mat::zeros(6*m_views,3*n_pts,CV_64F);
    ea = Mat::zeros(6*m_views,1,CV_64F);
    eb = Mat::zeros(3*n_pts,1,CV_64F);

    vector<vector<Mat>> A_;vector<vector<Mat>> B_;


    for(unsigned int j=0;j<m_views;j++){

        vector<Mat> A_j;vector<Mat> B_j;

        for(unsigned int i=0;i<n_pts;i++){

            Mat A_ij=Mat::zeros(2,6,CV_64F), B_ij=Mat::zeros(2,3,CV_64F);
            if(!visibility.at<uchar>(i,j)){
                A_j.push_back(A_ij);
                B_j.push_back(B_ij);
                continue;
            }

            pt3D pt = pts3D[i];
            A_ij = compute_dHde(cam_poses[j],pt) * 1/sigma;
            B_ij = compute_dHdp(cam_poses[j],pt) * 1/sigma;
            A_j.push_back(A_ij);
            B_j.push_back(B_ij);

           if(j<fixedFrames)
             continue;

           U[j] += (Matx66d)(Mat)(A_ij.t() * (Mat)(cov[i][j]) * A_ij);
           W(Range(j*6,j*6+6),Range(i*3,i*3+3)) += A_ij.t() * (Mat)(cov[i][j]) * B_ij;
           ea(Range(j*6,j*6+6),Range(0,1)) += A_ij.t() * (Mat)(cov[i][j]) * residuals.row(j*n_pts+i).t();
        }
        A_.push_back(A_j);
        B_.push_back(B_j);
    }
    for(unsigned int i=0;i<n_pts;i++)
        for(unsigned int j=0;j<m_views;j++)
            if(visibility.at<uchar>(i,j)){
                V[i] += (Matx33d) (Mat)(B_[j][i].t() * (Mat)(cov[i][j]) * B_[j][i]);
                eb(Range(i*3,i*3+3),Range(0,1)) += B_[j][i].t() * (Mat)(cov[i][j]) * residuals.row(j*n_pts+i).t();
            }
}

Matx33d skew(cv::Matx31d vec){

    Mat mat = (Mat_<double>(3,3) << 0,-vec(2),vec(1),vec(2),0,-vec(0),-vec(1),vec(0),0);
    return mat;
}

cv::Matx44d exp_map(const cv::Matx61d& eps){
    Mat A;
    double vx = eps(0);
    double vy = eps(1);
    double vz = eps(2);
    double vtux = eps(3);
    double vtuy = eps(4);
    double vtuz = eps(5);
    cv::Mat tu = (cv::Mat_<double>(3,1) << vtux, vtuy, vtuz); // theta u
    cv::Mat dR;
//    cv::Mat dt(3, 1, CV_64F);
    cv::Rodrigues(tu, dR);
    double theta = sqrt(tu.dot(tu));
    double sinc = (fabs(theta) < 1.0e-8) ? 1.0 : sin(theta) / theta;
    double mcosc = (fabs(theta) < 2.5e-4) ? 0.5 : (1.-cos(theta)) / theta / theta;
    double msinc = (fabs(theta) < 2.5e-4) ? (1./6.) : (theta-sin(theta)/theta) / theta / theta;

    Matx33d V = Matx33d::eye() + mcosc * skew(tu) + msinc * skew(tu) * skew(tu);

    Matx31d dt = V * Matx31d(vx,vy,vz);

    cv::hconcat(dR, dt, A);
    cv::Mat temp = (cv::Mat_<double>(1,4) << 0, 0, 0, 1);
    A.push_back(temp);

    return (Matx44d) A;
}

double MedianAbsoluteDeviation(const cv::Mat& squared_error){
    assert(squared_error.depth() == CV_64F && squared_error.cols == 1 );

    vector<double> res_squared;squared_error.copyTo(res_squared);
    if (res_squared.size() == 0)
        return -1;

    double phi_const{0.67449};  // Phi^-1(0.75);

    std::sort(res_squared.begin(), res_squared.end());

    int idx;
    double lmeds;
    if (res_squared.size() % 2 != 0) {
        idx = (res_squared.size() - 1)/2;
        lmeds = res_squared[idx];
    }
    else {
        idx = res_squared.size()/2;
        lmeds = (res_squared[idx - 1] + res_squared[idx])/2.;
    }

    double sigma = sqrt(lmeds)*1.f/phi_const;                                                               // Estimated std. dev.
    return sigma;
}

void tukeyMahalanobis(const cv::Mat& residuals, cv::Mat& new_residuals, const double &k, const std::vector<std::vector<cv::Matx22d>>& cov, std::vector<std::vector<cv::Matx22d>>& new_cov){

    new_residuals = Mat::zeros(residuals.rows*2,1,CV_64F);
    double res_norm, w_x,w_y;
    new_cov.clear();

    for(int i=0;i<cov.size();i++){
        vector<Matx22d> cov_i;
        for(int j=0;j<cov[i].size();j++){
            int row = i*cov[i].size()+j;
            res_norm = ((Mat)(residuals.row(row) * (Mat)cov[i][j] * residuals.row(row).t())).at<double>(0);
            if (res_norm <= k) {
                w_x = pow(1 - pow(residuals.at<double>(row,0)/k,2), 2);
                w_y = pow(1 - pow(residuals.at<double>(row,1)/k,2), 2);
                new_residuals.at<double>(row*2) = k*k/6 * (1 - pow(1-pow(residuals.at<double>(row,0)/k,2),3));
                new_residuals.at<double>(row*2+1) = k*k/6 * (1 - pow(1-pow(residuals.at<double>(row,1)/k,2),3));
            }
            else {
            w_x = 0.f;
            w_y = 0.f;
            new_residuals.at<double>(row*2) = k*k/6;
            new_residuals.at<double>(row*2+1) = k*k/6;
          }
          Matx22d cov_ij; cov_ij << w_y,0,0,w_y;
          cov_i.push_back(cov_ij);
        }
        new_cov.push_back(cov_i);
    }
}

void tukey(const cv::Mat& residuals, cv::Mat& new_residuals, const double &k, cv::Mat& W){

    W = cv::Mat::eye(residuals.rows*2,1,CV_64F);
    double res_norm, w_x,w_y;

    for(int i=0;i<residuals.rows;i++){
             res_norm = norm(residuals.row(i));
        if (res_norm <= k) {
            w_x = pow(1 - pow(residuals.at<double>(i,0)/k,2), 2);
            w_y = pow(1 - pow(residuals.at<double>(i,1)/k,2), 2);
            new_residuals.at<double>(i*2) = k*k/6 * (1 - pow(1-pow(residuals.at<double>(i,0)/k,2),3));
            new_residuals.at<double>(i*2+1) = k*k/6 * (1 - pow(1-pow(residuals.at<double>(i,1)/k,2),3));
        }
        else {
        w_x = 0.f;
        w_y = 0.f;
        new_residuals.at<double>(i*2) = k*k/6;
        new_residuals.at<double>(i*2+1) = k*k/6;
      }
      W.at<double>(i*2,1) = w_x;
      W.at<double>(i*2+1,1) = w_y;
    }
}

void least_squares(const cv::Mat& residuals, const double &sigma, const double &m_est_thresh, cv::Mat& W){
  W = cv::Mat::eye(residuals.rows,residuals.rows,CV_64F);
}

void showReprojectedPts(const cv::Mat& img, const std::vector<cv::Matx44d>& cam_poses, const std::vector<pt3D>& pts3D, const std::deque<std::vector<cv::Point2f>>& observations, const cv::Mat& visibility){

    for(int j=0;j<cam_poses.size();j++){
        Mat disp_img = img_.clone();
        cvtColor(disp_img,disp_img,CV_GRAY2BGR);
        Mat P;hconcat(K_,(Mat)Matx31d::zeros(), P);

        P *= (Mat) cam_poses[j];
        for(int i=0;i<observations.size();i++){
            if(visibility.at<uchar>(i,j)){
                ptH2D pt_ = (Matx34d) P * to_homogeneous(pts3D[i]);
                normalize(pt_);
                double norm_ = norm(observations[i][visibility.at<uchar>(i,j)-1]-Point2f(pt_(0),pt_(1)));
                if(norm_ > 5){
                    circle(disp_img,observations[i][visibility.at<uchar>(i,j)-1],2,Scalar(0,255,0));
                    circle(disp_img,Point2f(pt_(0),pt_(1)),2,Scalar(0,255,0));
                    line(disp_img,Point2f(pt_(0),pt_(1)),observations[i][visibility.at<uchar>(i,j)-1],Scalar(0,255,0));
                }else{
                    circle(disp_img,observations[i][visibility.at<uchar>(i,j)-1],2,Scalar(255,255,0));
                    circle(disp_img,Point2f(pt_(0),pt_(1)),2,Scalar(0,0,255));
                    line(disp_img,Point2f(pt_(0),pt_(1)),observations[i][visibility.at<uchar>(i,j)-1],Scalar(0,0,255));
                }
            }
        }
        imshow("test"+to_string(j),disp_img);
    }
    waitKey(100);
}


}
