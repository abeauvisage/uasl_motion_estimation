#include "windowedBA.h"
#include "utils.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/viz.hpp"
#include "opencv2/calib3d.hpp"

#include <fstream>
#include <iomanip>

using namespace cv;
using namespace std;

#define QUATERNIONS

namespace me{

static double e1=1e-5,e2=1e-3,e3=1e-1,e4=1e-8;
static int max_iter=500;
static Mat img_;
static vector<double> weights_;
static int start_;
static Matx33d K_;
static int fixedF_=0;
cv::viz::Viz3d viz("Camera poses and points");

#ifdef QUATERNIONS
static vector<Quatd> quaternion_bases;
#endif

static StereoVisualOdometry::parameters params_stereo;
static MonoVisualOdometry::parameters params_mono;


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

    bool success = optimize(observations,state);
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
/* Deprecated */
bool optimize(const std::deque<std::vector<cv::Point2f>>& observations, cv::Mat& state){

    int k=0;
    int result=0;
    double v=2,tau=1e-5,mu=1e-20;
    double abs_tol=1e-1,grad_tol=1e-9,incr_tol=1e-9,rel_tol=1e-3;
    StopCondition stop=StopCondition::NO_STOP;

    Mat Xa = state.rowRange(0,6*observations.size());
    Mat Xb = state.rowRange(6*observations.size(),state.rows);

    cout << "initial state " << Xa.t() << endl /*<< endl << Xb.t() << endl*/;

    do {
        state.rowRange(0,6*observations.size());
//        vector<Matx34d> pMat = computeProjectionMatrices(Xa);
        Mat residuals = compute_residuals(observations,Xb,Xa);
//        showReprojectedPts(img_,pMat,observations,Xb);

        double meanReprojError = sum((residuals.t()*residuals).diag())[0] / residuals.rows;

        if(meanReprojError < abs_tol)
            stop = StopCondition::SMALL_REPROJ_ERROR;

        Mat e,JJ;
        computeJacobian(Xa,Xb,residuals,JJ,e);

        if(norm(e,NORM_INF) < grad_tol)
            stop =StopCondition::SMALL_GRADIENT;

        cv::Mat X = Mat::zeros(Xa.rows+Xb.rows,1,CV_64F);


        if(k==0){
            double min_,max_;
            cv::minMaxLoc(JJ.diag(),&min_,&max_);
            mu = max(mu,max_);
            mu = tau * mu;
        }


//        cout << JJ << endl;
//        waitKey();


        for(;;){

            JJ += mu * Mat::eye(JJ.size(),CV_64F);

//            Mat schur = JJ(Range(0,Xa.rows),Range(0,Xa.rows)) - JJ(Range(0,Xa.rows),Range(Xa.rows,Xa.rows+Xb.rows))*JJ(Range(Xa.rows,Xa.rows+Xb.rows),Range(Xa.rows,Xa.rows+Xb.rows)).inv()*JJ(Range(Xa.rows,Xa.rows+Xb.rows),Range(0,Xa.rows));
//            cout  << "Schur " << schur(Range(0,12),Range(0,12)) << endl;
//            solve(schur,- JJ(Range(0,Xa.rows),Range(Xa.rows,Xa.rows+Xb.rows))*JJ(Range(Xa.rows,Xa.rows+Xb.rows),Range(Xa.rows,Xa.rows+Xb.rows)).inv()*);
//        solve JJ X = e (X = inv(J^2) * J * residual, for GN with step = 1)
//        if(solve(JJ(Range(0,6*observations.size()),Range(0,6*observations.size())),e(Range(0,6*observations.size()),Range(0,1)),X_a,DECOMP_QR)){
            if(solve(JJ,e,X,DECOMP_CHOLESKY)){

                Mat X_a(Xa.rows,1,CV_64F);

                Mat X_b(Xb.rows,1,CV_64F);

                X(Range(0,Xa.rows),Range(0,1)).copyTo(X_a);
                X(Range(Xa.rows,Xa.rows+Xb.rows),Range(0,1)).copyTo(X_b);

                if(norm(X) <= incr_tol * norm(e)){
                    stop = StopCondition::SMALL_INCREMENT;
                    break;
                }

                Mat xa_test = Xa + X_a;
                Mat xb_test = Xb + X_b;

                Mat res_ = compute_residuals(observations,xb_test,xa_test);

                        Mat rho = (residuals.t()*residuals - res_.t()*res_)/(X.t()*(mu*X+e));
                        if(rho.at<double>(0,0) >0 && rho.at<double>(1,1) > 0){ //threshold

                            mu *= max(0.333,1-pow(2*rho.at<double>(0,0)/rho.at<double>(1,1)-1,3));
                            v = 2;
                            if(pow(sum((residuals.t()*residuals - res_.t()*res_).diag())[0],2) < rel_tol * sum((residuals.t()*residuals).diag())[0])
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
                cout << "solve function failed" << endl;
            }
        }
        cout << "it: " << k << "\r";
        cout.flush();
    }while(!(k++ < max_iter?stop:stop=MAX_ITERATIONS));

    cout << "stop condition " << stop << endl;
    Xa.copyTo(state.rowRange(0,6*observations.size()));
    Xb.copyTo(state.rowRange(6*observations.size(),state.rows));
     cout << "final state " << Xa.t() << endl /*<< Xb.t() << endl*/;
//     showCameraPoses(Xa);
    if(stop == NO_CONVERGENCE || stop == MAX_ITERATIONS) // if failed or reached max iterations, return false (didn't work)
        return false;
    else{
        return true;
    }
}
/* Deprecated */
cv::Mat compute_residuals(const std::deque<std::vector<cv::Point2f>>& observations, const cv::Mat& Xb, const cv::Mat& Xa){

    Mat residuals = Mat::zeros(observations.size()*Xb.rows/3,2,CV_64F);
    for(unsigned int j=0;j<observations.size();j++){
        Euld orientation(Xa.at<double>(j*6),Xa.at<double>(j*6+1),Xa.at<double>(j*6+2));
        Mat Tr = (Mat) orientation.getR4();
        Xa.rowRange(j*6+3,j*6+6).copyTo(Tr(Range(0,3),Range(3,4)));
        for(unsigned int i=0;i<Xb.rows/3;i++){
            ptH3D pt = ptH3D(Xb.at<double>(i*3),Xb.at<double>(i*3+1),Xb.at<double>(i*3+2),1);
            ptH3D pt_next = (Matx44d)Tr*pt;
            normalize(pt_next);
            residuals.at<double>(j*Xb.rows/3+i,0) = (observations[j][i].x-(params_mono.fu * pt_next(0)/pt_next(2) + params_mono.cu));
            residuals.at<double>(j*Xb.rows/3+i,1) = (observations[j][i].y-(params_mono.fv * pt_next(1)/pt_next(2) + params_mono.cv));
        }
    }
    return residuals;
}

void computeJacobian(const cv::Mat& Xa,  const Mat& Xb, const Mat& residuals, Mat& JJ, Mat& e){

    int m_views = Xa.rows/6, n_pts = Xb.rows/3;
    Mat U = Mat::zeros(6*m_views,6*m_views,CV_64F);
    Mat V = Mat::zeros(3*n_pts,3*n_pts,CV_64F);
    Mat W = Mat::zeros(6*m_views,3*n_pts,CV_64F);
    JJ = Mat::zeros(U.rows+V.rows,U.cols+V.cols,CV_64F);
    e = Mat::zeros(JJ.rows,1,CV_64F);
    Mat cov_A=Mat::eye(2,2,CV_64F),cov_B=Mat::eye(2,2,CV_64F);

    vector<vector<Mat>> A_;vector<vector<Mat>> B_;

    for(unsigned int j=0;j<m_views;j++){

        Euld eul(Xa.at<double>(j*6,0),Xa.at<double>(j*6+1,0),Xa.at<double>(j*6+2,0));
        Mat Tr = (Mat) eul.getR4();
        Matx33d dRdx = eul.getdRdr();
        Matx33d dRdy = eul.getdRdp();
        Matx33d dRdz = eul.getdRdy();
        Xa.rowRange(j*6+3,j*6+6).copyTo(Tr(Range(0,3),Range(3,4)));

        vector<Mat> A_j;vector<Mat> B_j;
        for(unsigned int i=0;i<n_pts;i++){

            cov_B  = Mat::eye(2,2,CV_64F); //* weights_[i];
            if(j==0 /*|| j == 1 */)
                cov_A = Mat::zeros(2,2,CV_64F);
            else
                cov_A = Mat::eye(2,2,CV_64F);// * weights_[i];


            Mat A_ij(2,6,CV_64F), B_ij(2,3,CV_64F);
            ptH3D pt = ptH3D(Xb.at<double>(i*3),Xb.at<double>(i*3+1),Xb.at<double>(i*3+2),1);
            ptH3D pt_next = (Matx44d)Tr*pt;
            normalize(pt_next);

            pt3D pt_(pt(0),pt(1),pt(2));
            pt3D dpt_next,dpt_b;

            for(unsigned k=0;k<6;k++){ // derivation depending on element of the state (euler angles and translation)
                switch(k){
                    case 0: {dpt_next = dRdx*pt_;dpt_b=eul.getR3()*pt3D(1,0,0);break;}
                    case 1: {dpt_next = dRdy*pt_;dpt_b=eul.getR3()*pt3D(0,1,0);break;}
                    case 2: {dpt_next = dRdz*pt_;dpt_b=eul.getR3()*pt3D(0,0,1);break;}
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

/**** WBA Points ****/

void solveWindowedBA(vector<WBA_Ptf>& pts, const Matx33d& K, const Mat& img, std::vector<Euld>& ori, std::vector<Vec3d>& pos, int start,int window_size,int fixedFrames){

    assert(ori.size() == pos.size());
    assert(fixedFrames <= window_size);

    std::cout << "[Windowed BA] " << pts.size() << " pts" << std::endl;

    K_=K;
    fixedF_ = fixedFrames;
    img_ = img.clone();
    start_ = start;
    weights_.clear();
    #ifdef QUATERNIONS
    quaternion_bases.clear();
    vector<Euld> euler_bis;
    #endif

    cout << ori.size() << endl;
    assert(ori.size() == window_size);
    cout << "Window size " << window_size << endl;
    Mat state;
    //initialize state
    state = Mat::zeros(6*window_size+3*pts.size(),1,CV_64F);

    for(int j=0;j<ori.size();j++){
        #ifdef QUATERNIONS
        quaternion_bases.push_back(ori[j].getQuat().conj());
        euler_bis.push_back(Euld((Mat)ori[j].getR3().t()));
        cout << quaternion_bases[quaternion_bases.size()-1] << " " << ori[j] << endl;
        #else
        ((Mat) -ori[j].getVector()).copyTo(state.rowRange(j*6,j*6+3));
        #endif
        ((Mat) (- ori[j].getR3() * pos[j])).copyTo(state.rowRange(j*6+3,j*6+6));
    }

    for(uint i=0;i<pts.size();i++){
        ptH3D ptH = pts[i].get3DLocation();
        pt3D pt = to_euclidean(ptH);
        state.at<double>(window_size*6+i*3) = pt(0);
        state.at<double>(window_size*6+i*3+1) = pt(1);
        state.at<double>(window_size*6+i*3+2) = pt(2);
    }

    bool success = optimize(pts,state,window_size,fixedFrames);

    if(success){
        for(uint i=0;i<pts.size();i++){
            Vec3d pt = Vec3d(state.at<double>(window_size*6+i*3),state.at<double>(window_size*6+i*3+1),state.at<double>(window_size*6+i*3+2));
            pts[i].set3DLocation(ptH3D(pt(0),pt(1),pt(2),1));
        }

        for(uint j=0;j<window_size;j++){
            Vec3d vec = Vec3d(state.rowRange(j*6,j*6+3));
            #ifdef QUATERNIONS
            Euld e(-vec(0),-vec(1),-vec(2));
            cout << "eul " << e << endl;
            Mat new_rot = (Mat)(e.getR3().t() * quaternion_bases[j].getR3().t());
            cout << new_rot << endl;
            ori[j] = Euld(new_rot);
            cout << "ori " << ori[j] <<endl;
            Matx31d t_ = (Mat)-( new_rot.t() * state.rowRange(j*6+3,j*6+6));
            pos[j] = Vec3d(t_(0),t_(1),t_(2));
            #else
            ori[j] = (Euld(-vec(0),-vec(1),-vec(2)));
            vec = - (Euld(-vec(0),-vec(1),-vec(2)).getR3().t() * state.rowRange(j*6+3,j*6+6));
            pos[j] = vec;
            #endif
        }
    }
    else{
        cerr << "didn't work" << endl;
    }
}

bool optimize(const std::vector<WBA_Ptf>& pts, cv::Mat& state, const int window_size, const int fixedFrames){

    int k=0;
    int result=0;
    double v=2,tau=1e-3,mu=1e-20;
    double abs_tol=1e-3,grad_tol=1e-9,incr_tol=1e-9,rel_tol=1e-3;
    StopCondition stop=StopCondition::NO_STOP;

    Mat Xa = state.rowRange(0,6*window_size);
    Mat Xb = state.rowRange(6*window_size,state.rows);

    do {
//        state.rowRange(0,6*window_size);
//        vector<Matx34d> pMat = computeProjectionMatrices(Xa);
        Mat residuals = compute_residuals(pts,Xa,Xb);
//        cout << residuals.t() << endl;
//        showReprojectedPts(img_,pMat,observations,Xb);
        double e1 = sum((residuals.t()*residuals).diag())[0];
        double meanReprojError = e1 / (double)(residuals.rows);
//        cout << "error " << sum((residuals.t()*residuals).diag())[0] << " " << residuals.rows << " " <<  meanReprojError << endl;

        if(meanReprojError < abs_tol)
            stop = StopCondition::SMALL_REPROJ_ERROR;

//        cout << "Xa " << Xb.t() << endl;
        Mat e,JJ,U,V,W;
        computeJacobian(pts,Xa,Xb,residuals,JJ,U,V,W,e);
//        cout << JJ(Range(0,12),Range(0,12)) << endl;
//        cout << V(Range(0,6),Range(0,6)) << endl;
//        cout << e(Range(0,12),Range(0,1)) << endl << endl;
////
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

            if(solve(JJ,e,X,DECOMP_CHOLESKY)){

                Mat X_a(Xa.rows,1,CV_64F);
                Mat X_b(Xb.rows,1,CV_64F);
//                cout << X.rowRange(100,150).t() << endl;
//                Mat X_a = da;
////                ((Mat)Mat::zeros(60,1,CV_64F)).copyTo(X_a.rowRange(0,60));

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

cv::Mat compute_residuals(const std::vector<WBA_Ptf>& pts, const cv::Mat& Xa, const cv::Mat& Xb){

    Mat residuals = Mat::zeros(Xa.rows/6*Xb.rows/3,2,CV_64F);
    vector<Matx33d> Rs; vector<Vec3d> ts;
    for(unsigned int j=0;j<Xa.rows/6;j++){
        Euld orientation(Xa.at<double>(j*6),Xa.at<double>(j*6+1),Xa.at<double>(j*6+2));
        Vec3d t(Xa.at<double>(j*6+3),Xa.at<double>(j*6+4),Xa.at<double>(j*6+5));
        #ifdef QUATERNIONS
        Matx33d R = (Mat) (orientation.getR3().t() * quaternion_bases[j].getR3().t());
        Rs.push_back(R);ts.push_back(orientation.getR3().t() * t);
        #else
        Matx33d R = (Mat) orientation.getR3().t();
        Rs.push_back(R);ts.push_back(t);
        #endif
    }
    for(unsigned int i=0;i<pts.size();i++){
        pt3D pt = pt3D(Xb.at<double>(i*3),Xb.at<double>(i*3+1),Xb.at<double>(i*3+2));
        for(unsigned int k=0;k<pts[i].getNbFeatures();k++){
            int j = pts[i].getFrameIdx(k) - start_;
            if(j >= 0){
                pt3D pt_next = (Matx33d)Rs[j]*pt + ts[j];
                residuals.at<double>(j*Xb.rows/3+i,0) = (pts[i].getFeat(k).x-(K_(0,0) * pt_next(0)/pt_next(2) + K_(0,2)));
                residuals.at<double>(j*Xb.rows/3+i,1) = (pts[i].getFeat(k).y-(K_(1,1) * pt_next(1)/pt_next(2) + K_(1,2)));
//                pt3D pt_next = Rs[j]*pt - (Rs[j] * ts[j]);
//                pt2D res ( pts[i].getFeat(k).x -(params_mono.f * pt_next(0)/pt_next(2) + params_mono.cu),pts[i].getFeat(k).y-(params_mono.f* pt_next(1)/pt_next(2) + params_mono.cv));
//                ((Mat)res.t()).copyTo(residuals.row(i*Xa.rows/6+j));
            }
        }
    }
    return residuals;
}

void computeJacobian(const std::vector<WBA_Ptf>& pts, const cv::Mat& Xa,  const Mat& Xb, const Mat& residuals, Mat& JJ, Mat& U, cv::Mat& V, cv::Mat& W, Mat& e){

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
        ts.push_back(t);
//        Trs.push_back(eul.getR3());
//        Trs.push_back(dRdx);
//        Trs.push_back(dRdy);
//        Trs.push_back(dRdz);
        Trs.push_back(eul.getR3().t());
        Trs.push_back(dRdx.t());
        Trs.push_back(dRdy.t());
        Trs.push_back(dRdz.t());
        Trss.push_back(Trs);
    }

    for(unsigned int i=0;i<n_pts;i++){
        vector<Mat> A_i;vector<Mat> B_i;
        Mat cov = Mat::eye(2,2,CV_64F); /* pts[i].getReprojError()/* pts[i].get3DLocation()(2) / (pts[i].getCount());*/
        for(unsigned int k_pt=0;k_pt<pts[i].getNbFeatures();k_pt++){
            int j = pts[i].getFrameIdx(k_pt) - start_;
            Mat A_ij = Mat::zeros(2,6,CV_64F), B_ij = Mat::zeros(2,3,CV_64F);
            if(j >= 0 && pts[i].isTriangulated()){
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
                        case 0: {dpt_next = (Matx33d)Trss[j][1]*quaternion_bases[j].getR3().t()*pt_+((Matx33d)Trss[j][1]*ts[j]);dpt_b=(Matx33d)Trss[j][0]*quaternion_bases[j].getR3().t()*pt3D(1,0,0);break;} //normal
                        case 1: {dpt_next = (Matx33d)Trss[j][2]*quaternion_bases[j].getR3().t()*pt_+((Matx33d)Trss[j][2]*ts[j]);dpt_b=(Matx33d)Trss[j][0]*quaternion_bases[j].getR3().t()*pt3D(0,1,0);break;}
                        case 2: {dpt_next = (Matx33d)Trss[j][3]*quaternion_bases[j].getR3().t()*pt_+((Matx33d)Trss[j][3]*ts[j]);dpt_b=(Matx33d)Trss[j][0]*quaternion_bases[j].getR3().t()*pt3D(0,0,1);break;}
                        case 3: {dpt_next = (Matx33d)Trss[j][0]*pt3D(1,0,0);break;}
                        case 4: {dpt_next = (Matx33d)Trss[j][0]*pt3D(0,1,0);break;}
                        case 5: {dpt_next = (Matx33d)Trss[j][0]*pt3D(0,0,1);break;}
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
        for(unsigned int k_pt=0;k_pt<pts[i].getNbFeatures();k_pt++){
            int j = pts[i].getFrameIdx(k_pt) - start_;
            if(j >= fixedF_ && pts[i].isTriangulated()){
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

/**** Stereo ****/
/* Deprecated */
cv::Vec3d solveWindowedBA(const std::vector<std::vector<std::pair<me::ptH2D,me::ptH2D>>>& observations,const std::vector<ptH3D>& pts3D, const StereoVisualOdometry::parameters& params, const cv::Mat& img){
    assert(observations.size() > 0 && pts3D.size() == observations[0].size());
    std::cout << "[Windowed BA] " << observations.size() << " views" << std::endl;
    std::cout << "[Windowed BA] " << pts3D.size() << " pts" << std::endl;

    params_stereo = params;
    img_ = img.clone();

    Mat state = Mat::zeros(6*observations.size()/*+3*pts3D.size()*/,1,CV_64F);
    bool success = optimize_stereo(observations,pts3D,state);
    if(success){
        cout << "worked"<< endl;
        return Vec3d(state.at<double>(6*observations.size()-3),state.at<double>(6*observations.size()-2),state.at<double>(6*observations.size()-1));
    }
    else
        cout << "didn't work" << endl;


}
/* Deprecated */
bool optimize_stereo(const std::vector<std::vector<std::pair<me::ptH2D,me::ptH2D>>>& observations,const std::vector<ptH3D>& pts3D, cv::Mat& state){
    int k=0;
    int result=0;
    double lambda=1e-2;

    do {

        cout << "step " << k << endl;

        vector<Matx34d> pMat = computeProjectionMatrices(state);
        Mat residuals = compute_residuals_stereo(observations,pts3D,state);
//        showReprojectedPts(img_,pMat,observations,pts3D);
//        showCameraPoses(state(Range(0,6*observations.size()),Range(0,1)));

//        cout << residuals.rows << " obs" << endl;
//         cout << "residuals " << e << endl;
//        cout << "sums " << sum(residuals.col(0)) << " " << sum(residuals.col(1)) << " " << sum(residuals.col(2)) << " " << sum(residuals.col(3)) << " " << endl;


        Mat e,JJ;
        computeJacobian_stereo(state(Range(0,6*observations.size()),Range(0,1)),pts3D,residuals,JJ,e);
//        cout << "JJ " << JJ << endl << e.t() << endl;
//        waitKey();
//        cv::Mat X = Mat::zeros(6*observations.size()+3*pts3D.size(),1,CV_64F);
        cv::Mat X_(6*observations.size(),1,CV_64F);

//        cout << JJ(Range(0,6*observations.size()),Range(0,6*observations.size())) << endl;
        //solve JJ X = e (X = inv(J^2) * J * residual, for GN with step = 1)
        if(solve(JJ,e,X_,DECOMP_QR)){
//            cout << "X " << X_.t() << endl;
            double min,max,alpha=1;
            Mat res_,x_test;
            cout << X_.t() << endl;
            int it=0;
//                do{
                cout << "norm " << norm(X_) << endl;
                x_test = state + alpha * X_;
                vector<Matx34d> pMat_ = computeProjectionMatrices(x_test);
                res_ = compute_residuals_stereo(observations,pts3D,x_test);
//                showReprojectedPts(img_,pMat_,observations,pts3D);
                alpha /= 2;
//                    cout << "res " << res_.t() << endl;
                cout << "sum res " << sum((residuals.t()*residuals).diag()) << " " << sum((res_.t()*res_).diag()) << endl;
                it++;
//                }while(sum((residuals.t()*residuals).diag())[0] < sum((res_.t()*res_).diag())[0] && it < 15);
//                cout << "alpha " << alpha << endl;
//                waitKey();
            state  = x_test;
            cv::minMaxLoc(abs(X_),&min,&max);
            if(max < e1 || alpha < e4)
                result =1;

        }else{
            result = -1;
            cout << "solve failed" << endl;
        }
    }while(k++ < max_iter && result==0);

    cout << "result: " << result << endl;
    vector<Matx34d> pMat = computeProjectionMatrices(state(Range(0,6*observations.size()),Range(0,1)));
//    showReprojectedPts(img_,pMat,observations,pts3D);
    showCameraPoses(state(Range(0,6*observations.size()),Range(0,1)));
    cv::waitKey();

    if(result == -1 || k== max_iter) // if failed or reached max iterations, return false (didn't work)
        return false;
    else
        return true;
}


std::vector<Matx34d> computeProjectionMatrices(const cv::Mat& Xa){

        assert(!Xa.empty());

        vector<Matx34d> pMat;
        Mat P;
        for(unsigned int i=0;i<Xa.rows/6;i++){
            Euld orientation(Xa.at<double>(i*6),Xa.at<double>(i*6+1),Xa.at<double>(i*6+2));
            Matx33d K = Matx33d::eye();
            K(0,0) = params_mono.fu;K(0,2) = params_mono.cu;K(1,1) = params_mono.fv;K(1,2) = params_mono.cv;
            compPMat(K,orientation.getR3(),Xa.rowRange(i*6+3,i*6+6),P);
            pMat.push_back((Matx34d)P);
        }
        return pMat;
}
/* Deprecated */
cv::Mat compute_residuals_stereo(const std::vector<std::vector<std::pair<me::ptH2D,me::ptH2D>>>& observations, const std::vector<ptH3D>& pts3D, const cv::Mat& Xa){

    Mat residuals = Mat::zeros(observations.size()*pts3D.size(),4,CV_64F);
    for(unsigned int j=0;j<observations.size();j++){
        Euld orientation(Xa.at<double>(j*6),Xa.at<double>(j*6+1),Xa.at<double>(j*6+2));
        Mat Tr = (Mat) orientation.getR4();
        Xa.rowRange(j*6+3,j*6+6).copyTo(Tr(Range(0,3),Range(3,4)));

        for(unsigned int i=0;i<observations[j].size();i++){
            ptH3D pt = pts3D[i];
            ptH3D pt_next = (Matx44d)Tr*pt;
            normalize(pt_next);
            ptH2D ob1 = observations[j][i].first, ob2 = observations[j][i].second;
            normalize(ob1);normalize(ob2);

            residuals.at<double>(j*pts3D.size()+i,0) = (ob1(0)-(params_stereo.fu1 * pt_next(0)/pt_next(2) + params_stereo.cu1));
            residuals.at<double>(j*pts3D.size()+i,1) = (ob1(1)-(params_stereo.fv1 * pt_next(1)/pt_next(2) + params_stereo.cv1));
            residuals.at<double>(j*pts3D.size()+i,2) = (ob2(0)-(params_stereo.fu2 * (pt_next(0)-params_stereo.baseline)/pt_next(2) + params_stereo.cu2));
            residuals.at<double>(j*pts3D.size()+i,3) = (ob2(1)-(params_stereo.fv2 * pt_next(1)/pt_next(2) + params_stereo.cv2));
        }
    }
    return residuals;
}

/* Deprecated */
void computeJacobian_stereo(const cv::Mat& Xa,  const std::vector<ptH3D>& pts, const Mat& residuals, Mat& JJ, Mat& e){
    int m_views = Xa.rows/6, n_pts = pts.size();
    Mat U = Mat::zeros(6*m_views,6*m_views,CV_64F);
    Mat V = Mat::zeros(3*n_pts,3*n_pts,CV_64F);
    Mat W = Mat::zeros(6*m_views,3*n_pts,CV_64F);
//    JJ = Mat::zeros(U.rows+V.rows,U.cols+V.cols,CV_64F);
//    e.create(JJ.rows,1,CV_64F);
    JJ = Mat::zeros(U.rows,U.cols,CV_64F);
    e = Mat::zeros(JJ.rows,1,CV_64F);

    vector<vector<Mat>> A_;vector<vector<Mat>> B_;

    for(unsigned int j=0;j<m_views;j++){
        Euld eul(Xa.at<double>(j*6,0),Xa.at<double>(j*6+1,0),Xa.at<double>(j*6+2,0));
        Mat Tr = (Mat) eul.getR4();
        Matx33d dRdx = eul.getdRdr();
        Matx33d dRdy = eul.getdRdp();
        Matx33d dRdz = eul.getdRdy();
        Xa.rowRange(j*6+3,j*6+6).copyTo(Tr(Range(0,3),Range(3,4)));

        vector<Mat> A_j;vector<Mat> B_j;
        for(unsigned int i=0;i<n_pts;i++){
            Mat A_ij(4,6,CV_64F);
            ptH3D pt = pts[i];
            ptH3D pt_next = (Matx44d)Tr*pt;
            normalize(pt_next);

            pt3D pt_(pt(0),pt(1),pt(2));
            pt3D dpt_next,dpt_b;

            for(unsigned k=0;k<6;k++){ // derivation depending on element of the state (euler angles and translation)
                switch(k){
                    case 0: {dpt_next = dRdx*pt_;dpt_b=eul.getR3()*pt3D(1,0,0);break;}
                    case 1: {dpt_next = dRdy*pt_;dpt_b=eul.getR3()*pt3D(0,1,0);break;}
                    case 2: {dpt_next = dRdz*pt_;dpt_b=eul.getR3()*pt3D(0,0,1);break;}
                    case 3: {dpt_next = pt3D(1,0,0);break;}
                    case 4: {dpt_next = pt3D(0,1,0);break;}
                    case 5: {dpt_next = pt3D(0,0,1);break;}
                }
//                A_ij.at<double>(0,k) = K_(0,0)*(dpt_next(0)*pt_next(2)-pt_next(0)*dpt_next(2))/(pt_next(2)*pt_next(2));
//                A_ij.at<double>(1,k) = K_(1,1)*(dpt_next(1)*pt_next(2)-pt_next(1)*dpt_next(2))/(pt_next(2)*pt_next(2));
                A_ij.at<double>(0,k) = params_stereo.fu1*(dpt_next(0)*pt_next(2)-pt_next(0)*dpt_next(2))/(pt_next(2)*pt_next(2));
                A_ij.at<double>(1,k) = params_stereo.fv1*(dpt_next(1)*pt_next(2)-pt_next(1)*dpt_next(2))/(pt_next(2)*pt_next(2));
                A_ij.at<double>(2,k) = params_stereo.fu2*(dpt_next(0)*pt_next(2)-(pt_next(0)-params_stereo.baseline)*dpt_next(2))/(pt_next(2)*pt_next(2));
                A_ij.at<double>(3,k) = params_stereo.fv2*(dpt_next(1)*pt_next(2)-pt_next(1)*dpt_next(2))/(pt_next(2)*pt_next(2));
           }
           U(Range(j*6,j*6+6),Range(j*6,j*6+6)) += A_ij.t() * A_ij;
//           W(Range(j*6,j*6+6),Range(i*3,i*3+3)) += A_ij.t() * B_ij;
           e(Range(j*6,j*6+6),Range(0,1)) += A_ij.t() * residuals.row(j*n_pts+i).t();
           A_j.push_back(A_ij);
        }
        A_.push_back(A_j);
    }
    U.copyTo(JJ(Range(0,6*m_views),Range(0,6*m_views)));
}

std::deque<std::vector<cv::Point2f>> project_pts(const cv::Mat& Xb, const std::vector<cv::Matx34d>& pMat){
    deque<vector<Point2f>> pts2D;
    for(unsigned int i=0;i<pMat.size();i++){
        vector<Point2f> currentView;
//        cout << "Pi " << pMat[i] << endl;
        for(unsigned int j=0;j<Xb.rows/3;j++){
                ptH2D newPT = pMat[i] * ptH3D(Xb.at<double>(j*3),Xb.at<double>(j*3+1),Xb.at<double>(j*3+2),1);
                normalize(newPT);
//                if(i==0)
//                    cout << "new pt " << newPT.t() << endl;
                currentView.push_back(Point2f(newPT(0),newPT(1)));
        }
        pts2D.push_back(currentView);
    }
    return pts2D;
}

void showReprojectedPts(const cv::Mat& img, const std::vector<cv::Matx34d>& pMat, const deque<std::vector<cv::Point2f>>& observations,const Mat& Xb){

    deque<vector<Point2f>> pts2D = project_pts(Xb,pMat);
    for(int j=0;j<pMat.size();j++){
        Mat disp_img = img_.clone();
        cvtColor(disp_img,disp_img,CV_GRAY2BGR);
        for(int i=0;i<Xb.rows/3;i++){
            circle(disp_img,observations[j][i],2,Scalar(255,0,0));
            circle(disp_img,pts2D[j][i],2,Scalar(0,0,255));
        }
        imshow("test"+to_string(j),disp_img);
    }
    waitKey(1000);
}

void showCameraPoses(const Mat& Xa){

    cv::viz::Viz3d viz("Camera poses");
    viz.showWidget("Coordinate system", viz::WCoordinateSystem());

    for(uint i=0;i<Xa.rows/6;i++){
        viz::WCameraPosition cpw;
        cpw = viz::WCameraPosition(Vec2f(1,0.5));
        Euld orientation(Xa.at<double>(i*6),Xa.at<double>(i*6+1),Xa.at<double>(i*6+2));
        Mat cameraPose = (Mat) orientation.getR4();
        Xa(Range(i*6+3,i*6+6),Range(0,1)).copyTo(cameraPose(Range(0,3),Range(3,4)));
        viz.showWidget("Camera Widget"+to_string(i),cpw,Affine3d(cameraPose.inv()));
    }
    viz.spinOnce(0);
}

void showCameraPosesAndPoints(const std::vector<Euld>& ori ,const std::vector<cv::Vec3d>& pos, const std::vector<WBA_Ptf>& pts){

    viz.showWidget("Coordinate system", viz::WCoordinateSystem(1));

    for(int i=0;i<ori.size();i++){
        Mat cameraPose = (Mat) ori[i].getR4().t();
        Vec3d position = pos[i];
        Mat(position).copyTo(cameraPose(Range(0,3),Range(3,4)));
        viz::WCameraPosition cpw;
        cpw = viz::WCameraPosition(Vec2f(1,0.5));
        viz.showWidget("Camera Widget"+to_string(i),cpw,Affine3d(cameraPose));
    }

    Mat cloud = Mat::zeros(pts.size(),1,CV_64FC3);
    for(uint i=0;i<pts.size();i++){
        ptH3D pt = pts[i].get3DLocation();
        if(pt(2)>0 && pt(2) < 20)
            cloud.at<Matx31d>(i) = to_euclidean(pt);
    }

    Vec3d cam_pos(10,-10,-10), cam_focal_point(0,0,0), cam_y_dir(0,1,0);
    Affine3f cam_pose = viz::makeCameraPose(Vec3d(pos[pos.size()-1]+cam_pos),pos[pos.size()-1],cam_y_dir);

    viz.setViewerPose(cam_pose);

    viz::WCloud wcloud(cloud);
    viz.showWidget("CLOUD",wcloud);
//    viz.spinOnce(100);
    viz.spin();
}

void showCameraPosesAndPoints(const Euld& orientation,const cv::Vec3d& position, const std::vector<ptH3D>& pts){
    cv::viz::Viz3d viz("Camera poses and points");
    viz.showWidget("Coordinate system", viz::WCoordinateSystem(10));

    viz::WCameraPosition cpw;
    cpw = viz::WCameraPosition(Vec2f(1,0.5));
    Mat cameraPose = (Mat) orientation.getR4().t();
    Mat(position).copyTo(cameraPose(Range(0,3),Range(3,4)));
    viz.showWidget("Camera Widget",cpw,Affine3d(cameraPose));

    Mat cloud = Mat::zeros(pts.size(),1,CV_64FC3);
    for(uint i=0;i<pts.size();i++){
        ptH3D pt = pts[i];
        if(pt(2)>0 && pt(2) < 150)
            cloud.at<Matx31d>(i) = to_euclidean(pt);
    }

    viz::WCloud wcloud(cloud);
    viz.showWidget("CLOUD",wcloud);
    viz.spin();
}

void showCameraPosesAndPoints(const Mat& P, const std::vector<ptH3D>& pts){

    assert(P.rows == 3 && P.cols == 4);
    viz.showWidget("Coordinate system", viz::WCoordinateSystem(10));

    viz::WCameraPosition cpw;
    cpw = viz::WCameraPosition(Vec2f(1,0.5));
    viz.showWidget("Camera Widget",cpw,Affine3d(P));

    Mat cloud = Mat::zeros(pts.size(),1,CV_64FC3);
    for(uint i=0;i<pts.size();i++){
        ptH3D pt = pts[i];
        if(pt(2)>0 && pt(2) < 50)
        cloud.at<Matx31d>(i) = to_euclidean(pt);
    }

    viz::WCloud wcloud(cloud);
    viz.showWidget("CLOUD",wcloud);
    viz.spin();
}
}
