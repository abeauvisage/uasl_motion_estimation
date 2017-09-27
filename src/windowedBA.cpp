#include "windowedBA.h"
#include "utils.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/viz.hpp"
#include "opencv2/calib3d.hpp"

#include <fstream>

using namespace cv;
using namespace std;

namespace me{

//static Matx33d K_;
static double e1=1e-5,e2=1e-3,e3=1e-1,e4=1e-8;
static int max_iter=100;
static Mat img_;

enum StopCondition{NO_STOP,SMALL_GRADIENT,SMALL_INCREMENT,MAX_ITERATIONS,SMALL_DECREASE_FUNCTION,SMALL_REPROJ_ERROR,NO_CONVERGENCE};

static StereoVisualOdometry::parameters params_;

Vec3d solveWindowedBA(const std::vector<std::vector<cv::Point2f>>& observations, vector<ptH3D>& pts3D, const StereoVisualOdometry::parameters& params, const Mat& img, Vec3d dist){

    assert(observations.size() > 0 && pts3D.size() == observations[0].size());
    std::cout << "[Windowed BA] " << observations.size() << " views" << std::endl;
    std::cout << "[Windowed BA] " << pts3D.size() << " pts" << std::endl;

    params_ = params;
    img_ = img.clone();

    Mat state;
    //initialize state
    state = Mat::zeros(6*observations.size()+3*pts3D.size(),1,CV_64F);
    for(int j=0;j<observations.size();j++)
        state.at<double>(j*6+4,1) = (j+1)*dist(2)/observations.size();
    for(uint i=0;i<pts3D.size();i++){
        ptH3D pt = pts3D[i];
        normalize(pt);
        state.at<double>(observations.size()*6+i*3) = pt(0);
        state.at<double>(observations.size()*6+i*3+1) = pt(1);
        state.at<double>(observations.size()*6+i*3+2) = pt(2);
    }

    bool success = optimize(observations,pts3D,state);

    if(success){
        cout << "worked"<< endl;
        for(int i=0;i<pts3D.size();i++){
            ptH3D pt(state.at<double>(observations.size()*6+i*3),state.at<double>(observations.size()*6+i*3+1),state.at<double>(observations.size()*6+i*3+2),1);
            pts3D[i] = pt;
        }
        return Vec3d(state.at<double>(6*observations.size()-3),state.at<double>(6*observations.size()-2),state.at<double>(6*observations.size()-1));
    }
    else
        cout << "didn't work" << endl;

    return Vec3d(-1,-1,-1);
}

cv::Vec3d solveWindowedBA(const std::vector<std::vector<std::pair<me::ptH2D,me::ptH2D>>>& observations,const std::vector<ptH3D>& pts3D, const StereoVisualOdometry::parameters& params, const cv::Mat& img){
    assert(observations.size() > 0 && pts3D.size() == observations[0].size());
    std::cout << "[Windowed BA] " << observations.size() << " views" << std::endl;
    std::cout << "[Windowed BA] " << pts3D.size() << " pts" << std::endl;

    params_ = params;
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

bool optimize(const std::vector<std::vector<cv::Point2f>>& observations,const std::vector<ptH3D>& pts3D, cv::Mat& state){

    int k=0;
    int result=0;
    double v=2,tau=1e-3,mu=1e-20;
    double abs_tol=1e-3,grad_tol=1e-9,incr_tol=1e-9,rel_tol=1e-3;
    StopCondition stop=StopCondition::NO_STOP;

    Mat Xa = state.rowRange(0,6*observations.size());
    Mat Xb = state.rowRange(6*observations.size(),state.rows);

//    cout << "initial state " << Xa.t() << endl << endl << Xb.t() << endl;

    do {

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

//                cout << "Xa " << xa_test.t() << endl << endl;
//                cout << "Xb " << xb_test.t() << endl;
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
        cout << "it: " << k << endl;
    }while(!(k++ < max_iter?stop:stop=MAX_ITERATIONS));

    cout << "stop condition " << stop << endl;
    Xa.copyTo(state.rowRange(0,6*observations.size()));
    Xb.copyTo(state.rowRange(6*observations.size(),state.rows));
    if(stop == NO_CONVERGENCE || stop == MAX_ITERATIONS) // if failed or reached max iterations, return false (didn't work)
        return false;
    else{
        return true;
    }
}


std::vector<Matx34d> computeProjectionMatrices(const cv::Mat& Xa){

        assert(!Xa.empty());

        vector<Matx34d> pMat;
        Mat P;
        for(unsigned int i=0;i<Xa.rows/6;i++){
            Euld orientation(Xa.at<double>(i*6),Xa.at<double>(i*6+1),Xa.at<double>(i*6+2));
            Matx33d K = Matx33d::eye();
            K(0,0) = params_.f1;K(0,2) = params_.cu1;K(1,1) = params_.f1;K(1,2) = params_.cv1;
            sfm::projectionFromKRt(K,orientation.getR3(),Xa.rowRange(i*6+3,i*6+6),P);
            pMat.push_back((Matx34d)P);
        }
        return pMat;
}

cv::Mat compute_residuals_stereo(const std::vector<std::vector<std::pair<me::ptH2D,me::ptH2D>>>& observations, const std::vector<ptH3D>& pts3D, const cv::Mat& Xa){

    Mat residuals = Mat::zeros(observations.size()*pts3D.size(),4,CV_64F);
    for(unsigned int j=0;j<observations.size();j++){
        Euld orientation(Xa.at<double>(j*6),Xa.at<double>(j*6+1),Xa.at<double>(j*6+2));
        Mat Tr = (Mat) orientation.getR4();
        Xa.rowRange(j*6+3,j*6+6).copyTo(Tr(Range(0,3),Range(3,4)));
//        cout << "Tr " << Tr << endl;
        for(unsigned int i=0;i<observations[j].size();i++){
            ptH3D pt = pts3D[i];
            ptH3D pt_next = (Matx44d)Tr*pt;
            normalize(pt_next);
            ptH2D ob1 = observations[j][i].first, ob2 = observations[j][i].second;
            normalize(ob1);normalize(ob2);
//            cout << ob1.t() << " | ";
//            cout << ob1.t() << " - " << (K_(0,0) * pt_next(0)/pt_next(2) + K_(0,2)) << "," << (K_(1,1) * pt_next(1)/pt_next(2) + K_(1,2)) <<  " | ";
//            cout << K_(1,1) << " " << K_(1,2) << endl;
            residuals.at<double>(j*pts3D.size()+i,0) = (ob1(0)-(params_.f1 * pt_next(0)/pt_next(2) + params_.cu1));
            residuals.at<double>(j*pts3D.size()+i,1) = (ob1(1)-(params_.f1 * pt_next(1)/pt_next(2) + params_.cv1));
            residuals.at<double>(j*pts3D.size()+i,2) = (ob2(0)-(params_.f2 * (pt_next(0)-params_.baseline)/pt_next(2) + params_.cu2));
            residuals.at<double>(j*pts3D.size()+i,3) = (ob2(1)-(params_.f2 * pt_next(1)/pt_next(2) + params_.cv2));
//            cout << residuals.at<double>(j*pts3D.size()+i,0) << "," << residuals.at<double>(j*pts3D.size()+i,1) << " | ";
        }
    }
    return residuals;
}

cv::Mat compute_residuals(const std::vector<std::vector<cv::Point2f>>& observations, const cv::Mat& Xb, const cv::Mat& Xa){

    Mat residuals = Mat::zeros(observations.size()*Xb.rows/3,2,CV_64F);
    for(unsigned int j=0;j<observations.size();j++){
        Euld orientation(Xa.at<double>(j*6),Xa.at<double>(j*6+1),Xa.at<double>(j*6+2));
        Mat Tr = (Mat) orientation.getR4();
        Xa.rowRange(j*6+3,j*6+6).copyTo(Tr(Range(0,3),Range(3,4)));
        for(unsigned int i=0;i<observations[j].size();i++){
            ptH3D pt = ptH3D(Xb.at<double>(i*3),Xb.at<double>(i*3+1),Xb.at<double>(i*3+2),1);
            ptH3D pt_next = (Matx44d)Tr*pt;
            normalize(pt_next);
            residuals.at<double>(j*Xb.rows/3+i,0) = (observations[j][i].x-(params_.f1 * pt_next(0)/pt_next(2) + params_.cu1));
            residuals.at<double>(j*Xb.rows/3+i,1) = (observations[j][i].y-(params_.f1* pt_next(1)/pt_next(2) + params_.cv1));
        }
    }
    return residuals;
}

std::vector<std::vector<cv::Point2f>> project_pts(const cv::Mat& Xb, const std::vector<cv::Matx34d>& pMat){
    vector<vector<Point2f>> pts2D;
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
                A_ij.at<double>(0,k) = params_.f1*(dpt_next(0)*pt_next(2)-pt_next(0)*dpt_next(2))/(pt_next(2)*pt_next(2));
                A_ij.at<double>(1,k) = params_.f1*(dpt_next(1)*pt_next(2)-pt_next(1)*dpt_next(2))/(pt_next(2)*pt_next(2));
                A_ij.at<double>(2,k) = params_.f2*(dpt_next(0)*pt_next(2)-(pt_next(0)-params_.baseline)*dpt_next(2))/(pt_next(2)*pt_next(2));
                A_ij.at<double>(3,k) = params_.f2*(dpt_next(1)*pt_next(2)-pt_next(1)*dpt_next(2))/(pt_next(2)*pt_next(2));
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

void computeJacobian(const cv::Mat& Xa,  const Mat& Xb, const Mat& residuals, Mat& JJ, Mat& e){

    int m_views = Xa.rows/6, n_pts = Xb.rows/3;
    Mat U = Mat::zeros(6*m_views,6*m_views,CV_64F);
    Mat V = Mat::zeros(3*n_pts,3*n_pts,CV_64F);
    Mat W = Mat::zeros(6*m_views,3*n_pts,CV_64F);
    JJ = Mat::zeros(U.rows+V.rows,U.cols+V.cols,CV_64F);
    e = Mat::zeros(JJ.rows,1,CV_64F);
//    JJ = Mat::zeros(U.rows,U.cols,CV_64F);
//    e = Mat::zeros(JJ.rows,1,CV_64F);

    vector<vector<Mat>> A_;vector<vector<Mat>> B_;

    for(unsigned int j=0;j<m_views;j++){
        Euld eul(Xa.at<double>(j*6,0),Xa.at<double>(j*6+1,0),Xa.at<double>(j*6+2,0));
        Mat Tr = (Mat) eul.getR4();
        Matx33d dRdx = eul.getdRdr();
        Matx33d dRdy = eul.getdRdp();
        Matx33d dRdz = eul.getdRdy();
//        cout << Tr << endl;
        Xa.rowRange(j*6+3,j*6+6).copyTo(Tr(Range(0,3),Range(3,4)));

        vector<Mat> A_j;vector<Mat> B_j;
        for(unsigned int i=0;i<n_pts;i++){
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
                A_ij.at<double>(0,k) = params_.f1*(dpt_next(0)*pt_next(2)-pt_next(0)*dpt_next(2))/(pt_next(2)*pt_next(2));
                A_ij.at<double>(1,k) = params_.f1*(dpt_next(1)*pt_next(2)-pt_next(1)*dpt_next(2))/(pt_next(2)*pt_next(2));
                if(k<3){
                    B_ij.at<double>(0,k) = params_.f1*(dpt_b(0)*pt_next(2)-pt_next(0)*dpt_b(2))/(pt_next(2)*pt_next(2));
                    B_ij.at<double>(1,k) = params_.f1*(dpt_b(1)*pt_next(2)-pt_next(1)*dpt_b(2))/(pt_next(2)*pt_next(2));
                }
           }
           U(Range(j*6,j*6+6),Range(j*6,j*6+6)) += A_ij.t() * A_ij;
           W(Range(j*6,j*6+6),Range(i*3,i*3+3)) += A_ij.t() * B_ij;
           e(Range(j*6,j*6+6),Range(0,1)) += A_ij.t() * residuals.row(j*n_pts+i).t();
           A_j.push_back(A_ij);
           B_j.push_back(B_ij);
        }
        A_.push_back(A_j);
        B_.push_back(B_j);
    }
    for(unsigned int i=0;i<n_pts;i++)
        for(unsigned int j=0;j<m_views;j++){
            V(Range(i*3,i*3+3),Range(i*3,i*3+3)) += B_[j][i].t() * B_[j][i];
            e(Range(6*m_views+i*3,6*m_views+i*3+3),Range(0,1)) += B_[j][i].t() * residuals.row(j*n_pts+i).t();
        }

    U.copyTo(JJ(Range(0,6*m_views),Range(0,6*m_views)));
    V.copyTo(JJ(Range(6*m_views,6*m_views+3*n_pts),Range(6*m_views,6*m_views+3*n_pts)));
    W.copyTo(JJ(Range(0,6*m_views),Range(6*m_views,6*m_views+3*n_pts)));
    Mat Wt = W.t();
    Wt.copyTo(JJ(Range(6*m_views,6*m_views+3*n_pts),Range(0,6*m_views)));
}

void showReprojectedPts(const cv::Mat& img, const std::vector<cv::Matx34d>& pMat, const vector<std::vector<cv::Point2f>>& observations,const Mat& Xb){

    vector<vector<Point2f>> pts2D = project_pts(Xb,pMat);
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
        viz.showWidget("Camera Widget"+to_string(i),cpw,Affine3d(cameraPose));
    }
    viz.spinOnce();
}

/*
std::vector<std::vector<Point2f>> project_pts(const vector<ptH3D>& pts3D, const std::vector<cv::Matx34d>& pMat){
    vector<vector<Point2f>> pts2D;
    for(unsigned int i=0;i<pMat.size();i++){
        vector<Point2f> currentView;
        for(unsigned int j=0;j<pts3D.size();j++){
                ptH2D newPT = pMat[i] * pts3D[j];
                normalize(newPT);
                currentView.push_back(Point2f(newPT(0),newPT(1)));
        }
        pts2D.push_back(currentView);
    }
    return pts2D;
}

cv::Mat compute_residuals(const std::vector<std::vector<cv::Point2f>>& observations, const std::vector<ptH3D>& pts3D, const cv::Mat& Xa){

    Mat residuals = Mat::zeros(observations.size()*pts3D.size(),2,CV_64F);
    for(unsigned int j=0;j<observations.size();j++){
        Euld orientation(Xa.at<double>(j*6),Xa.at<double>(j*6+1),Xa.at<double>(j*6+2));
        Mat Tr = (Mat) orientation.getR4();
        Xa.rowRange(j*6+3,j*6+6).copyTo(Tr(Range(0,3),Range(3,4)));
        for(unsigned int i=0;i<observations[j].size();i++){
            ptH3D pt = pts3D[i];
            ptH3D pt_next = (Matx44d)Tr*pt;
            normalize(pt_next);
//            cout << observations[j][i] << " - " << (K_(0,0) * pt_next(0)/pt_next(2) + K_(0,2)) << "," << (K_(1,1) * pt_next(1)/pt_next(2) + K_(1,2)) << endl;
            residuals.at<double>(j*pts3D.size()+i,0) = (observations[j][i].x-(params_.f1 * pt_next(0)/pt_next(2) + params_.cu1));
            residuals.at<double>(j*pts3D.size()+i,1) = (observations[j][i].y-(params_.f1 * pt_next(1)/pt_next(2) + params_.cv1));
        }
    }
    return residuals;
}

void computeJacobian(const cv::Mat& Xa,  const std::vector<ptH3D>& pts, const Mat& residuals, Mat& JJ, Mat& e){

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

        cout << "Tr " << Tr << endl;

        vector<Mat> A_j;vector<Mat> B_j;
        for(unsigned int i=0;i<n_pts;i++){
            Mat A_ij(2,6,CV_64F), B_ij(2,3,CV_64F);
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
                A_ij.at<double>(0,k) = params_.f1*(dpt_next(0)*pt_next(2)-pt_next(0)*dpt_next(2))/(pt_next(2)*pt_next(2));
                A_ij.at<double>(1,k) = params_.f1*(dpt_next(1)*pt_next(2)-pt_next(1)*dpt_next(2))/(pt_next(2)*pt_next(2));
                if(k<3){
                    B_ij.at<double>(0,k) = params_.f1*(dpt_b(0)*pt_next(2)-pt_next(0)*dpt_b(2))/(pt_next(2)*pt_next(2));
                    B_ij.at<double>(1,k) = params_.f1*(dpt_b(1)*pt_next(2)-pt_next(1)*dpt_b(2))/(pt_next(2)*pt_next(2));
                }
           }
           U(Range(j*6,j*6+6),Range(j*6,j*6+6)) += A_ij.t() * A_ij;
           W(Range(j*6,j*6+6),Range(i*3,i*3+3)) += A_ij.t() * B_ij;
           e(Range(j*6,j*6+6),Range(0,1)) += A_ij.t() * residuals.row(j*n_pts+i).t();
           A_j.push_back(A_ij);
           B_j.push_back(B_ij);
        }
        A_.push_back(A_j);
        B_.push_back(B_j);
    }
//    for(unsigned int i=0;i<n_pts;i++)
//        for(unsigned int j=0;j<m_views;j++){
//            V(Range(i*3,i*3+3),Range(i*3,i*3+3)) += B_[j][i].t() * B_[j][i];
//            e(Range(6*m_views+i*3,6*m_views+i*3+3),Range(0,1)) += B_[j][i].t() * residuals.row(j*6+i*3).t();
//        }

    U.copyTo(JJ(Range(0,6*m_views),Range(0,6*m_views)));
//    V.copyTo(JJ(Range(6*m_views,6*m_views+3*n_pts),Range(6*m_views,6*m_views+3*n_pts)));
//    W.copyTo(JJ(Range(0,6*m_views),Range(6*m_views,6*m_views+3*n_pts)));
//    Mat Wt = W.t();
//    Wt.copyTo(JJ(Range(6*m_views,6*m_views+3*n_pts),Range(0,6*m_views)));
}

void showReprojectedPts(const cv::Mat& img, const std::vector<cv::Matx34d>& pMat, const vector<std::vector<cv::Point2f>>& observations,const std::vector<ptH3D>& pts3D){

    vector<vector<Point2f>> pts2D = project_pts(pts3D,pMat);
    for(int j=0;j<pMat.size();j++){
        Mat disp_img = img_.clone();
        cvtColor(disp_img,disp_img,CV_GRAY2BGR);
        for(int i=0;i<pts3D.size();i++){
            circle(disp_img,observations[j][i],2,Scalar(255,0,0));
            circle(disp_img,pts2D[j][i],2,Scalar(255,255,255));
        }
        imshow("test"+to_string(j),disp_img);
    }
    waitKey(1000);
}
*/
}
