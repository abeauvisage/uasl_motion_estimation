#include "viso_stereo.h"

#include <math.h>
#include <iostream>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

VisualOdometryStereo::VisualOdometryStereo (parameters param) : m_param(param) {
    srand(0);
    m_pose = Mat::eye(4,4,CV_64F);
}


bool VisualOdometryStereo::process (const vector<StereoOdoMatches<Point2f>>& matches) {

    int nb_matches = matches.size();
    if(nb_matches<6)
        return false;

    x = Mat::zeros(6,1,CV_64F);

    pts3D.clear();

    for (int i=0; i<nb_matches; i++) {
        double d = matches[i].f1.x - matches[i].f2.x;
        d = (d>0)?d:0.00001; // to avoid divison by 0;
        pts3D.push_back(Point3d((matches[i].f1.x-m_param.cu1)*m_param.baseline/d, (matches[i].f1.y-m_param.cv1)*m_param.baseline/d, m_param.f1*m_param.baseline/d));
    }

    inliers_idx.clear();

    for (int i=0;i<m_param.n_ransac;i++) {
        vector<int> selection;

        selection = randomIndexes(3,nb_matches);
        if((matches[selection[0]].f3.x*(matches[selection[1]].f3.y-matches[selection[2]].f3.y)+matches[selection[1]].f3.x*(matches[selection[2]].f3.y-matches[selection[0]].f3.y)+matches[selection[2]].f3.x*(matches[selection[0]].f3.y-matches[selection[1]].f3.y))/2 > 1000){
            x = Mat::zeros(6,1,CV_64F);

            if (optimize(matches,selection,false)) { // if optimization succeeded and more inliers obtained, inliers are saved
                vector<int> inliers_tmp = computeInliers(matches);
                if (inliers_tmp.size()>inliers_idx.size())
                    inliers_idx = inliers_tmp;
            }
        }
    }

    x = Mat::zeros(6,1,CV_64F);

//    for (int i=0; i<nb_matches; i++)
//        inliers_idx.push_back(i);

    cout << "nb inliers: " << inliers_idx.size() << endl;
    /** final optimization **/

    if (inliers_idx.size()>=6) // check that more than 6 inliers have been obtained
        if (optimize(matches,inliers_idx,false)) // optimize using inliers
            return true;
        else
            return false;
    else
        return false;
}

vector<int> VisualOdometryStereo::computeInliers(const vector<StereoOdoMatches<Point2f>>& matches) {

    vector<int> selection;
    for (int i=0;i<matches.size();i++)
        selection.push_back(i);

    projectionUpdate(matches,selection,false);

    vector<int> inliers_idx;
    for (int i=0; i<matches.size(); i++){
        double score=0;
        for(int j=0;j<4;j++)
            score += pow(residuals.at<double>(4*i+j),2);
        if (score < pow(m_param.inlier_threshold,2))
            inliers_idx.push_back(i);
    }

    return inliers_idx;
}

cv::Mat VisualOdometryStereo::applyFunction(const vector<StereoOdoMatches<Point2f>>& matches, cv::Mat& x_,  const vector<int>& selection){

    cv::Mat res = cv::Mat::zeros(4*selection.size(),1,CV_64F);

    Point3d p1,p2;

    // compute R, dR/dx dR/dy and dR/dz
    double* x_ptr = x_.ptr<double>();
    double tx = x_ptr[3], ty = x_ptr[4], tz = x_ptr[5];
    double sx = sin(x_ptr[0]), cx = cos(x_ptr[0]), sy = sin(x_ptr[1]), cy = cos(x_ptr[1]), sz = sin(x_ptr[2]), cz = cos(x_ptr[2]);


    /*** R matrix ***/

    double R_[9] = {    +cy*cz,             -cy*sz,             +sy,
                        +sx*sy*cz+cx*sz,    -sx*sy*sz+cx*cz,    -sx*cy,
                        -cx*sy*cz+sx*sz,    +cx*sy*sz+sx*cz,    +cx*cy};

    Mat R(3,3,CV_64FC1,&R_);

    for (unsigned int i=0; i<selection.size(); i++) {

        p1 = pts3D[selection[i]];

        double* R_ptr = R.ptr<double>();
        p2 = Point3d(R_ptr[0]*p1.x+R_ptr[1]*p1.y+R_ptr[2]*p1.z+tx, R_ptr[3]*p1.x+R_ptr[4]*p1.y+R_ptr[5]*p1.z+ty, R_ptr[6]*p1.x+R_ptr[7]*p1.y+R_ptr[8]*p1.z+tz);

        res.at<double>(4*i+0) = matches[selection[i]].f3.x-m_param.f1*p2.x/p2.z+m_param.cu1;
        res.at<double>(4*i+1) = matches[selection[i]].f3.y-m_param.f1*p2.y/p2.z+m_param.cv1;
        res.at<double>(4*i+2) = matches[selection[i]].f4.x-m_param.f2*(p2.x-m_param.baseline)/p2.z+m_param.cu2;
        res.at<double>(4*i+3) = matches[selection[i]].f4.y-m_param.f2*p2.y/p2.z+m_param.cv2;
    }
    return res;
}

void VisualOdometryStereo::projectionUpdate(const vector<StereoOdoMatches<Point2f>>& matches, const vector<int>& selection, bool weight){

    J          = cv::Mat::zeros(6,4*selection.size(),CV_64F);
    predictions  = cv::Mat::zeros(4*selection.size(),1,CV_64F);
    observations  = cv::Mat::zeros(4*selection.size(),1,CV_64F);
    residuals = cv::Mat::zeros(4*selection.size(),1,CV_64F);

    Point3d p1,p2,p2d;

    // compute R, dR/dx dR/dy and dR/dz

    double* x_ptr = x.ptr<double>();
    double tx = x_ptr[3], ty = x_ptr[4], tz = x_ptr[5];
    double sx = sin(x_ptr[0]), cx = cos(x_ptr[0]), sy = sin(x_ptr[1]), cy = cos(x_ptr[1]), sz = sin(x_ptr[2]), cz = cos(x_ptr[2]);


    /*** R matrix ***/

    double R_[9] = {    +cy*cz,             -cy*sz,             +sy,
                        +sx*sy*cz+cx*sz,    -sx*sy*sz+cx*cz,    -sx*cy,
                        -cx*sy*cz+sx*sz,    +cx*sy*sz+sx*cz,    +cx*cy};

    Mat R(3,3,CV_64FC1,&R_);

    /*** dR/dx matrix ***/

    double dRdx_[9] = { 0,                  0,                  0,
                        +cx*sy*cz-sx*sz,    -cx*sy*sz-sx*cz,    -cx*cy,
                        +sx*sy*cz+cx*sz,    -sx*sy*sz+cx*cz,    -sx*cy};

    Mat dRdx(3,3,CV_64F,dRdx_);

    /*** dR/dy matrix ***/

    double dRdy_[9] = { -sy*cz,     +sy*sz,     +cy,
                        +sx*cy*cz,  -sx*cy*sz,  +sx*sy,
                        -cx*cy*cz,  +cx*cy*sz,  -cx*sy};

    Mat dRdy(3,3,CV_64F,dRdy_);

    /*** dR/dz matrix ***/

    double dRdz_[9] = { -cy*sz,             -cy*cz,             0,
                        -sx*sy*sz+cx*cz,    -sx*sy*cz-cx*sz,    0,
                        +cx*sy*sz+sx*cz,    +cx*sy*cz-sx*sz,    0};

    Mat dRdz(3,3,CV_64F,dRdz_);


    for (unsigned int i=0; i<selection.size(); i++) {

        p1 = pts3D[selection[i]];
        // compute 3d point in current left coordinate system
        double* R_ptr = R.ptr<double>();
        p2 = Point3d(R_ptr[0]*p1.x+R_ptr[1]*p1.y+R_ptr[2]*p1.z+tx, R_ptr[3]*p1.x+R_ptr[4]*p1.y+R_ptr[5]*p1.z+ty, R_ptr[6]*p1.x+R_ptr[7]*p1.y+R_ptr[8]*p1.z+tz);
        float w;
        if(weight)
            w = 1/p1.z;
        else
            w = 1;

        for (int j=0; j<6; j++) {

            switch (j) {
                case 0: {
                            double* dRdx_ptr = dRdx.ptr<double>();
                            p2d = Point3d(0, dRdx_ptr[3]*p1.x+dRdx_ptr[4]*p1.y+dRdx_ptr[5]*p1.z, dRdx_ptr[6]*p1.x+dRdx_ptr[7]*p1.y+dRdx_ptr[8]*p1.z);
                            break;
                        }
                case 1: {
                            double* dRdy_ptr = dRdy.ptr<double>();
                            p2d = Point3d(dRdy_ptr[0]*p1.x+dRdy_ptr[1]*p1.y+dRdy_ptr[2]*p1.z, dRdy_ptr[3]*p1.x+dRdy_ptr[4]*p1.y+dRdy_ptr[5]*p1.z, dRdy_ptr[6]*p1.x+dRdy_ptr[7]*p1.y+dRdy_ptr[8]*p1.z);
                            break;
                        }
                case 2: {
                            double* dRdz_ptr = dRdz.ptr<double>();
                            p2d = Point3d(dRdz_ptr[0]*p1.x+dRdz_ptr[1]*p1.y, dRdz_ptr[3]*p1.x+dRdz_ptr[4]*p1.y, dRdz_ptr[6]*p1.x+dRdz_ptr[7]*p1.y);
                            break;
                        }
                case 3: p2d = Point3d(1,0,0); break;
                case 4: p2d = Point3d(0,1,0); break;
                case 5: p2d = Point3d(0,0,1); break;
            }

            // set jacobian entries (project via K)
            J.at<double>(j,4*i+0) = w*m_param.f1*(p2d.x*p2.z-p2.x*p2d.z)/(p2.z*p2.z); // left u'
            J.at<double>(j,4*i+1) = w*m_param.f1*(p2d.y*p2.z-p2.y*p2d.z)/(p2.z*p2.z); // left v'
            J.at<double>(j,4*i+2) = w*m_param.f2*(p2d.x*p2.z-(p2.x-m_param.baseline)*p2d.z)/(p2.z*p2.z); // right u'
            J.at<double>(j,4*i+3) = w*m_param.f2*(p2d.y*p2.z-p2.y*p2d.z)/(p2.z*p2.z); // right v'
    }

    observations.at<double>(4*i+0) = matches[selection[i]].f3.x;
    observations.at<double>(4*i+1) = matches[selection[i]].f3.y;
    observations.at<double>(4*i+2) = matches[selection[i]].f4.x;
    observations.at<double>(4*i+3) = matches[selection[i]].f4.y;

    predictions.at<double>(4*i+0) = m_param.f1*p2.x/p2.z+m_param.cu1; // left u
    predictions.at<double>(4*i+1) = m_param.f1*p2.y/p2.z+m_param.cv1; // left v
    predictions.at<double>(4*i+2) = m_param.f2*(p2.x-m_param.baseline)/p2.z+m_param.cu2; // right u
    predictions.at<double>(4*i+3) = m_param.f2*p2.y/p2.z+m_param.cv2; // right v

    for(int j=0;j<4;j++)
        residuals.at<double>(4*i+j) = w*(observations.at<double>(4*i+j)-predictions.at<double>(4*i+j));
  }

}

cv::Mat VisualOdometryStereo::getMotion(){

    double* x_ptr = x.ptr<double>();
    double tx = x_ptr[3], ty = x_ptr[4], tz = x_ptr[5];
    double sx = sin(x_ptr[0]), cx = cos(x_ptr[0]), sy = sin(x_ptr[1]), cy = cos(x_ptr[1]), sz = sin(x_ptr[2]), cz = cos(x_ptr[2]); //compute sine cosine from the state vector

    //create rigid-body transformation matrix (R|T) from state vector
    cv::Mat Rt(4,4,CV_64F);
    double* Rt_ptr = Rt.ptr<double>();
    Rt_ptr[0]  = +cy*cz;          Rt_ptr[1]  = -cy*sz;          Rt_ptr[2]  = +sy;    Rt_ptr[3]  = tx;
    Rt_ptr[4]  = +sx*sy*cz+cx*sz; Rt_ptr[5]  = -sx*sy*sz+cx*cz; Rt_ptr[6]  = -sx*cy; Rt_ptr[7]  = ty;
    Rt_ptr[8]  = -cx*sy*cz+sx*sz; Rt_ptr[9]  = +cx*sy*sz+sx*cz; Rt_ptr[10] = +cx*cy; Rt_ptr[11] = tz;
    Rt_ptr[12] = 0;               Rt_ptr[13] = 0;               Rt_ptr[14] = 0;      Rt_ptr[15] = 1;
    //create rigid-body transformation matrix (R|T) from state vector
//    double Rt_[16] = {   +cy*cz,             -cy*sz,             +sy,    x[3],
//                        +sx*sy*cz+cx*sz,    -sx*sy*sz+cx*cz,    -sx*cy, x[4],
//                        -cx*sy*cz+sx*sz,    +cx*sy*sz+sx*cz,    +cx*cy, x[5],
//                        0,                  0,                  0,      1};
//    Rt = cv::Mat(4,4,CV_64F,Rt_);

    return Rt;
}

vector<int> VisualOdometryStereo::randomIndexes(int nb_samples, int nb_tot) {

    assert(nb_samples < nb_tot);
    vector<int> samples_idx;

    int i=0;
    while(i<nb_samples){
        int idx = rand()% nb_tot; //select random number between 0 and nb_tot

        bool exists = false;
        for(int j=0;j<samples_idx.size();j++) // ckeck if the index is alredy included
            if(idx == samples_idx[j])
                exists = true;
        if(!exists){
            i++;
            samples_idx.push_back(idx);
        }
    }

  return samples_idx;
}

bool VisualOdometryStereo::optimize(const std::vector<StereoOdoMatches<Point2f>>& matches, const std::vector<int>& selection, bool weight){

    if (selection.size()<3) // if less than 3 points triangulation impossible
        return false;

    int k=0;
    int result=0;
    double lambda=1e-2;

    do {

        projectionUpdate(matches,selection,weight);

        // init
        cv::Mat A = J * J.t();
        cv::Mat B(6,1,CV_64F);
        cv::Mat X(6,1,CV_64F);

        for(int i=0;i<6;i++){
            Mat b = J.row(i) * residuals;
            B.at<double>(i) = b.at<double>(0);
        }

        if(m_param.method == LM)
            A += lambda * Mat::diag(A.diag());

        if(solve(A,B,X,DECOMP_QR)){

            if(m_param.method == GN){   // Gauss-Newton
                x += X;
                double min, max;
                cv::minMaxLoc(X,&min,&max);
                if(max < m_param.eps)
                    result = 1;
            }
            else{                       // Levenberg-Marquart
                Mat x_test = x + X;

                Mat r = applyFunction(matches,x_test,selection);
                Mat rho = (residuals.t()*residuals - r.t()*r)/(X.t()*(lambda*X+B));
                if(abs(rho.at<double>(0)) > m_param.e4){ //threshold
                    lambda = max(lambda/9,1.e-7);
                    x = x_test;
//                    cout << "updated!" << endl;
                }
                else
                    lambda = min(lambda*11,1.e7);

                double min, max, m1,m2,m3;
                cv::minMaxLoc(X,&min,&max);
                cv::minMaxLoc(B,&min,&m1);
                cv::minMaxLoc(x_test/x,&min,&m2);
                if(max < m_param.e1 || m1 < m_param.e2 || m2 < m_param.e3)
                    result = 1;

            }
        }else
            result = -1;

        // perform elimination
//        if (solve(A,B,X,DECOMP_QR)) {
//            bool converged = true;
//            for (int m=0; m<6; m++) {
//              double* x_ptr = x.ptr<double>();
//              x_ptr[m] += m_param.step_size*X.at<double>(m,0);
//              if (fabs(X.at<double>(m,0))>m_param.eps)
//                converged = false;
//            }
//            if (converged)
//              result = 1;
//
//        } else
//            result = -1;

    }while(k++ < m_param.max_iter && result==0);

    if(result == -1 || k== m_param.max_iter)
        return false;
    else
        return true;
}

void VisualOdometryStereo::updatePose(){
    Mat tmp_pose = getMotion();
    if(abs(tmp_pose.at<double>(0,3)) < 2 && abs(tmp_pose.at<double>(2,3)) < 2){
        Mat inv;invert(tmp_pose,inv);
        std::cout << inv << std::endl;
        m_pose *= inv;
    }
}
