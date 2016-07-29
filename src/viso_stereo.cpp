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

    x = vector<double>(6,0);
    J          = vector<double>(4*nb_matches*6);
    predictions  = vector<double>(4*nb_matches);
    observations  = vector<double>(4*nb_matches);
    residuals = vector<double>(4*nb_matches);


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
                for (int j=0; j<6; j++)
                    x[j] = 0;

                if (optimize(matches,selection,false)) { // if optimization succeeded and more inliers obtained, inliers are saved
                    vector<int> inliers_tmp = computeInliers(matches);
                    if (inliers_tmp.size()>inliers_idx.size())
                        inliers_idx = inliers_tmp;
                }
            }
        }

        for(int i=0;i<6;i++)
            x[i]=0;


        /** final optimization **/

        if (inliers_idx.size()>=6) // check that more than 6 inliers have been obtained
            if (optimize(matches,inliers_idx,true)) // optimize using inliers
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
            score += pow(residuals[4*i+j],2);
        if (score < pow(m_param.inlier_threshold,2))
            inliers_idx.push_back(i);
    }

    return inliers_idx;
}

void VisualOdometryStereo::projectionUpdate(const vector<StereoOdoMatches<Point2f>>& matches, const vector<int>& selection, bool weight){

    Point3d p1,p2,p2d;

    // compute R, dR/dx dR/dy and dR/

    double tx = x[3], ty = x[4], tz = x[5];
    double sx = sin(x[0]), cx = cos(x[0]), sy = sin(x[1]), cy = cos(x[1]), sz = sin(x[2]), cz = cos(x[2]);

    Mat R(3,3,CV_64F);
    R.at<double>(0,0) = +cy*cz;             R.at<double>(0,1) = -cy*sz;             R.at<double>(0,2) = +sy;
    R.at<double>(1,0) = +sx*sy*cz+cx*sz;    R.at<double>(1,1) = -sx*sy*sz+cx*cz;    R.at<double>(1,2) = -sx*cy;
    R.at<double>(2,0) = -cx*sy*cz+sx*sz;    R.at<double>(2,1) = +cx*sy*sz+sx*cz;    R.at<double>(2,2) = +cx*cy;

    Mat dRdx(3,3,CV_64F);
    dRdx.at<double>(0,0) = 0;               dRdx.at<double>(0,1) = 0;               dRdx.at<double>(0,2) = 0;
    dRdx.at<double>(1,0) = +cx*sy*cz-sx*sz; dRdx.at<double>(1,1) = -cx*sy*sz-sx*cz; dRdx.at<double>(1,2) = -cx*cy;
    dRdx.at<double>(2,0) = +sx*sy*cz+cx*sz; dRdx.at<double>(2,1) = -sx*sy*sz+cx*cz; dRdx.at<double>(2,2) = -sx*cy;

    Mat dRdy(3,3,CV_64F);
    dRdy.at<double>(0,0) = -sy*cz;      dRdy.at<double>(0,1) = +sy*sz;      dRdy.at<double>(0,2) = +cy;
    dRdy.at<double>(1,0) = +sx*cy*cz;   dRdy.at<double>(1,1) = -sx*cy*sz;   dRdy.at<double>(1,2)= +sx*sy;
    dRdy.at<double>(2,0) = -cx*cy*cz;   dRdy.at<double>(2,1) = +cx*cy*sz;   dRdy.at<double>(2,2) = -cx*sy;

    Mat dRdz(3,3,CV_64F);
    dRdz.at<double>(0,0) = -cy*sz;          dRdz.at<double>(0,1) = -cy*cz;          dRdz.at<double>(0,2) = 0;
    dRdz.at<double>(1,0) = -sx*sy*sz+cx*cz; dRdz.at<double>(1,1) = -sx*sy*cz-cx*sz; dRdz.at<double>(1,2) = 0;
    dRdz.at<double>(2,0) = +cx*sy*sz+sx*cz; dRdz.at<double>(2,1) = +cx*sy*cz-sx*sz; dRdz.at<double>(2,2) = 0;


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
            J[(4*i+0)*6+j] = w*m_param.f1*(p2d.x*p2.z-p2.x*p2d.z)/(p2.z*p2.z); // left u'
            J[(4*i+1)*6+j] = w*m_param.f1*(p2d.y*p2.z-p2.y*p2d.z)/(p2.z*p2.z); // left v'
            J[(4*i+2)*6+j] = w*m_param.f2*(p2d.x*p2.z-(p2.x-m_param.baseline)*p2d.z)/(p2.z*p2.z); // right u'
            J[(4*i+3)*6+j] = w*m_param.f2*(p2d.y*p2.z-p2.y*p2d.z)/(p2.z*p2.z); // right v'
    }

    observations[4*i+0] = matches[selection[i]].f3.x;
    observations[4*i+1] = matches[selection[i]].f3.y;
    observations[4*i+2] = matches[selection[i]].f4.x;
    observations[4*i+3] = matches[selection[i]].f4.y;

    predictions[4*i+0] = m_param.f1*p2.x/p2.z+m_param.cu1; // left u
    predictions[4*i+1] = m_param.f1*p2.y/p2.z+m_param.cv1; // left v
    predictions[4*i+2] = m_param.f2*(p2.x-m_param.baseline)/p2.z+m_param.cu2; // right u
    predictions[4*i+3] = m_param.f2*p2.y/p2.z+m_param.cv2; // right v

    for(int j=0;j<4;j++)
        residuals[4*i+j] = w*(observations[4*i+j]-predictions[4*i+j]);
  }
}

cv::Mat VisualOdometryStereo::getMotion(){

    double sx = sin(x[0]), cx = cos(x[0]), sy = sin(x[1]), cy = cos(x[1]), sz = sin(x[2]), cz = cos(x[2]); //compute sine cosine from the state vector

    //create rigid-body transformation matrix (R|T) from state vector
    cv::Mat Rt(4,4,CV_64F);
    double* Rt_ptr = Rt.ptr<double>();
    Rt_ptr[0]  = +cy*cz;          Rt_ptr[1]  = -cy*sz;          Rt_ptr[2]  = +sy;    Rt_ptr[3]  = x[3];
    Rt_ptr[4]  = +sx*sy*cz+cx*sz; Rt_ptr[5]  = -sx*sy*sz+cx*cz; Rt_ptr[6]  = -sx*cy; Rt_ptr[7]  = x[4];
    Rt_ptr[8]  = -cx*sy*cz+sx*sz; Rt_ptr[9]  = +cx*sy*sz+sx*cz; Rt_ptr[10] = +cx*cy; Rt_ptr[11] = x[5];
    Rt_ptr[12] = 0;               Rt_ptr[13] = 0;               Rt_ptr[14] = 0;      Rt_ptr[15] = 1;

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

    do {

        projectionUpdate(matches,selection,weight);

        // init
        cv::Mat A(6,6,CV_64F);
        cv::Mat B(6,1,CV_64F);
        cv::Mat X(6,1,CV_64F);

        // fill matrices A and B
        for (int32_t m=0; m<6; m++) {
            for (int32_t n=0; n<6; n++) {
              double a = 0;
              for (int32_t i=0; i<4*(int32_t)selection.size(); i++) {
                a += J[i*6+m]*J[i*6+n];
              }
              A.at<double>(m,n) = a;
            }
            double b = 0;
            for (int32_t i=0; i<4*(int32_t)selection.size(); i++) {
              b += J[i*6+m]*(residuals[i]);
            }
            B.at<double>(m,0) = b;
        }

        // perform elimination
        if (solve(A,B,X,DECOMP_QR)) {
            bool converged = true;
            for (int32_t m=0; m<6; m++) {
              x[m] += m_param.step_size*X.at<double>(m,0);
              if (fabs(X.at<double>(m,0))>m_param.eps)
                converged = false;
            }
            if (converged)
              result = 1;

        } else
            result = -1;

    }while(k++ < m_param.max_iter && result==0);

    if(result == -1 || k== m_param.max_iter)
        return false;
    else
        return true;
}

void VisualOdometryStereo::updatePose(){
    Mat tmp_pose = getMotion();
    if(abs(tmp_pose.at<double>(0,3)) < 10 && abs(tmp_pose.at<double>(2,3)) < 10){
        Mat inv;invert(tmp_pose,inv);
        m_pose *= inv;
    }
}
