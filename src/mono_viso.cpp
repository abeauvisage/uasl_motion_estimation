#include "mono_viso.h"

#include <opencv2/calib3d.hpp>

using namespace cv;

namespace me{

MonoVisualOdometry::MonoVisualOdometry(parameters param) : VisualOdometry(), m_param(param)
{
    m_Rt = Mat::eye(4,4,CV_64FC1);
    m_E = Mat::zeros(4,4,CV_64FC1);
}

bool MonoVisualOdometry::process(const std::vector<StereoMatch<cv::Point2f>>& matches){

    if(matches.size() >= 8){

        vector<Point2f> in,out;
        Mat E,R,T,mask;
        for(unsigned int i=0;i<matches.size();i++)
            if(matches[i].f1.x > 0 && matches[i].f2.x > 0){
                in.push_back(matches[i].f1);
                out.push_back(matches[i].f2);
            }

        if( m_param.inlier_threshold <= 0)
            m_param.inlier_threshold = 1.0;
        Point2d pp(m_param.cu,m_param.cv);
        m_E = findEssentialMat(in, out, m_param.f, pp, m_param.ransac?CV_RANSAC:CV_LMEDS, m_param.prob, m_param.inlier_threshold, mask);

        if(m_E.empty()){
            m_Rt = Mat::eye(4,4,CV_64FC1);
            cerr << "empty E matrix!" << endl;
            cerr << m_param.inlier_threshold << endl;
            return false;
        }
        recoverPose(m_E, in, out, R, T, m_param.f, pp, mask);

        int count=0;
        m_inliers.clear();
        for(unsigned int i = 0; i < out.size();i++)
            if( mask.at<uchar>(i) == 1 ){
                count++;
                m_inliers.push_back(i);
            }


        if(count < 10 /*|| fabs(T.at<double>(0)) > 0.5*/){
            m_Rt = Mat::eye(4,4,CV_64FC1);
            cerr << "not enough inliers!" << endl;
            cerr << m_param.inlier_threshold << endl;
            return false;
        }
        else{
//            Mat rot_T = -R * T;
            double* Rt_ptr = m_Rt.ptr<double>();
            double* R_ptr = R.ptr<double>();
            double* T_ptr = T.ptr<double>();
            for(unsigned int i=0;i<4;i++)
                for(unsigned int j=0;j<4;j++)
                    if(j == 3)
                        Rt_ptr[4*i+j] = T_ptr[i];
                    else if(i == 3)
                        Rt_ptr[4*i+j] = 0.0;
                    else
                        Rt_ptr[4*i+j] = R_ptr[3*i+j];

            Rt_ptr[15] = 1.0;
            return true;
        }
    }else{
        m_Rt = Mat::eye(4,4,CV_64FC1);
        cerr << "not enough matches!" << endl;
        return false;
    }
}

}
