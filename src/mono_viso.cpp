#include "mono_viso.h"

#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;

MonoVisualOdometry::MonoVisualOdometry(parameters param) : VisualOdometry(), m_param(param)
{
    m_Rt = Mat::eye(4,4,CV_64FC1);
}

bool MonoVisualOdometry::process(const std::vector<StereoMatch<cv::Point2f>>& matches){

    if(matches.size() >= 8){

        vector<Point2f> in,out;
        Mat E,R,T,mask;
        for(unsigned int i=0;i<matches.size();i++){
            in.push_back(matches[i].f1);
            in.push_back(matches[i].f2);
        }
        Point2d pp(m_param.cu,m_param.cv);
        E = findEssentialMat(in, out, m_param.f, pp, m_param.ransac?CV_RANSAC:CV_LMEDS, m_param.prob, m_param.inlier_threshold, mask);
        recoverPose(E, in, out, R, T, m_param.f, pp, mask);

        int count=0;
        for(unsigned int i = 0; i < out.size();i++)
            if( mask.at<uchar>(i) == 1 )
                count++;

        if(count < 10 || fabs(T.at<double>(0)) > 0.5)
            return false;
        else{
            double* Rt_ptr = m_Rt.ptr<double>();
            double* R_ptr = R.ptr<double>();
            double* T_ptr = T.ptr<double>();
            for(unsigned int i=0;i<4;i++)
                for(unsigned int j=0;j<4;j++)
                    if(j == 4)
                        Rt_ptr[4*i+j] = T_ptr[i];
                    else if(i == 4)
                        Rt_ptr[4*i+j] = 0;
                    else
                        Rt_ptr[4*i+j] = R_ptr[i];

            Rt_ptr[15] == 1;
            return true;
        }
    }else
        return false;
}
