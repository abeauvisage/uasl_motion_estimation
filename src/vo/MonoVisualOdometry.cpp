#include "vo/MonoVisualOdometry.h"

using namespace cv;

namespace me{

bool MonoVisualOdometry::process(const std::vector<StereoMatch<cv::Point2f>>& matches){

    if(matches.size() >= 8){
        std::vector<Point2f> in,out;
        Mat E,R,T,mask;
        for(unsigned int i=0;i<matches.size();i++)
            if(matches[i].f1.x > 0 && matches[i].f2.x > 0){
                in.push_back(matches[i].f1);
                out.push_back(matches[i].f2);
            }

        if( m_param.inlier_threshold <= 0)
            m_param.inlier_threshold = 1.0;
        m_E = findEssentialMat(in, out, m_K, m_param.ransac?RANSAC:LMEDS, m_param.prob, m_param.inlier_threshold, mask);

        if(m_E.empty()){
            m_Rt = Mat::eye(4,4,CV_64FC1);
            std::cerr << "empty E matrix!" << std::endl;
            return false;
        }

        Mat triangulated;
        recoverPose(m_E, in, out, m_K,R,T,500,mask,triangulated);

        m_inliers.clear();m_outliers.clear();m_pts.clear();
        std::vector<StereoMatchf> inliers;
        int k=0;
        for(unsigned int i = 0; i < matches.size();i++)
            if(matches[i].f1.x > 0 && matches[i].f2.x > 0){
                if(mask.at<uchar>(k))
                        m_inliers.push_back(i);
                else
                    m_outliers.push_back(i);
                k++;
            }
            else
                m_outliers.push_back(i);

        assert(m_inliers.size()+m_outliers.size() == matches.size());

        if(m_inliers.size() < 10){
            m_Rt = Mat::eye(4,4,CV_64FC1);
            std::cerr << "not enough inliers!" << std::endl;
            return false;
        }
        else{
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
        std::cerr << "not enough matches!" << std::endl;
        return false;
    }
}


double MonoVisualOdometry::findRelativeScale(std::vector<std::pair<me::ptH3D,me::ptH3D>>& pts){

    std::vector<pt3D> residuals;
    double s1 = 0.0, s2 = 0.0;
    for(uint i=0;i<pts.size();i++){
        pt3D p1 = to_euclidean(pts[i].first), p2 = to_euclidean(pts[i].second);
        s1 += fabs(1.0/p1(0)) * (p1(0)*p2(0) + p1(1)*p2(1) + p1(2)*p2(2));
        s2 += fabs(1.0/p1(0)) * (p2(0)*p2(0) + p2(1)*p2(1) + p2(2)*p2(2));
    }

    return s1/s2;
}


}
