#include "mono_viso.h"

#include <opencv2/calib3d.hpp>

using namespace cv;

namespace me{

MonoVisualOdometry::MonoVisualOdometry(parameters param) : VisualOdometry(), m_param(param)
{
    m_Rt = Mat::eye(4,4,CV_64FC1);
    m_E = Mat::zeros(4,4,CV_64FC1);
    m_K = (Mat_<double>(3,3) << param.fu,0,param.cu,0,param.fv,param.cv,0,0,1);
}

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
        m_E = findEssentialMat(in, out, m_K, m_param.ransac?CV_RANSAC:CV_LMEDS, m_param.prob, m_param.inlier_threshold, mask);

        if(m_E.empty()){
            m_Rt = Mat::eye(4,4,CV_64FC1);
            std::cerr << "empty E matrix!" << std::endl;
            return false;
        }

//        std::cout << matches.size() << " current matches" << std::endl;
//        std::cout << m_inliers.size() << " prev matches" << std::endl;
//        std::cout << m_pts.size() << " prev pts" << std::endl;

        Matx44d prev_Rt = m_Rt;
//        std::vector<ptH3D> prev_pts = m_pts;
//        std::vector<std::pair<ptH3D,ptH3D>> pairs;
        Mat triangulated;
        recoverPose(m_E, in, out, m_K,R,T,500,mask,triangulated);

        m_inliers.clear();m_outliers.clear();m_pts.clear();
        std::vector<StereoMatchf> inliers;
        int k=0;
//        std::cout << "process " << prev_pts.size() << std::endl;
        for(unsigned int i = 0; i < matches.size();i++)
            if(matches[i].f1.x > 0 && matches[i].f2.x > 0){
                    if(mask.at<uchar>(k)){
//                        inliers.push_back(matches[i]);
//                        if(i>=matches.size()-prev_pts.size()){
////                            pairs.push_back(std::pair<ptH3D,ptH3D>((ptH3D)triangulated.col(k),prev_pts[i-(matches.size()-prev_pts.size())]));
////                            std::cout << "p1: " << (prev_Rt * (ptH3D)(triangulated.col(k))).t() << " p2: " << prev_pts[i-(matches.size()-prev_pts.size())].t() << std::endl;
//                            m_outliers.push_back(i);
//                        }else{
                            m_inliers.push_back(i);
//                            m_pts.push_back((ptH3D)triangulated.col(k));
//                        }
                    }
                    else
                        m_outliers.push_back(i);
                k++;
            }
            else
                m_outliers.push_back(i);

//            double scale=0.2;
//            if(!pairs.empty())
//                scale = 1.0/findRelativeScale(pairs);
////            T *= scale;
//            std::cout << "rel scale: " << scale << std::endl;
//
//            for(uint i=0;i<m_pts.size();i++)
//                m_pts[i](2) *= scale;
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
        std::cout << p1.t() << std::endl;
        s1 += fabs(1.0/p1(0)) * (p1(0)*p2(0) + p1(1)*p2(1) + p1(2)*p2(2));
        s2 += fabs(1.0/p1(0)) * (p2(0)*p2(0) + p2(1)*p2(1) + p2(2)*p2(2));
    }

    return s1/s2;
}


}
