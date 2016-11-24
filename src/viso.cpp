#include "viso.h"

using namespace cv;

VisualOdometry::VisualOdometry(){
    m_pose = Mat::eye(4,4,CV_64F);
}

void VisualOdometry::updatePose(){
    Mat tmp_pose = getMotion();
    if(abs(tmp_pose.at<double>(0,3)) < 2 && abs(tmp_pose.at<double>(2,3)) < 2 && tmp_pose.at<double>(2,3) < 0){
        Mat inv;invert(tmp_pose,inv);
        m_pose *= inv;
    }
}
