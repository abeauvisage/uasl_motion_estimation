#include "gui/gui_utils.h"

using namespace std;
using namespace me;
using namespace cv;

namespace me{

//! display features in the latest image
cv::Mat show(const vector<WBA_Ptf>& pts, const vector<CamPose_qd>& poses, const cv::Mat& img){
    CamPose_qd pose = poses[poses.size()-1];
    Mat img_color;img.convertTo(img_color,CV_8UC3);
    cvtColor(img_color,img_color,CV_GRAY2BGR);
    for(auto& pt : pts)
        if((int) pt.getLastFrameIdx() == pose.ID){
            int color = (to_euclidean(pt.get3DLocation())(2)-5)*(255/45.0);
            circle(img_color,pt.getLastFeat(),2,Scalar(0,color,255),-1);
        }

    imshow("img",img_color);
    waitKey(1);
    return img_color;
}

//! display features in the latest stereo pair
cv::Mat show(const vector<WBA_Ptf>& pts, const pair<vector<CamPose_qd>,vector<CamPose_qd>>& poses, const pair<cv::Mat,cv::Mat>& img){
    CamPose_qd pose = *(poses.first.end()-1);
    Mat img_color(img.first.rows,img.first.cols+img.second.cols,CV_8U);
    img.first.convertTo(img_color.colRange(Range(0,img.first.cols)),CV_8UC3);
    img.second.convertTo(img_color.colRange(Range(img.first.cols,img.first.cols+img.second.cols)),CV_8UC3);
    cvtColor(img_color,img_color,CV_GRAY2BGR);

     int nb_pts[2] = {0,0};
    for(auto& pt : pts)
        if((int) pt.getLastFrameIdx() == pose.ID){
            nb_pts[pt.getCameraID()]++;
            int color = ((pose.orientation * to_euclidean(pt.get3DLocation())+pose.position)(2)-5)*(255/45.0);
            circle(img_color,pt.getLastFeat()+pt.getCameraID()*Point2f(img.first.cols,0),2,Scalar(0,color,255),-1);
        }
    imshow("img",img_color);
    waitKey(1);
    return img_color;
}

//! display features in the latest stereo pair
cv::Mat show_stereo_reproj(const vector<WBA_Ptf>& pts, const pair<vector<CamPose_qd>,vector<CamPose_qd>>& poses, const pair<cv::Mat,cv::Mat>& img, const cv::Matx33d& K){
    CamPose_qd pose_left = *(poses.first.end()-1);
    CamPose_qd pose_right = *(poses.second.end()-1);
    Mat img_color(img.first.rows,img.first.cols+img.second.cols,CV_8U);
    img.first.convertTo(img_color.colRange(Range(0,img.first.cols)),CV_8UC3);
    img.second.convertTo(img_color.colRange(Range(img.first.cols,img.first.cols+img.second.cols)),CV_8UC3);
    cvtColor(img_color,img_color,CV_GRAY2BGR);

    int nb_pts[2] = {0,0};
    for(auto& pt : pts){
        if((int) pt.getLastFrameIdx() == pose_left.ID){
            int color = pt.getCameraID()*255;
            nb_pts[pt.getCameraID()]++;

            pt3D p3_left = (pose_left.orientation * to_euclidean(pt.get3DLocation()) + pose_left.position);normalize(p3_left);
            pt3D p3_right = (pose_right.orientation * to_euclidean(pt.get3DLocation()) + pose_right.position);normalize(p3_right);
            pt2D p_left = to_euclidean(K * p3_left);
            pt2D p_right = to_euclidean(K * p3_right);

            circle(img_color,Point2f(p_left(0),p_left(1)),2,Scalar(0,color,255),-1);
            circle(img_color,Point2f(p_right(0),p_right(1))+Point2f(img.first.cols,0),2,Scalar(0,color,255),-1);
        }
    }
    imshow("img reproj",img_color);
    waitKey(1);
    return img_color;
}

//! display features in the latest stereo pair
cv::Mat show_stereo_reproj_scaled(const pair<vector<WBA_Ptf>,vector<WBA_Ptf>>& pts, const pair<vector<CamPose_qd>,vector<CamPose_qd>>& poses, const pair<cv::Mat,cv::Mat>& img, const pair<cv::Matx33d,cv::Matx33d>& K, double baseline, double scale){

    CamPose_qd pose_left = *(poses.first.end()-1);
    CamPose_qd pose_right = *(poses.second.end()-1);
    Mat img_color(img.first.rows,img.first.cols+img.second.cols,CV_8U);
    img.first.convertTo(img_color.colRange(Range(0,img.first.cols)),CV_8UC3);
    img.second.convertTo(img_color.colRange(Range(img.first.cols,img.first.cols+img.second.cols)),CV_8UC3);
    cvtColor(img_color,img_color,CV_GRAY2BGR);
    int window_size = 1;
    Rect bb(window_size,window_size,img.first.cols-2*window_size-1,img.first.rows-2*window_size-1);

    //features are reprojected in the last frame only
    uint lframe = poses.first[0].ID + poses.first.size()-1;
    //for all features extracted from the left camera
    for(uint i=0;i<pts.first.size();i++){
        if(pts.first[i].isTriangulated()){
            ptH3D pt = pts.first[i].get3DLocation();
            if(pts.first[i].getLastFrameIdx() == lframe){ // if has been observed in the last keyframe
                int f_idx = poses.first.size()-1;
                Mat Tr = (Mat) poses.first[f_idx].orientation.getR4();
                ((Mat)poses.first[f_idx].position).copyTo(Tr(Range(0,3),Range(3,4)));
                Matx44d Tr_ = Tr;
                ptH2D feat = K.first * Matx34d::eye() * scale *(Tr_ * pt);
                Point2f feat_left(to_euclidean(feat)(0),to_euclidean(feat)(1)); // reprojection in the left image
                ptH2D feat_ =  (K.second * Matx34d::eye()) * (scale * (Tr_ * pt) - Matx41d(baseline,0,0,0));
                Point2f feat_right(to_euclidean(feat_)(0),to_euclidean(feat_)(1)); // reprojection in the right image
                if(bb.contains(feat_left) && bb.contains(feat_right)){
                    circle(img_color,feat_left,2,Scalar(0,0,255),-1);
                    circle(img_color,feat_right+Point2f(img.first.cols,0),2,Scalar(0,0,255),-1);
                }
            }
        }
    }

    for(uint i=0;i<pts.second.size();i++){
        if(pts.second[i].isTriangulated()){
            ptH3D pt = pts.second[i].get3DLocation()-Matx41d(baseline,0,0,0);
            if(pts.second[i].getLastFrameIdx() == lframe){ // if has been observed in the last keyframe
                int f_idx = poses.second.size()-1;
                Mat Tr = (Mat) poses.second[f_idx].orientation.getR4();
                ((Mat)poses.second[f_idx].position).copyTo(Tr(Range(0,3),Range(3,4)));
                Tr.colRange(3,4) += (Mat)(poses.second[f_idx].orientation.getR4() * Matx41d(baseline,0,0,0));
                Matx44d Tr_ = Tr;
                ptH2D feat = K.second * Matx34d::eye() * scale *(Tr_ * pt);
                Point2f feat_right(to_euclidean(feat)(0),to_euclidean(feat)(1));
                ptH2D feat_ =  (K.first * Matx34d::eye()) * (scale * (Tr_ * pt) + Matx41d(baseline,0,0,0));
                Point2f feat_left(to_euclidean(feat_)(0),to_euclidean(feat_)(1));
                if(bb.contains(feat_right) && bb.contains(feat_left)){
                    circle(img_color,feat_left,2,Scalar(0,255,255),-1);
                    circle(img_color,feat_right+Point2f(img.first.cols,0),2,Scalar(0,255,255),-1);
                }
            }
        }
    }

    imshow("img reproj",img_color);
    waitKey(1);
    return img_color;
}


cv::Mat show(const cv::Mat& img, const std::vector<cv::Point2f>& pts){

    assert(img.type() == CV_8UC1 && "image should be 8-bit 1 channel");

    namedWindow("stereo_matches",0);
    cv::RNG rng_cv( 0xFFFFFFFF );

    Mat color_img(img.size(),CV_8U);
    img.convertTo(color_img,CV_8UC3);
    cvtColor(color_img,color_img,CV_GRAY2BGR);

    for(auto& pt : pts){
        int icolor = (unsigned) rng_cv;
        cv::Scalar color( icolor&255, (icolor>>8)&255, (icolor>>16)&255 );
        circle(color_img,pt,10,color,5);
    }
    imshow("stereo_matches", color_img);
    resizeWindow("stereo_matches",1280,480);
    waitKey(100);
    return color_img;
}

cv::Mat display_cov(const string& wname, const cv::Mat& cov, double s){

    cv::RotatedRect ellipse_;
    auto getErrorEllipse = [&](double chisquare_val, const cv::Point2f& mean, const cv::Mat& covmat){

        cv::Mat eigenvalues, eigenvectors;
        cv::eigen(covmat, eigenvalues, eigenvectors);

        //Calculate the angle between the largest eigenvector and the x-axis
        double angle = atan2(eigenvectors.at<double>(0,1), eigenvectors.at<double>(0,0));
        //Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
        if(angle < 0)
            angle += 6.28318530718;
        angle = 180*angle/3.14159265359;

        //Calculate the size of the minor and major axes
        double halfmajoraxissize=chisquare_val*sqrt(eigenvalues.at<double>(0));
        double halfminoraxissize=chisquare_val*sqrt(eigenvalues.at<double>(1));
        //The -angle is used because OpenCV defines the angle clockwise instead of anti-clockwise
        ellipse_ = cv::RotatedRect(mean, cv::Size2f(halfmajoraxissize*40, halfminoraxissize*40), -angle);
    };

    cv::Mat cov_cv = cov(Range(0,3),Range(0,3));
    //Covariance matrix of our data
	cv::Mat covmatxy = (cv::Mat_<double>(2,3) << 1,0,0,0,1,0) * cov_cv * (cv::Mat_<double>(3,2) << 1,0,0,1,0,0);
	cv::Mat covmatxz = (cv::Mat_<double>(2,3) << 1,0,0,0,0,1) * cov_cv * (cv::Mat_<double>(3,2) << 1,0,0,0,0,1);
	cv::Mat covmatyz = (cv::Mat_<double>(2,3) << 0,1,0,0,0,1) * cov_cv * (cv::Mat_<double>(3,2) << 0,0,1,0,0,1);

	double min_,max_;
	cv::Point min_loc,max_loc; minMaxLoc(cov_cv,&min_,&max_,&min_loc,&max_loc);
	s = 1.0/pow(10,floor(log10(sqrt(max_)+1e-20)))/3;

	//Show the result
	cv::Mat visualizeimage(240, 320, CV_8UC3, cv::Scalar::all(0));
	getErrorEllipse(s*2.4477, cv::Point2f(60,120), covmatxz);
	arrowedLine(visualizeimage,cv::Point2f(60,120),cv::Point2f(100,120),cv::Scalar(0,0,255));putText(visualizeimage,"x",cv::Point2f(100,120),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,0,255));
	arrowedLine(visualizeimage,cv::Point2f(60,120),cv::Point2f(60,80),cv::Scalar(255,0,0));putText(visualizeimage,"z",cv::Point2f(60,80),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(255,0,0));
	cv::ellipse(visualizeimage, ellipse_, cv::Scalar::all(255), 2);
	getErrorEllipse(s*2.4477, cv::Point2f(160,120), covmatxy);
	arrowedLine(visualizeimage,cv::Point2f(160,120),cv::Point2f(200,120),cv::Scalar(0,0,255));putText(visualizeimage,"x",cv::Point2f(200,120),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,0,255));
	arrowedLine(visualizeimage,cv::Point2f(160,120),cv::Point2f(160,80),cv::Scalar(0,255,0));putText(visualizeimage,"y",cv::Point2f(160,80),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,255,0));
	cv::ellipse(visualizeimage, ellipse_, cv::Scalar::all(255), 2);
	getErrorEllipse(s*2.4477, cv::Point2f(240,120), covmatyz);
	arrowedLine(visualizeimage,cv::Point2f(240,120),cv::Point2f(240,80),cv::Scalar(255,0,0));putText(visualizeimage,"z",cv::Point2f(240,80),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(255,0,0));
	arrowedLine(visualizeimage,cv::Point2f(240,120),cv::Point2f(280,120),cv::Scalar(0,255,0));putText(visualizeimage,"y",cv::Point2f(280,120),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,255,0));
	cv::ellipse(visualizeimage, ellipse_, cv::Scalar::all(255), 2);
	cv::putText(visualizeimage,"x"+to_string((int)s),cv::Point(280,20),CV_FONT_HERSHEY_COMPLEX,0.4,cv::Scalar::all(255));
	cv::imshow(wname, visualizeimage);
	cv::waitKey(10);
    return visualizeimage;
}

}
