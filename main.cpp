#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "opencv2/opencv.hpp"
//#include <opencv2/viz.hpp>
#include <iomanip>

#include "viso_stereo.h"
#include "Graph2D.h"
#include "matcher.h"

using namespace std;
using namespace cv;

ifstream logFile;


cv::RNG rng( 0xFFFFFFFF );

int main()
{
//    string dir = "/home/abeauvisage/Insa/PhD/datasets/Indoor-2nd_calib_2nd_rec/";
//    string dir = "/home/abeauvisage/Insa/PhD/datasets/16_01_15/seq_MO_15_01_16_rec";
//    string dir = "/home/abeauvisage/Insa/PhD/datasets/16_03_04/seq2_rectified";
//    string dir = "/home/abeauvisage/Insa/PhD/datasets/16_01_15/seq_MO_15_01_1_4_rec";
//    string dir = "/home/abeauvisage/Insa/PhD/datasets/seq_MO_5-06-15_4_rectified";
//    string dir = "/home/abeauvisage/Insa/PhD/datasets/seq_11_01_3_rec4";
//    string dir = "/home/abeauvisage/Insa/PhD/datasets/KAIST_new2_rectified";
//    string dir = "/home/abeauvisage/Insa/PhD/datasets/seq_MO_original_rectified";
//    string dir = "/home/abeauvisage/Documents/datasets/16_05_26/test_16_05_26-a_rec";
//    string dir = "/home/abeauvisage/Insa/PhD/datasets/16_03_14/test_16_03_14_2_rectified";
//    string dir = "/home/abeauvisage/Documents/datasets/16_08_30/test_16_08_30-b_rec";
    string dir = "/home/abeauvisage/Insa/PhD/datasets/seq_MO_original_rectified";

    ifstream data(dir+"/matches_7_bud.csv");
    ifstream gps(dir+"/gps_traj.txt");

    ofstream file,proj;
    file.open(dir+"/new_coord.txt");
//    proj.open(dir+"/projection.csv");

    Graph2D g("Trajectory",2);
    g.addLegend("estimation",1);
    g.addLegend("gps",2);
    vector<CvPoint2D32f> pts;

    VisualOdometryStereo::parameters param;

//    param.calib.f  = 1162.1022; // focal length in pixels
//    param.calib.cu = 247.034; // principal point (u-coordinate) in pixels
//    param.calib.cv = 159.740; // principal point (v-coordinate) in pixels
//    param.base     = 0.122; // baseline in meters

//    param.calib.f     = 581.84094209989632;
//    param.calib.cu    = 331.95452117919922;
//    param.calib.cv    = 245.56638336181641;
//    param.base  = 0.22528522596647409;

//    param.calib.f     = 595.58540493568330;
//    param.calib.cu    = 299.91595268249512;
//    param.calib.cv    = 252.80833435058594;
//    param.base  = 0.22685299763003069;

//    param.calib.f     = 518.39660666433156/1.6;
//    param.calib.cu    = 646.37947463989258/1.6;
//    param.calib.cv    = 526.57693481445312/1.6;
//    param.base  = 0.25;

//    param.calib.f     = 3235.57313348954/2;
//    param.calib.cu    = 577.682064808143/2;
//    param.calib.cv    = 482.450866361120/2;
//    param.base  = 0.2460029;

//    param.calib.f     = 1457.5949601852;
//    param.calib.cu    = 310.461045453217;
//    param.calib.cv    = 245.800749377711;
//    param.base  = 0.2460029;

//    param.calib.f     = 3238.02421555377/2;
//    param.calib.cu    = 551.399086833990/2;
//    param.calib.cv    = 514.299013570319/2;
//    param.base  = 0.2460029;

//    param.f1     = 5.9578119773793446e+02;
//    param.f2     = 5.9578119773793446e+02;
//    param.cu1    = 3.3541476821899414e+02;
//    param.cu2    = 3.3541476821899414e+02;
//    param.cv1    = 2.3775906753540039e+02;
//    param.cv2    = 2.3775906753540039e+02;
//    param.baseline  = 2.8330214554049502e-01;

//    param.f1     = 5.9255299209557393e+02;
//    param.f2     = 5.9255299209557393e+02;
//    param.cu1    = 3.3553755187988281e+02;
//    param.cu2    = 3.3553755187988281e+02;
//    param.cv1    = 2.3892620849609375e+02;
//    param.cv2    = 2.3892620849609375e+02;
//    param.baseline  = 3.0877265416681990e-01;
//    param.n_ransac = 0;

//calib 16_08_30
//    param.f1     = 5.7900686502448991e+02;
//    param.f2     = 5.7900686502448991e+02;
//    param.cu1    = 3.4719164657592773e+02;
//    param.cu2    = 3.4719164657592773e+02;
//    param.cv1    = 2.3975083160400391e+02;
//    param.cv2    = 2.3975083160400391e+02;
//    param.baseline  = 2.8724344758286230e-01;
//    param.method = VisualOdometryStereo::LM;

//calib spanish
    param.f1     = 1162.1022;
    param.f2     = 1162.1022;
    param.cu1    = 247.034;
    param.cu2    = 247.034;
    param.cv1    = 159.740;
    param.cv2    = 159.740;
    param.baseline  = 0.122;
    param.method = VisualOdometryStereo::GN;
//    param.reweighting = true;
//    param.method = VisualOdometryStereo::LM;
    param.ransac = true;
    param.inlier_threshold = 2;


//    param.optim = 2;
//        param.reweighting = false;
//    param.inlier_threshold = 10;

    VisualOdometryStereo viso(param);
    Mat pose = Mat::eye(Size(4,4),CV_64F);

    string line;

    namedWindow("imgL",0);
    namedWindow("imgR",0);

    cv::moveWindow("imgL",900,0);
    cv::moveWindow("imgR",900,400);


    double x=0,y=0,z=0;
    vector<float> errors;

    if(!gps.is_open())
        cout << "cannot get gps" << endl;

    if(!data.is_open())
        cout << "cannot get matches" << endl;

    /**************/

    int skip =7;
    int test_frame=0;
    int nframe=65;  //dataset 03_07 -> d1: 250 d2: 230 d4-> 600

    /***************/
    string init_gx,init_gy;
    string gpsdata;
    if(gps.is_open()){
        for(int i=0;i<nframe;i++)
            getline(gps,gpsdata);
        stringstream ss(gpsdata);
        getline(ss,init_gx,',');
        getline(ss,init_gy,',');
    }

    /*** viz ***/

//    viz::Viz3d myWindow("coordinate frame");
//    myWindow.showWidget("coordinate widget", viz::WCoordinateSystem());
//    myWindow.spin();

    /***********/


//    waitKey(0);

    if(data.is_open()){
        vector<Matcher::p_match> p_matched;
        vector<StereoOdoMatches<cv::Point2f>> matched;
        double mean_matches=0;
        double mean_inliers=0;
        int nb =0;
        while ( getline (data,line))
        {
            if(line == ""){

                /******* gps data ******/

                string gx,gy;
                 if(gps.is_open()){
                    getline(gps,gpsdata);
                    stringstream ss(gpsdata);
                    getline(ss,gx,',');
                    getline(ss,gy,',');

                    for(int i=0;i<skip-1;i++)
                        getline(gps,gpsdata);

                }

                /*************/

                cout << "new line" << endl;

                if(nframe > test_frame){

                    stringstream num;num <<  std::setfill('0') << std::setw(5) << nframe; // frame number with 5 digit format
                    cout << "### frame  " << nframe << " ###" << endl;

                    Mat imgLc = imread(dir+"/cam0_image"+num.str()+"_rec.png",0);
                    Mat imgRc = imread(dir+"/cam1_image"+num.str()+"_rec.png",0);


                    cv::Mat imgL_color(imgLc.size(),CV_8UC3),imgR_color(imgLc.size(),CV_8UC3);
                    imgLc.convertTo(imgL_color,CV_8UC3);
                    cv::cvtColor(imgL_color,imgL_color,CV_GRAY2BGR);
                    imgRc.convertTo(imgR_color,CV_8UC3);
                    cv::cvtColor(imgR_color,imgR_color,CV_GRAY2BGR);

                    for(int i=0;i<p_matched.size();i++){
                        circle(imgL_color,Point(p_matched[i].u1c,p_matched[i].v1c),3,cv::Scalar(0,0,255));
                        circle(imgR_color,Point(p_matched[i].u2c,p_matched[i].v2c),3,cv::Scalar(0,0,255));
                    }

                    cv::line(imgL_color,Point(param.cu1,0),Point(param.cu1,imgL_color.rows),cv::Scalar(0,0,0));
                    cv::line(imgL_color,Point(0,param.cv1),Point(imgL_color.cols,param.cv1),cv::Scalar(0,0,0));
                    cv::line(imgR_color,Point(param.cu2,0),Point(param.cu2,imgL_color.rows),cv::Scalar(0,0,0));
                    cv::line(imgR_color,Point(0,param.cv2),Point(imgL_color.cols,param.cv2),cv::Scalar(0,0,0));

                    /**** processing 3D pts ****/

                    if(viso.process(matched)){
                        viso.updatePose();
                        pose = viso.getPose();
                    }
                    else
                        cout << "failed" << endl;
//
//                    Mat new_pose =viso.getMotion();
//                    proj << new_pose << endl;
//                    Mat inv;invert(new_pose,inv);
//                    if(abs(new_pose.at<double>(0,3)) < 10 && abs(new_pose.at<double>(2,3)) < 10)
//                        pose *= inv;

//                    pose = viso.getPose();
//                    cout << pose << endl;

                    x = pose.at<double>(0,3);
                    y = pose.at<double>(2,3);
                    z = pose.at<double>(1,3);
                    vector<int> indices = viso.getInliers_idx();
                    cout << "nb inliers " << indices.size() << endl;
                    mean_inliers += indices.size();
                    mean_matches += p_matched.size();
                    nb++;

                    for(int i=0;i<indices.size();i++){
                        int icolor = (unsigned) rng;
                        cv::Scalar color( icolor&255, (icolor>>8)&255, (icolor>>16)&255 );
                        circle(imgL_color,Point(p_matched[indices[i]].u1c,p_matched[indices[i]].v1c),3,cv::Scalar(255,0,0));
                        circle(imgR_color,Point(p_matched[indices[i]].u2c,p_matched[indices[i]].v2c),3,cv::Scalar(255,0,0));
    //                    cout << param.base*param.calib.f/(p_matched[i].u1c-p_matched[i].u2c) << " " << p_matched[i].u1c-p_matched[i].u2c << endl;
    //                    imshow("imgL",imgL_color);
    //                    imshow("imgR",imgR_color);
    //                    waitKey(0);
                    }

                    imshow("imgL",imgL_color);
                    imshow("imgR",imgR_color);


                    file << x << ";" << y << ";" << z << ";" << nframe<< endl;

                    CvPoint2D32f point;
                    point.x=(float)x;point.y=(float)y;
                    pts.push_back(point);

    //                circle(traj, Point((int)(x*5+ 600), (int)(y*5+ 1000)) ,1, CV_RGB(255,0,0), 2);
                    Point2f p1(x,y);
                    cout << "[" << x << "," << y << "]" << endl;
                    g.addValue(p1,1);
                    if(gps.is_open()){
    //                    Point2f p2(stof(gx)-0.024667,stof(gy)-1.966435);
                        Point2f p2(stof(gx)+stof(init_gx),stof(gy)-stof(init_gy));
                        g.addValue(p2,2);
                        cout << "GPS " << p2 << endl;
                        cout << "diff = " << sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2)) << endl;
                        cout << "error = " << sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2))*100/g.getLength() << " %" << endl;
                        errors.push_back(sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2)));
                    }
                    else{
                        Point2f p2(pose.at<double>(2,1) + pose.at<double>(0,3),pose.at<double>(0,1) + pose.at<double>(2,3));
                        g.addValue(p2,2);
                    }
                waitKey(10);
                }
                nframe+=skip;
                p_matched.clear();
                matched.clear();
            }
            else{
                stringstream ss(line);

                string m11,m12,m21,m22,m31,m32,m41,m42,f;
                getline(ss,m11,',');
                getline(ss,m12,',');
                getline(ss,m21,',');
                getline(ss,m22,',');
                getline(ss,m31,',');
                getline(ss,m32,',');
                getline(ss,m41,',');
                getline(ss,m42,',');

                if(stof(m12) < 400 && stof(m12) > 100){
//                if( param.base*param.calib.f/(stof(m11)-stof(m21)) > 0 && param.base*param.calib.f/(stof(m11)-stof(m21)) < 30)
                    p_matched.push_back(Matcher::p_match(stof(m11),stof(m12),0,stof(m21),stof(m22),0,stof(m31),stof(m32),0,stof(m41),stof(m42),0));
                    matched.push_back(StereoOdoMatches<Point2f>(Point2f(stof(m11),stof(m12)),Point2f(stof(m21),stof(m22)),Point2f(stof(m31),stof(m32)),Point2f(stof(m41),stof(m42))));
//                    cout << p_matched[p_matched.size()-1].u1p << "," << p_matched[p_matched.size()-1].v1p << "," << p_matched[p_matched.size()-1].u2p << "," << p_matched[p_matched.size()-1].v2p << endl;
//                    cout << "frame: " << f << endl;
                }
            }
        }
        cout << "mean_macthes: " << mean_matches/nb << endl;
        cout << "mean_inliers: " << mean_inliers/nb << endl;
        data.close();
    }

    float mean_error =0;
    for(int i=0;i<errors.size();i++)
        mean_error += errors[i];
    mean_error /= errors.size();

    cout << " Global error = " << mean_error*100/g.getLength() << " %" << endl;
    cout << "distance: " << g.getLength() << endl;

    file.close();
    return waitKey();
}
