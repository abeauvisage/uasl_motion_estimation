#include "Graph3D.h"

#include <opencv2/highgui.hpp>
#include <iostream>

#include "utils.h"

using namespace std;
using namespace cv;

namespace me{

Graph3D::Graph3D(const string& name, bool traj, bool coordSyst) : m_viz(name), m_traj(traj), m_coordSyst(coordSyst)
{
    m_poses.push_back(Matx44d::eye());
    m_t = thread(&Graph3D::mainloop,this);
}

void Graph3D::mainloop(){

    m_viz.showWidget("Coordinate system", viz::WCoordinateSystem(1));
    m_viz.showWidget("Trajectory",viz::WTrajectory(vector<Affine3d>(1,Affine3d::Identity())));
//    m_viz.showWidget("GPS track",viz::WTrajectory(vector<Affine3d>(1,Affine3d::Identity())));
    m_viz.showWidget("IMU track",viz::WTrajectory(vector<Affine3d>(1,Affine3d::Identity())));
    while(!m_viz.wasStopped()){
        m_viz.spinOnce(100);
    }
}


void Graph3D::refresh(){

    Affine3d correction;
    if(!m_coordSyst){
        Euler<double> e(0,-90,90,false);
        correction = Affine3d(e.getR4());
    }else
        correction = Affine3d(Matx44d::eye());

//    for(uint i=0;i<m_poses.size();i++){
//

//    m_viz.showWidget("Coordinate system", viz::WCoordinateSystem(/*m_poses.size()/10.0<1?1:current_pose.matrix(2,3)/10.0*/));

    //set viewer pose to follow the cemera
//    if(m_coordSyst){
//        Vec3d pos(current_pose.matrix(0,3)+5.0,current_pose.matrix(1,3)-5.0,current_pose.matrix(2,3)+10.0), f(current_pose.matrix(0,3),current_pose.matrix(1,3),current_pose.matrix(2,3)), y(0.0,1.0,0.0);
//        Affine3f cam_pose = viz::makeCameraPose(pos,f,y);
//        m_viz.setViewerPose(cam_pose);
//    }else{
//        Vec3d pos(current_pose.matrix(0,3)-5.0,current_pose.matrix(1,3)-2.0,current_pose.matrix(2,3)+10.0), f(current_pose.matrix(0,3),current_pose.matrix(1,3),current_pose.matrix(2,3)), y(0.0,0.0,-1.0);
//        Affine3f cam_pose = viz::makeCameraPose(pos,f,y);
//        m_viz.setViewerPose(cam_pose);
//    }
//    m_viz.spinOnce(1,true);
}

void Graph3D::addGPSPosition(const cv::Vec3d& position){

    Matx44d pose = Matx44d::eye();
    pose(0,3) = position(0);
    pose(1,3) = position(2);
    pose(2,3) = position(1);
    m_gps.push_back(Affine3d(pose));
//    m_viz.showWidget("GPS track",viz::WTrajectorySpheres(Mat(m_gps),10,0.05,viz::Color::red(),viz::Color::red()));
}

void Graph3D::addIMUPose(const Quatd& ori, const cv::Vec3d& position){
    Matx44d pose = ori.getR4().t();
    pose(0,3) = position(0);
    pose(1,3) = position(1);
    pose(2,3) = position(2);
    m_imu.push_back(Affine3d(pose));
//    m_viz.removeWidget("IMU track");
//    m_viz.showWidget("IMU track",viz::WTrajectorySpheres(m_imu,10,0.05,viz::Color::yellow(),viz::Color::yellow()));
}

void Graph3D::addCameraPose(const Quatd& ori, const cv::Vec3d& position){
    Matx44d pose = ori.getR4().t();
    pose(0,3) = position(0);
    pose(1,3) = position(1);
    pose(2,3) = position(2);
    viz::WCameraPosition cpw;
    cpw = viz::WCameraPosition(Vec2f(1,0.5));
    m_poses.push_back(Affine3d(pose));
    m_viz.showWidget("Camera_widget_"+to_string(m_poses.size()-1),cpw,Affine3d(pose));
//    if(m_traj){
//        m_viz.removeWidget("Trajectory");
//        m_viz.showWidget("Trajectory",viz::WTrajectory(m_poses,viz::WTrajectory::PATH,1.0, viz::Color::green()));
//    }
}


void Graph3D::addCameraPose(const cv::Matx44d& pose){
    viz::WCameraPosition cpw;
    cpw = viz::WCameraPosition(Vec2f(1,0.5));
    m_poses.push_back(Affine3d(pose));
    m_viz.showWidget("Camera_widget_"+to_string(m_poses.size()-1),cpw,Affine3d(pose));
    if(m_traj)
        m_viz.showWidget("Trajectory",viz::WTrajectory(Mat(m_poses),viz::WTrajectory::PATH,1.0, viz::Color::green()));
}

void Graph3D::addCameraPose(const cv::Affine3d& pose){
    viz::WCameraPosition cpw;
    cpw = viz::WCameraPosition(Vec2f(1,0.5));
    m_poses.push_back(pose);
    m_viz.showWidget("Camera_widget_"+to_string(m_poses.size()-1),cpw,pose);
    if(m_traj)
        m_viz.showWidget("Trajectory",viz::WTrajectory(Mat(m_poses),viz::WTrajectory::PATH,1.0, viz::Color::green()));
}

std::istream& operator>>(std::istream& is, Graph3D& g){
    string ts="";
    for(int i=0;i<15;i++){
        string s;is >> s;
        ts += s;
    }
    viz::WText text(ts,Point(0,10),10);
    g.m_viz.showWidget("Text Widget",text);

    return is;
}

}
