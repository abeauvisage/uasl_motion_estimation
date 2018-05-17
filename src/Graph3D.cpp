#include "Graph3D.h"

#include <opencv2/highgui.hpp>
#include <iostream>

#include "utils.h"

using namespace std;
using namespace cv;

namespace me{

Graph3D::Graph3D(const string& name, bool traj, bool coordSyst) : m_viz(name), m_traj(traj), m_coordSyst(coordSyst)
{
    m_viz.setBackgroundColor(viz::Color::black(),viz::Color::gray());
    m_viz.showWidget("Coordinate system", viz::WCoordinateSystem(1));
    m_viz.showWidget("Camera_widget",viz::WTrajectoryFrustums(m_poses,Vec2d(1,0.5)));
    m_viz.resetCamera();
    m_t = thread(&Graph3D::mainloop,this);
}

void Graph3D::mainloop(){

    while(!m_viz.wasStopped()){
        {
        lock_guard<mutex> lock(m_mutex);
        m_viz.spinOnce(10);
        }
        this_thread::sleep_for(chrono::milliseconds(5));
    }
}


void Graph3D::refresh(){

    lock_guard<mutex> lock(m_mutex);


    if(m_traj)
        m_viz.showWidget("Trajectory",viz::WTrajectory(m_poses,viz::WTrajectory::PATH,1.0,viz::Color::green()));
    m_viz.showWidget("GPS track",viz::WTrajectorySpheres(m_gps,0.05,0.007,viz::Color::blue(),viz::Color::green()));
    m_viz.showWidget("IMU track",viz::WTrajectorySpheres(m_imu,0.05,0.07,viz::Color::red(),viz::Color::red()));
    m_viz.showWidget("Camera_widget",viz::WTrajectoryFrustums(m_poses,Vec2d(1,0.5),1.0,viz::Color::blue()));

    if(!m_pts.empty()){
        viz::WCloud wcloud(m_pts);
        m_viz.showWidget("CLOUD",wcloud);
        m_viz.setRenderingProperty("CLOUD",viz::POINT_SIZE,3);
    }
}

void Graph3D::add3Dpts(const vector<me::pt3D>& points){

    Mat new_pts = Mat::zeros(points.size(),1,CV_64FC3);
    for(uint i=0;i<points.size();i++)
        new_pts.at<Matx31d>(i) = points[i];

   m_pts.push_back(new_pts);
}

void Graph3D::update3Dpts(const vector<me::pt3D>& points){

    m_pts = Mat::zeros(points.size(),1,CV_64FC3);
    for(uint i=0;i<points.size();i++)
        m_pts.at<Matx31d>(i) = points[i];
}

void Graph3D::addGPSPosition(const cv::Vec3d& position){

    Matx44d pose = Matx44d::eye();
    pose(0,3) = position(0);
    pose(1,3) = position(2);
    pose(2,3) = position(1);
    m_gps.push_back(Affine3d(pose));
}

void Graph3D::addIMUPose(const Quatd& ori, const cv::Vec3d& position){
    Matx44d pose = ori.getR4().t();
    pose(0,3) = position(0);
    pose(1,3) = position(1);
    pose(2,3) = position(2);
    m_imu.push_back(Affine3d(pose));
}

void Graph3D::addCameraPose(const Quatd& ori, const cv::Vec3d& position){

    Matx44d pose = ori.getR4().t();
    pose(0,3) = position(0);
    pose(1,3) = position(1);
    pose(2,3) = position(2);
    m_poses.push_back(Affine3d(pose));
    refresh();
}


void Graph3D::addCameraPose(const cv::Matx44d& pose){
    m_poses.push_back(Affine3d(pose));
}

void Graph3D::addCameraPose(const cv::Affine3d& pose){
    m_poses.push_back(pose);
}

void Graph3D::updateCameraPose(const Quatd& ori, const cv::Vec3d& position, int idx){
    Matx44d pose = ori.getR4().t();
    pose(0,3) = position(0);
    pose(1,3) = position(1);
    pose(2,3) = position(2);
    m_poses[idx] = Affine3d(pose);
    refresh();
}

void Graph3D::updateCameraPose(const cv::Matx44d& pose, int idx){
    m_poses[idx] = Affine3d(pose);
}

void Graph3D::updateCameraPose(const cv::Affine3d& pose, int idx){
    m_poses[idx] = pose;
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
