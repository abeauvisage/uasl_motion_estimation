#include "Graph3D.h"

#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Graph3D::Graph3D(const string& name, bool traj) : m_viz(name), m_traj(traj)
{
    m_poses.push_back(Matx44d::eye());
    refresh();
}


void Graph3D::refresh(){

    Affine3d current_pose = m_poses[m_poses.size()-1];

    //displaying camera object, coordinate system and trajectory
    viz::WCameraPosition cpw(Vec2f(1,0.5));
    m_viz.showWidget("Camera Widget",cpw,current_pose);
    m_viz.showWidget("Coordinate system", viz::WCoordinateSystem(/*m_poses.size()/10.0<1?1:current_pose.matrix(2,3)/10.0*/));
    if(m_traj)
        m_viz.showWidget("Trajectory",viz::WTrajectory(Mat(m_poses),viz::WTrajectory::PATH,1.0, viz::Color::green()));

    //set viewer pose to follow the cemera
    Vec3d pos(current_pose.matrix(0,3)+5.0,current_pose.matrix(1,3)-5.0,current_pose.matrix(2,3)+10.0), f(current_pose.matrix(0,3),current_pose.matrix(1,3),current_pose.matrix(2,3)), y(0.0,1.0,0.0);
    Affine3f cam_pose = viz::makeCameraPose(pos,f,y);
    m_viz.setViewerPose(cam_pose);
    m_viz.spinOnce(1,true);
}

void Graph3D::addPose(const cv::Matx44d& pose){
    m_poses.push_back(Affine3d(pose));
    refresh();
}

void Graph3D::addPose(const cv::Affine3d& pose){
    m_poses.push_back(pose);
    refresh();
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
