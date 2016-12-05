#ifndef GRAPH3D_H
#define GRAPH3D_H

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/viz.hpp>

class Graph3D
{
    public:
        Graph3D(const std::string& name);

        void addPose(const cv::Matx44d& pose);
        void show(){m_viz.spin();}

    private:

    void refresh();

    cv::viz::WCube m_cube;
    cv::viz::Viz3d m_viz;
    std::vector<cv::Affine3d> m_poses;

};

#endif // GRAPH3D_H
