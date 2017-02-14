#ifndef GRAPH3D_H
#define GRAPH3D_H

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/viz.hpp>

#include <istream>

namespace me{

class Graph3D
{
    public:
        Graph3D(const std::string& name, bool traj=true, bool coordSyst=true);

        void addPose(const cv::Matx44d& pose);
        void addPose(const cv::Affine3d& pose);
        void addImage(const cv::Mat& img){m_image = img.clone();}
        void resetView(){m_viz.resetCamera();}
        void show(){m_viz.spin();}

        friend std::istream& operator>>(std::istream& is, Graph3D& g);


    private:

    void refresh();

    bool m_traj;
    bool m_coordSyst;
    cv::Mat m_image;
    cv::viz::WCube m_cube;
    cv::viz::Viz3d m_viz;
    std::vector<cv::Affine3d> m_poses;

};

}

#endif // GRAPH3D_H
