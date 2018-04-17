#ifndef GRAPH3D_H
#define GRAPH3D_H

#include <string>
#include <istream>
#include <thread>

#include "utils.h"

#include <opencv2/core.hpp>
#include <opencv2/viz.hpp>

namespace me{

//! Class for creating 3D graphs.

/*! Create and diplay a camera in a 3D openGL graph. The camera can be noved by adding a new pose (using a 4x4 transformation matrix or an OpenCV Affine3d).*/
class Graph3D
{
    public:
        //! Main constructor. Takes as input the name of the window to be created, boolean to display the trajectory or not and the coordinate system to be used.
        Graph3D(const std::string& name, bool traj=true, bool coordSyst=true);
        ~Graph3D(){m_viz.close();m_t.join();}

        //! Update the camera pose.
        void addCameraPose(const cv::Matx44d& pose);
        //! Update the camera pose.
        void addCameraPose(const cv::Affine3d& pose);
        void addCameraPose(const Quatd& ori, const cv::Vec3d& position);
        void addGPSPosition(const cv::Vec3d& position);
        void addIMUPose(const Quatd& ori, const cv::Vec3d& position);
        //!Add 3D vector
        void add3Vector(const cv::Point3d& vec){cv::viz::WArrow vec_(cv::Point3d(0,0,0),vec);m_viz.showWidget("Vector Widget",vec_);}
        //! Add an image to be displayed in the camera image plane.
        void addImage(const cv::Mat& img){m_image = img.clone();}
        //! reset the view of the widget to get a global view of the trajectory.
        void resetView(){m_viz.resetCamera();}
        void mainloop();

        //! stream operator to add text to the graph (only one line).
        friend std::istream& operator>>(std::istream& is, Graph3D& g);


    private:

    void refresh();

    cv::viz::Viz3d m_viz; //!< OpenGL widget used to modify the 3D scene.
    bool m_traj; //!< Display or not thet trajectory.
    bool m_coordSyst; //!< Coordinate system. True: OpenCV (z-axis forward). False: standard (x-axis forward).
    cv::Mat m_image; //!< Image to be displayed
    std::vector<cv::Affine3d> m_poses; //!< Set of all camera poses.
    std::vector<cv::Affine3d> m_gps; //!< Set of all camera poses.
    std::vector<cv::Affine3d> m_imu; //!< Set of all camera poses.
    cv::Vec3d m_gps_position;
    std::thread  m_t;

};

}

#endif // GRAPH3D_H
