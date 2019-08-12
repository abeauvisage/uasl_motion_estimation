#ifndef GRAPH3D_H
#define GRAPH3D_H

/** \file Graph3D.h
*   \brief Creates and update an OpenCV viz 3D graph
*
*	add/update 6-dof camera and IMU poses
*	add/update 3D point cloud and GPS coordinates
*
*    \author Axel Beauvisage (axel.beauvisage@gmail.com)
*/

#include "core/feature_types.h"

#include <opencv2/viz.hpp>

#include <istream>
#include <thread>
#include <mutex>


namespace me{

//! Class for creating 3D graphs.

/*! Create and diplay a camera in a 3D openGL graph. The camera can be noved by adding a new pose (using a 4x4 transformation matrix or an OpenCV Affine3d).*/
class Graph3D
{
    public:
        //! Main constructor. Takes as input the name of the window to be created, boolean to display the trajectory or not and the coordinate system to be used.
        Graph3D(const std::string& name, bool traj=true);
        //! destructor. Stops viz and join the thread
        ~Graph3D(){m_viz.close();m_t.join();}

        //! Add a new camera pose.
        void addCameraPose(const cv::Matx44d& pose);
        //! Add a new camera pose.
        void addCameraPose(const cv::Affine3d& pose);
        //! Add a new camera pose.
        void addCameraPose(const Quatd& ori, const cv::Vec3d& position);
        //! Update the camera pose.
        void updateCameraPose(const cv::Affine3d& pose,unsigned int idx);
        //! Update the camera pose.
        void updateCameraPose(const Quatd& ori, const cv::Vec3d& position, unsigned int idx);
        //! Update the camera pose.
        void updateCameraPose(const cv::Matx44d& pose, unsigned int idx);
        //! Add a new GPS coordinate.
        void addGPSPosition(const cv::Vec3d& position);
        //! Add a new IMU pose.
        void addIMUPose(const Quatd& ori, const cv::Vec3d& position);
        //! Add a cloud of 3D points.
        void add3Dpts(const std::vector<me::pt3D>& points);
        //! Update the cloud of 3D points.
        void update3Dpts(const std::vector<me::pt3D>& points);
        //! Add 3D vector
        void add3Vector(const cv::Point3d& vec){std::lock_guard<std::mutex> lock(m_mutex);cv::viz::WArrow vec_(cv::Point3d(0,0,0),vec);m_viz.showWidget("Vector Widget",vec_);}
        //! Add an image to be displayed in the camera image plane.
        void addImage(const cv::Mat& img){m_image = img.clone();}
        //! Reset the view of the widget to get a global view of the trajectory.
        void resetView(){std::lock_guard<std::mutex> lock(m_mutex);m_viz.resetCamera();}
        //! Remove all widgets and reset view.
        void reset(){std::lock_guard<std::mutex> lock(m_mutex);m_viz.removeAllWidgets();resetView();}
        //! viz display mainloop.
        void mainloop();
        //! controls the position of the window
        void moveWindow(const int x, const int y){m_viz.setWindowPosition(cv::Point(x,y));}
		//! retrieve the current view of the graph
        cv::Mat getScreenshot(){std::lock_guard<std::mutex> lock(m_mutex);return m_viz.getScreenshot();}
        //! returns the number of poses saved
        int getNbCameraPoses(){return m_poses.size();}
        //! display a message at the bottom of the window
        void display_text(const std::string& txt);
        //! stream operator to add text to the graph (only one line).
        friend std::istream& operator>>(std::istream& is, Graph3D& g);


    private:

	//! main function which should be called every time the object is updated
    void refresh();

    cv::viz::Viz3d m_viz; //!< OpenGL widget used to modify the 3D scene.
    bool m_traj; //!< Display or not thet trajectory.
    bool m_coordSyst; //!< Coordinate system. True: OpenCV (z-axis forward). False: standard (x-axis forward).
    cv::Mat m_image; //!< Image to be displayed
    std::vector<cv::Affine3d> m_poses; //!< Set of all camera poses.
    std::vector<cv::Affine3d> m_gps; //!< Set of all GPS coordinates.
    std::vector<cv::Affine3d> m_imu; //!< Set of all IMU poses.
    cv::Mat m_pts; //!< Set of 3D points.
    std::thread  m_t; //!< Thread to run display loop.
    std::mutex m_mutex; //!< Mutex for multi-threading display/update.

};

}

#endif // GRAPH3D_H
