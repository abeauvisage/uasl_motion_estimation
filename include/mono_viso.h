#ifndef MONOVISUALODOMETRY_H
#define MONOVISUALODOMETRY_H

#include "viso.h"
#include "featureType.h"

namespace me{

//! Visual Odometry class for stereovision

/*! This class compute motion from monocular images.
    It computes and extract the rotation and translation from the Essential matrix using the OpenCV library.
    A RANSAC outlier rejection scheme is used to keep only good matches.
 */
class MonoVisualOdometry : public VisualOdometry
{

public:
    //! Mono parameters
    /*! contains calibration parameters and info about the method used. */
    struct parameters  {
        bool ransac;                //! use of RANSAC or not.
        double prob;                //! level of confidence for the estimation (probability).
        double  inlier_threshold;   //! inlier threshold for RANSAC.
        double fu;                   //! focal length of the camera.
        double fv;                   //! focal length of the camera.
        double cu;                  //! principal point.
        double cv;                  //! principal point.
        parameters () {
            ransac=true;
            prob=0.99;
            inlier_threshold = 1.0;
            fu=1;
            fv=1;
            cu=0;
            cv=0;
        }
    };
        /*! Main constructor. Takes a set of stereo parameters as input. */
        MonoVisualOdometry(parameters param=parameters());

        /*! Function processing a set of matches. Estimate the Essential matrix and extract R and t.
            Returns true if a motion is successfuly computed, false otherwise. */
        bool process(const std::vector<StereoMatch<cv::Point2f>>& matches);
        virtual cv::Mat getMotion(){return m_Rt;} //!< Returns the Transformation matrix.
        std::vector<int> getInliersIdx(){return m_inliers;} //!< Returns the set of inlier indices.
        std::vector<int> getOutliersIdx(){return m_outliers;} //!< Returns the set of outlier indices.
        std::vector<ptH3D> get3Dpts(){return m_pts;} //!< Returns thet set of 3D features.
        cv::Mat getEssentialMat(){return m_E;} //!< Returns the Essential matrix.

    private:
        parameters m_param;         //!< Mono parameters.
        cv::Mat m_K;                //! intrinsic Matrix.
        cv::Mat m_Rt;               //!< Transformation matrix.
        cv::Mat m_E;                //!< Essential matrix.
        std::vector<int> m_inliers; //!< List of inliers.
        std::vector<int> m_outliers; //!< List of outliers.
        std::vector<ptH3D> m_pts; //! List of 3D points

        double findRelativeScale(std::vector<std::pair<me::ptH3D,me::ptH3D>>& pts);

};
}

#endif // MONOVISUALODOMETRY_H
