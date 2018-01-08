#ifndef STEREO_VISO_H
#define STEREO_VISO_H

#include <iostream>

#include "viso.h"
#include "featureType.h"

namespace me{

//! Visual Odometry class for stereovision

/*! This class compute motion from stereo images using a bundle ajdjustment approach.
    It is minimizing the reprojection error of different features in the images.
    Features needs to be matched in the two consecutive stereo pairs (quad-mathing).
    A RANSAC outlier rejection scheme is used to keep only good matches.
 */
class StereoVisualOdometry: public VisualOdometry {

public:
    //! Method used for optimization. Either Gauss-Newton or Levenberg-Maquart.
    enum Method {GN, LM};

    //! Stereo parameters
    /*! contains calibration parameters and info about the method used. */
    struct parameters  {
        double  baseline;           //!< distance between cameras.
        Method method;              //!< optimization method used.
        int n_ransac;               //!< nb of iterations used for the RANSAC process.
        bool ransac;                //!< use of not if RANSAC.
        double  inlier_threshold;   //!< error threshold for a match to be considered inlier.
        bool    reweighting;        //!< use weight deping on feature distance (deprecated).
        double fu1,fv1,fu2,fv2;     //!< focal length of each camera.
        double cu1,cu2;             //!< principal point in u (horizontal).
        double cv1,cv2;             //!< principal point in v (vertical).
        double step_size;           //!< step size for optimization (TO BE USED).
        double eps,e1,e2,e3,e4;     //!< optimization thresholds.
        int max_iter;               //!< max_iteration for optim algo to converge.
        parameters () {
            method = GN;
            fu1=1;fv1=1;fu2=1;fv2=1;
            cu1=0;cu2=0;
            cv1=0;cv2=0;
            baseline = 1.0;
            n_ransac = 200;
            inlier_threshold = 2.0;
            ransac=true;
            reweighting = true;
            step_size = 1;
            eps = 1e-9;
            e1 = 1e-3;
            e2 = 1e-12;
            e3 = 1e-12;
            e4 = 1e-15;
            max_iter=100;
        }
    };

    /*! Main constructor. Takes a set of stereo parameters as input. */
    StereoVisualOdometry(parameters param=parameters()):m_param(param){};

    /*! Function processing a set of matches. Project features into 3D and run optimization.
        Uses RANSAC if specified. Matches should be StereoOdoMatches. Returns true if optimization worked and
        a motion has been computed, false otherwise. */
    bool process(const std::vector<StereoOdoMatchesf>& matches, cv::Mat init = cv::Mat::zeros(6,1,CV_64F));
    //! returns the transformation matrix corresponding to the current camera pose.
    virtual cv::Mat getMotion();

    //getters
    std::vector<ptH3D> getPts3D(){return m_pts3D;}
    std::vector<int> getInliers_idx(){return m_inliers_idx;}
    std::vector<std::pair<ptH2D,ptH2D>> getPredictions(){return reproject(m_state,m_inliers_idx);}
    parameters getParams(){return m_param;}


private:

    std::vector<ptH3D> m_pts3D; //!< set of 3D features in previous coordinate frame.
    std::vector<float> m_disparities;  //!< disparity of each feature.
    std::vector<int> m_inliers_idx; //!< indices of inliers.

    parameters m_param; //!< Stereo parameters

    cv::Mat m_J;    //!< Jacobian matrix (6x4).
    cv::Matx61d m_state; //!< State vector. Contains the 3 Euler angles and the translation.
    std::vector<std::pair<ptH2D,ptH2D>> m_obs; //!< location of observed features in the current stereo pair.
    cv::Mat m_res; //!< residual matrix (1x4*nb_matches).

    /*! project a set of StereoMatch (2 matches) into 3D points. StereoMatch has float precision. */
    void project3D(const std::vector<StereoMatchf>& features);
    /*! project a set of StereoOdoMatch (4 matches) into 3D points. StereoOdoMatch has float precision. */
    void project3D(const std::vector<StereoOdoMatchesf>& features);
    /*! optimization function. Update the state vector with the matches corresponding to the indices in the selection vector.
        Returns true if the optimization worked, false otherwise. */
    bool optimize(const std::vector<int>& selection, bool weight);
    /*! reproject a set of 3D points (from the indices in selection).
        The transformation described by the state vector is applied to the 3D points before been reprojected.
        Returns a set of 2D locations in the current stereo pair.*/
    std::vector<std::pair<ptH2D,ptH2D>> reproject(cv::Matx61d& state, const std::vector<int>& selection);
    /*! Update the observation vector from the matches. */
    void updateObservations(const std::vector<StereoOdoMatchesf>& matches);
    /*! Compute and upadate the Jacobian matrix from the current state vector and the matches provided */
    void updateJacobian(const std::vector<int>& selection);
    /*! compute the inliers form the current state vector. Matches are selected as inliers when their reprojection error
        is smaller than the specified threshold.
        Returns the list of inliers indices.*/
    std::vector<int> computeInliers();
    /*! Select N features randomly to be used in the RANSAC process.
        nb_tot corresponds to the total nb of features being processed at the time the function is called.
        Returns a list of indices. */
    std::vector<int> selectRandomIndices(int nb_samples, int nb_tot);
};

}
#endif // VSTEREO_VISO_H

