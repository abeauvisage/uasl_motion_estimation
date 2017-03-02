#ifndef STEREO_VISO_H
#define STEREO_VISO_H

#include <iostream>

#include "viso.h"
#include "featureType.h"

namespace me{

class StereoVisualOdometry: public VisualOdometry {

public:

    enum Method {GN, LM};

    struct parameters  {
        double  baseline;
        Method method;
        int n_ransac;
        bool ransac;
        double  inlier_threshold;
        bool    reweighting;
        double f1,f2;
        double cu1,cu2;
        double cv1,cv2;
        double step_size;
        double eps,e1,e2,e3,e4;
        int max_iter;
        parameters () {
            method = GN;
            f1=1;f2=1;
            cu1=0;cu2=0;
            cv1=0;cv2=0;
            baseline = 1.0;
            n_ransac = 200;
            inlier_threshold = 2.0;
            ransac=true;
            reweighting = true;
            step_size = 1;
            eps = 1e-8;
            e1 = 1e-3;
            e2 = 1e-3;
            e3 = 1e-1;
            e4 = 1e-1;
            max_iter=20;
        }
    };


    StereoVisualOdometry(parameters param=parameters());
    ~StereoVisualOdometry(){};

    bool process(const std::vector<StereoOdoMatchesf>& matches, cv::Mat init = cv::Mat::zeros(6,1,CV_64F));
    virtual cv::Mat getMotion();

    std::vector<ptH3D> getPts3D(){return m_pts3D;}
    std::vector<int> getInliers_idx(){return m_inliers_idx;}
    std::vector<std::pair<ptH2D,ptH2D>> getPredictions(){return reproject(m_state,m_inliers_idx);}


private:

    std::vector<ptH3D> m_pts3D;
    std::vector<float> m_disparities;
    std::vector<int> m_inliers_idx; // inliers indexes

    parameters m_param;

    cv::Mat m_J;
    cv::Matx61d m_state;
    std::vector<std::pair<ptH2D,ptH2D>> m_obs;
    cv::Mat m_res;

    void project3D(const std::vector<StereoMatchf>& features);
    void project3D(const std::vector<StereoOdoMatchesf>& features);
    bool optimize(const std::vector<int>& selection, bool weight);
//    void projectionUpdate(const std::vector<StereoOdoMatchesf>& matches, const std::vector<int>& selection, bool weight);
    void computeReprojErrors(const std::vector<int>& inliers);
    std::vector<std::pair<ptH2D,ptH2D>> reproject(cv::Matx61d& state, const std::vector<int>& selection);
    void updateObservations(const std::vector<StereoOdoMatchesf>& matches);
    void updateJacobian(const vector<int>& selection);
    std::vector<int> computeInliers();
    std::vector<int> selectRandomIndexes(int nb_samples, int nb_tot);
};

}
#endif // VSTEREO_VISO_H

