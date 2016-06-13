#ifndef VISO_STEREO_H
#define VISO_STEREO_H

#include <iostream>

#include "opencv2/core/core.hpp"
#include "featureType.h"

class VisualOdometryStereo {

public:

    struct parameters  {
        double  baseline;
        int n_ransac;
        double  inlier_threshold;
        bool    reweighting;
        double f1,f2;
        double cu1,cu2;
        double cv1,cv2;
        double step_size;
        double eps;
        int max_iter;
        parameters () {
            f1=1;f2=1;
            cu1=0;cu2=0;
            cv1=0;cv2=0;
            baseline = 1.0;
            n_ransac = 200;
            inlier_threshold = 2.0;
            reweighting = true;
            step_size = 1;
            eps = 1e-8;
            max_iter=20;
        }
    };


    VisualOdometryStereo(parameters param);
    ~VisualOdometryStereo(){};

    bool process(const std::vector<StereoOdoMatches<cv::Point2f>>& matches);

    cv::Mat getMotion();
    std::vector<cv::Point3d> getPts3D(){return pts3D;}
    std::vector<int> getInliers_idx(){return inliers_idx;}

private:

    std::vector<cv::Point3d> pts3D;
    std::vector<int> inliers_idx; // inliers indexes

    parameters m_param;

    std::vector<double> J;
    std::vector<double> x;
    std::vector<double> observations;
    std::vector<double> predictions;
    std::vector<double> residuals;

    bool optimize(const std::vector<StereoOdoMatches<cv::Point2f>>& matches, const std::vector<int32_t>& selection);
    void projectionUpdate(const std::vector<StereoOdoMatches<cv::Point2f>>& matches, const std::vector<int32_t>& selection);
    std::vector<int> computeInliers(const std::vector<StereoOdoMatches<cv::Point2f>>& matches);
    std::vector<int> randomIndexes(int nb_samples, int nb_tot);
};

#endif // VISO_STEREO_H

