#ifndef MONOVISUALODOMETRY_H
#define MONOVISUALODOMETRY_H

#include "viso.h"
#include "featureType.h"

namespace me{

class MonoVisualOdometry : public VisualOdometry
{

    public:
    struct parameters  {
        bool ransac;
        double prob;
        double  inlier_threshold;
        double f;
        double cu;
        double cv;
        parameters () {
            ransac=true;
            prob=0.999;
            inlier_threshold = 1.0;
            f=1;
            cu=0;
            cv=0;
        }
    };

        MonoVisualOdometry(parameters param=parameters());
        ~MonoVisualOdometry(){};

        bool process(const std::vector<StereoMatch<cv::Point2f>>& matches);
        virtual cv::Mat getMotion(){return m_Rt;}
        std::vector<int> getInliersIdx(){return m_inliers;}
        cv::Mat getEssentialMat(){return m_E;}

    private:
        parameters m_param;
        cv::Mat m_Rt;
        cv::Mat m_E;
        std::vector<int> m_inliers;
};

}

#endif // MONOVISUALODOMETRY_H
