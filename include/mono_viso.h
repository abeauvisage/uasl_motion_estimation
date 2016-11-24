#ifndef MONOVISUALODOMETRY_H
#define MONOVISUALODOMETRY_H

#include "viso.h"
#include "featureType.h"

class MonoVisualOdometry : VisualOdometry
{
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
            inlier_threshold = 2.0;
            f=1;
            cu=0;
            cv=0;
        }
    };

    public:
        MonoVisualOdometry(parameters param=parameters());
        ~MonoVisualOdometry(){};

        bool process(const std::vector<StereoMatch<cv::Point2f>>& matches);
        virtual cv::Mat getMotion(){return m_Rt;}

    private:
        parameters m_param;
        cv::Mat m_Rt;
};

#endif // MONOVISUALODOMETRY_H
