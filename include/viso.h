#ifndef VISO_H
#define VISO_H

#include "opencv2/core/core.hpp"

class VisualOdometry {

public:

    void updatePose();
    cv::Mat getPose(){return m_pose;};

    virtual cv::Mat getMotion() = 0;

    VisualOdometry();
    virtual ~VisualOdometry(){};

protected:
    cv::Mat m_pose;

};

#endif // VISO_H

