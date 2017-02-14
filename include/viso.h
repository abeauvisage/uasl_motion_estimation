#ifndef VISO_H
#define VISO_H

#include "opencv2/core.hpp"

namespace me{

class VisualOdometry {

public:

    virtual cv::Mat getMotion() = 0;

    VisualOdometry(){};
    virtual ~VisualOdometry(){};

};

}

#endif // VISO_H

