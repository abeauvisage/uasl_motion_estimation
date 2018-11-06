#ifndef VISOME_H
#define VISOME_H

#include "opencv2/core.hpp"

namespace me{

//! VisualOdometry class
/** generic class to compute motion from mono or stereo setup */
class VisualOdometry {

public:

    /*! function return the egomotion of a camera(s) from its relative initial position.
     *  This function is a pure virtual function and is implemented differently for mono and stereo.
     */
    virtual cv::Mat getMotion() = 0;

    VisualOdometry(){};
    virtual ~VisualOdometry(){}; //!< empty destructor

};

}

#endif // VISO_H

