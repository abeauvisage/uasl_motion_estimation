#ifndef VISOME_H
#define VISOME_H

#include "opencv2/core/core.hpp"

namespace me{

//! VisualOdometry class
/** generic class to compute motion from mono or stereo setup */
class VisualOdometry {

public:

    //! Method used for optimization. Either Gauss-Newton or Levenberg-Maquart.
    enum class Method {GN, LM};

    //! Mono parameters
    /*! contains info about the optimisation paramters on methods to be used. */
    struct parameters {

        Method method;              //!< optimization method used.
        double step_size;           //!< step size for optimization (TO BE USED).
        double eps,e1,e2,e3,e4;     //!< optimization thresholds.
        int max_iter;               //!< max_iteration for optim algo to converge.
        int nb_fixed_frames;        //! number of fixed frame for BA

        bool ransac;                //! use of RANSAC or not.
        int n_ransac;               //!< nb of iterations used for the RANSAC process.
        double  inlier_threshold;   //! inlier threshold for RANSAC.

        //! default constructor
        parameters(): method(Method::GN),step_size(1.0),eps(1e-9),e1(1e-3),e2(1e-12),e3(1e-12),e4(1e-15),max_iter(100),nb_fixed_frames(2),ransac(true),n_ransac(200),inlier_threshold(2.0){}
    };

    /*! function return the egomotion of a camera(s) from its relative initial position.
     *  This function is a pure virtual function and is implemented differently for mono and stereo.
     */
    virtual cv::Mat getMotion() = 0;

    VisualOdometry(){};
    virtual ~VisualOdometry(){}; //!< empty destructor

};

}

#endif // VISO_H

