#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

/** \file rotation_utils.h
*   \brief Defines useful classes and functions to deal with different rotation representations.
*
*   useful applying rotation and converting between euler angles, quaterions, rotation matrices and rotation vectors.
*
*    \author Axel Beauvisage (axel.beauvisage@gmail.com)
*/

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#define PI 3.14156592

namespace me{

const cv::Mat TRef = (cv::Mat_<double>(3,3) << 0,-1,0,0,0,-1,1,0,0);
enum StopCondition{NO_STOP,SMALL_GRADIENT,SMALL_INCREMENT,MAX_ITERATIONS,SMALL_DECREASE_FUNCTION,SMALL_REPROJ_ERROR,NO_CONVERGENCE};

constexpr double deg2Rad(double deg){return deg * PI/180;}
constexpr double rad2Deg(double rad){return rad * 180/PI;}


template <typename T>
cv::Matx<T,4,3> Gq_v(const cv::Vec<T,3>& v);

template <typename T>
cv::Matx<T,3,3> skew(const cv::Vec<T,3>& mat){return cv::Matx<T,3,3>(0, -mat(2), mat(1), mat(2), 0, -mat(0), -mat(1), mat(0), 0);}

//! template class Quat
/*! represents a quaternion. Must be a float-point type (float or double), results with other data type are not guaranteed */
template <typename T>
class Quat;

//! template class Euler
/*! represents Euler angles in 3 axis (x,y,z). Must be a float-point type (float or double), results with other data type are not guaranteed */
template <typename T>
class Euler;

typedef Euler<double> Euld; //!< type Euler angles with double precision.
typedef Euler<float> Eulf;
typedef Quat<double> Quatd;
typedef Quat<float> Quatf;

template<typename T>
class Euler {

private:

	T m_roll; //!< angle around the x-axis
	T m_pitch; //!< angle around the y-axis.
	T m_yaw; //!< angle around the z-axis.

    inline void deg2Rad(){m_roll *= PI/180; m_pitch *= PI/180; m_yaw *= PI/180;} //!< converting the angles into rad units.
	inline void rad2Deg(){m_roll *= 180/PI; m_pitch *= 180/PI; m_yaw *= 180/PI;} //!< converting the angles into degrees units.
	inline void computeCosSin(T& cr, T& sr, T& cp, T& sp, T& cy, T& sy) const{cr=cos(m_roll);sr=sin(m_roll);cp=cos(m_pitch);sp=sin(m_pitch);cy=cos(m_yaw);sy=sin(m_yaw);} //!<< computing sine and cosine for each axis.

public:
    /*! Main constructor. Takes 3 angles as input and the unit. By default  all angles are 0 and expressed in radians. */
	Euler(T r=0, T p=0, T y=0, bool rad=true ): m_roll(r), m_pitch(p), m_yaw(y) {if(!rad)deg2Rad();}
	/*! Main constructor. Takes a vector as input and the unit. By default  all angles are 0 and expressed in radians. */
	Euler(const cv::Vec<T,3>& vec, bool rad=true ): m_roll(vec(0)), m_pitch(vec(1)), m_yaw(vec(2)) {if(!rad)deg2Rad();}
	/*! retrieves Euler angles from a rotation matrix and create an Euler object. */
	Euler(const cv::Mat& M){fromMat(M);}
	/*! copy constructor. */
	Euler(const Euler& e): m_roll(e.roll()), m_pitch(e.pitch()), m_yaw(e.yaw()){}

	//displaying
	friend std::ostream& operator<<(std::ostream& os, const Euler<T>& e){
        os << "[" << e.m_roll << "," << e.m_pitch << "," << e.m_yaw << ", rad]";
        return os;
    }
	std::string getDegrees(); /*!< return the degree angles as a string (for display purposes). */

	cv::Matx<T,3,3> getR3() const;      //!< 3x3 R Matrix.
	cv::Matx<T,4,4> getR4() const;      //!< 4x4 Rt Matrix, t is null because only the angle is known.
	cv::Matx<T,3,3> getE() const;       //!< 3x3 E matrix.
	cv::Matx<T,3,3> getdRdr() const;    //!< 3x3 derivative of R along the x axis (roll).
	cv::Matx<T,3,3> getdRdp() const;    //!< 3x3 derivative of R along the y axis (pitch).
	cv::Matx<T,3,3> getdRdy() const;    //!< 3x3 derivative of R along the z axis (yaw).
	void fromMat(const cv::Mat& R);     //!< compute Euler angles from a rotation matrix R.
    Quat<T> getQuat() const;            //!< convert Euler angles to a Quaternion of the same type.
    cv::Vec<T,3> getVector() const;         //!< returns a 3-axis vector contains the different angles.
    Euler inv() const {return Euler(-roll(),-pitch(),-yaw());}

    //operator
    void operator+=(const Euler& e); //!< concatenate with another Euler object.
    cv::Vec<T,3> operator*(const cv::Vec<T,3>& v) const; //!< rotate a 3-vector with the rotation described by the object.
	cv::Vec<T,4> operator*(const cv::Vec<T,4>& v) const; //!< rotate a 4-vector with the rotation described by the object.
	cv::Matx<T,3,1> operator*(const cv::Matx<T,3,1>& v) const;
	cv::Matx<T,4,1> operator*(const cv::Matx<T,4,1>& v) const;
	void operator*=(const Euler& e){}
	void operator*=(const double& d){}
	Euler operator*(const Euler& e) const{return Euler();}
	Euler operator*(const double d) const{return Euler();}

    //access
    T roll() const {return m_roll;}
    T pitch() const {return m_pitch;}
    T yaw() const {return m_yaw;}
};

template<typename T>
Quat<T> exp_map_Quat(const cv::Vec<T,3>& vec);

template<typename T>
cv::Vec<T,3> log_map_Quat(const Quat<T>& quat);

template<typename T>
cv::Matx<T,3,3> exp_map_Mat(const cv::Vec<T,3>& vec);

template<typename T>
cv::Vec<T,3> log_map_Mat(const cv::Matx<T,3,3>& mat);

template <typename T>
class Quat {

private:

	T m_w; //!< real component.
	T m_x; //!< x-axis.
	T m_y; //!< y-axis.
	T m_z; //!< z-axis.

public:

    /*! Main constructor. Default values are 1 for the real part and 0 for the axis components. */
	Quat(T w=1, T x=0, T y=0, T z=0): m_w(w), m_x(x), m_y(y), m_z(z){normalize();}
	/*! Create a Quat object from a rotation matrix or vector(quaternion/rot vector) and normalize it. */
	Quat(const cv::Mat& M){
	    if(M.cols == 1){ // if M is a vector
            if(M.rows == 4){ // with 4 quaternion components
                const T* ptr = M.ptr<T>();
                m_w = ptr[0];m_x = ptr[1];m_y = ptr[2];m_z = ptr[3];
            }else if(M.rows == 3) // or 3 rotation vector components
                *this = exp_map_Quat<T>(M);
	    }
        else
            fromMat(M);
        normalize();
    }
	/*! copy constructor. */
	Quat(const Quat& q): m_w(q.w()), m_x(q.x()), m_y(q.y()), m_z(q.z()){normalize();}

	void normalize(); //!< normalize the object
	Quat conj() const {return Quat(m_w,-m_x,-m_y,-m_z);} //!< returns the conjugate of the object.
	Quat inv() const {return conj();} //!< returns the conjugate of the object.

	friend std::ostream& operator<<(std::ostream& os, const Quat<T>& q){
        os << "[" << q.m_w << "|" << q.m_x << "," << q.m_y << "," << q.m_z << "] angle: " << 2*acos(q.m_w);
        return os;
    }

    // conversions
    inline cv::Matx<T,4,4> getQr() const;        //!< Q matrix to multiply with another quaternion.
    inline cv::Matx<T,4,4> getQl() const;       //!< inverse of Q. Represents the inverse rotation.
    inline cv::Matx<T,3,4> getH() const;        //!< rotation vector derivative wrt quaternion.
    inline cv::Matx<T,3,3> getH_qvec(const cv::Vec<T,3> x) const; //!< quaternion-vector multiplication derivative wrt rotation vector.
    inline cv::Matx<T,4,3> getG() const;        //!< quaternion derivative wrt rotation vector.

    inline cv::Vec<T,4> getCoeffs() const {return cv::Vec<T,4>(m_w,m_x,m_y,m_z);} //! return the vector of coefficients [q_w, q_x, q_y, q_z]
	inline cv::Matx<T,3,3> getR3() const;   //!< returns the 3x3 corresponding rotation matrix.
	inline cv::Matx<T,4,4> getR4() const;   //!< returns the 4x4 corresponding rotation matrix.
	void fromMat(const cv::Mat& M);         //!< update thet object from a rotation matrix.
	Euler<T> getEuler();                    //!< convert the quaternion object into Euler angles.

	//operators
	void operator*=(const Quat& q);
	void operator*=(const double& d){m_w*=d;m_x*=d;m_y*=d;m_z*=d;}
	Quat operator*(const Quat& q) const;
	Quat operator*(const double d) const;
	cv::Vec<T,3> operator*(const cv::Vec<T,3>& v) const;
	cv::Matx<T,3,1> operator*(const cv::Matx<T,3,1>& v) const;
	Quat operator+(const Quat& q) const;
	void operator+=(const Quat& q);
	void operator-=(const Quat& q);
	void operator/=(double nb);
	Quat<T>& operator=(const Quat<T>& q){m_w = q.w();m_x = q.x();m_y = q.y();m_z = q.z();return *this;}

	//access
	double w() const {return m_w;}
	double x() const {return m_x;}
	double y() const {return m_y;}
	double z() const {return m_z;}
	cv::Vec<T,3> vec() const {return cv::Vec<T,3>(m_x,m_y,m_z);}
};

template<typename T>
Quat<T> exp_map_Quat(const cv::Vec<T,3>& vec){
    double norm = sqrt(pow(vec(0),2)+pow(vec(1),2)+pow(vec(2),2));
    double theta = (norm < 1e-10 ? 1e-10:norm);
    Quat<T> q(cos(theta/2),vec(0)/theta*sin(theta/2),vec(1)/theta*sin(theta/2),vec(2)/theta*sin(theta/2));
    q.normalize();
    return q;
}

template<typename T>
cv::Vec<T,3> log_map_Quat(const Quat<T>& quat){
    double norm = sqrt(pow(quat.x(),2)+pow(quat.y(),2)+pow(quat.z(),2));
    double theta = (norm < 1e-10 ? 1e-10:norm);
    return acos(quat.w()) * 2.0 * (cv::Vec3d(quat.x(),quat.y(),quat.z())/theta);
}

template<typename T>
cv::Matx<T,3,3> exp_map_Mat(const cv::Vec<T,3>& vec){
    cv::Mat R;
    cv::Rodrigues(vec,R);
    return (cv::Matx<T,3,3>) R;
}

template<typename T>
cv::Vec<T,3> log_map_Mat(const cv::Matx<T,3,3>& mat){
    cv::Vec<T,3> vec;
    cv::Rodrigues(mat,vec);
    return vec;
}

/**** inline funcitons ****/

template <typename T>
inline cv::Matx<T,4,4> Quat<T>::getR4() const{

    return typename cv::Matx<T,4,4>::Matx(  m_w*m_w + m_x*m_x - m_y*m_y - m_z*m_z,  2*(m_x*m_y - m_w*m_z),                  2*(m_x*m_z + m_w*m_y),                  0,
                                            2*(m_x*m_y + m_w*m_z),                  m_w*m_w - m_x*m_x + m_y*m_y - m_z*m_z,  2*(m_y*m_z - m_w*m_x),                  0,
                                            2*(m_x*m_z - m_w*m_y),                  2*(m_y*m_z + m_w*m_x),			        m_w*m_w - m_x*m_x - m_y*m_y + m_z*m_z,  0,
                                            0,                                      0,						                0,					                    1);
}

template <typename T>
inline cv::Matx<T,3,3> Quat<T>::getR3() const{

    return typename cv::Matx<T,3,3>::Matx(  m_w*m_w + m_x*m_x - m_y*m_y - m_z*m_z,  2*(m_x*m_y - m_w*m_z),                  2*(m_x*m_z + m_w*m_y),
                                            2*(m_x*m_y + m_w*m_z),                  m_w*m_w - m_x*m_x + m_y*m_y - m_z*m_z,  2*(m_y*m_z - m_w*m_x),
                                            2*(m_x*m_z - m_w*m_y),                  2*(m_y*m_z + m_w*m_x),			        m_w*m_w - m_x*m_x - m_y*m_y + m_z*m_z);
}

template <typename T>
inline cv::Matx<T,4,4> Quat<T>::getQl() const{
    return typename cv::Matx<T,4,4>::Matx(m_w,-m_x,-m_y,-m_z,m_x,m_w,-m_z,m_y,m_y,m_z,m_w,-m_x,m_z,-m_y,m_x,m_w);
}

template <typename T>
inline cv::Matx<T,4,4> Quat<T>::getQr() const{
    return typename cv::Matx<T,4,4>::Matx(m_w,-m_x,-m_y,-m_z,m_x,m_w,m_z,-m_y,m_y,-m_z,m_w,m_x,m_z,m_y,-m_x,m_w);
}

template <typename T>
inline cv::Matx<T,4,3> Quat<T>::getG() const{
    return Gq_v(log_map_Quat(*this));
}

template <typename T>
inline cv::Matx<T,3,4> Quat<T>::getH() const{
    double c = 1.0/(1-m_w*m_w+1e-20);
    double d = acos(m_w)/sqrt(1-m_w*m_w+1e-20);
    return typename cv::Matx<T,3,4>::Matx(2*c*m_x*(d*m_w-1), 2*d, 0, 0, 2*c*m_y*(d*m_w-1), 0, 2*d, 0, 2*c*m_z*(d*m_w-1), 0, 0, 2*d);
}

template <typename T>
inline cv::Matx<T,3,3> Quat<T>::getH_qvec(const cv::Vec<T,3> x) const{
    cv::Vec<T,3> q_vec = vec();
    cv::Mat dqxdq(3,4,CV_64F);
    ((cv::Mat)( 2 * m_w * x + 2 * skew(q_vec) * x)).copyTo(dqxdq.colRange(0,1));
    ((cv::Mat)( 2 * ((q_vec.t()*x)[0] * cv::Matx<T,3,3>::eye()+q_vec*x.t()-x*q_vec.t()-m_w * skew(x)))).copyTo(dqxdq.colRange(1,4));
    return (cv::Matx<T,3,4>) dqxdq * Gq_v(log_map_Quat(*this));
}

template<typename T>
void convertToOpenCV(Euler<T>& e);
template<typename T>
void convertToXYZ(Euler<T>& e);
template<typename T>
void convertToOpenCV(Quat<T>& q);
template<typename T>
void convertToXYZ(Quat<T>& q);
template<typename T>
void convertToOpenCV(cv::Vec<T,3>& e);
template<typename T>
void convertToXYZ(cv::Vec<T,3>& e);

}// namespace me

#endif // UTILS_H_INCLUDED
