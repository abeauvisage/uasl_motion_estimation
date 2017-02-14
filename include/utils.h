#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <ostream>

#include <opencv2/core.hpp>

#define PI 3.14156592

namespace me{

template <typename T>
class Quat;

template<typename T>
class Euler {

private:

	T m_roll;
	T m_pitch;
	T m_yaw;

    inline void deg2Rad(){m_roll *= PI/180; m_pitch *= PI/180; m_yaw *= PI/180;}
	inline void rad2Deg(){m_roll *= 180/PI; m_pitch *= 180/PI; m_yaw *= 180/PI;}
	inline void computeCosSin(T& cr, T& sr, T& cp, T& sp, T& cy, T& sy) const{cr=cos(m_roll);sr=sin(m_roll);cp=cos(m_pitch);sp=sin(m_pitch);cy=cos(m_yaw);sy=sin(m_yaw);}

public:

	Euler(T r=0, T p=0, T y=0, bool rad=true ): m_roll(r), m_pitch(p), m_yaw(y) {if(!rad)deg2Rad();}
	Euler(const cv::Mat& M){fromMat(M);}
	Euler(const Euler& e): m_roll(e.roll()), m_pitch(e.pitch()), m_yaw(e.yaw()){}

	//displaying
	friend std::ostream& operator<<(std::ostream& os, const Euler<T>& e){
        os << "[" << e.m_roll << "," << e.m_pitch << "," << e.m_yaw << ", rad]" << std::endl;
        return os;
    }
	std::string getDegrees();

	//conversions
	cv::Matx<T,3,3> getR3() const;
	cv::Matx<T,4,4> getR4() const;
	cv::Matx<T,3,3> getE() const;
	cv::Matx<T,3,3> getdRdr() const;
	cv::Matx<T,3,3> getdRdp() const;
	cv::Matx<T,3,3> getdRdy() const;
	void fromMat(const cv::Mat& R);
    Quat<T> getQuat() const;

    //operator
    void operator+=(Euler& e);

    //access
    T roll() const {return m_roll;}
    T pitch() const {return m_pitch;}
    T yaw() const {return m_yaw;}
};

template <typename T>
class Quat {

private:

	T m_w;
	T m_x;
	T m_y;
	T m_z;

public:

    // constructors and copy constructor
	Quat(T w=1, T x=0, T y=0, T z=0): m_w(w), m_x(x), m_y(y), m_z(z){}
	Quat(const cv::Mat& M){fromMat(M);norm();}
	Quat(const Quat& q): m_w(q.w()), m_x(q.x()), m_y(q.y()), m_z(q.z()){norm();}

	//normalize, conjugate and display functions
	void norm();
	Quat conj() const {return Quat(m_w,-m_x,-m_y,-m_z);}
	friend std::ostream& operator<<(std::ostream& os, const Quat<T>& q){
        os << "[" << q.m_w << "|" << q.m_x << "," << q.m_y << "," << q.m_z << "] angle: " << 2*acos(q.m_w) << std::endl;
        return os;
    }

    // conversions
    inline cv::Matx<T,4,4> getQ() const;
    inline cv::Matx<T,4,4> getQ_() const;
    inline cv::Matx<T,4,4> getdQdq0() const;
	inline cv::Matx<T,4,4> getdQ_dq0() const;
	inline cv::Matx<T,4,4> getdQdq1() const;
	inline cv::Matx<T,4,4> getdQ_dq1() const;
	inline cv::Matx<T,4,4> getdQdq2() const;
	inline cv::Matx<T,4,4> getdQ_dq2() const;
	inline cv::Matx<T,4,4> getdQdq3() const;
	inline cv::Matx<T,4,4> getdQ_dq3() const;

	inline cv::Matx<T,3,3> getR3() const;
	inline cv::Matx<T,4,4> getR4() const;
	void fromMat(const cv::Mat& M);
	Euler<T> getEuler();

	//operators
	void operator*=(const Quat& q);
	void operator*=(const double& d){m_w*=d;m_x*=d;m_y*=d;m_z*=d;}
	Quat operator*(const Quat& q) const;
	Quat operator*(const double d) const;
	cv::Vec<T,3> operator*(const cv::Vec<T,3>& v);
	cv::Vec<T,4> operator*(const cv::Vec<T,4>& v);
	Quat operator+(const Quat& q) const;
	void operator+=(const Quat& q);
	void operator-=(const Quat& q);
	void operator/=(double nb);

	//access
	double w() const {return m_w;}
	double x() const {return m_x;}
	double y() const {return m_y;}
	double z() const {return m_z;}
};

}

#endif // UTILS_H_INCLUDED
