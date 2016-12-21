#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <ostream>

#include <opencv2/core.hpp>

class Quat;
class Euler {

private:

	double m_roll;
	double m_pitch;
	double m_yaw;
	bool m_rad;

    void deg2Rad();
	void rad2Deg();

public:

	Euler(double r=0, double p=0, double y=0, bool rad=true ): m_roll(r), m_pitch(p), m_yaw(y), m_rad(rad){}
	Euler(const cv::Mat& M){fromMat(M);}
	Euler(const Euler& e): m_roll(e.roll()), m_pitch(e.pitch()), m_yaw(e.yaw()), m_rad(e.isRad()){}

	bool isRad() const { return m_rad;}
	friend std::ostream& operator<<(std::ostream& os, const Euler& e);
	std::string to_str(bool rad);

	//conversions
	cv::Mat1f getMat3f();
	cv::Mat1f getMat4f();
	cv::Mat1d getMat3d();
	cv::Mat1d getMat4d();
	cv::Mat getMat();
	void fromMat(const cv::Mat& R);
    Quat getQuat();

    //operator
    void operator+=(Euler& e);

    //access
    double roll() const {return m_roll;}
    double pitch() const {return m_pitch;}
    double yaw() const {return m_yaw;}
};

class Quat {

private:

	double m_w;
	double m_x;
	double m_y;
	double m_z;

public:

    // constructors and copy constructor
	Quat(double w=1, double x=0, double y=0, double z=0): m_w(w), m_x(x), m_y(y), m_z(z){}
	Quat(const cv::Mat& M){fromMat(M);norm();}
	Quat(const Quat& q): m_w(q.w()), m_x(q.x()), m_y(q.y()), m_z(q.z()){norm();}

	//normalize, conjugate and display functions
	void norm();
	Quat conj() const {return Quat(m_w,-m_x,-m_y,-m_z);}
	friend std::ostream& operator<<(std::ostream& os, const Quat& e);

    // conversions
    cv::Matx44f getQMatf() const;
    cv::Matx44d getQMatd() const;
    cv::Matx44f getQ_Matf() const;
    cv::Matx44d getQ_Matd() const;
    cv::Matx44d getdQdq0() const;
	cv::Matx44d getdQ_dq0() const;
	cv::Matx44d getdQdq1() const;
	cv::Matx44d getdQ_dq1() const;
	cv::Matx44d getdQdq2() const;
	cv::Matx44d getdQ_dq2() const;
	cv::Matx44d getdQdq3() const;
	cv::Matx44d getdQ_dq3() const;

	cv::Mat1f getMat3f() const;
	cv::Mat1f getMat4f();
	cv::Mat1d getMat3d() const;
	cv::Mat1d getMat4d() const;
	void fromMat(const cv::Mat& M);
	Euler getEuler();

	//operators
	void operator*=(const Quat& q);
	Quat operator*(const Quat& q);
	cv::Vec3d operator*(const cv::Vec3d& v);
	cv::Vec4d operator*(const cv::Vec4d& v);
	void operator+=(const Quat& q);
	void operator-=(const Quat& q);

	//access
	double w() const {return m_w;}
	double x() const {return m_x;}
	double y() const {return m_y;}
	double z() const {return m_z;}
};

#endif // UTILS_H_INCLUDED

//Euler operator+(const Euler& e1, const Euler& e2);

