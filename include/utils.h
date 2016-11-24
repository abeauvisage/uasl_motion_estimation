#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <opencv2/core/core.hpp>

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
	Euler(const Euler& e): m_roll(e.roll()), m_pitch(e.pitch()), m_yaw(e.yaw()), m_rad(e.isRad()){}
    void init();
	void show() const;
	bool isRad() const { return m_rad;}

	//conversions
	cv::Mat getMat3f();
	cv::Mat getMat4f();
	cv::Mat getMat3d();
	cv::Mat getMat4d();
	void fromMat(const cv::Mat &R);
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

	Quat(double w=1, double x=0, double y=0, double z=0): m_w(w), m_x(x), m_y(y), m_z(z){norm();}
	Quat(const Quat& q): m_w(q.w()), m_x(q.x()), m_y(q.y()), m_z(q.z()){norm();}
	void init();
	void norm();
	Quat conj() const {return Quat(m_w,-m_x,-m_y,-m_z);}
	void show() const;

    // conversions
	cv::Mat getMat3d() const;
	cv::Mat getMat4d() const;
	void fromMat(const cv::Mat& M);
	Euler getEuler();

	//operators
	void operator*=(const Quat& q);
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
