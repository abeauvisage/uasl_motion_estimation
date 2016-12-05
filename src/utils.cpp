#include "utils.h"

#include <iostream>

#define PI 3.14156592

using namespace cv;
using namespace std;


/**** Euler class ****/

void Euler::deg2Rad(){
    if(!m_rad){m_roll *= PI/180; m_pitch *= PI/180; m_yaw *= PI/180; m_rad = true;}
}

void Euler::rad2Deg(){
    if(m_rad){m_roll *= 180/PI; m_pitch *= 180/PI; m_yaw *= 180/PI; m_rad = false;}
}

Mat1f Euler::getMat3f(){

		Mat1f R(3,3);

		deg2Rad();

		float cr = cos(m_roll), cp= cos(m_pitch), cy= cos(m_yaw);
		float sr = sin(m_roll), sp= sin(m_pitch), sy= sin(m_yaw);

		R(0,0)= cp*cy;	R(0,1)= sr*sp*cy-cr*sy;	R(0,2)= cr*sp*cy+sr*sy;
		R(1,0)= cp*sy;	R(1,1)= sr*sp*sy+cr*cy;	R(1,2)= cr*sp*sy-sr*cy;
		R(2,0)= -sp;	R(2,1)= sr*cp;   		R(2,2)= cr*cp;

		return R;
}

Mat1f Euler::getMat4f(){

		Mat4f R(4,4);

		deg2Rad();

		float cr = cos(m_roll), cp= cos(m_pitch), cy= cos(m_yaw);
		float sr = sin(m_roll), sp= sin(m_pitch), sy= sin(m_yaw);

		R(0,0)= cp*cy;	R(0,1)= sr*sp*cy-cp*sy;	R(0,2)= cr*sp*cy+sp*sy;	R(0,3)= 0;
		R(1,0)= cp*sy;	R(1,1)= sr*sp*sy+cp*cy;	R(1,2)= cr*sp*sy-sp*cy;	R(1,3)= 0;
		R(2,0)= -sp;	R(2,1)= sy*sp;   		R(2,2)= cr*cp;			R(2,3)= 0;
		R(3,0)= 0;		R(3,1)= 0;				R(3,2)= 0;				R(3,3)= 1;

		return R;
}

Mat1d Euler::getMat3d(){

		Mat1d R(3,3);

		deg2Rad();

		double cr = cos(m_roll), cp= cos(m_pitch), cy= cos(m_yaw);
		double sr = sin(m_roll), sp= sin(m_pitch), sy= sin(m_yaw);

		R(0,0)= cp*cy;	R(0,1)= sr*sp*cy-cr*sy;	R(0,2)= cr*sp*cy+sr*sy;
		R(1,0)= cp*sy;	R(1,1)= sr*sp*sy+cr*cy;	R(1,2)= cr*sp*sy-sr*cy;
		R(2,0)= -sp;	R(2,1)= sr*cp;   		R(2,2)= cr*cp;

		return R;
}

Mat1d Euler::getMat4d(){

		Mat4d R(4,4);

		deg2Rad();

		double cr = cos(m_roll), cp= cos(m_pitch), cy= cos(m_yaw);
		double sr = sin(m_roll), sp= sin(m_pitch), sy= sin(m_yaw);

		R(0,0)= cp*cy;	R(0,1)= sr*sp*cy-cp*sy;	R(0,2)= cr*sp*cy+sp*sy;	R(0,3)= 0;
		R(1,0)= cp*sy;	R(1,1)= sr*sp*sy+cp*cy;	R(1,2)= cr*sp*sy-sp*cy;	R(1,3)= 0;
		R(2,0)= -sp;	R(2,1)= sy*sp;   		R(2,2)= cr*cp;			R(2,3)= 0;
		R(3,0)= 0;		R(3,1)= 0;				R(3,2)= 0;				R(3,3)= 1;

		return R;
}

void Euler::fromMat(const Mat1f &R){

//    m_roll = atan2(R(2,1),R(2,2));
//    m_pitch = atan2(-R(2,0),sqrt(pow(R(2,1),2)+pow(R(2,2),2)));
//    m_yaw = atan2(R(1,0),(0,0));
    m_roll = atan2(R(0,2),R(1,2));
    m_pitch = acos(R(2,2));
    m_yaw = atan2(R(2,0),-R(2,1));
    m_rad = true;
}

void Euler::fromMat(const Mat1d &R){

    if (!(sqrt(pow(R(0,0),2)+pow(R(1,0),2)) < 1e-6)){
        m_roll = atan2(R(2,1),R(2,2));
        m_pitch = atan2(-R(2,0),sqrt(pow(R(0,0),2)+pow(R(1,0),2)));
        m_yaw = atan2(R(1,0),R(0,0));
//        m_roll = atan2(R(1,2),R(2,2));
//        m_pitch = acos(R(2,2));
//        m_yaw = atan2(R(0,1),R(0,0));

    }else{
        m_roll = atan2(-R(1,2),R(1,1));
        m_pitch = atan2(-R(2,0),sqrt(pow(R(0,0),2)+pow(R(1,0),2)));
        m_yaw = 0;
    }
    m_rad = true;
}

Quat Euler::getQuat(){

    deg2Rad();

    double t0 = cos(m_yaw * 0.5f);
    double t1 = sin(m_yaw * 0.5f);
    double t2 = cos(m_roll * 0.5f);
    double t3 = sin(m_roll * 0.5f);
    double t4 = cos(m_pitch * 0.5f);
    double t5 = sin(m_pitch * 0.5f);

    return Quat(t0*t2*t4+t1*t3*t5,t0*t3*t4-t1*t2*t5,t0*t2*t5+t1*t3*t4,t1*t2*t4-t0*t3*t5);
}

//Euler operator+(const Euler& e1, const Euler& e2){
//    if(e1.isRad()){
//        if(!e2.isRad())
//            e2.deg2Rad();
//    }
//    else
//        if(e2.isRad())
//            e2.rad2Deg();
//
//    return Euler(e1.roll()+e2.roll(),e1.pitch()+e2.pitch(),e1.yaw()+e2.yaw());
//}

void Euler::operator+=(Euler& e){
    if(isRad()){
        if(!e.isRad())
            e.deg2Rad();
    }else
        if(e.isRad())
            e.rad2Deg();

    m_roll += e.roll();
    m_pitch += e.pitch();
    m_yaw += e.yaw();
}



/***********************
        Quat class

************************/

void Quat::norm(){
    double n = sqrt(m_w*m_w + m_x*m_x + m_y*m_y + m_z*m_z);
    if(n != 1.0){
        m_w /= n;
        m_x /= n;
        m_y /= n;
        m_z /= n;
    }
}

cv::Mat1f Quat::getMat4f() const{

    Mat1f Q(4,4);

    Q(0,0)= m_w*m_w + m_x*m_x - m_y*m_y - m_z*m_z;	Q(0,1)= 2*(m_x*m_y + m_w*m_z);			        Q(0,2)= 2*(m_x*m_z - m_w*m_y);			        Q(0,3)= 0;
    Q(1,0)= 2*(m_x*m_y - m_w*m_z);			        Q(1,1)= m_w*m_w - m_x*m_x + m_y*m_y - m_z*m_z;	Q(1,2)= 2*(m_w*m_x + m_y*m_z);			        Q(1,3)= 0;
    Q(2,0)= 2*(m_x*m_z + m_w*m_y);			        Q(2,1)= 2*(m_w*m_x - m_y*m_z);			        Q(2,2)= m_w*m_w - m_x*m_x - m_y*m_y + m_z*m_z;	Q(2,3)= 0;
    Q(3,0)= 0;						                Q(3,1)= 0;						                Q(3,2)= 0;						                Q(3,3)= 1;

    return Q;
}

cv::Mat1f Quat::getMat3f() const{

    Mat1f Q(3,3);

    Q(0,0)= m_w*m_w + m_x*m_x - m_y*m_y - m_z*m_z;	Q(0,1)= 2*(m_x*m_y + m_w*m_z);			        Q(0,2)= 2*(m_x*m_z - m_w*m_y);
    Q(1,0)= 2*(m_x*m_y - m_w*m_z);			        Q(1,1)= m_w*m_w - m_x*m_x + m_y*m_y - m_z*m_z;	Q(1,2)= 2*(m_w*m_x + m_y*m_z);
    Q(2,0)= 2*(m_x*m_z + m_w*m_y);			        Q(2,1)= 2*(m_w*m_x - m_y*m_z);			        Q(2,2)= m_w*m_w - m_x*m_x - m_y*m_y + m_z*m_z;

    return Q;
}

cv::Mat1d Quat::getMat4d() const{

    Mat1d Q(4,4);

    Q(0,0)= m_w*m_w + m_x*m_x - m_y*m_y - m_z*m_z;	Q(0,1)= 2*(m_x*m_y + m_w*m_z);			        Q(0,2)= 2*(m_x*m_z - m_w*m_y);			        Q(0,3)= 0;
    Q(1,0)= 2*(m_x*m_y - m_w*m_z);			        Q(1,1)= m_w*m_w - m_x*m_x + m_y*m_y - m_z*m_z;	Q(1,2)= 2*(m_w*m_x + m_y*m_z);			        Q(1,3)= 0;
    Q(2,0)= 2*(m_x*m_z + m_w*m_y);			        Q(2,1)= 2*(m_w*m_x - m_y*m_z);			        Q(2,2)= m_w*m_w - m_x*m_x - m_y*m_y + m_z*m_z;	Q(2,3)= 0;
    Q(3,0)= 0;						                Q(3,1)= 0;						                Q(3,2)= 0;						                Q(3,3)= 1;

    return Q;
}

cv::Mat1d Quat::getMat3d() const{

    Mat1d Q(3,3);

    Q(0,0)= m_w*m_w + m_x*m_x - m_y*m_y - m_z*m_z;	Q(0,1)= 2*(m_x*m_y + m_w*m_z);			        Q(0,2)= 2*(m_x*m_z - m_w*m_y);
    Q(1,0)= 2*(m_x*m_y - m_w*m_z);			        Q(1,1)= m_w*m_w - m_x*m_x + m_y*m_y - m_z*m_z;	Q(1,2)= 2*(m_w*m_x + m_y*m_z);
    Q(2,0)= 2*(m_x*m_z + m_w*m_y);			        Q(2,1)= 2*(m_w*m_x - m_y*m_z);			        Q(2,2)= m_w*m_w - m_x*m_x - m_y*m_y + m_z*m_z;

    return Q;
}

void Quat::fromMat(const cv::Mat& M){

    assert((M.type() == CV_32F || M.type() == CV_64F) && M.channels() == 1);

    //define type of matrix M
    if(M.type() == CV_64F){
        #define DOUBLE
    }

    #ifdef DOUBLE
        Mat1d m = M;
    #else
        Mat1f m = M;
    #endif


    if(m(1,1) > -m(2,2) && m(0,0) > - m(1,1) && m(0,0) > -m(2,2)){
        double norm = sqrt(1+m(0,0)+m(1,1)+m(2,2));
        m_w = norm/2;
        m_x = (m(1,2)-m(2,1))/(2*norm);
        m_y = (m(2,0)-m(0,2))/(2*norm);
        m_z = (m(0,1)-m(1,0))/(2*norm);
    } else if (m(1,1) < -m(2,2) && m(0,0) > m(1,1) && m(0,0) > m(2,2)){
        double norm = sqrt(1+m(0,0)-m(1,1)-m(2,2));
        m_w = (m(1,2)-m(2,1))/(2*norm);
        m_x = norm/2;
        m_y = (m(0,1)+m(1,0))/(2*norm);
        m_z = (m(2,0)+m(0,2))/(2*norm);
    } else if (m(1,1) > m(2,2) && m(0,0) < m(1,1) && m(0,0) < -m(2,2)){
        double norm = sqrt(1-m(0,0)+m(1,1)-m(2,2));
        m_w = (m(2,0)-m(0,2))/(2*norm);
        m_x = (m(0,1)+m(1,0))/(2*norm);
        m_y = norm/2;
        m_z = (m(1,2)+m(2,1))/(2*norm);
    } else{
        double norm = sqrt(1-m(0,0)-m(1,1)+m(2,2));
        m_w = (m(0,1)-m(1,0))/(2*norm);
        m_x = (m(2,0)+m(0,2))/(2*norm);
        m_y = (m(1,2)+m(2,1))/(2*norm);
        m_z = norm/2;
    }
}

Euler Quat::getEuler(){
    norm();
    Euler e(atan2(2*m_y*m_z+2*m_w*m_x,m_z*m_z-m_y*m_y-m_x*m_x+m_w*m_w),-asin(2*m_x*m_z-2*m_w*m_y),atan2(2*m_x*m_y+2*m_w*m_z,m_x*m_x+m_w*m_w-m_z*m_z-m_y*m_y));

    return e;
}

void Quat::operator*=(const Quat& q){
    m_w = m_w*q.w()-(m_x*q.x()+m_y*q.y()+m_z*q.z());
    m_x = q.w()*m_x+m_w*q.x()+m_y*q.z()-m_z*q.y();
    m_y = q.w()*m_y+m_w*q.y()+m_y*q.w()-m_z*q.x();
    m_z = q.w()*m_y+m_w*q.z()+m_y*q.x()-m_z*q.w();
    norm();
}

void Quat::operator+=(const Quat& q){
    m_w += q.w();
    m_x += q.x();
    m_y += q.y();
    m_z += q.z();
    norm();
}

void Quat::operator-=(const Quat& q){
    m_w -= q.w();
    m_x -= q.x();
    m_y -= q.y();
    m_z -= q.z();
    norm();
}


ostream& operator<<(ostream& os, const Euler& e){
    os << "[" << e.m_roll << "," << e.m_pitch << "," << e.m_yaw << ",";
    if(e.m_rad) os << "rad";else os << "deg"; os << "]" << endl;
    return os;
}

ostream& operator<<(ostream& os, const Quat& q){
    os << "[" << q.m_w << "|" << q.m_x << "," << q.m_y << "," << q.m_z << "] , angle: " << 2*acos(q.m_w) << endl;
    return os;
}
