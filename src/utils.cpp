#include "utils.h"

#include <iostream>

#define PI 3.14156592

using namespace cv;
using namespace std;


/**** Euler class ****/

void Euler::init(){
    m_roll = 0; m_pitch = 0; m_yaw =0; m_rad = true;
}

void Euler::deg2Rad(){
    if(!m_rad){m_roll *= PI/180; m_pitch *= PI/180; m_yaw *= PI/180; m_rad = true;}
}

void Euler::rad2Deg(){
    if(m_rad){m_roll *= 180/PI; m_pitch *= 180/PI; m_yaw *= 180/PI; m_rad = false;}
}

void Euler::show() const{
    cout << "e = [ " << m_roll << " " << m_pitch << " " << m_yaw << " ]t" << endl;
}

Mat Euler::getMat3f(){

		Mat R(3,3,CV_32FC1);

		deg2Rad();

		float cr = cos(m_roll), cp= cos(m_pitch), cy= cos(m_yaw);
		float sr = sin(m_roll), sp= sin(m_pitch), sy= sin(m_yaw);

		R.at<float>(0,0)= cp*cy;	R.at<float>(0,1)= sr*sp*cy-cr*sy;	R.at<float>(0,2)= cr*sp*cy+sr*sy;
		R.at<float>(1,0)= cp*sy;	R.at<float>(1,1)= sr*sp*sy+cr*cy;	R.at<float>(1,2)= cr*sp*sy-sr*cy;
		R.at<float>(2,0)= -sp;		R.at<float>(2,1)= sr*cp;   		    R.at<float>(2,2)= cr*cp;

		return R;
}

Mat Euler::getMat4f(){

		Mat R(4,4,CV_32FC1);

		deg2Rad();

		float cr = cos(m_roll), cp= cos(m_pitch), cy= cos(m_yaw);
		float sr = sin(m_roll), sp= sin(m_pitch), sy= sin(m_yaw);

		R.at<float>(0,0)= cp*cy;	R.at<float>(0,1)= sr*sp*cy-cp*sy;	R.at<float>(0,2)= cr*sp*cy+sp*sy;	R.at<float>(0,3)= 0;
		R.at<float>(1,0)= cp*sy;	R.at<float>(1,1)= sr*sp*sy+cp*cy;	R.at<float>(1,2)= cr*sp*sy-sp*cy;	R.at<float>(1,3)= 0;
		R.at<float>(2,0)= -sp;		R.at<float>(2,1)= sy*sp;   		    R.at<float>(2,2)= cr*cp;			R.at<float>(2,3)= 0;
		R.at<float>(3,0)= 0;		R.at<float>(3,1)= 0;				R.at<float>(3,2)= 0;				R.at<float>(3,3)= 1;

		return R;
}

Mat Euler::getMat3d(){

		Mat R(3,3,CV_64FC1);

		deg2Rad();

		double cr = cos(m_roll), cp= cos(m_pitch), cy= cos(m_yaw);
		double sr = sin(m_roll), sp= sin(m_pitch), sy= sin(m_yaw);

		R.at<double>(0,0)= cp*cy;	R.at<double>(0,1)= sr*sp*cy-cr*sy;	R.at<double>(0,2)= cr*sp*cy+sr*sy;
		R.at<double>(1,0)= cp*sy;	R.at<double>(1,1)= sr*sp*sy+cr*cy;	R.at<double>(1,2)= cr*sp*sy-sr*cy;
		R.at<double>(2,0)= -sp;		R.at<double>(2,1)= sr*cp;   		R.at<double>(2,2)= cr*cp;

		return R;
}

Mat Euler::getMat4d(){

		Mat R(4,4,CV_64FC1);

		deg2Rad();

		double cr = cos(m_roll), cp= cos(m_pitch), cy= cos(m_yaw);
		double sr = sin(m_roll), sp= sin(m_pitch), sy= sin(m_yaw);

		R.at<double>(0,0)= cp*cy;	R.at<double>(0,1)= sr*sp*cy-cp*sy;	R.at<double>(0,2)= cr*sp*cy+sp*sy;	R.at<double>(0,3)= 0;
		R.at<double>(1,0)= cp*sy;	R.at<double>(1,1)= sr*sp*sy+cp*cy;	R.at<double>(1,2)= cr*sp*sy-sp*cy;	R.at<double>(1,3)= 0;
		R.at<double>(2,0)= -sp;		R.at<double>(2,1)= sy*sp;   		R.at<double>(2,2)= cr*cp;			R.at<double>(2,3)= 0;
		R.at<double>(3,0)= 0;		R.at<double>(3,1)= 0;				R.at<double>(3,2)= 0;				R.at<double>(3,3)= 1;

		return R;
}

void Euler::fromMat(const Mat &R){

    m_roll = atan2(R.at<float>(2,1),R.at<float>(2,2));
    m_pitch = atan2(-R.at<float>(2,0),sqrt(pow(R.at<float>(2,1),2)+pow(R.at<float>(2,2),2)));
    m_yaw = atan2(R.at<float>(1,0),R.at<float>(0,0));
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

/**** Quat class ****/

void Quat::init(){
    m_w =1; m_x=0; m_y=0; m_z=0;
}

void Quat::norm(){
    double n = sqrt(m_w*m_w + m_x*m_x + m_y*m_y + m_z*m_z);
    m_w /= n;
    m_x /= n;
    m_y /= n;
    m_z /= n;
}

void Quat::show() const{
    cout << "q = [ " << m_w << " " << m_x << " " << m_y << " " << m_z  << " ]t" << endl;
}

cv::Mat Quat::getMat4d() const{

    Mat Q(4,4,CV_64FC1);
//    double q2w = m_w*m_w, q2x = m_x*m_x, q2m_y = m_y*m_y, q2m_z = m_z*m_z;

    Q.at<double>(0,0)= m_w*m_w + m_x*m_x - m_y*m_y - m_z*m_z;	Q.at<double>(0,1)= 2*(m_x*m_y - m_w*m_z);			Q.at<double>(0,2)= 2*(m_x*m_z + m_w*m_y);			Q.at<double>(0,3)= 0;
    Q.at<double>(1,0)= 2*(m_x*m_y + m_w*m_z);			Q.at<double>(1,1)= m_w*m_w - m_x*m_x + m_y*m_y - m_z*m_z;	Q.at<double>(1,2)= 2*(m_w*m_x - m_y*m_z);			Q.at<double>(1,3)= 0;
    Q.at<double>(2,0)= 2*(m_x*m_z - m_w*m_y);			Q.at<double>(2,1)= 2*(m_w*m_x + m_y*m_z);			Q.at<double>(2,2)= m_w*m_w - m_x*m_x - m_y*m_y + m_z*m_z;	Q.at<double>(2,3)= 0;
    Q.at<double>(3,0)= 0;						Q.at<double>(3,1)= 0;						Q.at<double>(3,2)= 0;						Q.at<double>(3,3)= 1;

    return Q;
}

cv::Mat Quat::getMat3d() const{

//    Quat anglq;
//    double ang = 2*acos(anglq.w());
//    double X = anglq.x()/sqrt(1-anglq.w()*anglq.w()), Y = anglq.y()/sqrt(1-anglq.w()*anglq.w()), Z = anglq.z()/sqrt(1-anglq.w()*anglq.w());

    Mat Q(3,3,CV_64FC1);
    double q2m_w = m_w*m_w, q2x = m_x*m_x, q2m_y = m_y*m_y, q2m_z = m_z*m_z;

    Q.at<double>(0,0)= 1-2*(m_y*m_y+m_z*m_z);	        Q.at<double>(0,1)= 2*(m_x*m_y - m_w*m_z);			Q.at<double>(0,2)= 2*(m_x*m_z + m_w*m_y);
    Q.at<double>(1,0)= 2*(m_x*m_y + m_w*m_z);			Q.at<double>(1,1)= 1-2*(m_x*m_x+m_z*m_z);	        Q.at<double>(1,2)= 2*(m_w*m_x - m_y*m_z);
    Q.at<double>(2,0)= 2*(m_x*m_z - m_w*m_y);			Q.at<double>(2,1)= 2*(m_w*m_x + m_y*m_z);			Q.at<double>(2,2)= 1-2*(m_x*m_x+m_y*m_y);

//    Q.at<double>(0,0)= cos(ang)+X*X*(1-cos(ang));	    Q.at<double>(0,1)= X*Y*(1-cos(ang))-Z*sin(ang);	    Q.at<double>(0,2)= X*Z*(1-cos(ang))+Y*sin(ang);
//    Q.at<double>(1,0)= Y*X*(1-cos(ang))+Z*sin(ang);		Q.at<double>(1,1)= cos(ang)+Y*Y*(1-cos(ang));	    Q.at<double>(1,2)= Y*Z*(1-cos(ang))-X*sin(ang);
//    Q.at<double>(2,0)= Z*X*(1-cos(ang))-Y*sin(ang);		Q.at<double>(2,1)= Z*Y*(1-cos(ang))+X*sin(ang);		Q.at<double>(2,2)= cos(ang)+Z*Z*(1-cos(ang));


    return Q;
}

void Quat::fromMat(const cv::Mat& m){
    m_w = sqrt(1+m.at<float>(0,0)+m.at<float>(1,1)+m.at<float>(2,2))/2;
    m_x =  (m.at<float>(2,1)-m.at<float>(1,2))/(4*m_w);
    m_y =  (m.at<float>(0,2)-m.at<float>(2,0))/(4*m_w);
    m_z =  (m.at<float>(1,0)-m.at<float>(0,1))/(4*m_w);
}

Euler Quat::getEuler(){
    norm();
    double t0 = -2.0f * (m_y*m_y + m_z*m_z)+1.0f;
    double t1 = +2.0f * (m_x*m_y - m_w*m_z);
    double t2 = -2.0f * (m_x*m_z + m_w*m_y);
    double t3 = +2.0f * (m_y*m_z - m_w*m_x);
    double t4 = -2.0f * (m_x*m_x + m_y*m_y)+1.0f;

    t2 = t2 > 1.0f?1.0f:t2;
    t2 = t2 < -1.0f?-1.0f:t2;

    Euler e(asin(t2),atan2(t3,t4),atan2(t1,t0));

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
