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

		Mat1f R(4,4);

		deg2Rad();

		float cr = cos(m_roll), cp= cos(m_pitch), cy= cos(m_yaw);
		float sr = sin(m_roll), sp= sin(m_pitch), sy= sin(m_yaw);

//		R(0,0)= cp*cy;	R(0,1)= sr*sp*cy-cp*sy;	R(0,2)= cr*sp*cy+sp*sy;	R(0,3)= 0;
//		R(1,0)= cp*sy;	R(1,1)= sr*sp*sy+cp*cy;	R(1,2)= cr*sp*sy-sp*cy;	R(1,3)= 0;
//		R(2,0)= -sp;	R(2,1)= sy*sp;   		R(2,2)= cr*cp;			R(2,3)= 0;
//		R(3,0)= 0;		R(3,1)= 0;				R(3,2)= 0;				R(3,3)= 1;

        R(0,0)= cr*cy+sr*sp*sy;	R(0,1)= sr*cp;	R(0,2)= -cr*sy+sr*sp*cy; R(0,3)= 0;
		R(1,0)= -sr*cy+cr*sp*sy;R(1,1)= cr*cp;	R(1,2)= sr*sy+cr*sp*cy; R(1,3)= 0;
		R(2,0)= sy*cp;	        R(2,1)=  sp;   	R(2,2)= cy*cp;          R(2,3)= 0;
		R(3,0)= 0;		        R(3,1)= 0;		R(3,2)= 0;				R(3,3)= 1;


		return R;
}

Mat1d Euler::getMat3d(){

		Mat1d R(3,3);

		deg2Rad();

		double cr = cos(m_roll), cp= cos(m_pitch), cy= cos(m_yaw);
		double sr = sin(m_roll), sp= sin(m_pitch), sy= sin(m_yaw);

//		R(0,0)= cp*cy;	R(0,1)= sr*sp*cy-cr*sy;	R(0,2)= cr*sp*cy+sr*sy;
//		R(1,0)= cp*sy;	R(1,1)= sr*sp*sy+cr*cy;	R(1,2)= cr*sp*sy-sr*cy;
//		R(2,0)= -sp;	R(2,1)= sr*cp;   		R(2,2)= cr*cp;

        R(0,0)= cp*cy;	        R(0,1)= cp*sy;	        R(0,2)= -sp;
		R(1,0)= sp*sr*cy-cr*sy;	R(1,1)= sr*sp*sy+cr*cy;	R(1,2)= cp*sr;
		R(2,0)= cr*sp*cy+sr*sy;	R(2,1)= cr*sp*sy-sr*cy;	R(2,2)= cp*cr;

		return R;
}

Mat1d Euler::getMat4d(){

		Mat1d R(4,4);

		deg2Rad();

		double cr = cos(m_roll), cp= cos(m_pitch), cy= cos(m_yaw);
		double sr = sin(m_roll), sp= sin(m_pitch), sy= sin(m_yaw);

		R(0,0)= cp*cy;	        R(0,1)= cp*sy;      	R(0,2)= -sp;	R(0,3)= 0;
		R(1,0)= sr*sp*cy-cr*sy;	R(1,1)= sr*sp*sy+cr*cy;	R(1,2)= cp*sr;	R(1,3)= 0;
		R(2,0)= cr*sp*cy+sr*sy;	R(2,1)= cr*sp*sy-sr*cy; R(2,2)= cr*cp;	R(2,3)= 0;
		R(3,0)= 0;		        R(3,1)= 0;				R(3,2)= 0;		R(3,3)= 1;

		return R;
}

Mat  Euler::getMat(){
    return Mat();
}

void Euler::fromMat(const Mat& M){

    assert((M.type() == CV_32F || M.type() == CV_64F) && M.channels() == 1);

    //define type of matrix M
    if(M.type() == CV_64F){
        #define DOUBLE
    }

    #ifdef DOUBLE
        Mat1d R = M;
    #else
        Mat1f R = M;
    #endif

    if (!(sqrt(pow(R(0,0),2)+pow(R(1,0),2)) < 1e-6)){
        m_roll = atan2(R(2,1),R(2,2));
        m_pitch = atan2(-R(2,0),sqrt(pow(R(0,0),2)+pow(R(1,0),2)));
        m_yaw = atan2(R(1,0),R(0,0));
    }else{
        m_roll = atan2(-R(1,2),R(1,1));
        m_pitch = atan2(-R(2,0),sqrt(pow(R(0,0),2)+pow(R(1,0),2)));
        m_yaw = 0;
    }
    if(m_roll > PI)
            m_roll -= PI;
        if(m_roll < -PI)
            m_roll += PI;
        if(m_pitch > PI)
            m_pitch -= PI;
        if(m_yaw < -PI)
            m_yaw += PI;
        if(m_yaw > PI)
            m_yaw -=PI;
        if(m_yaw < -PI)
            m_yaw +=PI;
    m_rad = true;
}

Quat Euler::getQuat(){

    deg2Rad();

    double cy = cos(m_roll * 0.5f);
    double sy = sin(m_roll * 0.5f);
    double cr = cos(m_yaw * 0.5f);
    double sr = sin(m_yaw * 0.5f);
    double cp = cos(m_pitch * 0.5f);
    double sp = sin(m_pitch * 0.5f);

    return Quat(cp*cr*cy-sp*sr*sy,-cp*sr*sy+cr*cy*sp,cp*cy*sr+sp*cr*sy,cp*cr*sy-sp*cy*sr);
//    return Quat(cp*cr*cy-sr*sp*sy,cr*cp*sy+sr*cy*sp,cr*cy*sp-sr*cp*sy,cr*sp*sy+cp*cy*sr);
}


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

std::string Euler::to_str(bool rad){

    std::stringstream ss;
    if(rad)
        deg2Rad();
    else
        rad2Deg();
    ss <<"[" << m_roll << "," << m_pitch << "," << m_yaw << ",";
    if(rad)
        ss << "rad";
    else
        ss << "deg";
    ss << "]";
    return ss.str();
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

cv::Mat1f Quat::getMat4f() {

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
    Q(1,0)= 2*(m_x*m_y - m_w*m_z);			        Q(1,1)= m_w*m_w - m_x*m_x + m_y*m_y - m_z*m_z;	Q(1,2)= 2*(m_w*m_x - m_y*m_z);			        Q(1,3)= 0;
    Q(2,0)= 2*(m_x*m_z + m_w*m_y);			        Q(2,1)= -2*(m_w*m_x + m_y*m_z);			        Q(2,2)= m_w*m_w - m_x*m_x - m_y*m_y + m_z*m_z;	Q(2,3)= 0;
    Q(3,0)= 0;						                Q(3,1)= 0;						                Q(3,2)= 0;						                Q(3,3)= 1;

    return Q;
}

cv::Mat1d Quat::getMat3d() const{

    Mat1d Q(3,3);

    Q(0,0)= m_w*m_w + m_x*m_x - m_y*m_y - m_z*m_z;	Q(0,1)= 2*(m_x*m_y + m_w*m_z);			        Q(0,2)= 2*(m_x*m_z - m_w*m_y);
    Q(1,0)= 2*(m_x*m_y - m_w*m_z);			        Q(1,1)= m_w*m_w - m_x*m_x + m_y*m_y - m_z*m_z;	Q(1,2)= 2*(m_w*m_x + m_y*m_z);
    Q(2,0)= 2*(m_x*m_z + m_w*m_y);			        Q(2,1)= 2*(m_w*m_x + m_y*m_z);			        Q(2,2)= m_w*m_w - m_x*m_x - m_y*m_y + m_z*m_z;

    return Q;
}

cv::Matx44f Quat::getQMatf() const{
    return Matx44f(m_w,-m_x,-m_y,-m_z,m_x,m_w,-m_z,m_y,m_y,m_z,m_w,-m_x,m_z,-m_y,m_x,m_w);
}

cv::Matx44d Quat::getQMatd() const{
    return Matx44d(m_w,-m_x,-m_y,-m_z,m_x,m_w,-m_z,m_y,m_y,m_z,m_w,-m_x,m_z,-m_y,m_x,m_w);
}

cv::Matx44f Quat::getQ_Matf() const{
    return Matx44f(m_w,-m_x,-m_y,-m_z,m_x,m_w,m_z,-m_y,m_y,-m_z,m_w,m_x,m_z,m_y,-m_x,m_w);
}

cv::Matx44d Quat::getQ_Matd() const{
    return Matx44d(m_w,-m_x,-m_y,-m_z,m_x,m_w,m_z,-m_y,m_y,-m_z,m_w,m_x,m_z,m_y,-m_x,m_w);
}

cv::Matx44d Quat::getdQdq0() const{
    return Matx44d::eye();
}

cv::Matx44d Quat::getdQ_dq0() const{
    return Matx44d::eye();
}

cv::Matx44d Quat::getdQdq1() const{
    return Matx44d(0,-1,0,0,1,0,0,0,0,0,0,-1,0,0,1,0);
}

cv::Matx44d Quat::getdQ_dq1() const{
    return Matx44d(0,-1,0,0,1,0,0,0,0,0,0,1,0,0,-1,0);
}

cv::Matx44d Quat::getdQdq2() const{
    return Matx44d(0,0,-1,0,0,0,0,1,1,0,0,0,0,-1,0,0);
}

cv::Matx44d Quat::getdQ_dq2() const{
    return Matx44d(0,0,-1,0,0,0,0,-1,1,0,0,0,0,1,0,0);
}

cv::Matx44d Quat::getdQdq3() const{
    return Matx44d(0,0,0,-1,0,0,-1,0,0,1,0,0,1,0,0,0);
}

cv::Matx44d Quat::getdQ_dq3() const{
    return Matx44d(0,0,0,-1,0,0,1,0,0,-1,0,0,1,0,0,0);
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
    m_x = m_x*q.w()+m_w*q.x()-m_z*q.y()+m_y*q.z();
    m_y = m_y*q.w()+m_z*q.x()+m_w*q.y()-m_x*q.z();
    m_z = m_z*q.w()-m_y*q.x()+m_x*q.y()+m_w*q.z();
    norm();
}

Quat Quat::operator*(const Quat& q){
    return Quat(m_w*q.w()-(m_x*q.x()+m_y*q.y()+m_z*q.z()),
                m_x*q.w()+m_w*q.x()-m_z*q.y()+m_y*q.z(),
                m_y*q.w()+m_z*q.x()+m_w*q.y()-m_x*q.z(),
                m_z*q.w()-m_y*q.x()+m_x*q.y()+m_w*q.z());
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

Vec3d Quat::operator*(const Vec3d& v){

    norm();
    Quat vq(0,v[0],v[1],v[2]);
    Quat res = *(this) * vq * ((*(this)).conj());
    return Vec3d(res.m_x,res.m_y,res.m_z);

}

Vec4d Quat::operator*(const Vec4d& v){

    norm();
    Quat vq(v[0],v[1],v[2],v[3]);
    Quat res = *(this) * vq * ((*(this)).conj());
    return Vec4d(res.w(),res.x(),res.y(),res.z());

}

ostream& operator<<(ostream& os, const Euler& e){
    os << "[" << e.m_roll << "," << e.m_pitch << "," << e.m_yaw << ",";
    if(e.m_rad) os << "rad";else os << "deg"; os << "]" << endl;
    return os;
}

ostream& operator<<(ostream& os, const Quat& q){
    os << "[" << q.m_w << "|" << q.m_x << "," << q.m_y << "," << q.m_z << "] angle: " << 2*acos(q.m_w) << endl;
    return os;
}
