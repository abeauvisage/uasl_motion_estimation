#include "utils.h"

#include <iostream>



using namespace cv;
using namespace std;


/******************

     Euler class

*******************/

template <typename T>
Matx<T,3,3> Euler<T>::getR3() const{

    T cr,sr,cp,sp,cy,sy;
    computeCosSin(cr,sr,cp,sp,cy,sy);

    return typename Matx<T,3,3>::Matx(  cp*cy,          cp*sy,	        -sp,
                                        sp*sr*cy-cr*sy, sr*sp*sy+cr*cy, cp*sr,
                                        cr*sp*cy+sr*sy, cr*sp*sy-sr*cy, cp*cr);
}

template <typename T>
Matx<T,4,4> Euler<T>::getR4() const{

    T cr,sr,cp,sp,cy,sy;
    computeCosSin(cr,sr,cp,sp,cy,sy);

    return typename Matx<T,4,4>::Matx(  cp*cy,          cp*sy,	        -sp,    0,
                                        sp*sr*cy-cr*sy, sr*sp*sy+cr*cy, cp*sr,  0,
                                        cr*sp*cy+sr*sy, cr*sp*sy-sr*cy, cp*cr,  0,
                                        0,              0,              0,      1);
}

template <typename T>
Matx<T,3,3> Euler<T>::getE() const{

    T cr,sr,cp,sp,cy,sy;
    computeCosSin(cr,sr,cp,sp,cy,sy);

    return typename Matx<T,3,3>::Matx(cp*cr,-sy,0,cp*sy,cy,0,-sp,0,1);

}

template <typename T>
Matx<T,3,3> Euler<T>::getdRdr() const{

    T cr,sr,cp,sp,cy,sy;
    computeCosSin(cr,sr,cp,sp,cy,sy);

    return typename Matx<T,3,3>::Matx(   0,                  0,              0,
                                        cr*sp*cy+sr*sy,     cr*sp*sy-sr*cy, cr*cp,
                                        -sr*sp*cy+cr*sy,    -sr*sp-sr*cy,   -sr*cp);

}

template <typename T>
Matx<T,3,3> Euler<T>::getdRdp() const{

    T cr,sr,cp,sp,cy,sy;
    computeCosSin(cr,sr,cp,sp,cy,sy);

    return typename Matx<T,3,3>::Matx(  -cy*sp,     -sy*sp,     -cp,
                                        sr*cp*cy,   sr*cp*sy,   -sr*sp,
                                        cr*cp*cy,   cr*cp*sy,   -cr*sp);

}

template <typename T>
Matx<T,3,3> Euler<T>::getdRdy() const{

    T cr,sr,cp,sp,cy,sy;
    computeCosSin(cr,sr,cp,sp,cy,sy);

    return typename Matx<T,3,3>::Matx(  -cp*sy,             cp*cy,          0,
                                        -sr*sp*sy-cr*cy,    sr*sp*cy,       0,
                                        -cr*sp*sy+sr*cy,    cr*sp*cy+sr*sy, 0);

}

template <>
void Euler<float>::fromMat(const Mat& M){

    assert((M.type() == CV_32F || M.type() == CV_64F) && M.channels() == 1);

    Mat_<float> R;

    if(M.type() == CV_64F)
        M.convertTo(R,CV_32F);
    else
        R = M;

    m_roll = atan2(R(1,2),R(2,2));
    m_pitch = -asin(R(0,2));
    m_yaw = atan2(R(0,1),R(0,0));

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
}

template <>
void Euler<double>::fromMat(const Mat& M){

    assert((M.type() == CV_32F || M.type() == CV_64F) && M.channels() == 1);

    Mat_<double> R;

    if(M.type() == CV_32F)
        M.convertTo(R,CV_64F);
    else
        R = M;

    m_roll = atan2(R(1,2),R(2,2));
    m_pitch = -asin(R(0,2));
    m_yaw = atan2(R(0,1),R(0,0));

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
}

template <typename T>
Quat<T> Euler<T>::getQuat() const{

    T cy = cos(m_roll * 0.5f);
    T sy = sin(m_roll * 0.5f);
    T cr = cos(m_yaw * 0.5f);
    T sr = sin(m_yaw * 0.5f);
    T cp = cos(m_pitch * 0.5f);
    T sp = sin(m_pitch * 0.5f);

    return Quat<T>(cr*cp*cy+sr*sp*sy,-cr*sp*sy+cp*cy*sr,cr*cy*sp+sr*cp*sy,cr*cp*sy-sr*cy*sp);
}

template <typename T>
void Euler<T>::operator+=(Euler& e){

    m_roll += e.roll();
    m_pitch += e.pitch();
    m_yaw += e.yaw();
}

template <typename T>
std::string Euler<T>::getDegrees(){

    std::stringstream ss;
    rad2Deg();
    ss <<"[" << m_roll << "," << m_pitch << "," << m_yaw << ", deg]";
    deg2Rad();

    return ss.str();
}

/***********************

        Quat class

************************/

template <typename T>
void Quat<T>::norm(){
    T n = sqrt(m_w*m_w + m_x*m_x + m_y*m_y + m_z*m_z);
    if(n != 0.0){
        m_w /= n;
        m_x /= n;
        m_y /= n;
        m_z /= n;
    }
}

template <typename T>
inline cv::Matx<T,4,4> Quat<T>::getR4() const{

    return typename Matx<T,4,4>::Matx(  m_w*m_w + m_x*m_x - m_y*m_y - m_z*m_z,  2*(m_x*m_y + m_w*m_z),                  2*(m_x*m_z - m_w*m_y),                  0,
                                        2*(m_x*m_y - m_w*m_z),                  m_w*m_w - m_x*m_x + m_y*m_y - m_z*m_z,  2*(m_w*m_x + m_y*m_z),                  0,
                                        2*(m_x*m_z + m_w*m_y),			        2*(m_w*m_x - m_y*m_z),			        m_w*m_w - m_x*m_x - m_y*m_y + m_z*m_z,  0,
                                        0,                                      0,						                0,					                    1);
}

template <typename T>
inline cv::Matx<T,3,3> Quat<T>::getR3() const{

    return typename Matx<T,3,3>::Matx(  m_w*m_w + m_x*m_x - m_y*m_y - m_z*m_z,  2*(m_x*m_y + m_w*m_z),                  2*(m_x*m_z - m_w*m_y),
                                        2*(m_x*m_y - m_w*m_z),                  m_w*m_w - m_x*m_x + m_y*m_y - m_z*m_z,  2*(m_w*m_x + m_y*m_z),
                                        2*(m_x*m_z + m_w*m_y),			        2*(m_w*m_x - m_y*m_z),			        m_w*m_w - m_x*m_x - m_y*m_y + m_z*m_z);
}

template <typename T>
inline cv::Matx<T,4,4> Quat<T>::getQ() const{
    return typename Matx<T,4,4>::Matx(m_w,-m_x,-m_y,-m_z,m_x,m_w,-m_z,m_y,m_y,m_z,m_w,-m_x,m_z,-m_y,m_x,m_w);
}

template <typename T>
inline cv::Matx<T,4,4> Quat<T>::getQ_() const{
    return typename Matx<T,4,4>::Matx(m_w,-m_x,-m_y,-m_z,m_x,m_w,m_z,-m_y,m_y,-m_z,m_w,m_x,m_z,m_y,-m_x,m_w);
}

template <typename T>
inline cv::Matx<T,4,4> Quat<T>::getdQdq0() const{
    return Matx<T,4,4>::eye();
}

template <typename T>
inline cv::Matx<T,4,4> Quat<T>::getdQ_dq0() const{
    return Matx<T,4,4>::eye();
}

template <typename T>
inline cv::Matx<T,4,4> Quat<T>::getdQdq1() const{
    return typename Matx<T,4,4>::Matx(0,-1,0,0,1,0,0,0,0,0,0,-1,0,0,1,0);
}

template <typename T>
inline cv::Matx<T,4,4> Quat<T>::getdQ_dq1() const{
    return typename Matx<T,4,4>::Matx(0,-1,0,0,1,0,0,0,0,0,0,1,0,0,-1,0);
}

template <typename T>
inline cv::Matx<T,4,4> Quat<T>::getdQdq2() const{
    return typename Matx<T,4,4>::Matx(0,0,-1,0,0,0,0,1,1,0,0,0,0,-1,0,0);
}

template <typename T>
inline cv::Matx<T,4,4> Quat<T>::getdQ_dq2() const{
    return typename Matx<T,4,4>::Matx(0,0,-1,0,0,0,0,-1,1,0,0,0,0,1,0,0);
}

template <typename T>
inline cv::Matx<T,4,4> Quat<T>::getdQdq3() const{
    return typename Matx<T,4,4>::Matx(0,0,0,-1,0,0,-1,0,0,1,0,0,1,0,0,0);
}

template <typename T>
inline cv::Matx<T,4,4> Quat<T>::getdQ_dq3() const{
    return typename Matx<T,4,4>::Matx(0,0,0,-1,0,0,1,0,0,-1,0,0,1,0,0,0);
}

template <>
void Quat<float>::fromMat(const cv::Mat& M){

    assert((M.type() == CV_32F || M.type() == CV_64F) && M.channels() == 1);

    Mat_<float> m;

    if(M.type() == CV_64F)
        M.convertTo(m,CV_32F);
    else
        m = M;


    if(m(1,1) > -m(2,2) && m(0,0) > - m(1,1) && m(0,0) > -m(2,2)){
        float norm = sqrt(1+m(0,0)+m(1,1)+m(2,2));
        m_w = norm/2;
        m_x = (m(1,2)-m(2,1))/(2*norm);
        m_y = (m(2,0)-m(0,2))/(2*norm);
        m_z = (m(0,1)-m(1,0))/(2*norm);
    } else if (m(1,1) < -m(2,2) && m(0,0) > m(1,1) && m(0,0) > m(2,2)){
        float norm = sqrt(1+m(0,0)-m(1,1)-m(2,2));
        m_w = (m(1,2)-m(2,1))/(2*norm);
        m_x = norm/2;
        m_y = (m(0,1)+m(1,0))/(2*norm);
        m_z = (m(2,0)+m(0,2))/(2*norm);
    } else if (m(1,1) > m(2,2) && m(0,0) < m(1,1) && m(0,0) < -m(2,2)){
        float norm = sqrt(1-m(0,0)+m(1,1)-m(2,2));
        m_w = (m(2,0)-m(0,2))/(2*norm);
        m_x = (m(0,1)+m(1,0))/(2*norm);
        m_y = norm/2;
        m_z = (m(1,2)+m(2,1))/(2*norm);
    } else{
        float norm = sqrt(1-m(0,0)-m(1,1)+m(2,2));
        m_w = (m(0,1)-m(1,0))/(2*norm);
        m_x = (m(2,0)+m(0,2))/(2*norm);
        m_y = (m(1,2)+m(2,1))/(2*norm);
        m_z = norm/2;
    }
}

template <>
void Quat<double>::fromMat(const cv::Mat& M){

    assert((M.type() == CV_32F || M.type() == CV_64F) && M.channels() == 1);

    Mat_<double> m;

    if(M.type() == CV_32F)
        M.convertTo(m,CV_64F);
    else
        m = M;


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

template <typename T>
Euler<T> Quat<T>::getEuler(){

    norm();
    return Euler<T>(atan2( 2*(m_w*m_x + m_y*m_z),m_w*m_w - m_x*m_x - m_y*m_y + m_z*m_z),-asin(2*(m_x*m_z - m_w*m_y)),atan2(2*(m_x*m_y + m_w*m_z),m_w*m_w + m_x*m_x - m_y*m_y - m_z*m_z));
}

template <typename T>
void Quat<T>::operator*=(const Quat& q){
    m_w = m_w*q.w()-(m_x*q.x()+m_y*q.y()+m_z*q.z());
    m_x = m_x*q.w()+m_w*q.x()-m_z*q.y()+m_y*q.z();
    m_y = m_y*q.w()+m_z*q.x()+m_w*q.y()-m_x*q.z();
    m_z = m_z*q.w()-m_y*q.x()+m_x*q.y()+m_w*q.z();
}

template <typename T>
Quat<T> Quat<T>::operator*(const Quat& q) const{
    return Quat(m_w*q.w()-(m_x*q.x()+m_y*q.y()+m_z*q.z()),
                m_x*q.w()+m_w*q.x()-m_z*q.y()+m_y*q.z(),
                m_y*q.w()+m_z*q.x()+m_w*q.y()-m_x*q.z(),
                m_z*q.w()-m_y*q.x()+m_x*q.y()+m_w*q.z());
}

template <typename T>
Quat<T> Quat<T>::operator*(const double d) const{
    return Quat(m_w*d,m_x*d,m_y*d,m_z*d);
}

template <typename T>
Quat<T> Quat<T>::operator+(const Quat& q) const{
    return Quat(m_w+q.w(),m_x+q.x(),m_y+q.y(),m_z+q.z());
}

template <typename T>
void Quat<T>::operator+=(const Quat<T>& q){
    m_w += q.w();
    m_x += q.x();
    m_y += q.y();
    m_z += q.z();
    norm();
}

template <typename T>
void Quat<T>::operator-=(const Quat<T>& q){
    m_w -= q.w();
    m_x -= q.x();
    m_y -= q.y();
    m_z -= q.z();
    norm();
}

template <typename T>
void Quat<T>::operator/=(double nb){
    m_w /=nb;
    m_x /=nb;
    m_y /=nb;
    m_z /=nb;
    norm();
}

template <typename T>
Vec<T,3> Quat<T>::operator*(const Vec<T,3>& v){

    Quat vq(0,v[0],v[1],v[2]);
    Quat res = *(this) * vq * ((*(this)).conj());
    return Vec<T,3>(res.m_x,res.m_y,res.m_z);

}

template <typename T>
Vec<T,4> Quat<T>::operator*(const Vec<T,4>& v){

    Quat vq(v[0],v[1],v[2],v[3]);
    Quat res = *(this) * vq * ((*(this)).conj());
    return Vec<T,4>(res.w(),res.x(),res.y(),res.z());

}

template class Euler<float>;
template class Euler<double>;
template class Quat<float>;
template class Quat<double>;
