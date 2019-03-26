#include "utils.h"

using namespace cv;
using namespace std;

namespace me{


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
                                        -sr*sp*cy+cr*sy,    -sr*sp*sy-cr*cy,   -sr*cp);

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
                                        -sr*sp*sy-cr*cy,    sr*sp*cy-cr*sy,       0,
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

    T cy = cos(m_yaw * 0.5f);
    T sy = sin(m_yaw * 0.5f);
    T cr = cos(m_roll * 0.5f);
    T sr = sin(m_roll * 0.5f);
    T cp = cos(m_pitch * 0.5f);
    T sp = sin(m_pitch * 0.5f);

    return Quat<T>(cr*cp*cy+sr*sp*sy,-cr*sp*sy+cp*cy*sr,cr*cy*sp+sr*cp*sy,cr*cp*sy-sr*cy*sp);
}

template <typename T>
Vec<T,3> Euler<T>::getVector() const{
    return Vec<T,3>(m_roll,m_pitch,m_yaw);
}

template <typename T>
void Euler<T>::operator+=(const Euler& e){

    m_roll += e.roll();
    m_pitch += e.pitch();
    m_yaw += e.yaw();
}

template <typename T>
cv::Vec<T,3> Euler<T>::operator*(const cv::Vec<T,3>& v) const{
    return getR3() * v;
}

template <typename T>
cv::Matx<T,3,1> Euler<T>::operator*(const cv::Matx<T,3,1>& v) const{
    return getR3() * v;
}

template <typename T>
cv::Vec<T,4> Euler<T>::operator*(const cv::Vec<T,4>& v) const{
    return getR4() * v;
}

template <typename T>
cv::Matx<T,4,1> Euler<T>::operator*(const cv::Matx<T,4,1>& v) const{
    return getR4() * v;
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
void Quat<T>::normalize(){
    T n = sqrt(m_w*m_w + m_x*m_x + m_y*m_y + m_z*m_z);
    if(n != 0.0){
        m_w /= n;
        m_x /= n;
        m_y /= n;
        m_z /= n;
    }
}


template <typename T>
void Quat<T>::fromMat(const cv::Mat& M){

    Vec3d vec;
    cv::Rodrigues(M,vec);

    T norm = sqrt(pow(vec(0),2)+pow(vec(1),2)+pow(vec(2),2));
    T theta = (norm < 1e-10 ? 1e-10:norm);
    m_w = cos(theta/2.0);
    m_x = vec(0)/theta*sin(theta/2.0);
    m_y = vec(1)/theta*sin(theta/2.0);
    m_z = vec(2)/theta*sin(theta/2.0);
    this->normalize();
}

template <typename T>
Euler<T> Quat<T>::getEuler(){

    normalize();
    return Euler<T>(atan2( 2*(m_w*m_x + m_y*m_z),m_w*m_w - m_x*m_x - m_y*m_y + m_z*m_z),-asin(2*(m_x*m_z - m_w*m_y)),atan2(2*(m_x*m_y + m_w*m_z),m_w*m_w + m_x*m_x - m_y*m_y - m_z*m_z));
}

template <typename T>
void Quat<T>::operator*=(const Quat& q){
    m_w = m_w*q.w()-(m_x*q.x()+m_y*q.y()+m_z*q.z());
    m_x = m_w*q.x()+m_x*q.w()+m_y*q.z()-m_z*q.y();
    m_y = m_w*q.y()-m_x*q.z()+m_y*q.w()+m_z*q.x();
    m_z = m_w*q.z()+m_x*q.y()-m_y*q.x()+m_z*q.w();
}

template <typename T>
Quat<T> Quat<T>::operator*(const Quat& q) const{
    return Quat(m_w*q.w()-(m_x*q.x()+m_y*q.y()+m_z*q.z()),
                m_w*q.x()+m_x*q.w()+m_y*q.z()-m_z*q.y(),
                m_w*q.y()-m_x*q.z()+m_y*q.w()+m_z*q.x(),
                m_w*q.z()+m_x*q.y()-m_y*q.x()+m_z*q.w());
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
    normalize();
}

template <typename T>
void Quat<T>::operator-=(const Quat<T>& q){
    m_w -= q.w();
    m_x -= q.x();
    m_y -= q.y();
    m_z -= q.z();
    normalize();
}

template <typename T>
void Quat<T>::operator/=(double nb){
    m_w /=nb;
    m_x /=nb;
    m_y /=nb;
    m_z /=nb;
    normalize();
}

template <typename T>
Vec<T,3> Quat<T>::operator*(const Vec<T,3>& v) const{

    Vec<T,4> vq(0,v[0],v[1],v[2]);
    Vec<T,4> v_r = (*this) * vq;
    return Vec<T,3>(v_r(1),v_r(2),v_r(3));

}

template <typename T>
Matx<T,3,1> Quat<T>::operator*(const Matx<T,3,1>& v) const{

    Matx41d vq(0,v(0),v(1),v(2));
    Matx41d v_r = (*this) * vq;
    return Matx<T,3,1>(v_r(1),v_r(2),v_r(3));

}

template <typename T>
Matx<T,4,1> Quat<T>::operator*(const Matx<T,4,1>& v) const{
    return getQl().t() * getQr() * v;
}

template <typename T>
Vec<T,4> Quat<T>::operator*(const Vec<T,4>& v) const{
    return getQl().t() * getQr() * v;
}

template class Euler<float>;
template class Euler<double>;
template class Quat<float>;
template class Quat<double>;

template<typename T>
void convertToOpenCV(Euler<T>& e){

    Vec<T,3> vec(e.roll(),e.pitch(),e.yaw());
    vec = (Matx<T,3,3>)TRef * vec;
    e = Euler<T>(vec(0),vec(1),vec(2));
}

template<typename T>
void convertToXYZ(Euler<T>& e){
    Vec<T,3> vec(e.roll(),e.pitch(),e.yaw());
    vec = ((Matx<T,3,3>)(TRef)).t() * vec;
    e = Euler<T>(vec(0),vec(1),vec(2));
}

template<typename T>
void convertToOpenCV(Quat<T>& q){
    Quat<T> p(TRef);
    q = p*q;
}

template<typename T>
void convertToXYZ(Quat<T>& q){
    q *= Quat<T>(0.5,-0.5,0.5,-0.5).conj();
}

template<typename T>
void convertToOpenCV(Vec<T,3>& vec){
    vec = (Matx<T,3,3>)TRef * vec;
}

template<typename T>
void convertToXYZ(Vec<T,3>& vec){
    vec = ((Matx<T,3,3>)(TRef)).t() * vec;
}

template <typename T>
cv::Matx<T,4,3> Gq_v(const Vec<T,3>& rot_vec){

    double snorm = rot_vec[0]*rot_vec[0]+rot_vec[1]*rot_vec[1]+rot_vec[2]*rot_vec[2]; //squared norm
    double norm = sqrt(snorm)+1e-20;
    double a =cos(0.5*norm)*norm-2*sin(0.5*norm);

    return 1/(2*pow(norm,3)) * cv::Matx<T,4,3>( -rot_vec[0]*snorm*sin(0.5*norm),                -rot_vec[1]*snorm*sin(0.5*norm),                -rot_vec[2]*snorm*sin(0.5*norm),
                                                2*snorm*sin(0.5*norm)+rot_vec[0]*rot_vec[0]*a,  rot_vec[0]*rot_vec[1]*a,                        rot_vec[0]*rot_vec[2]*a,
                                                rot_vec[0]*rot_vec[1]*a,                        2*snorm*sin(0.5*norm)+rot_vec[1]*rot_vec[1]*a,  rot_vec[1]*rot_vec[2]*a,
                                                rot_vec[0]*rot_vec[2]*a,                        rot_vec[1]*rot_vec[2]*a,                        2*snorm*sin(0.5*norm)+rot_vec[2]*rot_vec[2]*a);
}

template cv::Matx43d Gq_v(const Vec3d& v);
template cv::Matx43f Gq_v(const Vec3f& v);

template void convertToOpenCV(Euler<double>& e);
template void convertToXYZ(Euler<double>& e);
template void convertToOpenCV(Euler<float>& e);
template void convertToXYZ(Euler<float>& e);
template void convertToOpenCV(Quat<double>& q);
template void convertToXYZ(Quat<double>& q);
template void convertToOpenCV(Quat<float>& q);
template void convertToXYZ(Quat<float>& q);
template void convertToOpenCV(Vec3d& vec);
template void convertToXYZ(Vec3d& vec);
template void convertToOpenCV(Vec3f& vec);
template void convertToXYZ(Vec3f& vec);

}
