#include "utils.h"

#include <iostream>



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
cv::Vec<T,3> Euler<T>::operator*(const cv::Vec<T,3>& v){
    return getR3() * v;
}

template <typename T>
cv::Vec<T,4> Euler<T>::operator*(const cv::Vec<T,4>& v){
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

std::vector<pt2D> nonMaxSupScanline3x3(const cv::Mat& input, cv::Mat& output)
{
    //Identify the regional maxima in the image
    //Regional maxima are connected components of pixels with the same
    //intensity value, t, whose external boundary pixels all have a value
    //less than t.
    //An 8 connection is used here
    //The input is a double precision matrix
    //The output is a binary matrix whose intensity value is 1 for maxima, 0 otherwise

    //Algorithm can be found here http://download.springer.com/static/pdf/269/chp%253A10.1007%252F978-3-642-17688-3_41.pdf?originUrl=http%3A%2F%2Flink.springer.com%2Fchapter%2F10.1007%2F978-3-642-17688-3_41&token2=exp=1434641760~acl=%2Fstatic%2Fpdf%2F269%2Fchp%25253A10.1007%25252F978-3-642-17688-3_41.pdf%3ForiginUrl%3Dhttp%253A%252F%252Flink.springer.com%252Fchapter%252F10.1007%252F978-3-642-17688-3_41*~hmac=19ed69bc8952599cdc45d31cdf1486bef2269679b596ee784b9a76f35f511dd4
    //(scanline)
    std::vector<pt2D> maxima;
    output.create(input.size(), CV_8U);
    output.setTo(cv::Scalar(0));

    int h = input.rows;
    int w = input.cols;

    cv::Mat skip(2, w, CV_8U, cv::Scalar(0));
    int cur = 0;
    int next = 1;

    for (int r = 1; r < h - 1; r++)
    {
        int c = 1;
        uchar * ptrOutput = output.ptr<uchar>(r);
        const double * ptrInput = input.ptr<double>(r);
        while (c < w-1)
        {
            uchar * ptrSkip = skip.ptr<uchar>(cur);
            if (ptrSkip[c])
            {
                c++;//Skip current pixel
                continue;
            }
            if (ptrInput[c] <= ptrInput[c+1])
            {
                c++;
                while (c < w - 1 && ptrInput[c] <= ptrInput[c+1]) c++;
                if (c == w-1) break;
            }
            else if (ptrInput[c] <= ptrInput[c-1]) //compare to pixel on the right
            {
                c++;
                continue;
            }
            ptrSkip[c+1] = 1; // skip next pixel in the scanline

            ptrSkip = skip.ptr<uchar>(next);
            //compare to 3 future then 3 past neighbors
            const double * ptrInputRp1 = input.ptr<double>(r + 1);
            if (ptrInput[c] <= ptrInputRp1[c-1])
            {
                c++;
                continue;
            }
            ptrSkip[c-1] = 1; // skip future neighbors only
            if (ptrInput[c] <= ptrInputRp1[c])
            {
                c++;
                continue;
            }
            ptrSkip[c] = 1;
            if (ptrInput[c] <= ptrInputRp1[c + 1])
            {
                c++;
                continue;
            }
            ptrSkip[c + 1] = 1;

            const double * ptrInputRm1 = input.ptr<double>(r - 1);
            if (ptrInput[c] <= ptrInputRm1[c-1])
            {
                c++;
                continue;
            }
            if (ptrInput[c] <= ptrInputRm1[c])
            {
                c++;
                continue;
            }
            if (ptrInput[c] <= ptrInputRm1[c +1])
            {
                c++;
                continue;
            }
            ptrOutput[c] = 255;//a new local maximum is found
            double sub_v = c+0.5+(ptrInput[c+1]-ptrInput[c-1])/(ptrInput[c-1]+ptrInput[c]+ptrInput[c+1]);
            double sub_u = r+0.5+(ptrInputRp1[c]-ptrInputRm1[c])/(ptrInputRm1[c]+ptrInput[c]+ptrInputRp1[c]);
            maxima.push_back(pt2D(sub_u,sub_v));
            c++;
        }
        std::swap(cur, next); //swap mask indices
        skip.row(next).setTo(0);//reset next scanline mask
    }

    return maxima;
}

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
