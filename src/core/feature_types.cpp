/** \file feature_types.cpp
*   \brief Defines useful classes and structs to represents image features
*
*   Defines:  - homogeneous point coordinates.
*             - stereo and quad matches.
*             - points and camera poses for windowed bundle adjustment
*
*    \author Axel Beauvisage (axel.beauvisage@gmail.com)
*/

#include "core/feature_types.h"

namespace me{

namespace core{

template class WBA_Point<cv::Point2f>;

template<>
cv::Matx44f CamPose_ef::TrMat() const {
    cv::Mat Tr = (cv::Mat) orientation.getR4();
    ((cv::Mat) position).copyTo(Tr(cv::Range(0,3),cv::Range(3,4)));
    return Tr;
}

template<>
cv::Matx44d CamPose_ed::TrMat() const {
    cv::Mat Tr = (cv::Mat) orientation.getR4();
    ((cv::Mat) position).copyTo(Tr(cv::Range(0,3),cv::Range(3,4)));
    return Tr;
}

template<>
cv::Matx44f CamPose_qf::TrMat() const {
    cv::Mat Tr = (cv::Mat) orientation.getR4();
    ((cv::Mat) position).copyTo(Tr(cv::Range(0,3),cv::Range(3,4)));
    return Tr;
}

template<>
cv::Matx44d CamPose_qd::TrMat() const {
    cv::Mat Tr = (cv::Mat) orientation.getR4();
    ((cv::Mat) position).copyTo(Tr(cv::Range(0,3),cv::Range(3,4)));
    return Tr;
}

template<>
cv::Matx44d CamPose_md::TrMat() const {
    cv::Mat Tr = cv::Mat::eye(4,4,CV_64F);
    ((cv::Mat)orientation).copyTo(Tr(cv::Range(0,3),cv::Range(0,3)));
    ((cv::Mat) position).copyTo(Tr(cv::Range(0,3),cv::Range(3,4)));
    return Tr;
}

template<>
cv::Matx44f CamPose_mf::TrMat() const {
    cv::Mat Tr = cv::Mat::eye(4,4,CV_32F);
    ((cv::Mat)orientation).copyTo(Tr(cv::Range(0,3),cv::Range(0,3)));
    ((cv::Mat) position).copyTo(Tr(cv::Range(0,3),cv::Range(3,4)));
    return Tr;
}

template<>
CamPose_qd CamPose_qd::inv() const{
    return CamPose_qd(ID,orientation.conj(),-(orientation.conj()*position),Cov);
}

template<>
CamPose_qf CamPose_qf::inv() const{
    return CamPose_qf(ID,orientation.conj(),-(orientation.conj()*position),Cov);
}

template<>
CamPose_md CamPose_md::inv() const{
    return CamPose_md(ID,orientation.t(),-orientation.t()*position,Cov);
}

template<>
CamPose_mf CamPose_mf::inv() const{
    return CamPose_mf(ID,orientation.t(),-orientation.t()*position,Cov);
}

template <>
cv::Mat CamPose_qd::JacobianMult(const CamPose_qd& old_pose) const{
    CamPose_qd new_pose = old_pose * (*this);
    cv::Mat J_mul = cv::Mat::zeros(6,12,CV_64F);
    J_mul(cv::Range(0,3),cv::Range(0,3)) = cv::Mat::eye(3,3,CV_64F);
    ((cv::Mat) old_pose.orientation.getH_qvec(position)).copyTo(J_mul(cv::Range(0,3),cv::Range(3,6)));
    ((cv::Mat) old_pose.orientation.getR3()).copyTo(J_mul(cv::Range(0,3),cv::Range(6,9)));
    ((cv::Mat) (new_pose.orientation.getH() * orientation.getQr() * old_pose.orientation.getG())).copyTo(J_mul(cv::Range(3,6),cv::Range(3,6)));
    ((cv::Mat) (new_pose.orientation.getH() * old_pose.orientation.getQl() * orientation.getG())).copyTo(J_mul(cv::Range(3,6),cv::Range(9,12)));

    return J_mul;
}

template <>
cv::Mat CamPose_qf::JacobianMult(const CamPose_qf& old_pose) const{
    CamPose_qf new_pose = old_pose * (*this);
    cv::Mat J_mul = cv::Mat::zeros(6,12,CV_64F);
    J_mul(cv::Range(0,3),cv::Range(0,3)) = cv::Mat::eye(3,3,CV_64F);
    ((cv::Mat) old_pose.orientation.getH_qvec(position)).copyTo(J_mul(cv::Range(0,3),cv::Range(3,6)));
    ((cv::Mat) old_pose.orientation.getR3()).copyTo(J_mul(cv::Range(0,3),cv::Range(6,9)));
    ((cv::Mat) (new_pose.orientation.getH() * orientation.getQr() * old_pose.orientation.getG())).copyTo(J_mul(cv::Range(3,6),cv::Range(3,6)));
    ((cv::Mat) (new_pose.orientation.getH() * old_pose.orientation.getQl() * orientation.getG())).copyTo(J_mul(cv::Range(3,6),cv::Range(9,12)));

    return J_mul;
}

template <>
cv::Mat CamPose_qd::JacobianMultReverse(const CamPose_qd& old_pose) const{
    CamPose_qd new_pose = (*this) * old_pose;
    cv::Mat J_mul = cv::Mat::zeros(6,12,CV_64F);
    ((cv::Mat) orientation.getR3()).copyTo(J_mul(cv::Range(0,3),cv::Range(0,3)));
    J_mul(cv::Range(0,3),cv::Range(6,9)) = cv::Mat::eye(3,3,CV_64F);
    ((cv::Mat) orientation.getH_qvec(old_pose.position)).copyTo(J_mul(cv::Range(0,3),cv::Range(9,12)));
    ((cv::Mat) (new_pose.orientation.getH() * orientation.getQl() * old_pose.orientation.getG())).copyTo(J_mul(cv::Range(3,6),cv::Range(3,6)));
    ((cv::Mat) (new_pose.orientation.getH() * old_pose.orientation.getQr() * orientation.getG())).copyTo(J_mul(cv::Range(3,6),cv::Range(9,12)));

    return J_mul;
}

template <>
cv::Mat CamPose_qf::JacobianMultReverse(const CamPose_qf& old_pose) const{
    CamPose_qf new_pose = (*this) * old_pose;
    cv::Mat J_mul = cv::Mat::zeros(6,12,CV_32F);
    ((cv::Mat) orientation.getR3()).copyTo(J_mul(cv::Range(0,3),cv::Range(0,3)));
    J_mul(cv::Range(0,3),cv::Range(6,9)) = cv::Mat::eye(3,3,CV_32F);
    ((cv::Mat) orientation.getH_qvec(old_pose.position)).copyTo(J_mul(cv::Range(0,3),cv::Range(9,12)));
    ((cv::Mat) (new_pose.orientation.getH() * orientation.getQl() * old_pose.orientation.getG())).copyTo(J_mul(cv::Range(3,6),cv::Range(3,6)));
    ((cv::Mat) (new_pose.orientation.getH() * old_pose.orientation.getQr() * orientation.getG())).copyTo(J_mul(cv::Range(3,6),cv::Range(9,12)));

    return J_mul;
}

template <>
cv::Mat CamPose_qd::JacobianInv() const{
    cv::Mat J_inv = cv::Mat::zeros(6,6,CV_64F);
    ((cv::Mat) -orientation.conj().getR3()).copyTo(J_inv(cv::Range(0,3),cv::Range(0,3)));
    ((cv::Mat) orientation.conj().getH_qvec(position)).copyTo(J_inv(cv::Range(0,3),cv::Range(3,6)));
    J_inv(cv::Range(3,6),cv::Range(3,6)) = -cv::Mat::eye(3,3,CV_64F);

    return J_inv;
}

template <>
cv::Mat CamPose_qf::JacobianInv() const{
    cv::Mat J_inv = cv::Mat::zeros(6,6,CV_32F);
    ((cv::Mat) -orientation.conj().getR3()).copyTo(J_inv(cv::Range(0,3),cv::Range(0,3)));
    ((cv::Mat) -orientation.conj().getH_qvec(position)).copyTo(J_inv(cv::Range(0,3),cv::Range(3,6)));
    J_inv(cv::Range(3,6),cv::Range(3,6)) = -cv::Mat::eye(3,3,CV_32F);

    return J_inv;
}

template <>
cv::Mat CamPose_qd::JacobianScale(double scale) const{
    cv::Mat J_scale = cv::Mat::eye(6,7,CV_64F);
    J_scale(cv::Range(0,3),cv::Range(0,3)) *= scale;
    ((cv::Mat)position).copyTo(J_scale(cv::Range(0,3),cv::Range(6,7)));

    return J_scale;
}

template <>
cv::Mat CamPose_qf::JacobianScale(float scale) const{
    cv::Mat J_scale = cv::Mat::eye(6,7,CV_32F);
    J_scale(cv::Range(0,3),cv::Range(0,3)) *= scale;
    ((cv::Mat)position).copyTo(J_scale(cv::Range(0,3),cv::Range(6,7)));

    return J_scale;
}

template<>
CamPose_qd poseMultiplicationWithCovariance(const CamPose_qd& p1, const CamPose_qd& p2, int ID){

    assert(!p1.Cov.empty() && !p2.Cov.empty() && "Poses cannot be mulitplied (empty Cov matrix)");
    // P3 = P1 * P2 = R1 R2 | R1 t2 + t1
    CamPose_qd p3 = p1 * p2;
    cv::Mat augmented_cov = cv::Mat::zeros(12,12,CV_64F);
    p1.Cov.copyTo(augmented_cov(cv::Range(0,6),cv::Range(0,6)));
    p2.Cov.copyTo(augmented_cov(cv::Range(6,12),cv::Range(6,12)));

    cv::Mat J = cv::Mat::zeros(6,12,CV_64F);
    // J = [ I dR1t2/dq1 R1 0; 0 dq3/dq1 0 dq3/dq2]
    ((cv::Mat) cv::Mat::eye(3,3,CV_64F)).copyTo(J(cv::Range(0,3),cv::Range(0,3)));
    ((cv::Mat)/* cv::Mat::eye(3,3,CV_64F)*/p1.orientation.getH_qvec(p2.position)).copyTo(J(cv::Range(0,3),cv::Range(3,6)));
    ((cv::Mat) p1.orientation.getR3()).copyTo(J(cv::Range(0,3),cv::Range(6,9)));
    ((cv::Mat) (p3.orientation.getH() * p2.orientation.getQr() * p1.orientation.getG())).copyTo(J(cv::Range(3,6),cv::Range(3,6)));
    ((cv::Mat) (p3.orientation.getH() * p1.orientation.getQl() * p2.orientation.getG())).copyTo(J(cv::Range(3,6),cv::Range(9,12)));

    p3.ID = ID;
    cv::Mat new_cov = J * augmented_cov * J.t();
    new_cov.copyTo(p3.Cov);
    return p3;
}

template<>
CamPose_qd poseMultiplicationWithCovarianceReverse(const CamPose_qd& p1, const CamPose_qd& p2, int ID){

    assert(!p1.Cov.empty() && !p2.Cov.empty() && "Poses cannot be multiplied (empty Cov matrix)");
    // P3 = P2 * P1 = R2 R1 | R2 t1 + t2
    CamPose_qd p3 = p2 * p1;
    cv::Mat augmented_cov = cv::Mat::zeros(12,12,CV_64F);
    p1.Cov.copyTo(augmented_cov(cv::Range(0,6),cv::Range(0,6)));
    p2.Cov.copyTo(augmented_cov(cv::Range(6,12),cv::Range(6,12)));

    cv::Mat J = cv::Mat::zeros(6,12,CV_64F);
    // J = [ R2 0 I dR2t1/dq2; 0 dq3/dq1 0 dq3/dq2]
    ((cv::Mat) p2.orientation.getR3()).copyTo(J(cv::Range(0,3),cv::Range(0,3)));
    ((cv::Mat) cv::Mat::eye(3,3,CV_32F)).copyTo(J(cv::Range(0,3),cv::Range(6,9)));
    ((cv::Mat) p2.orientation.getH_qvec(p1.position)).copyTo(J(cv::Range(0,3),cv::Range(9,12)));
    ((cv::Mat) (p3.orientation.getH() * p2.orientation.getQl() * p1.orientation.getG())).copyTo(J(cv::Range(3,6),cv::Range(3,6)));
    ((cv::Mat) (p3.orientation.getH() * p1.orientation.getQr() * p2.orientation.getG())).copyTo(J(cv::Range(3,6),cv::Range(9,12)));

    p3.ID = ID;
    cv::Mat new_cov = J * augmented_cov * J.t();
    new_cov.copyTo(p3.Cov);
    return p3;
}

template<>
void invertPoseWithCovariance(CamPose_qd& p){

    assert(!p.Cov.empty() && "Pose cannot be inverted (empty Cov matrix)");

    cv::Mat J = cv::Mat::zeros(6,6,CV_64F);
    //J = [-R^T dR^Tt/dq; 0 -I]
    ((cv::Mat) -p.orientation.conj().getR3()).copyTo(J(cv::Range(0,3),cv::Range(0,3)));
    ((cv::Mat) p.orientation.conj().getH_qvec(p.position)).copyTo(J(cv::Range(0,3),cv::Range(3,6)));
    ((cv::Mat) -cv::Mat::eye(3,3,CV_32F)).copyTo(J(cv::Range(3,6),cv::Range(3,6)));

    p.position = -(p.orientation.conj() * p.position);
    p.orientation = p.orientation.conj();
    cv::Mat new_cov = J * p.Cov * J.t();
    new_cov.copyTo(p.Cov);
}

template<>
void ScalePoseWithCovariance(CamPose_qd& p, const std::pair<double,double>& scale){

    assert(!p.Cov.empty() && "Pose cannot be scaled (empty Cov matrix)");

    cv::Mat augmentedCov = cv::Mat::zeros(7,7,CV_64F);
    p.Cov.copyTo(augmentedCov(cv::Range(0,6),cv::Range(0,6)));
    augmentedCov.at<double>(6,6) = scale.second;
    cv::Mat J = cv::Mat::zeros(6,7,CV_64F);
    ((cv::Mat)(cv::Mat::eye(3,3,CV_64F) * scale.first)).copyTo(J(cv::Range(0,3),cv::Range(0,3)));
    ((cv::Mat)cv::Mat::eye(3,3,CV_64F)).copyTo(J(cv::Range(3,6),cv::Range(3,6)));
    ((cv::Mat) p.position).copyTo(J(cv::Range(0,3),cv::Range(6,7)));
    cv::Mat new_cov = J * augmentedCov * J.t();
    new_cov.copyTo(p.Cov);
    p.position *= scale.first;
}

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

}// namespace core
}// namespace me
