#include "featureType.h"

namespace me{

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


}
