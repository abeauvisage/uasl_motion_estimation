#include "mutualInformation.h"

using namespace std;
using namespace cv;
using namespace me;

float comparePC(const Mat& PC1, const Mat& PC2){

    float sum=0,sum1=0,sum2=0;
    for(int i=0;i<PC1.rows;i++)
        for(int j=0;j<PC1.cols;j++){
            sum += PC1.at<float>(i,j) * PC2.at<float>(i,j);
            sum1 += pow(PC1.at<float>(i,j),2);
            sum2 += pow(PC2.at<float>(i,j),2);
        }

    return sum/sqrt(sum1*sum2);
}


float computeEntropy(const Mat& img){

    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    int histSize[] = {20};

    Mat hist;
    calcHist( &img, 1, 0, Mat(), hist, 1, histSize, &histRange);

    hist /= img.rows*img.cols;

    float entropy=0;
    for(int i=0;i<hist.rows;i++)
        if(hist.at<float>(i) > 0)
            entropy += hist.at<float>(i)*log2(hist.at<float>(i));

    return -entropy;
}


void quantise(cv::Mat& img, const std::pair<uchar,uchar>& range){
    int nb_samples = img.rows * img.cols;
    uchar* img_ptr = img.ptr<uchar>();
    for(int i=0;i<nb_samples;i++)
        img_ptr[i] = (uchar) (img_ptr[i] / (256/(int)(range.second-range.first)))+range.first;
}

float computeMutualInformation(const Mat& imgL, const Mat& imgR){

    assert(!imgL.empty() && !imgR.empty());

    Mat imgL_,imgR_;
    imgL.copyTo(imgL_);imgR.copyTo(imgR_);
    imgL_.convertTo(imgL_,CV_8U);imgR_.convertTo(imgR_,CV_8U);
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    const float* histRangeJoint[] = { range, range };
    int histSize = 20;
    int histSizeJoint[] = {histSize,histSize};
    Mat imgs[] = {imgL_,imgR_};

    Mat histL, histR, histJoint;
    calcHist( &imgL_, 1, 0, Mat(), histL, 1, &histSize, &histRange);
    calcHist( &imgR_, 1, 0, Mat(), histR, 1, &histSize, &histRange);
    calcHist( imgs, 2, 0, Mat(), histJoint, 2, histSizeJoint, histRangeJoint);

    histL /= imgL.rows*imgL.cols;
    histR /= imgR.rows*imgR.cols;
    histJoint /= imgL.rows*imgL.cols;

    float MI=0;

    for(int i=0;i<histJoint.rows;i++)
        for(int j=0;j<histJoint.cols;j++)
            if(histJoint.at<float>(i,j) > 0 && histL.at<float>(i) > 0 && histR.at<float>(j) > 0)
                MI += histJoint.at<float>(i,j)*log2(histJoint.at<float>(i,j)/(histL.at<float>(i)*histR.at<float>(j)));

    return MI;
}

void jointDistribution(const Mat& imgL, const Mat& imgR){

    float range[] = { 0, 256 } ;
    float rangeJoint[] = { 0, 256, 0, 256 } ;
    const float* histRange = { range };
    const float* histRangeJoint = { rangeJoint };
    int histSize[] = {11};
    int histSizeJoint[] = {11,11};
    Mat imgs[] = {imgL,imgR};

    Mat histL, histR, histJoint;
    calcHist( &imgL, 1, 0, Mat(), histL, 1, histSize, &histRange);
    calcHist( &imgR, 1, 0, Mat(), histR, 1, histSize, &histRange);
    calcHist( imgs, 2, 0, Mat(), histJoint, 2, histSizeJoint, &histRangeJoint);

    histL /= imgL.rows*imgL.cols;
    histR /= imgR.rows*imgR.cols;
    histJoint /= imgL.rows*imgL.cols;

    float max=0;
    int max_x=0,max_y=0;
    for(int i=0;i<histJoint.rows;i++)
        for(int j=0;j<histJoint.cols;j++)
            if(histJoint.at<float>(i,j) > max){
                max= histJoint.at<float>(i,j);
                max_x=i;max_y=j;
            }

    Mat ROILc(imgL.size(),CV_8UC3);
    Mat ROIRc(imgR.size(),CV_8UC3);
    imgL.convertTo(ROILc,CV_8UC3);
    imgR.convertTo(ROILc,CV_8UC3);
    cvtColor(imgL,ROILc,CV_GRAY2BGR);
    cvtColor(imgR,ROIRc,CV_GRAY2BGR);

    ROILc.at<Vec3b>(0,0).val[0] = 255;

    for(int i=0;i<imgL.rows;i++)
        for(int j=0;j<imgL.cols;j++)
            if(imgL.at<uchar>(i,j) == max_x*25)
                ROILc.at<Vec3b>(i,j) = Vec3b(255,0,0);

    for(int i=0;i<imgR.rows;i++)
        for(int j=0;j<imgR.cols;j++)
            if(imgR.at<uchar>(i,j) == max_y*25)
                ROIRc.at<Vec3b>(i,j) = Vec3b(255,0,0);
}

float applyCCOEFFNormed(const Mat& r1, const Mat& r2){
    Mat r1_ = (r1-1)/(r1.rows*r1.cols)*sum(r1)[0];
    Mat r2_ = (r2-1)/(r2.rows*r2.cols)*sum(r2)[0];
    return sum(r1_.mul(r2_))[0]/sqrt(sum(r1_.mul(r1_))[0]*sum(r2_.mul(r2_))[0]);
}
