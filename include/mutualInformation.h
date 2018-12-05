#ifndef MUTUALINFORMATION_H_INCLUDED
#define MUTUALINFORMATION_H_INCLUDED

#include "featureType.h"
//#include "matching.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


double parzenWin(const double* data, const int nb_samples, const double mean, const double h, const double std);
double parzenWin(std::pair<const double*,const double*> data, const int nb_samples, const cv::Matx21d& means, const cv::Matx22d& cov);
double computeEntropy(const cv::Mat& img,const std::pair<int,int>& range);
std::vector<double> computeJointEntropy(const std::pair<cv::Mat,cv::Mat>& img,const std::pair<int,int>& range);

void quantise(cv::Mat& img, const std::pair<uchar,uchar>& range);

float computeEntropy(const cv::Mat& img);

float computeMutualInformation(const cv::Mat& imgL, const cv::Mat& imgR);
void jointDistribution(const cv::Mat& imgL, const cv::Mat& imgR);


std::vector<me::StereoMatch<me::featurePC>> computeMIStereoMatching(std::vector<me::featurePC>& f1, std::vector<me::featurePC>& f2, const cv::Mat& imgL, const cv::Mat& imgR, const cv::Mat& PC1, const cv::Mat& PC2);
//void computeMIStereoMatching(std::vector<cv::Point2f>& f1, std::vector<cv::Point2f>& f2, const cv::Mat& imgL, const cv::Mat& imgR, const cv::Mat& PC1, const cv::Mat& PC2);
std::vector<me::StereoMatch<me::featurePC>> computeMIStereoMatchingFeatures(std::vector<me::featurePC>& f1, std::vector<me::featurePC>& f2, const cv::Mat& imgL, const cv::Mat& imgR, const cv::Mat& PC1, const cv::Mat& PC2);
std::vector<me::StereoMatch<me::featurePC>> computeMIStereoMatchingMean(std::vector<me::featurePC>& f1, std::vector<me::featurePC>& f2, const cv::Mat& imgL, const cv::Mat& imgR, const cv::Mat& PC1, const cv::Mat& PC2, int max_disp);
std::vector<me::StereoMatch<me::featurePCOnly>> computeMIStereoMatchingDichoto(std::vector<me::featurePCOnly>& f1, std::vector<me::featurePCOnly>& f2, const cv::Mat& imgL, const cv::Mat& imgR, const cv::Mat& PC1, const cv::Mat& PC2, int max_disp);

float comparePC(const cv::Mat& PC1, const cv::Mat& PC2);
float applyCCOEFFNormed(const cv::Mat& r1, const cv::Mat& r2);
#endif // MUTUALINFORMATION_H_INCLUDED
