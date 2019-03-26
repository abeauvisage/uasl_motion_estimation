#ifndef MUTUALINFORMATION_H_INCLUDED
#define MUTUALINFORMATION_H_INCLUDED

#include "featureType.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


float computeEntropy(const cv::Mat& img);

void quantise(cv::Mat& img, const std::pair<uchar,uchar>& range);

float computeMutualInformation(const cv::Mat& imgL, const cv::Mat& imgR);
void jointDistribution(const cv::Mat& imgL, const cv::Mat& imgR);

float comparePC(const cv::Mat& PC1, const cv::Mat& PC2);
float applyCCOEFFNormed(const cv::Mat& r1, const cv::Mat& r2);
#endif // MUTUALINFORMATION_H_INCLUDED
