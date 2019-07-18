#ifndef MUTUALINFORMATION_H_INCLUDED
#define MUTUALINFORMATION_H_INCLUDED

/** \file mutual information.h
*   \brief utilities to compute mutual information from images
*
*    \author Axel Beauvisage (axel.beauvisage@gmail.com)
*/

#include "core/feature_types.h"

#include <opencv2/imgproc/imgproc.hpp>

namespace me{

namespace core{

float computeEntropy(const cv::Mat& img);

void quantise(cv::Mat& img, const std::pair<uchar,uchar>& range);

float computeMutualInformation(const cv::Mat& imgL, const cv::Mat& imgR);
void jointDistribution(const cv::Mat& imgL, const cv::Mat& imgR);

float comparePC(const cv::Mat& PC1, const cv::Mat& PC2);
float applyCCOEFFNormed(const cv::Mat& r1, const cv::Mat& r2);

}// namespace core
}// namespace me

#endif // MUTUALINFORMATION_H_INCLUDED
