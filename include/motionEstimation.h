#ifndef MOTIONESTIMATION_H_INCLUDED
#define MOTIONESTIMATION_H_INCLUDED

#include "featureType.h"
#include "matcher.h"
#include "viso_stereo.h"
#include "matrix.h"

using namespace cv;

Matrix estimateNewPose(VisualOdometryStereo& viso, const vector<StereoOdoMatches<featurePC>>& matches, const vector<Matcher::p_match>& p_matched, vector<StereoMatch<featurePC>>& inliers,const VisualOdometryStereo::parameters& param);
Matrix estimateNewPose(VisualOdometryStereo& viso, const vector<Matcher::p_match>& p_matched, const VisualOdometryStereo::parameters& param);
Matrix estimateNewPose(VisualOdometryStereo& viso, const vector<StereoOdoMatches<Point2f>>& matches,const VisualOdometryStereo::parameters& param);
void updatePose(Matrix& pose, const Matrix& new_pose);


#endif // MOTIONESTIMATION_H_INCLUDED
