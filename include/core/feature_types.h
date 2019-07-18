#ifndef FEATURETYPE_H
#define FEATURETYPE_H

#include <iostream>
#include <iomanip>
#include <iterator>
#include <assert.h>
#include <deque>

#include "core/rotation_utils.h"

#define DESCRIPTOR_SIZE 32

namespace me{

//!< Representation of a 2D point as a Matrix.
/*! It is better to use this class rather than the opencv Point when dealing with matrices (multiplication, etc...). */
typedef cv::Matx21d pt2D;
typedef cv::Matx31d pt3D; //!< Representation of a 3D point as a Matrix.
typedef cv::Matx31d ptH2D; //!< Representation of a 2D homogeneous point as a Matrix.
typedef cv::Matx41d ptH3D; //!< Representation of a 3D homogeneous point as a Matrix.

/*! Normalize a 2D homogeneous point. */
inline void normalize(ptH2D& pt){
    pt(0)/=pt(2);
    pt(1)/=pt(2);
    pt(2)/=pt(2);
    assert(pt(2) == 1);
}

/*! Normalize a 3D homogeneous point. */
inline void normalize(ptH3D& pt){
    pt(0)/=pt(3);
    pt(1)/=pt(3);
    pt(2)/=pt(3);
    pt(3)/=pt(3);
    assert(pt(3) == 1);
}

/*! Normalize a 2D homogeneous point. */
inline ptH2D normalize(const ptH2D& pt_){
    ptH2D pt(pt_);
    pt(0)/=pt(2);
    pt(1)/=pt(2);
    pt(2)/=pt(2);
    assert(pt(2) == 1);
    return pt;
}

/*! Normalize a 3D homogeneous point. */
inline ptH3D normalize(const ptH3D& pt_){
    ptH3D pt(pt_);
    pt(0)/=pt(3);
    pt(1)/=pt(3);
    pt(2)/=pt(3);
    pt(3)/=pt(3);
    assert(pt(3) == 1);
    return pt;
}
/*! converts a 2D homogeneous point to non-homogeneous coordinate*/
inline pt2D to_euclidean(const ptH2D& pt_){
    ptH2D pt =pt_;
    normalize(pt);
    return pt2D(pt(0),pt(1));
}
/*! converts a 3D homogeneous point to non-homogeneous coordinate*/
inline pt3D to_euclidean(const ptH3D& pt_){
    ptH3D pt = pt_;
    normalize(pt);
    return pt3D(pt(0),pt(1),pt(2));
}
/*! converts a 2D point to homogeneous coordinate*/
inline ptH2D to_homogeneous(const pt2D& pt){
    return ptH2D(pt(0),pt(1),1);
}
/*! converts a 3D point to homogeneous coordinate*/
inline ptH3D to_homogeneous(const pt3D& pt){
    return ptH3D(pt(0),pt(1),pt(2),1);
}

/*! Function used to display elements of a descriptor. */
template <typename T>
void printVector(T * data, size_t s, size_t width = 0)
{
	for (size_t i = 0; i < s; i++)
	{
		std::cout << data[i] <<" ";
		if (width && (i+1)%width == 0) std::cout << std::endl;
	}
	std::cout << std::endl;
}

//! Class describing features.
/*! It contains the location and description of the feature.
    The type of the class corresponds to the type of data stored. */
template <typename T>
struct featureType {
	int32_t u;   //!< u coordinate.
	int32_t v;   //!< v coordinate.
	T * d;       //!< pointer to the data

	featureType() {d = new T();}
	featureType(int32_t u, int32_t v) :u(u), v(v) {}
};

//! Structure for storing SURF interest points
//! Size : 64
struct featureFREAK : featureType<unsigned char> {


	featureFREAK(const featureFREAK& feat) : featureType(feat.u,feat.v)
    {
        d = new unsigned char[64];
        for(int i=0;i<64;i++)
            std::memcpy( d, feat.d, 64 );
    }

	featureFREAK(int32_t u = 0, int32_t v = 0) : featureType(u, v)
	{
		d = new unsigned char[64];
	}

	featureFREAK& operator=(const featureFREAK& feat)
    {
        if (this != &feat) {
            memcpy( d, feat.d, 64 );
        }
        return *this;

    }

	~featureFREAK()
	{
		if(d) delete[] d;
	}
};

//! Structure for storing SURF interest points
//! Size : 128
struct featureSURF : featureType<float> {

	featureSURF(const featureSURF& feat) : featureType(feat.u,feat.v)
    {
        d = new float[128];
        for(int i=0;i<128;i++)
            std::memcpy( d, feat.d, 128 );
    }

	featureSURF(int32_t u = 0, int32_t v = 0) : featureType(u, v)
	{
		d = new float[128];
	}

	featureSURF& operator=(const featureSURF& feat)
    {
        if (this != &feat) {
            memcpy( d, feat.d, 128 );
        }
        return *this;

    }

	~featureSURF()
	{
		if (d) delete[] d;
	}
};

//! Structure for storing Phase Congruency interest points
//! Size : 104
struct featurePC : featureType<float> {

    featurePC(const featurePC& feat) : featureType(feat.u,feat.v)
    {
        d = new float[104];
        for(int i=0;i<104;i++)
            std::memcpy( d, feat.d, 104 );
    }

    featurePC(int32_t u=0, int32_t v=0) : featureType(u,v)
    {
        d = new float[104];
        for(int i=0;i<104;i++)
            d[i]=0;
    }

    featurePC& operator=(const featurePC& feat)
    {
        if (this != &feat) {
            memcpy( d, feat.d, 104 );
        }
        return *this;

    }

    ~featurePC()
    {

        if (d) delete d;
    }
};

//! Structure for storing Phase Congruency interest points (without amplitudes and orientations)
//! Size : 24
struct featurePCOnly : featureType<float> {

    featurePCOnly(const featurePCOnly& feat) : featureType(feat.u,feat.v)
    {
        d = new float[24];
        std::memcpy( d, feat.d, 24 );
    }

    featurePCOnly(int32_t u=0, int32_t v=0) : featureType(u,v)
    {
        d = new float[24];
        for(int i=0;i<24;i++)
            d[i]=0;
    }

    featurePCOnly& operator=(const featurePCOnly& feat)
    {
        if (this != &feat) {
            memcpy( d, feat.d, 24 );
        }
        return *this;

    }

    ~featurePCOnly()
    {
        if (d) delete[] d;
    }
};

//! Class storing a match between two images.
/*! Contains the feature in both image and a matching score (similarity). By default the score is -1. */
template <typename T>
struct StereoMatch{
	T f1;
	T f2;
	float m_score;

    StereoMatch(){};
	StereoMatch(const T& feat1, const T& feat2, float score=-1):f1(feat1),f2(feat2),m_score(score){}
	StereoMatch(const StereoMatch& sm):f1(sm.f1),f2(sm.f2),m_score(sm.m_score){}
};

//! Class storing a match between four images.
/*! Contains the feature in each image and a matching score (similarity). By default the score is -1. */
template <typename T>
struct StereoOdoMatches : public StereoMatch<T>{
    T f3;
    T f4;
    StereoOdoMatches(){};
	StereoOdoMatches(const T& feat1, const T& feat2, const T& feat3, const T& feat4, float score=-1):StereoMatch<T>(feat1,feat2,score),f3(feat3),f4(feat4){}
	StereoOdoMatches(const StereoOdoMatches& som):StereoMatch<T>(som.f1,som.f2,som.m_score),f3(som.f3),f4(som.f4){}
};

typedef StereoMatch<cv::Point2f> StereoMatchf;              //! float precision StereoMatch
typedef StereoMatch<cv::Point2d> StereoMatchd;              //! double precision StereoMatch
typedef StereoOdoMatches<cv::Point2f> StereoOdoMatchesf;    //! float precision StereoOdoMatch
typedef StereoOdoMatches<cv::Point2d> StereoOdoMatchesd;    //! double precision StereOdooMatch


//! Structure to store WBA matches
/*! Represent a Point with its 3D location and the list of features and the corresponding frame indices in which they appear */
template <typename T>
struct WBA_Point{

private:
std::deque<T> features;
std::deque<unsigned int> indices;
std::deque<cv::Matx22d> cov;
ptH3D pt;
int count=0;
int ID;
int cam_num;

static int latestID;

public:
WBA_Point(const T match, const int frame_nb, const int cam_number=0, const cv::Matx22d& cov_=cv::Matx22d::zeros(), const ptH3D pt_=ptH3D(0,0,0,1)): count(1), ID(latestID++), cam_num(cam_number){features.push_back(match);indices.push_back(frame_nb);cov.push_back(cov_);pt=pt_;}
WBA_Point(const WBA_Point& point): count(point.count), ID(point.ID), cam_num(point.cam_num){features=point.features;indices=point.indices;cov=point.cov;pt=point.pt;}
void addMatch(const T match, const int frame_nb, const cv::Matx22d& cov_=cv::Matx22d::zeros()){features.push_back(match);indices.push_back(frame_nb);cov.push_back(cov_);count++;assert(indices.size() == features.size()); if(indices.size() != getLastFrameIdx()-getFirstFrameIdx()+1) std::cout << indices.size() << " [] " << getLastFrameIdx() << " " << getFirstFrameIdx() << std::endl;assert(indices.size() == getLastFrameIdx()-getFirstFrameIdx()+1); assert(cov.size() == indices.size());}
void pop(){features.pop_front();indices.pop_front();cov.pop_front();assert(indices.size() == features.size() && indices.size() == cov.size() && (indices.size() == 0 || indices.size() == getLastFrameIdx()-getFirstFrameIdx()+1));}
bool isValid() const {return !features.empty();}
bool isTriangulated() const {return !(pt(0)==0 && pt(1)==0 && pt(2)==0 && pt(3)==1);}
T getLastFeat() const {if(isValid()) return features[features.size()-1]; else return T();}
T getFirstFeat() const {if(isValid()) return features[0]; else return T();}
T getFeat(unsigned int idx) const {assert(idx < features.size()); return features[idx];}
cv::Matx22d getCov(unsigned int idx) const {assert(idx < cov.size()); return cov[idx];}
bool findFeat(unsigned int idx, T& feat) const {
    for(uint k=0;k<indices.size();k++){
        if(indices[k] == idx){
            feat = features[k];
            return true;
        }
    }
    return false;
}
void removeLastFeat(){features.pop_back();indices.pop_back();cov.pop_back();assert(indices.size() == features.size() && indices.size() == cov.size() && (indices.size() == 0 || indices.size() == getLastFrameIdx()-getFirstFrameIdx()+1));}
unsigned int getLastFrameIdx() const {if(isValid())return indices[indices.size()-1];else return -1;}
unsigned int getFirstFrameIdx() const {if(isValid())return indices[0];else return -1;}
unsigned int getFrameIdx(unsigned int idx) const {assert(idx < indices.size() && idx>=0);return indices[idx];}
unsigned int getNbFeatures() const {return features.size();}
int getCount() const {return count;}
int getID() const {return ID;}
int getCameraNum() const {return cam_num;};
ptH3D get3DLocation() const {return pt;}
void set3DLocation(const ptH3D& pt_){pt = pt_;}
void setCameraNum(int i){cam_num = i;};

friend void swap(WBA_Point& pt1, WBA_Point& pt2){
    std::swap(pt1.features,pt2.features);
    std::swap(pt1.indices,pt2.indices);
    std::swap(pt1.cov,pt2.cov);
    std::swap(pt1.pt,pt2.pt);
    int tmp=pt1.ID;pt1.ID=pt2.ID;pt2.ID=tmp;
}

WBA_Point& operator=(const WBA_Point& point){ WBA_Point tmp(point);swap(*this,tmp); return *this;}


friend std::ostream& operator<<(std::ostream& os, const WBA_Point& pt){
    os << "Point " << pt.getID() << ": " << pt.getNbFeatures() << " feats (from " << pt.getFirstFrameIdx() << " to " << pt.getLastFrameIdx() << ")";
    return os;
}

};

typedef WBA_Point<cv::Point2f> WBA_Ptf;
typedef WBA_Point<std::pair<cv::Point2f,cv::Point2f>> WBA_stereo_Ptf;

template<typename T>
int WBA_Point<T>::latestID=0;

template<class O, class T>
class CamPose{
    public:
    //params
    O orientation;
    cv::Vec<T,3> position;
    cv::Mat Cov;
    int ID;
    //constructors
    CamPose<O,T>(int id=0, const O& e=O(), const cv::Vec<T,3>& v=cv::Vec<T,3>(), const cv::Mat& c=cv::Mat()):orientation(e),position(v),Cov(c),ID(id){}
    CamPose<O,T>(const CamPose<O,T>& cp):orientation(cp.orientation),position(cp.position),Cov(6,6,CV_64F),ID(cp.ID){cp.Cov.copyTo(Cov);}

    //member functions
    CamPose<O,T>& operator=(const CamPose<O,T>& cp){
        orientation = cp.orientation;
        position = cp.position;
        ID=cp.ID;
        Cov = cv::Mat::zeros(6,6,CV_64F);
        cp.Cov.copyTo(Cov);
        return *this;
    }
    CamPose<O,T> operator*(const CamPose<O,T>& pose) const{
        CamPose<O,T> new_pose(*this);
        new_pose.orientation = orientation * pose.orientation;
        new_pose.position = orientation * pose.position + position;
        return new_pose;
    }
    cv::Mat JacobianMult(const CamPose<O,T>& pose) const;
    cv::Mat JacobianMultReverse(const CamPose<O,T>& pose) const;
    cv::Mat JacobianInv() const;
    cv::Mat JacobianScale(T scale) const;

    cv::Matx<T,4,4> TrMat() const;
    CamPose<O,T> inv() const;
    void copyPoseOnly(const CamPose<O,T>& pose){orientation=pose.orientation;position=pose.position;}

    friend std::ostream& operator<<(std::ostream& os, const CamPose& pose){
    os << "ID: " << pose.ID << std::endl << "orientation: " << pose.orientation << std::endl << "position: " << pose.position << std::endl;
    return os;
}
};

template<class O, class T>
std::ostream& operator<<(std::ostream& out, const std::vector<CamPose<O,T>>& v){
    out << "Poses [" << std::endl;
    if(!v.empty())
        std::copy(v.begin(),v.end(), std::ostream_iterator<CamPose<O,T>>(out, ""));
    out << "\b\b]";
    return out;
}

template<class O, class T>
CamPose<O,T> poseMultiplicationWithCovariance(const CamPose<O,T>& p1, const CamPose<O,T>& p2, int ID);
template<class O, class T>
CamPose<O,T> poseMultiplicationWithCovarianceReverse(const CamPose<O,T>& p1, const CamPose<O,T>& p2, int ID);
template<class O, class T>
void invertPoseWithCovariance(CamPose<O,T>& p);
template<class O, class T>
void ScalePoseWithCovariance(CamPose<O,T>& p, const std::pair<double,double>& scale);

typedef CamPose<Euld,double> CamPose_ed;
typedef CamPose<Eulf,float> CamPose_ef;
typedef CamPose<Quatd,double> CamPose_qd;
typedef CamPose<Quatf,float> CamPose_qf;
typedef CamPose<cv::Matx33d,double> CamPose_md;
typedef CamPose<cv::Matx33f,float> CamPose_mf;

std::vector<pt2D> nonMaxSupScanline3x3(const cv::Mat& input, cv::Mat& output);

}

#endif