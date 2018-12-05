#ifndef OPTIMISATION_H_INCLUDED
#define OPTIMISATION_H_INCLUDED

#include <Eigen/Core>

#include "featureType.h"
#include "utils.h"

namespace me{
enum OptimType{GN,LM};

struct OptimisationParams{
    int MAX_NB_ITER;
    double v,tau,mu;
    double abs_tol,grad_tol,incr_tol,rel_tol;
    double alpha;
    OptimType type;
    bool minim;

//    OptimisationParams(it=500,v_=2,t=1e-3,m=1e-20,e1=1e-6,e2=1e-6,e3=1-6,e4=1e-6,a=1.0,type_=LM): MAX_NB_ITER(it),v(v_),tau(t),mu(m),abs_tol(e1),grad_tol(e2),incr_tol(e3),rel_tol(e4),alpha(a),type(type_){}
    OptimisationParams(OptimType type_=OptimType::LM,bool min=true, int it=20,double v_=2,double t=1e-3,double m=1e-20,double e1=1e-4,double e2=1e-4,double e3=1e-3, double e4=1e-6,double a=2.0): MAX_NB_ITER(it),v(v_),tau(t),mu(m),abs_tol(e1),grad_tol(e2),incr_tol(e3),rel_tol(e4),alpha(a),type(type_),minim(min){}
};

struct OptimState{

    int nb_params;

    OptimState(int nb_params_):nb_params(nb_params_){}

//    virtual Eigen::MatrixXd compute_residuals() =0;
    virtual void update(const Eigen::MatrixXd& dX) =0;
    virtual std::string show_params() const =0;
};

struct StructureState: OptimState{
    Eigen::MatrixXd state_poses;
    Eigen::MatrixXd state_pts;

    StructureState(): OptimState(5){}
};

struct PosesState{
    Eigen::MatrixXd poses;
    int param_poses;
};

struct StereoState: OptimState{

    std::pair<cv::Matx33d,cv::Matx33d> K;
    CamPose_qd pose;
    std::vector<ptH3D> pts;
    double baseline;

    StereoState(): OptimState(6){}

    void update(const Eigen::MatrixXd& dX) override{
        assert(dX.rows() == 6 && dX.cols() == 1);
        pose.orientation *= exp_map_Quat(cv::Vec3d(dX(3),dX(4),dX(5)));
        pose.position += exp_map_Mat(cv::Vec3d(dX(3),dX(4),dX(5))) * cv::Vec3d(dX(0),dX(1),dX(2));
    }

    std::string show_params() const override{
        return "["+std::to_string(pose.orientation.w())+"|"+std::to_string(pose.orientation.x())+","+std::to_string(pose.orientation.y())+","+std::to_string(pose.orientation.z())+"] ["+std::to_string(pose.position(0))+","+std::to_string(pose.position(1))+","+std::to_string(pose.position(2))+"]";
    }
};

struct ScaleState: OptimState{
    std::pair<std::vector<WBA_Ptf>,std::vector<WBA_Ptf>> pts;
    std::pair<cv::Matx33d,cv::Matx33d> K;
    std::pair<std::vector<CamPose_qd>,std::vector<CamPose_qd>> poses;
    double scale;
    double baseline;
    int window_size;

    ScaleState(): OptimState(1){}

    void update(const Eigen::MatrixXd& dX) override{
        assert(dX.rows() == 1 && dX.cols() == 1);
        scale += dX(0);
    }

    std::string show_params() const override{
     return std::to_string(scale);
    }

//    Eigen::MatrixXd compute_residuals() override {return Eigen::MatrixXd();}
};

template<class S, class T>
class Optimiser{

public:
    Optimiser(const T& observations, const OptimisationParams& params=OptimisationParams()): m_obs(observations), m_params(params), m_stop(StopCondition::NO_STOP){}

    StopCondition optimise(S& state, const bool test=false, const Eigen::VectorXi& mask=Eigen::VectorXi());
    std::vector<int> compute_inliers(const double threshold);

private:

Eigen::MatrixXd compute_residuals(const S& state);
void compute_normal_equations(const Eigen::MatrixXd& residuals, Eigen::MatrixXd& JJ, Eigen::VectorXd& e);

void run_GN_step(Eigen::MatrixXd& JJ, Eigen::VectorXd& e, Eigen::VectorXd& dX);
void run_LM_step(Eigen::MatrixXd& JJ, Eigen::VectorXd& e, Eigen::VectorXd& dX, const double e1);

S m_state;
T m_obs;
Eigen::VectorXi m_mask;
OptimisationParams m_params;
StopCondition m_stop;

};

//template<, class T>
//class Optimiser{
//
//public:
//    Optimiser(S& state, const T& observations, const OptimisationParams& params): m_state(state), m_obs(observations),m_params(params){}
//
//    S optimise();
//
//private:
//
//Eigen::MatrixXd compute_residuals();
//void compute_jacobian(Eigen::MatrixXd& JJ, Eigen::MatrixXd& e);
//
//void update_state(const Eigen::MatrixXd& dX);
//
//void run_GN_step(Eigen::MatrixXd& JJ, Eigen::MatrixXd& e, Eigen::MatrixXd dX);
//void run_LM_step(Eigen::MatrixXd& JJ, Eigen::MatrixXd& e, Eigen::MatrixXd dX);
//
//S m_state;
//T m_obs;
//OptimisationParams m_params;
//
//};

}

#endif
