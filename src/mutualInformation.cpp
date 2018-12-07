#include "mutualInformation.h"

#include <chrono>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;
using namespace me;


double parzenWin(const double* data, const int nb_samples, const double mean, const double h, const double std){

    double res=0;
    double coeff = 1.0/(sqrt(2*M_PI)*h*std*nb_samples);
    for(int i=0;i<nb_samples;i++)
        res += exp(-0.5*pow(data[i]-mean,2)/(h*h*std*std));
    return coeff * res;
}

double parzenWin(std::pair<const double*,const double*> data, const int nb_samples, const cv::Matx21d& means, const Matx22d& cov){

    double res=0;
    double coeff = 1.0/(sqrt(2*M_PI*cov(0,0)*cov(1,1))*nb_samples);
    Matx22d cov_inv = cov.inv();
    for(int i=0;i<nb_samples;i++){
        Matx21d xy(data.first[i],data.second[i]);
        res += exp(-0.5 * (xy.t() * cov_inv * xy)(0));
    }
    return coeff * res;
}

constexpr double root = 1.0/sqrt(2*M_PI);

auto GaussKernel = [](const double x, const double mean, const double h, double sum){

    sum +=  exp(-0.5 * (pow(x,2)/h));
};

auto parzen = [&root](const Eigen::MatrixXd& img, const double mean, const double h) -> double{

//        double inv_h = 1.0/h;
//        double det_h = h;//.determinant();
        Eigen::MatrixXd sum_mat(img.rows(),img.cols());
        sum_mat.setZero();
        double sum=0.0;
//        Mat sum_mat = (img.t() * inv_h * img).diag();//Mat::zeros(1,img.cols,CV_64F);
        auto tp1 = chrono::steady_clock::now();
        for(int i=0;i<img.rows();i++){
                GaussKernel(img(i,0),mean,h,sum);
//                sum_mat(i,0) = (img(i,0)-mean) * inv_h * (img(i,0)-mean);
//                cout << img.col(i)-mean << endl;
//            cout << (img.col(i)-mean).t() * inv_h * (img.col(i)-mean) << endl;

//            sum_mat.col(i) = img.col(i).t() * inv_h * img.col(i);
//            GaussKernel(img.col,mean,inv_h,det_h,sum_mat);
////        }
        }
        auto tp2 = chrono::steady_clock::now();
//        exp(-0.5 * sum_mat,sum_mat);
//        sum_mat = (-0.5*sum_mat).array().exp();
//        sum_mat *= root / sqrt(det_h);
        sum *= root / sqrt(h);
        auto tp3 = chrono::steady_clock::now();

        cout << "for loop " << chrono::duration_cast<chrono::milliseconds>(tp2-tp1).count() << endl;
        cout << "exp " << chrono::duration_cast<chrono::milliseconds>(tp3-tp2).count() << endl;
        return (sum/(double)img.cols());
};

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

    auto t1 = chrono::steady_clock::now();

    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    int histSize[] = {20};

    Mat hist;
    calcHist( &img, 1, 0, Mat(), hist, 1, histSize, &histRange);

    hist /= img.rows*img.cols;
    cout << "hist: " << hist.t() << endl;

    float entropy=0;
    for(int i=0;i<hist.rows;i++)
        if(hist.at<float>(i) > 0)
            entropy += hist.at<float>(i)*log2(hist.at<float>(i));

    auto t2 = chrono::steady_clock::now();
    cout << img.size() << endl;
    cout << "entropy: "  << chrono::duration_cast<chrono::microseconds>(t2-t1).count() << " ms" << endl;

    return -entropy;
}

double computeEntropy(const cv::Mat& img,const std::pair<int,int>& range){

    assert(img.type() == CV_8U);

    int nb_samples = img.rows * img.cols;
    int nb_pix_int = range.second - range.first;

    cv::Mat img_q;img.copyTo(img_q);
    int alpha = 256/(int)(range.second-range.first);

    uchar* img_ptr = img_q.ptr<uchar>();
    for(int j=0;j<nb_samples;j++){
        img_ptr[j] = (img_ptr[j]/alpha)+range.first;
    }

    img_q.convertTo(img_q,CV_64F);
    img_q = img_q.reshape(1,1);
//    cv::Scalar test(15);
    const double* img_ptrd = img_q.ptr<double>();

    cv::Scalar mean_mat,stddev_mat;
    cv::meanStdDev(img_q,mean_mat,stddev_mat);

    cout << "std " << stddev_mat << endl;

    double h = 1.06*stddev_mat[0]*pow(nb_samples,-0.2);
    double entropy=0;double sum=0;
//    Mat h_ = Mat_<double>(1,1) << 25;
    Eigen::Matrix<double,1,1> h_; h_ << 25;

    Eigen::MatrixXd img_eigen;
    cv2eigen(img_q.t(),img_eigen);

    for(int i= (int)range.first;i<=(int)range.second;i++){
        Eigen::Matrix<double,1,1> i_mat;i_mat << i;
//        entropy +=  log(parzen(img_q,Mat_<double>(1,1)<<(double)i,h_).at<double>(0,0));
        entropy +=  log(parzen(img_eigen,i,25));
//        entropy += log(parzenWin(img_ptrd,nb_samples,(double)i,h,stddev_mat[0]));
//        cout << parzenWin(img_ptrd,nb_samples,(double)i,h,stddev_mat[0]) << ",";
        sum += log(parzenWin(img_ptrd,nb_samples,(double)i,h,stddev_mat[0]));
    }
    cout << endl;
    cout << -1.0/nb_pix_int * sum << endl;
    return -1.0/nb_pix_int * entropy;
}

std::vector<double> computeJointEntropy(const std::pair<cv::Mat,cv::Mat>& imgs,const std::pair<int,int>& range){

    assert(imgs.first.size == imgs.second.size && imgs.first.type() == CV_8U && imgs.second.type() == CV_8U);

    int nb_samples = imgs.first.rows * imgs.first.cols;
    int nb_pix_int = range.second - range.first;

    std::pair<cv::Mat,cv::Mat> img_q;
    imgs.first.copyTo(img_q.first);
    imgs.second.copyTo(img_q.second);
    int alpha = 256/(int)(range.second-range.first);

    std::pair<uchar*,uchar*> ptrs = make_pair(img_q.first.ptr<uchar>(),img_q.second.ptr<uchar>());

    for(int j=0;j<nb_samples;j++){
        ptrs.first[j] = (ptrs.first[j]/alpha)+range.first;
        ptrs.second[j] = (ptrs.second[j]/alpha)+range.first;
    }

    img_q.first.convertTo(img_q.first,CV_64F);
    img_q.second.convertTo(img_q.second,CV_64F);
    std::pair<const double*, const double*> ptrd = make_pair(img_q.first.ptr<double>(),img_q.second.ptr<double>());

    std::pair<cv::Scalar,cv::Scalar> means,stddevs;
    cv::meanStdDev(img_q.first,means.first,stddevs.first);
    cv::meanStdDev(img_q.second,means.second,stddevs.second);

    Matx22d cov = Matx22d::zeros();
    cov(0,0) = (stddevs.first*stddevs.first)[0];
    cov(1,1) = (stddevs.second*stddevs.second)[0];

    std::pair<double,double> h = make_pair(1.06*stddevs.first[0]*pow(nb_samples,-0.2),1.06*stddevs.second[0]*pow(nb_samples,-0.2));
    vector<double> entropy(3,0);

    for(int i= (int)range.first;i<=(int)range.second;i++){
        entropy[0] += log(parzenWin(ptrd.first,nb_samples,(double)i,h.first,stddevs.first[0]));
        entropy[1] += log(parzenWin(ptrd.second,nb_samples,(double)i,h.second,stddevs.second[0]));
        for(int j= (int)range.first;j<=(int)range.second;j++){
            entropy[2] += log(parzenWin(ptrd,nb_samples,Matx21d(i,j),cov));
        }
    }

    for(uint i=0;i<3;i++)
        entropy[i] *= -1.0/nb_pix_int;

    return entropy;
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
    int histSize = 10;
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
                MI += histJoint.at<float>(i,j)*log(histJoint.at<float>(i,j)/(histL.at<float>(i)*histR.at<float>(j)));

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
//    Mat ROILc,ROIRc;
    imgL.convertTo(ROILc,CV_8UC3);
    imgR.convertTo(ROILc,CV_8UC3);
    cvtColor(imgL,ROILc,CV_GRAY2BGR);
    cvtColor(imgR,ROIRc,CV_GRAY2BGR);

    ROILc.at<Vec3b>(0,0).val[0] = 255;

    cout << "max " << max_x << endl;

    for(int i=0;i<imgL.rows;i++)
        for(int j=0;j<imgL.cols;j++)
            if(imgL.at<uchar>(i,j) == max_x*25)
                ROILc.at<Vec3b>(i,j) = Vec3b(255,0,0);

    for(int i=0;i<imgR.rows;i++)
        for(int j=0;j<imgR.cols;j++)
            if(imgR.at<uchar>(i,j) == max_y*25)
                ROIRc.at<Vec3b>(i,j) = Vec3b(255,0,0);

    imshow("L",ROILc);
    imshow("R",ROIRc);
    waitKey(0);
}

vector<StereoMatch<featurePC>> computeMIStereoMatching(vector<featurePC>& f1, vector<featurePC>& f2, const Mat& imgL, const Mat& imgR, const Mat& PC1, const Mat& PC2){

    vector<StereoMatch<featurePC>> matches;
    int windowSize=80;
    int max_disp =80;

    for(uint i=0;i<f1.size();i++){
//        cout << i << "/" << f1.size() << endl;
        float max_mi=0.0;
        int max_idx=0;
        vector<float> miv;
        float mean=0;
        cv::Mat ROIL =  PC1(cv::Rect(f1[i].v-windowSize/2,f1[i].u-windowSize/2,windowSize,windowSize)).clone();
        ROIL.convertTo(ROIL,CV_8U);
        cv::Mat ROIL2 =  imgL(cv::Rect(f1[i].v-windowSize/2,f1[i].u-windowSize/2,windowSize,windowSize)).clone();

        for(int j=0; j< max_disp; j++){
            if(f1[i].v-j-windowSize/2 > 0 && PC2.at<float>(f1[i].u,f1[i].v-j-windowSize/2) > 10){
                cv::Mat ROIR =  PC2(cv::Rect(f1[i].v-j-windowSize/2,f1[i].u-windowSize/2,windowSize,windowSize)).clone();
                featurePC currentDescriptor(f1[i].u,f1[i].v-j);
                ROIR.convertTo(ROIR,CV_8U);
                cv::Mat ROIR2 =  imgR(cv::Rect(f1[i].v-j-windowSize/2,f1[i].u-windowSize/2,windowSize,windowSize)).clone();

                float mi = computeMutualInformation(ROIL,ROIR) * computeMutualInformation(ROIL2,ROIR2);
                mean += mi;
                if(mi > max_mi){max_mi=mi;max_idx=j;}
            }
        }

        mean /= max_disp;

        featurePC desc(f1[i].u,f1[i].v-max_idx);
        f2.push_back(desc);
        StereoMatch<featurePC> f(f1[i],f2[i],max_mi);
        if(max_idx>0 /*&& max_mi > 2*mean*/)
            matches.push_back(f);
    }

    return matches;
}

std::vector<StereoMatch<featurePC>> computeMIStereoMatchingMean(std::vector<featurePC>& f1, std::vector<featurePC>& f2, const cv::Mat& imgL, const cv::Mat& imgR, const cv::Mat& PC1, const cv::Mat& PC2, int max_disp){

    vector<StereoMatch<featurePC>> matches;
    int windowSize = 80;
    for(uint i=0;i<f1.size();i++){
        float max_mi=0, max_pc=0;
        float min_mi=1000, min_pc=1000;
        int max_idx=0;
        vector<float> miv;
        vector<float> pcv;
        float mean_mi=0,mean_pc=0;
        cv::Mat ROIL =  PC1(cv::Rect(f1[i].v-windowSize/2,f1[i].u-windowSize/2,windowSize,windowSize));
        ROIL.convertTo(ROIL,CV_8U);
        cv::Mat ROIL2 =  imgL(cv::Rect(f1[i].v-windowSize/2,f1[i].u-windowSize/2,windowSize,windowSize));

        for(int j=0; j< max_disp; j++)
            if(f1[i].v-j-windowSize/2 > 0){
                cv::Mat ROIR =  PC2(cv::Rect(f1[i].v-j-windowSize/2,f1[i].u-windowSize/2,windowSize,windowSize));
                ROIR.convertTo(ROIR,CV_8U);
                cv::Mat ROIR2 =  imgR(cv::Rect(f1[i].v-j-windowSize/2,f1[i].u-windowSize/2,windowSize,windowSize));

                float mi = computeMutualInformation(ROIL,ROIR);
                float pc = computeMutualInformation(ROIL2,ROIR2);
                miv.push_back(mi);
                pcv.push_back(pc);
                mean_mi += mi;
                mean_pc += pc;

                if(mi > max_mi){max_mi=mi;max_idx=j;}
                if(pc > max_pc)max_pc=pc;
                if(mi < min_mi)min_mi=mi;
                if(pc < min_pc)min_pc=pc;

            }
        mean_mi /= miv.size();
        mean_pc /= pcv.size();

        float max_tot=0;
        for(uint k=0;k<miv.size();k++)
            if(miv[k]/mean_mi+pcv[k]/mean_pc >max_tot){max_tot=miv[k]/mean_mi+pcv[k]/mean_pc;max_idx=k;}


        featurePC desc(f1[i].u,f1[i].v-max_idx);
        f2.push_back(desc);
        StereoMatch<featurePC> f(f1[i],f2[i],max_mi);
        if(max_tot > 0.4 * (-computeEntropy(ROIL2)) && max_idx < max_disp -1 && max_idx>0 /*&& max_mi > 2*mean*/)
            matches.push_back(f);
    }

    return matches;
}

std::vector<StereoMatch<featurePCOnly>> computeMIStereoMatchingDichoto(std::vector<featurePCOnly>& f1, std::vector<featurePCOnly>& f2, const cv::Mat& imgL, const cv::Mat& imgR, const cv::Mat& PC1, const cv::Mat& PC2, int max_disp){

    vector<StereoMatch<featurePCOnly>> matches;
    int windowSize = 80;
    for(uint i=0;i<f1.size();i++){
        float max_mi=0, max_pc=0;
        float min_mi=1000, min_pc=1000;
        int max_idx=0;
        vector<float> miv;
        vector<float> pcv;
        float mean_mi=0,mean_pc=0;
        cv::Mat ROIL =  PC1(cv::Rect(f1[i].v-windowSize/2,f1[i].u-windowSize/2,windowSize,windowSize));
        ROIL.convertTo(ROIL,CV_8U);
        cv::Mat ROIL2 =  imgL(cv::Rect(f1[i].v-windowSize/2,f1[i].u-windowSize/2,windowSize,windowSize));

        int l_left = f1[i].v-max_disp > 0 ? f1[i].v-max_disp : 0;
        int l_right = f1[i].v+windowSize/2;

        float prev = 0;

        while(l_right - l_left > 20){
            cv::Mat ref =  imgL(cv::Rect(f1[i].v-(l_right-l_left)/2,f1[i].u-windowSize/2,3*(l_right-l_left)/4,windowSize));
            cv::Mat ref2 =  PC1(cv::Rect(f1[i].v-(l_right-l_left)/2,f1[i].u-windowSize/2,3*(l_right-l_left)/4,windowSize));
            cv::Mat r1 =  imgR(cv::Rect(l_left,f1[i].u-windowSize/2,3*(l_right-l_left)/4,windowSize));
            cv::Mat r2 = imgR(cv::Rect(l_left+(l_right-l_left)/4,f1[i].u-windowSize/2,3*(l_right-l_left)/4,windowSize));
            cv::Mat r3 =  PC2(cv::Rect(l_left,f1[i].u-windowSize/2,3*(l_right-l_left)/4,windowSize));
            cv::Mat r4 = PC2(cv::Rect(l_left+(l_right-l_left)/4,f1[i].u-windowSize/2,3*(l_right-l_left)/4,windowSize));
        Mat img = imgL.clone();
        circle(img,cv::Point(f1[i].v,f1[i].u),3,Scalar(0));
        imshow("Test", img);
        imshow("Test1", r1);
        imshow("Test2", r2);
        imshow("Test3", ref);

            cout << computeMutualInformation(ref,r1)*computeMutualInformation(ref2,r3) << " " << comparePC(ref2,r3) << " " << comparePC(ref2,r4) << endl;
//        cout << computeMutualInformation(ref,r1)*computeMutualInformation(ref2,r3) << " " << computeMutualInformation(ref,r2)*computeMutualInformation(ref2,r4) << endl;

            if(computeMutualInformation(ref,r1)+comparePC(ref2,r3) > computeMutualInformation(ref,r2)+comparePC(ref2,r4)){
                l_right = l_left + 3*(l_right-l_left)/4;
                cout << "left" << endl;
                if(computeMutualInformation(ref,r1)*computeMutualInformation(ref2,r3)<prev)
                    cout << "div" << endl;
                prev = computeMutualInformation(ref,r1)*computeMutualInformation(ref2,r3);
            }
            else{
                cout << "right" << endl;
                l_left = l_left+(l_right-l_left)/4;
                if(computeMutualInformation(ref,r2)*computeMutualInformation(ref2,r4)<prev)
                    cout << "div" << endl;
                prev = computeMutualInformation(ref,r1)*computeMutualInformation(ref2,r3);
            }
        }
        Mat imgR_color;
        imgR.convertTo(imgR_color,CV_8UC3);
        cv::cvtColor(imgR_color,imgR_color,CV_GRAY2BGR);
        cv::line(imgR_color,cv::Point(l_left,f1[i].u),cv::Point(l_right,f1[i].u),cv::Scalar(0,255,0));
        imshow("color",imgR_color);
        waitKey(0);

        for(int j=0; j< max_disp; j++)
            if(f1[i].v-j-windowSize/2 > 0){
                cv::Mat ROIR =  PC2(cv::Rect(f1[i].v-j-windowSize/2,f1[i].u-windowSize/2,windowSize,windowSize));
                ROIR.convertTo(ROIR,CV_8U);
                cv::Mat ROIR2 =  imgR(cv::Rect(f1[i].v-j-windowSize/2,f1[i].u-windowSize/2,windowSize,windowSize));

                float mi = computeMutualInformation(ROIL,ROIR);
                float pc = computeMutualInformation(ROIL2,ROIR2);
                miv.push_back(mi);
                pcv.push_back(pc);
                mean_mi += mi;
                mean_pc += pc;

                if(mi > max_mi){max_mi=mi;max_idx=j;}
                if(pc > max_pc)max_pc=pc;
                if(mi < min_mi)min_mi=mi;
                if(pc < min_pc)min_pc=pc;

            }
        mean_mi /= miv.size();
        mean_pc /= pcv.size();

        float max_tot=0;
        for(uint k=0;k<miv.size();k++)
            if(miv[k]/mean_mi+pcv[k]/mean_pc >max_tot){max_tot=miv[k]/mean_mi+pcv[k]/mean_pc;max_idx=k;}


        featurePCOnly desc(f1[i].u,f1[i].v-max_idx);
        f2.push_back(desc);
        StereoMatch<featurePCOnly> f(f1[i],f2[i],max_mi);
        if(max_tot > 0.4 * (-computeEntropy(ROIL2)) && max_idx < max_disp -1 && max_idx>0 /*&& max_mi > 2*mean*/)
            matches.push_back(f);
    }

    return matches;
}

vector<StereoMatch<featurePC>> computeMIStereoMatchingFeatures(vector<featurePC>& f1, vector<featurePC>& f2, const Mat& imgL, const Mat& imgR, const Mat& PC1, const Mat& PC2){

    vector<StereoMatch<featurePC>> matches;
    int windowSize=80;
    int max_disp=80;

    for(uint i=0;i<f1.size();i++){
        float max_mi=0.01;
        int max_idx=-1;
        for(uint j=0;j<f2.size();j++){
            if(abs(f1[i].u-f2[j].u) < 3 && abs(f1[i].v-f2[j].v) < max_disp){
                cv::Mat ROIL =  PC1(cv::Rect(f1[i].v-windowSize/2,f1[i].u-windowSize/2,windowSize,windowSize)).clone();
                cv::Mat ROIL2 =  imgL(cv::Rect(f1[i].v-windowSize/2,f1[i].u-windowSize/2,windowSize,windowSize)).clone();
                cv::Mat ROIR =  PC2(cv::Rect(f2[j].v-windowSize/2,f2[j].u-windowSize/2,windowSize,windowSize)).clone();
                cv::Mat ROIR2 =  imgR(cv::Rect(f2[j].v-windowSize/2,f2[j].u-windowSize/2,windowSize,windowSize)).clone();
                ROIL.convertTo(ROIL,CV_8U);
                ROIR.convertTo(ROIR,CV_8U);
                float mi = computeMutualInformation(ROIL,ROIR) * computeMutualInformation(ROIL2,ROIR2);//* compareDescriptors(f1[i],f2[j]);
//                if(f1[i].u > 400){
//                cout << compareDescriptors(f1[i],f2[j]) << endl;
//                    imshow("L",ROIL2);
//                    imshow("R",ROIR2);
////                    cout << computeMutualInformation(ROIL,ROIR) << " " << computeMutualInformation(ROIL2,ROIR2) << endl;
//                    waitKey(0);
//                }
                if(mi > max_mi){max_mi=mi;max_idx=j;}
            }
        }

        if(max_mi>0.01 && max_idx > 0){
            StereoMatch<featurePC> f(f1[i],f2[max_idx],max_mi);
            matches.push_back(f);
        }
    }

    cout << matches.size() << endl;

    return matches;
}

float applyCCOEFFNormed(const Mat& r1, const Mat& r2){
    Mat r1_ = (r1-1)/(r1.rows*r1.cols)*sum(r1)[0];
    Mat r2_ = (r2-1)/(r2.rows*r2.cols)*sum(r2)[0];
    return sum(r1_.mul(r2_))[0]/sqrt(sum(r1_.mul(r1_))[0]*sum(r2_.mul(r2_))[0]);
}
