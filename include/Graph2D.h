#ifndef GRAPH2D_H
#define GRAPH2D_H

#include <vector>
#include <string>
#include <sstream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

namespace me{

class Graph2D
{
    public:
        Graph2D(std::string name, const int nb=1, bool orth=true, cv::Size s=cv::Size(640,480));
        void refresh();
        void addValue(cv::Point2f& v, int idx=1);
        void addValue(cv::Point2d& v, int idx=1){cv::Point2f pf(v.x,v.y);addValue(pf,idx);}
        void addValue(double v, int idx=1){cv::Point2f p(count,v);addValue(p,idx);count++;}
        void addLegend(std::string s, int idx=1){m_legend[idx-1]=s;}
        void clearGraph();
        void saveGraph(std::string filename){imwrite(filename,m_image);}
        float getLength(){return m_length;}
        float getNbValues(int idx){return m_values[idx].size();}
        float getNbCurves(){return m_nb_curves;}
        ~Graph2D();

    private:

    // attributes
    int height;
    int width;
    int m_nb_curves;
    cv::Mat m_image;
    std::vector<std::vector<cv::Point2f>> m_values;
    std::vector<cv::Scalar> m_colours;
    std::vector<std::string> m_legend;
//    std::vector<float> m_length;
    int m_margin = 90;
    float m_max_pts = 50;
    int count=0;
    std::string m_name;
    float m_min_x;
    float m_min_y;
    float m_max_x;
    float m_max_y;
    float m_length;
    bool m_orthogonal;

    // functions
    static cv::Scalar randomColor(cv::RNG& rng){int icolor=(unsigned)rng;return cv::Scalar(icolor&255,(icolor>>8)&255,(icolor>>16)&255);}
    void plot_background();
    void plot_legend();
    void plot_axis(cv::Mat& bg, const int dx, const int dy);
};

}

#endif // GRAPH2D_H
