#ifndef GRAPH2D_H
#define GRAPH2D_H

#include <vector>
#include <string>
#include <sstream>
#include "opencv2/opencv.hpp"

class Graph2D
{
    public:
        Graph2D(std::string name, const int nb=1);
        void refresh();
        void addValue(cv::Point2f& v, int idx=1);
        void addLegend(std::string s, int idx=1){m_legend[idx-1]=s;}
        void clearGraph();
        void saveGraph(std::string filename){imwrite(filename,m_image);}
        float getLength(){return m_length;}
        ~Graph2D();

    private:

    // attributes
    int height=480;
    int width=640;
    int m_nb_curves;
    cv::Mat m_image;
    std::vector<std::vector<cv::Point2f>> m_values;
    std::vector<cv::Scalar> m_colours;
    std::vector<std::string> m_legend;
//    std::vector<float> m_length;
    int m_margin = 90;
    float m_max_pts = 50;
    std::string m_name;
    float m_min_x;
    float m_min_y;
    float m_max_x;
    float m_max_y;
    float m_length;

    // functions
    static cv::Scalar randomColor(cv::RNG& rng){int icolor=(unsigned)rng;return cv::Scalar(icolor&255,(icolor>>8)&255,(icolor>>16)&255);}
    void plot_background();
    void plot_legend();
    void plot_axis(cv::Mat& bg, const int dx, const int dy);
};

#endif // GRAPH2D_H
