#include "Graph2D.h"

using namespace cv;
using namespace std;

namespace me{

Graph2D::Graph2D(string name, const int nb, bool orth, Type t, cv::Size s): m_name(name), m_nb_curves(nb), m_orthogonal(orth),width(s.width),height(s.height), m_type(t)
{
    m_min_x=0.0;
    m_min_y=0.0;
    m_max_x=0.0;
    m_max_y=0.0;
    m_length = vector<float>(nb,0.0);

    RNG rng(0xFFFFFFFF);

    for(int i=0;i<m_nb_curves;i++){
        vector<Point2f> pts;
//        pts.push_back(Point2f(0,0));
        m_values.push_back(pts);
        m_colours.push_back(randomColor(rng));
        m_legend.push_back("");
    }
}

void Graph2D::clearGraph(){
    for(int i=0;i<m_nb_curves;i++)
    m_values[i].clear();
    m_min_x=0.0;
    m_min_y=0.0;
    m_max_x=0.0;
    m_max_y=0.0;
    Mat bg;
    plot_axis(bg,0,0);
}

void Graph2D::refresh(){

    if(m_image.empty())
        namedWindow(m_name,CV_WINDOW_NORMAL);

    float dx=0,dy=0;
    if(m_min_x < 0)
        dx = abs(m_min_x)*(width-2*m_margin)/(m_max_x-m_min_x);
    if(m_min_y < 0)
        dy = abs(m_min_y)*(height-2*m_margin)/(m_max_y-m_min_y);

    plot_axis(m_image,(int)dx,(int)dy);
    for(int k=0;k<m_nb_curves;k++){
        Point2f prev;
        int pitch = (m_values[k].size()/m_max_pts)+1;
        for(unsigned int i=0;i< m_values[k].size();i+=pitch){
            Point2f p(round(m_margin+(m_values[k][i].x-m_min_x)*(width-m_margin-m_margin)/(m_max_x-m_min_x)),round(height-m_margin-(m_values[k][i].y-m_min_y)*(height-m_margin-m_margin)/(m_max_y-m_min_y)));
            if(m_type != LINE)
                circle(m_image, p ,1,m_colours[k], 2);
            if(m_type != DOT && i>0)
                line(m_image,prev,p,m_colours[k],1);
            prev=p;
        }
    }
    plot_legend();
    imshow(m_name,m_image);
    waitKey(2);
}

void Graph2D::addValue(const cv::Vec3d& v, int idx){

    if(idx < 1)
        idx =1;
    for(int i=0;i<3;i++){
        int value_idx = m_values[idx+i-1].size();
        m_values[idx+i-1].push_back(Point2f(value_idx,v[i]));

        if(value_idx < m_min_x)
            m_min_x = value_idx;
        if(v[i] < m_min_y)
            m_min_y = v[i];
        if(value_idx > m_max_x)
            m_max_x = value_idx;
        if(v[i] > m_max_y)
            m_max_y = v[i];

        if(m_orthogonal){
            if(m_max_x > m_max_y)
                m_max_y = m_max_x;
            else
                m_max_x = m_max_y;
            if(m_min_x < m_min_y)
                m_min_y = m_min_x;
            else
                m_min_x = m_min_y;
        }
    }

    refresh();
}

void Graph2D::addValue(const cv::Point2f& v, int idx){

    if(idx < 1)
        idx =1;

    if(m_values[idx-1].size()>0)
        m_length[idx-1] += (float) sqrt(pow(v.x-m_values[idx-1][m_values[idx-1].size()-1].x,2)+pow(v.y-m_values[idx-1][m_values[idx-1].size()-1].y,2));
    m_values[idx-1].push_back(v);

    if(v.x < m_min_x)
        m_min_x = v.x;
    if(v.y < m_min_y)
        m_min_y = v.y;
    if(v.x > m_max_x)
        m_max_x = v.x;
    if(v.y > m_max_y)
        m_max_y = v.y;

    if(m_orthogonal){
        if(m_max_x > m_max_y)
            m_max_y = m_max_x;
        else
            m_max_x = m_max_y;
        if(m_min_x < m_min_y)
            m_min_y = m_min_x;
        else
            m_min_x = m_min_y;
    }

    refresh();
}

void Graph2D::plot_background(){
    Mat bg;
    plot_axis(bg,0,0);
    imshow(m_name,bg);
}

void Graph2D::plot_legend(){
    for(int k=0;k<m_nb_curves;k++){
        putText(m_image,m_legend[k],Point(30+100*k,10),FONT_HERSHEY_SIMPLEX,0.3,m_colours[k]);
        line(m_image,Point(10+100*k,7),Point(20+100*k,7),m_colours[k],2);
    }
}

void Graph2D::plot_axis(cv::Mat& bg, const int dx, const int dy){

    bg.create(height,width,CV_8UC3);
    bg.setTo(Scalar(255,255,255));
    line(bg,Point(m_margin+dx,m_margin),Point(m_margin+dx,height-m_margin),Scalar(0,0,0));
    line(bg,Point(width-m_margin,height-m_margin-dy),Point(m_margin,height-m_margin-dy),Scalar(0,0,0));
    putText(bg,to_string(m_min_x).substr(0,to_string(m_min_x).rfind(".")+3),Point(5,height-m_margin-dy), FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,0));
    putText(bg,to_string(m_max_x).substr(0,to_string(m_max_x).rfind(".")+3),Point(width-m_margin+5,height-m_margin-dy), FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,0));
    putText(bg,to_string(m_max_y).substr(0,to_string(m_max_y).rfind(".")+3),Point(m_margin/2+dx,m_margin-5), FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,0));
    putText(bg,to_string(m_min_y).substr(0,to_string(m_min_y).rfind(".")+3),Point(m_margin/2+dx,height-m_margin/2), FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,0));
}

Graph2D::~Graph2D(){
	if(!m_image.empty())
	    destroyWindow(m_name);
}

}
