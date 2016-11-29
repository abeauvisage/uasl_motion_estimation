#include "Graph2D.h"

using namespace cv;
using namespace std;


Graph2D::Graph2D(string name, const int nb): m_name(name), m_nb_curves(nb)
{
    m_min_x=0.0;
    m_min_y=0.0;
    m_max_x=0.0;
    m_max_y=0.0;
    m_length=0.0;

    RNG rng(0xFFFFFFFF);

    for(int i=0;i<m_nb_curves;i++){
        vector<Point2f> pts;
        pts.push_back(Point2f(0,0));
        m_values.push_back(pts);
        m_colours.push_back(randomColor(rng));
        m_legend.push_back("");
    }

    namedWindow(m_name,0);
    plot_background();
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

    float dx=0,dy=0;
    if(m_min_x < 0)
        dx = abs(m_min_x)*(width-2*m_margin)/(m_max_x-m_min_x);
    if(m_min_y < 0)
        dy = abs(m_min_y)*(height-2*m_margin)/(m_max_y-m_min_y);

    plot_axis(m_image,(int)dx,(int)dy);
//    Scalar color(255,0,0)
    for(int k=0;k<m_nb_curves;k++){
        Point2f prev;
        float pitch = trunc(m_values.size()/m_max_pts)+1;
        for(int i=0;i< m_values[k].size();i+=pitch){
            Point2f p(round(m_margin+(m_values[k][i].x-m_min_x)*(width-m_margin-m_margin)/(m_max_x-m_min_x)),round(height-m_margin-(m_values[k][i].y-m_min_y)*(height-m_margin-m_margin)/(m_max_y-m_min_y)));
            circle(m_image, p ,1, /*CV_RGB(255,255/m_nb_curves*k,0)*/m_colours[k], 2);
            if(i>0)
                line(m_image,prev,p,/*Scalar(0,255/m_nb_curves*k,255)*/m_colours[k]);
            prev=p;
        }
    }
    plot_legend();
    imshow(m_name,m_image);
}

void Graph2D::addValue(cv::Point2f& v, int idx){

    if(idx < 1)
        idx =1;

    if(idx ==1 && m_values[0].size()>0)
        m_length += (float) sqrt(pow(v.x-m_values[0][m_values[0].size()-1].x,2)+pow(v.y-m_values[0][m_values[0].size()-1].y,2));
    m_values[idx-1].push_back(v);

    if(v.x < m_min_x)
        m_min_x = v.x;
    if(v.y < m_min_y)
        m_min_y = v.y;
    if(v.x > m_max_x)
        m_max_x = v.x;
    if(v.y > m_max_y)
        m_max_y = v.y;

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
        line(m_image,Point(10+100*k,7),Point(20+100*k,7),m_colours[k]);
    }
}

void Graph2D::plot_axis(cv::Mat& bg, const int dx, const int dy){

    bg.create(height,width,CV_8UC3);
    bg.setTo(Scalar(255,255,255));
    line(bg,Point(m_margin+dx,m_margin),Point(m_margin+dx,height-m_margin),Scalar(0,0,0));
    line(bg,Point(width-m_margin,height-m_margin-dy),Point(m_margin,height-m_margin-dy),Scalar(0,0,0));
    putText(bg,to_string(m_min_x),Point(5,height-m_margin-dy), FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,0));
    putText(bg,to_string(m_max_x),Point(width-m_margin,height-m_margin-dy), FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,0));
    putText(bg,to_string(m_max_y),Point(m_margin/2+dx,m_margin), FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,0));
    putText(bg,to_string(m_min_y),Point(m_margin/2+dx,height-m_margin/2), FONT_HERSHEY_SIMPLEX,0.5,Scalar(0,0,0));
}

Graph2D::~Graph2D(){
    destroyWindow(m_name);
}
