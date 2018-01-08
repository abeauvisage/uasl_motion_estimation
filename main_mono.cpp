/** opencv **/
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

/** Motion estimation **/
#include "mono_viso.h"
#include "featureType.h"
#include "Graph2D.h"
#include "data_utils.h"
#include "gps_utils.h"
#include "utils.h"
#include "fileIO.h"

/** std **/
#include <sstream>
#include <fstream>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace cv;
using namespace me;

int delay=0;

int wait(Graph2D& g, const string& filename){
    int wk = cv::waitKey(delay);
    if((char)wk == 'p')
        if(!delay)delay=10;else delay=0;
    if((char)wk == 's'){
        g.saveGraph(filename);
        cout << "image saved!" << endl;
    }
    return wk;
}


/**	To run the example:
	cd build/
	./mono_example ../examples/mono/data
**/

int main(int argc, char** argv){

	if(argc < 2){
		cout << "please provide an image directory" << endl;
		exit(-1);
	}
	string dir = argv[1];

    if(!loadYML(dir+"/../config.yml")){
        cout << "couldn't load "+dir+"/../config.yml !" << endl; return -1;
    }

    /**** variables declaration ****/

    Graph2D g("Trajectory",2,true,Graph2D::LINE);
    Graph2D gError("Error",1,false);
    gError.addLegend("Error VO",1);
    g.addLegend("VO");
    g.addLegend("GPS",2);

    int image_nb=1;
    double image_stamp=0, mean_error_VO=0;;
    Mat Rt_tot = (Mat)Matx44d::eye();
    Euld R_tot(0,0,0);
    Vec3d t_tot;

    GpsData gps;

    /**** loading inages and IO files ****/

    // uncomment to read images
    /*pair<Mat,Mat> imgs = loadImages(dir,fi.fframe);
    if(imgs.first.empty())
        return -1;*/

    ImageFile imgF(dir+"/image_data.csv");
    GpsFile gpsF(dir+"/gps_data.csv");

    std::string savingFile = dir+"/matches_"+to_string(fi.skip)+".yml";
    FileStorage matchesFile(savingFile, FileStorage::READ);
	
    openLogFile(dir+"/log.txt");
    
	while(image_nb > 0 && image_nb < fi.fframe)
        imgF.readData(image_nb,image_stamp);

	cout << "image nb" << endl;
	cout << image_nb << " " << image_stamp << endl;

	gpsF.getNextData(image_stamp, gps);
	setOrigin(Point2d(gps.lat,gps.lon));
	setAngle(-150*3.14159265/180);

    MonoVisualOdometry mono_viso(param_mono);

	/**** main loop ****/

    for(int i=fi.fframe+fi.skip;i<fi.lframe;i+=fi.skip){

        cout << " #### processing pair "  << i << " ####" << endl;

        writeLogFile("#### Frame "+to_string(i)+" ####\n");

        /**** readin images, gps and matches ****/
		while(image_nb > 0 && image_nb < i)
			imgF.readData(image_nb,image_stamp);

        gpsF.getNextData(image_stamp,gps);

        vector<StereoMatch<Point2f>> savedMatches;

		if(matchesFile.isOpened()){
			FileNode matches = matchesFile["image_"+to_string(i)];
	        FileNodeIterator it_matches=matches.begin(),it_matches_end=matches.end();
	        cout << matches.size() << " matches found" << endl;
        	for(it_matches=matches.begin();it_matches!=it_matches_end;it_matches++){
            	savedMatches.push_back(StereoMatch<Point2f>(Point2f((float)(*it_matches)["f1x"],(*it_matches)["f1y"]),Point2f((float)(*it_matches)["f2x"],(float)(*it_matches)["f2y"])));
        	}
        }

        chrono::time_point<chrono::system_clock> start,read,stop;
        start = chrono::system_clock::now();

	// uncomment to read images
        /*pair<Mat,Mat> imgs = loadImages(dir,i);
        if(imgs.first.empty())
            return -1;*/

        read = chrono::system_clock::now();

        Matx44d Tr;

	cout << "images loaded" << endl;
        mono_viso.process(savedMatches);
        Tr = mono_viso.getMotion();
		Vec3d t(Tr(0,3),Tr(1,3),Tr(2,3));
		Euld rot((Mat)Tr);
		cout << rot << endl;
		R_tot = Euld(R_tot.roll()-rot.roll(),R_tot.pitch()-rot.pitch(),R_tot.yaw()-rot.yaw());
		t_tot += -R_tot.getR3() * t;
		cout << "Tr :" << rot << R_tot << " " << t_tot << endl;
		//cout << "Tr :" << rot.getR3() << " " << -rot.getR3() * t << endl;

		Rt_tot *= (Mat)Tr.inv();


        /**** graph updates ****/

        Point2f p1(Rt_tot.at<double>(0,3),Rt_tot.at<double>(2,3));
        g.addValue(p1);

        Point2d gps_xy = getCartesianCoordinate(Point2d(gps.lat,gps.lon));
        double error_VO =  sqrt(pow(gps_xy.x-p1.x,2)+pow(gps_xy.y-p1.y,2));
        mean_error_VO += error_VO;
        gError.addValue(error_VO,1);
        g.addValue(gps_xy,2);
        cout << "error VO" << error_VO << endl;



        /**** computation time analysis ****/
        stop = chrono::system_clock::now();
        chrono::duration<double> loading_time = read-start;
        chrono::duration<double> processing_time = stop-read;
        chrono::duration<double> tot_time = stop-start;
        cout << "loading time: " << loading_time.count() << endl;
        cout << "processing time: " << processing_time.count() << endl;
        cout << "total time: " << tot_time.count() << endl;
        if((char) wait(g,dir) == 'q')
            break;
    }

    cout << "mean error VO: " << mean_error_VO/g.getNbValues(2) << endl;
    cout << "Total length: " << g.getLength() << endl;
    writeLogFile("mean error VO: "+to_string(mean_error_VO/g.getNbValues(2))+"\n");
    writeLogFile("Total length: "+to_string(g.getLength(3))+"\n");

    delay=0;
    return wait(g,dir+"/trajectory.png");
}
