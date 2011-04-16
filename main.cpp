#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <fstream>
#include <ctype.h>
#include <algorithm>



#include "siftCorner.h"
#include "backGroundmodel.h"
#include "reSiftValidate.h"
#include "timerMacros.h"
#include "Tracker.h"
#include "LKTracker.h"
#include "Object.h"
using namespace cv;
using namespace std;



void PrintPoint(Point aim)
{
    cout<<'('<<aim.x<<' '<<aim.y<<')'<<endl;
}

int main( int argc, char* argv[] )
{
    VideoCapture cap;
    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
    Size winSize(10,10);


    bool needToInit =true;
    bool nightMode = false;
    bool reSift=true;

    if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
        cap.open(argc == 2 ? argv[1][0] - '0' : 0);
    else if( argc == 2 )
        cap.open(argv[1]);

    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
        return 0;
    }

    Mat gray, prevGray, image;
    vector<Point2f> points[2];
    VideoWriter wri;


    siftCorner cornerFinder;
    if(!cornerFinder.Init("track.config"))
    {
        cout<<"Can not Init cornerFinder"<<endl;
        return 0;
    }


    const string backGroundFilename="background.jpg";
    const float alpha=0.85;
    Mat backGround;
    backGround=imread(backGroundFilename,CV_LOAD_IMAGE_GRAYSCALE);

    backGroundModel bgModel;
    bgModel.Init(alpha,backGround);

    namedWindow("Track",1);
    const int step=1;

    reSiftValidate validator;
    Tracker *tracker=NULL;
    Object curObj;
    tracker=new LKTracker;
    //validator.init();
    //
    DECLARE_TIMING(myTimer);
    START_TIMING(myTimer);
    DECLARE_TIMING(siftTimer);
    for(;;)
    {
        Mat frame;
        for(int ii(0);ii<step;++ii)
        	cap >> frame;
        if( frame.empty() )
            break;

        frame.copyTo(image);
        cvtColor(image, gray, CV_BGR2GRAY);

	    if( nightMode )
            image = Scalar::all(0);

        medianBlur(gray,gray,3);
        bgModel.renewModel(gray);

        if( needToInit )
        {
            char fileNameBuffer[30];
            time_t rawtime;
            struct tm * timeinfo;

            time ( &rawtime );
            timeinfo = localtime ( &rawtime );

            sprintf(fileNameBuffer
                    ,"output/%d_%d_%d_%d_%d_%d.avi"
                    ,timeinfo->tm_year+1900,timeinfo->tm_mon,timeinfo->tm_mday,timeinfo->tm_hour,timeinfo->tm_min,timeinfo->tm_sec);
            wri.open(fileNameBuffer,CV_FOURCC('X','V','I','D'),50,image.size(),true);
            if(!wri.isOpened())
            {
                cout<<"can not init the writer"<<endl;
                return 0;
            }
            needToInit = false;
            tracker->Init(gray);
        }

        if(reSift)
        {
            START_TIMING(siftTimer);
            cout<<"reSift"<<endl;

            Mat Mask;
            bgModel.substractModel(gray,Mask);
            reSift=false;

            cornerFinder.goodFeatures(gray,curObj,Mask);

            cout<<"reSift Done"<<endl;
            STOP_TIMING(siftTimer);
            tracker->setObject(curObj);
        }
        else
        {
        	tracker->Process(gray);
        	curObj=tracker->getObject();
        	curObj.draw(image);
            reSift=!validator.validate(curObj);
        }
        imshow("Track", image);
        wri<<image;

        char c;
        c=(char)waitKey(2);
        if( c == 27 )
            break;
        switch( c )
        {
        case 'r':
        case 'c':
        case 'R':
        case 'C':
            points[1].clear();
            reSift=true;
            cout<<"reSift Type four"<<endl;
            break;
        case 'n':
            nightMode = !nightMode;
            break;
        case ' ':
            waitKey(-1);
            break;
        default:
            ;
        }

        std::swap(points[1], points[0]);
        swap(prevGray, gray);
    }
    STOP_TIMING(myTimer);
    printf("Execution time: %f ms.\n", GET_TIMING(myTimer));
    printf("sift Execution time: %f ms.\n", GET_TIMING(siftTimer));
    printf("sift average Execution time: %f ms.\n", GET_AVERAGE_TIMING(siftTimer));

    if(!tracker)
    	delete tracker;
    return 0;
}
