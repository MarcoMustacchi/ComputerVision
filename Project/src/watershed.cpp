
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include "read_BB_matrix.h"
#include "removeOutliers.h"
#include "fillMaskHoles.h"
#include "insertMask.h"  
#include "randomColorMask.h" 
#include <numeric>


using namespace cv;
using namespace std;


Mat markerMask, img;
Point prevPt(-1, -1);



int mostFrequent(int *arr, int n) {
  // code here
  int maxcount=0;
  int element_having_max_freq;
  for(int i=0;i<n;i++)
  {
    int count=0;
    for(int j=0;j<n;j++)
    {
      if(arr[i] == arr[j])
        count++;
    }
 
    if(count>maxcount)
    {
      maxcount=count;
      element_having_max_freq = arr[i];
    }
     
  }
 
  return element_having_max_freq;
}

static void onMouse( int event, int x, int y, int flags, void* )
{
    if( x < 0 || x >= img.cols || y < 0 || y >= img.rows )
        return;
    if( event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON) )
        prevPt = Point(-1,-1);
    else if( event == EVENT_LBUTTONDOWN )
        prevPt = Point(x,y);
    else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON) )
    {
        Point pt(x, y);
        if( prevPt.x < 0 )
            prevPt = pt;
        line( markerMask, prevPt, pt, Scalar::all(255), 5, 8, 0 );
        line( img, prevPt, pt, Scalar::all(255), 5, 8, 0 );
        prevPt = pt;
        imshow("image", img);
    }
}
int main( int argc, char** argv )
{
    std::string image_number;
    
    std::cout << "Insert image number from 01 to 30" << std::endl;
    std::cin >> image_number; 
	
	//___________________________ Load Dataset image ___________________________ //

	cv::Mat img0 = cv::imread("../Dataset/rgb/" + image_number + ".jpg", cv::IMREAD_COLOR);
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", img0);
	cv::waitKey(0);
	
	//___________________________ Load Dataset bounding box coordinates ___________________________ //

	std::vector<std::vector<int>> coordinates_bb;
	
	coordinates_bb = read_sort_BB_matrix("../Dataset/det/" + image_number + ".txt");
	
	int n_hands = coordinates_bb.size(); // return number of rows
	std::cout << "Number of hands detected are " << n_hands << std::endl;

    
	//___________________________ Important parameters declaration ___________________________//
    std::vector<cv::Mat> img_roi_BGR(n_hands);
    std::vector<cv::Mat> img_roi_thr(n_hands);
    cv::Mat tempROI;
    
	int x, y, width, height;	
		
	for (int i=0; i<n_hands; i++) 
	{
	    //_________ ROI extraction _________//
    	x = coordinates_bb[i][0];
	    y = coordinates_bb[i][1];
	    width = coordinates_bb[i][2];
	    height = coordinates_bb[i][3];
	
		cv::Range colonna(x, x+width);
        cv::Range riga(y, y+height);
	    tempROI = img0(riga, colonna);
	    img_roi_BGR[i] = tempROI.clone(); // otherwise matrix will not be continuos
	    
      	cv::namedWindow("ROI BGR");
	    cv::imshow("ROI BGR", img_roi_BGR[i]);
	    cv::waitKey(0);
	}
	
	Mat imgGray;
	img_roi_BGR[0].copyTo(img);
    cvtColor(img, markerMask, COLOR_BGR2GRAY);
	cv::namedWindow("markerMask");
	cv::imshow("markerMask", markerMask);
	cv::waitKey(0);
    cvtColor(markerMask, imgGray, COLOR_GRAY2BGR);
	cv::namedWindow("imgGray");
	cv::imshow("imgGray", imgGray);
	cv::waitKey(0);
    markerMask = Scalar::all(0);
    imshow( "image", img );
    setMouseCallback( "image", onMouse, 0 );
    
    

	Mat mask;
	
	double th = cv::threshold(markerMask, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    double threshold1 = th-200;	//canny lower threshold
	double threshold2 = th+200;	//canny higher threshold
	int apertureSize = 3;
	bool L2gradient = false;
	Mat edges;
	cv::blur(img, img, cv::Size(3, 3));
    
    cv::cvtColor(img, img, cv::COLOR_BGR2HSV);
    
	Canny(img, edges, threshold1, threshold2, apertureSize, L2gradient);	//find edges with canny
	cv::namedWindow("Edges");
	imshow("Edges", edges);
    cv::waitKey(0);   
    
    // Create a structuring element
    int morph_size = 1;
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(2*morph_size + 1,
             2*morph_size + 1),
        cv::Point(morph_size,
              morph_size));    
    // Closing
    for (int i=0; i<n_hands; i++) {
        cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 1);
        cv::imshow("closing", edges);
        cv::waitKey(0);
    }  
    
    int i, j, compCount = 0;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);

    Mat markers(markerMask.size(), CV_32S);
    markers = Scalar::all(0);
    int idx = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0], compCount++ )
        drawContours(markers, contours, idx, Scalar::all(compCount+1), -1, 8, hierarchy, FILLED); 
    
    
    vector<Vec3b> colorTab;
    for( i = 0; i < compCount; i++ )
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    

    watershed( img_roi_BGR[0], markers );
    Mat wshed(markers.size(), CV_8UC3);
    
    // paint the watershed image
    for( i = 0; i < markers.rows; i++ )
        for( j = 0; j < markers.cols; j++ )
        {
            int index = markers.at<int>(i,j);
            if( index == -1 )
                wshed.at<Vec3b>(i,j) = Vec3b(255,255,255);
            else if( index <= 0 || index > compCount )
                wshed.at<Vec3b>(i,j) = Vec3b(0,0,0);
            else
                wshed.at<Vec3b>(i,j) = colorTab[index - 1];
        }

	cv::namedWindow("watershed transform");
    imshow( "watershed transform", wshed );
    cv::waitKey(0);
    
	//_____________________________________________ Remove outliers from segmentation _____________________________________________   
    
    return 0;
}
