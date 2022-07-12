/**
 * @file Project.cpp
 *
 * @brief  Hand Segmentation
 *
 * @author Marco Mustacchi
 *
 */

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include "read_sort_BB_matrix.h"
#include "removeOutliers.h"
#include "fillMaskHoles.h"
#include "insertMask.h"  
#include "randomColorMask.h" 


// #include <opencv2/core/utility.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <cstdio>
// #include <numeric>

using namespace cv;
using namespace std;

Mat markerMask;


int getMaxAreaContourId(vector <vector<cv::Point>> contours) {
    double maxArea = 0;
    int maxAreaContourId = -1;
    for (int j = 0; j < contours.size(); j++) {
        double newArea = cv::contourArea(contours.at(j));
        if (newArea > maxArea) {
            maxArea = newArea;
            maxAreaContourId = j;
        } // End if
    } // End for
    return maxAreaContourId;
} // End function


void prova(const cv::Mat& wshed, vector<Vec3b> colorTab) {
    
    Vec3b maxColor;
    int num = 0;
    int max = 0;
    
    for (int k=0; k<colorTab.size(); k++) {
        for( int i = 0; i < wshed.rows; i++ ) {
            for( int j = 0; j < wshed.cols; j++ ) {   
                if( wshed.at<Vec3b>(i,j) == colorTab[k]) {
                    num++;
                } 
            }
        }  
        if(num > max)
            max = num;
            maxColor = colorTab[k];
    }
    
    std::cout<< maxColor << std::endl;
}

void segmentationCanny(cv::Mat& img)
{
	
    std::string image_number;
    
    std::cout << "Insert image number from 01 to 30" << std::endl;
    std::cin >> image_number; 
	
	//___________________________ Load Dataset image ___________________________ //

	img = cv::imread("../Dataset/rgb/" + image_number + ".jpg", cv::IMREAD_COLOR);
	
	std::cout << img.rows << std::endl;
	std::cout << img.cols << std::endl;
	
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", img);
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
	    tempROI = img(riga, colonna);
	    img_roi_BGR[i] = tempROI.clone(); // otherwise matrix will not be continuos
	    
      	cv::namedWindow("ROI BGR");
	    cv::imshow("ROI BGR", img_roi_BGR[i]);
	    cv::waitKey(0);
	}
	
	//__________________________ Change image color space __________________________//
	std::vector<cv::Mat> img_roi_HSV(n_hands);    
    std::vector<cv::Mat> img_roi_YCrCb(n_hands);

    for (int i=0; i<n_hands; i++) 
	{
		cv::cvtColor(img_roi_BGR[i], img_roi_HSV[i], cv::COLOR_BGR2HSV);
		
      	// cv::namedWindow("ROI HSV");
	    // cv::imshow("ROI HSV", img_roi_HSV[i]);
	    // cv::waitKey(0);

	}
    
	
    for (int i=0; i<n_hands; i++) 
	{
		cv::cvtColor(img_roi_BGR[i], img_roi_YCrCb[i], cv::COLOR_BGR2YCrCb);
		
      	// cv::namedWindow("ROI");
	    // cv::imshow("ROI", img_roi_YCrCb[i]);
	    // cv::waitKey(0);

	}
	
	std::vector<cv::Mat> dilate(n_hands);
	std::vector<cv::Mat> final_ROI(n_hands);
	
    for (int k=0; k<n_hands; k++) 
	{

    	cv::Mat imgGray;
	
        cvtColor(img_roi_BGR[k], markerMask, COLOR_BGR2GRAY);
	    cv::namedWindow("markerMask");
	    cv::imshow("markerMask", markerMask);
	    cv::waitKey(0);
        markerMask = Scalar::all(0);
        
        //___________________________ canny often double edges -> bad result ___________________________//
        //___________________________ This method one edges -> better ___________________________//
	    Mat mask;
	    Mat edges;
	    
        /*
        Mat sharp_kernel = (Mat_<float>(3,3) << 
        -1,  -1, -1,
        -1,  9,  -1,
        -1,  -1, -1); 

        cv::filter2D(img_roi_BGR[k], img_roi_BGR[k], -1, sharp_kernel);
        
        cv::imshow("sharpened", img_roi_BGR[k]);
        cv::waitKey(0); 
        */
        
	    Mat gray;
        cv::cvtColor(img_roi_BGR[k], gray, cv::COLOR_BGR2GRAY);
        
        Mat filtered;
        cv::bilateralFilter(gray, filtered, 11, 75, 90, cv::BORDER_DEFAULT); // important for threshold
        
        /*
        Mat sharpened;
        cv::GaussianBlur(gray, filtered, cv::Size(0, 0), 5);
        cv::addWeighted(gray, 1.5, filtered, -0.5, 0, sharpened);
        
        cv::imshow("sharpened", sharpened);
        cv::waitKey(0); 
        */
        
        double th = cv::threshold(filtered, gray, 150, 200, cv::THRESH_BINARY | cv::THRESH_OTSU);
        
        std::cout << "threshold is " << th << std::endl;

        cv::Mat thresh;
        cv::threshold(gray, thresh, th, 255, cv::THRESH_BINARY);
        
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7));
             
        cv::morphologyEx(thresh, dilate[k], cv::MORPH_DILATE, kernel);
        
        cv::Mat diff;
        cv::absdiff(dilate[k], thresh, diff);

        edges = 255 - diff;

        cv::imshow("thresh", thresh);
        cv::imshow("dilate", dilate[k]);
        // cv::imshow("diff", diff);
        // cv::imshow("edges", edges);
        cv::waitKey(0);  
        
        cv::destroyAllWindows(); 
        
        
        /*
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
        */
        
        //________________________ watershed -> da fare se troviamo Overlap, semantic segmentation //
        int i, j, compCount = 0;
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(edges, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);

        // find max contour
        // int maxAreaContourId = getMaxAreaContourId(contours);

        Mat markers(markerMask.size(), CV_32S);
        markers = Scalar::all(0);
        
        int idx = 0;
        for( ; idx >= 0; idx = hierarchy[idx][0], compCount++ )
            drawContours(markers, contours, idx, Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);
        
        vector<Vec3b> colorTab;
        for( i = 0; i < compCount; i++ )
        {
            int b = theRNG().uniform(0, 255);
            int g = theRNG().uniform(0, 255);
            int r = theRNG().uniform(0, 255);
            colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
        }
        
        watershed( img_roi_BGR[k], markers );
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
    
        final_ROI[k] = wshed.clone();
	    cv::namedWindow("watershed transform");
        imshow( "watershed transform", final_ROI[k] );
        cv::waitKey(0);
        
        prova(wshed, colorTab);
        
	}	

    cv::destroyAllWindows();
    
    
    
    /*
    // Create a structuring element
    int morph_size = 1;
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(2*morph_size + 1,
             2*morph_size + 1),
        cv::Point(morph_size,
              morph_size));
              
    // Opnening
    for (int i=0; i<n_hands; i++) {
        cv::morphologyEx(img_roi_HSV[i], img_roi_HSV[i], cv::MORPH_OPEN, element, cv::Point(-1, -1), 2);
        cv::imshow("Opening", img_roi_HSV[i]);
        cv::waitKey(0);
    }
    */
        
    //___________________________________ Floodfill ___________________________________//
    // Floodfill from point (0, 0) (background) -> check if it is black (no more since draw rectangle)
    // I have to put all the background connected to point (0,0) -> all extreme pixel contour black
    
    /*
    std::vector<cv::Mat> mask_final_ROI(n_hands);

    for (int i=0; i<n_hands; i++) 
    {
        mask_final_ROI[i] = cv::Mat::zeros(dilate[i].rows, dilate[i].cols, CV_8UC1);
        mask_final_ROI[i] = dilate[i];
        //fillMaskHoles(img_roi_HSV[i], mask_final_ROI[i]);
        
      	// cv::namedWindow("filled");
        // cv::imshow("filled", mask_filled[i]);
        // cv::waitKey(0);
    }
    
    
	//______________ insert binary mask in a image same dimension of the original ______________//
	cv::Mat mask_final(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
    insertBinaryMask(img, mask_final_ROI, mask_final, coordinates_bb, n_hands);
    
	//_____________________________ generate random color and color mask _____________________________//
	std::vector<cv::Mat> img_ROI_color(n_hands);
	
	std::vector<cv::Scalar> randColor(n_hands);
	
	randomColorMask(mask_final_ROI, img_ROI_color, randColor, n_hands);
	
	//____________________ Inserisci maschera immagine colorata in immagine nera stessa dimensione originale _____________________//
	cv::Mat mask_color_final(img.rows, img.cols, CV_8UC3, cv::Scalar::all(0)); 
	insertColorMask(img, img_ROI_color, mask_color_final, coordinates_bb, n_hands);
	
	//____________________ Unisci maschera con immagine di partenza _____________________//
	// quando pixel diverso da zero, vuol dire che ho maschera mano, quindi sostituisco pixel
	for(int i=0; i<img.rows; i++) {
        for(int j=0; j<img.cols; j++) {
            if(mask_color_final.at<cv::Vec3b>(i,j)[0] != 0 && mask_color_final.at<cv::Vec3b>(i,j)[1] != 0 && mask_color_final.at<cv::Vec3b>(i,j)[2] != 0) {
                img.at<cv::Vec3b>(i,j)[0] = mask_color_final.at<cv::Vec3b>(i,j)[0];
                img.at<cv::Vec3b>(i,j)[1] = mask_color_final.at<cv::Vec3b>(i,j)[1];
                img.at<cv::Vec3b>(i,j)[2] = mask_color_final.at<cv::Vec3b>(i,j)[2];
            }
        }
	}
	
	cv::namedWindow("Image final");
    cv::imshow("Image final", img);
	cv::waitKey(0);
    */
    
	
}


