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
#include <opencv2/core/utils/filesystem.hpp> //include per utils::fs::glob, anche se ho gia' core.hpp
#include <iostream>
#include <fstream>
#include <vector>
#include "read_BB_matrix.h"
#include "removeOutliers.h"
#include "fillMaskHoles.h"
#include "insertMask.h"  
#include "randomColorMask.h" 


using namespace cv;
using namespace std;


int main()
{

    vector<String> result;

    utils::fs::glob ("../Dataset/rgb/",
                    "*.jpg",
                    result,
                    false,
                    false 
                    );	
                    
    for (int i=0; i<result.size(); i++) { 
        cout << result[i] << endl;
    }                 

	int num_img = result.size();				//number of the images
	vector<Mat> images(num_img);
	
	for (int i = 0; i < num_img; i++) {
		images[i] = imread(result[i]);
		if ((images[i].cols == 0) || (images[i].rows == 0)) {		//if the program can't load an image, stop
			cout << "Immagine " + result[i] + " non trovata" << endl << endl;
			waitKey();
			return -1;
		};
	};
		
}

using namespace cv;

cv::Scalar detectedColor;

Mat equalizeIntensity(const Mat& inputImage)
{
    if(inputImage.channels() >= 3)
    {
        Mat ycrcb;

        cvtColor(inputImage,ycrcb, cv::COLOR_BGR2YCrCb);

        std::vector<Mat> channels;
        split(ycrcb,channels);

        equalizeHist(channels[0], channels[0]);

        Mat result;
        merge(channels,ycrcb);

        cvtColor(ycrcb,result, cv::COLOR_YCrCb2BGR);

        return result;
    }
    return Mat();
}


void CallBackFunc(int event, int x, int y, int flags, void* param)
{
		
	cv::Mat* imgColor = (cv::Mat *)param;
	 
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ") with colors: \n" 
		<< "B: " << (int)(*imgColor).at<cv::Vec3b>(y, x)[0] << "\n"
		<< "G: " << (int)(*imgColor).at<cv::Vec3b>(y, x)[1] << "\n"
		<< "R: " << (int)(*imgColor).at<cv::Vec3b>(y, x)[2] << "\n" 
		<< std::endl;
		
		int blue = (int)(*imgColor).at<cv::Vec3b>(y, x)[0];
		int green = (int)(*imgColor).at<cv::Vec3b>(y, x)[1];
		int red = (int)(*imgColor).at<cv::Vec3b>(y, x)[2];
		
        cv::Scalar temp(blue, green, red);
        
        detectedColor = temp;
	    
	}
	
}

int main(int argc, char* argv[])
{
	
    std::string image_number;
    
    std::cout << "Insert image number from 01 to 30" << std::endl;
    std::cin >> image_number; 
	
	//___________________________ Load Dataset image ___________________________ //

	cv::Mat img = cv::imread("../Dataset/rgb/" + image_number + ".jpg", cv::IMREAD_COLOR);
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
	
	// img_roi_BGR[0] = equalizeIntensity(img_roi_BGR[0]);
  	// cv::namedWindow("ROI equalize");
    // cv::imshow("ROI equalize", img_roi_BGR[0]);
    // cv::waitKey(0);
	
	
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
	
	
    for (int i=0; i<n_hands; i++) 
	{
	
        //####################(  Create a window and bind the callback function to that window )##########################
        cv::namedWindow("My Window", 1); 	    //Create a window
        cv::setMouseCallback("My Window", CallBackFunc, &img_roi_HSV[i]);    //set the callback function for any mouse event
        cv::imshow("My Window", img_roi_HSV[i]);     //show the image
        // Wait until user press some key
        cv::waitKey(0); 
        
        
        cv::Scalar hsvlow(0,0,0), hsvhigh(180,255,255);
        float change[3] = { 20, 30, 40 };
        
        for (int i=0; i<3; i++) { 
                hsvlow[i]  = detectedColor[i] - change[i]; 
                hsvhigh[i] = detectedColor[i] + change[i];
        }
        
        std::cout << hsvlow << std::endl;
        std::cout << hsvhigh << std::endl;
        
        // cv::Mat bw; 
        
        cv::inRange(img_roi_HSV[i], hsvlow, hsvhigh, img_roi_HSV[i]);

	}
	
  	cv::namedWindow("test");
    cv::imshow("test", img_roi_HSV[0]);
    cv::waitKey(0);
    
  	cv::namedWindow("test");
    cv::imshow("test", img_roi_HSV[1]);
    cv::waitKey(0);
    
   
	//_____________________________________________ Remove outliers from segmentation _____________________________________________
    for (int i=0; i<n_hands; i++) {
         removeOutliers(img_roi_HSV[i]);
    }
    
    cv::destroyAllWindows();
    
    
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
    

    
        
    //___________________________________ Floodfill ___________________________________//
    // Floodfill from point (0, 0) (background) -> check if it is black (no more since draw rectangle)
    // I have to put all the background connected to point (0,0) -> all extreme pixel contour black
    
    
    std::vector<cv::Mat> mask_final_ROI(n_hands);

    for (int i=0; i<n_hands; i++) 
    {
        mask_final_ROI[i] = cv::Mat::zeros(img_roi_HSV[i].rows, img_roi_HSV[i].cols, CV_8UC1);
        mask_final_ROI[i] = img_roi_HSV[i];
        //fillMaskHoles(img_roi_HSV[i], mask_final_ROI[i]);
        
      	// cv::namedWindow("filled");
        // cv::imshow("filled", mask_filled[i]);
        // cv::waitKey(0);
    }
    
    
	//______________ insert binary mask in a image same dimension of the original ______________//
	cv::Mat mask_final(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
    insertBinaryMask(img, mask_final_ROI, mask_final, coordinates_bb, n_hands);
    
    /*
    //____________________________________ Distance transform and watershed ____________________________________//
	cv::Mat distTransf;
	cv::distanceTransform(mask_final, distTransf, cv::DIST_L2, 3);
	cv::normalize(distTransf, distTransf, 0, 1.0, cv::NORM_MINMAX);
		
	cv::namedWindow("Mask transform");
    cv::imshow("Mask transform", distTransf);
	cv::waitKey(0);
	
	cv::Mat dist;
	cv::threshold(distTransf, dist, 0.6, 1.0, cv::THRESH_BINARY);
	
	cv::namedWindow("Mask transform");
    cv::imshow("Mask transform", dist);
	cv::waitKey(0);
	
    //from each blob create a seed for watershed algorithm
    cv::Mat dist8u, markers8u;
    cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32SC1);
    dist.convertTo(dist8u, CV_8U);
    
    //find total markers
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dist8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    //number of contours
    int ncomp = static_cast<int>(contours.size());
    std::printf("Contours: %d\n", ncomp);

    //draw foreground markers
    for(int i=0; i<ncomp; ++i) {
        cv::drawContours(markers, contours, i, cv::Scalar(i+1), -1);
    }
    
    markers.convertTo(markers8u, CV_8U, 10);
    cv::imshow("Markers", markers8u);
    cv::waitKey(0);
    
    //draw background markers
    cv::circle(markers, cv::Point(5, 5), 3, cv::Scalar(255), -1);
    markers.convertTo(markers8u, CV_8U, 10);
    cv::imshow("Markers", markers8u);
    cv::waitKey(0);
    */
    
	//_____________________________ generate random color and color mask _____________________________//
	std::vector<cv::Mat> img_ROI_color(n_hands);
	
	std::vector<cv::Scalar> randColor(n_hands);
	
	randomColorMask(mask_final_ROI, img_ROI_color, randColor, n_hands);
	
	//____________________ Inserisci maschera immagine colorata in immagine nera stessa dimensione originale _____________________//
	cv::Mat mask_color_final(img.rows, img.cols, CV_8UC3, cv::Scalar::all(0)); 
	insertColorMask(img, img_ROI_color, mask_color_final, coordinates_bb, n_hands);
	
	/*
	//_________________________________________ parte 2 watershed _________________________________________//
	cv::cvtColor(mask_final, mask_final, cv::COLOR_GRAY2BGR);
	
	//apply the watershed algorithm
    cv::Mat result = mask_final.clone();
    cv::watershed(result, markers);
    cv::Mat mark;
    markers.convertTo(mark, CV_8U);
    cv::bitwise_not(mark, mark);

    //generate random colors
    cv::RNG rng (12345);
    std::vector<cv::Vec3b> colors;
    for(int i=0; i<ncomp; ++i) {
        uchar b = static_cast<uchar>(rng.uniform(0, 255));
        uchar g = static_cast<uchar>(rng.uniform(0, 255));
        uchar r = static_cast<uchar>(rng.uniform(0, 255));
        //insert new color
        colors.push_back(cv::Vec3b(b, g, r));
    }
    
    cv::Mat output = mask_final.clone();
    
	cv::namedWindow("Mask tram");
    cv::imshow("Mask tram", output);
	cv::waitKey(0);

    cv::destroyAllWindows();
    
    //create output image
    for(int i=0; i<markers.rows; ++i) {
        for(int j=0; j<markers.cols; ++j) {
            int index = markers.at<int>(i, j);
            if(index > 0 && index <= ncomp) {
                output.at<cv::Vec3b>(i, j) = colors[index-1];
            }
        }
    }
    
	cv::namedWindow("transform");
    cv::imshow("transform", output);
	cv::waitKey(0);
	
	//_________________________________ Combine watershed with Histogram threshold for better result _________________________________//
	/*
	for(int i=0; i<img.rows; i++) {
        for(int j=0; j<img.cols; j++) {
            if((mask_color_final.at<cv::Vec3b>(i,j)[0] != 0 && mask_color_final.at<cv::Vec3b>(i,j)[1] != 0 && mask_color_final.at<cv::Vec3b>(i,j)[2] != 0)
                &&((mask_color_final.at<cv::Vec3b>(i,j)[0] != randColor[0][0] && mask_color_final.at<cv::Vec3b>(i,j)[1] != randColor[0][1] && mask_color_final.at<cv::Vec3b>(i,j)[2] != randColor[0][2])
                ||(mask_color_final.at<cv::Vec3b>(i,j)[0] != randColor[1][0] && mask_color_final.at<cv::Vec3b>(i,j)[1] != randColor[1][1] && mask_color_final.at<cv::Vec3b>(i,j)[2] != randColor[1][2])))      
            {
              
                mask_color_final.at<cv::Vec3b>(i,j)[0] = output.at<cv::Vec3b>(i,j)[0];
                mask_color_final.at<cv::Vec3b>(i,j)[1] = output.at<cv::Vec3b>(i,j)[1];
                mask_color_final.at<cv::Vec3b>(i,j)[2] = output.at<cv::Vec3b>(i,j)[2];
                
            }
        }
	}
	
	std::cout << randColor[0] + randColor[1] << std::endl;
	
	for(int i=0; i<img.rows; i++) {
        for(int j=0; j<img.cols; j++) {
            if(mask_color_final.at<cv::Vec3b>(i,j)[0] == 254 && mask_color_final.at<cv::Vec3b>(i,j)[1] == 239 && mask_color_final.at<cv::Vec3b>(i,j)[2] == 239)      
            {
              
                mask_color_final.at<cv::Vec3b>(i,j)[0] = output.at<cv::Vec3b>(i,j)[0];
                mask_color_final.at<cv::Vec3b>(i,j)[1] = output.at<cv::Vec3b>(i,j)[1];
                mask_color_final.at<cv::Vec3b>(i,j)[2] = output.at<cv::Vec3b>(i,j)[2];
                
            }
        }
	}
	
	
	cv::namedWindow("combination");
    cv::imshow("combination", mask_color_final);
	cv::waitKey(0);
	*/
	
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
    
	return 0;
