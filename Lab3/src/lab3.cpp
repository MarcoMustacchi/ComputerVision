#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>


void CallBackFunc(int event, int x, int y, int flags, void* param)
{
		
	cv::Mat* imgColor = (cv::Mat *)param;
	
	double mean_B;
	double mean_G;
	double mean_R;
	 
    if  ( event == cv::EVENT_LBUTTONDOWN )
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ") with colors: \n" 
		<< "B: " << (int)(*imgColor).at<cv::Vec3b>(y, x)[0] << "\n"
		<< "G: " << (int)(*imgColor).at<cv::Vec3b>(y, x)[1] << "\n"
		<< "R: " << (int)(*imgColor).at<cv::Vec3b>(y, x)[2] << "\n" 
		<< std::endl;
		
		// #########(  Mean of the B, G and R values around clicked pixel  )#########
		
		for (int i = -4; i <= 4; i++) // in this way is a 9x9
		{
			for (int j = -4; j <= 4; j++)
			{
				mean_B += (int)(*imgColor).at<cv::Vec3b>(y+i, x+j)[0];
				mean_G += (int)(*imgColor).at<cv::Vec3b>(y+i, x+j)[1];
				mean_R += (int)(*imgColor).at<cv::Vec3b>(y+i, x+j)[2];
			}
		}
		
		mean_B = mean_B / 81;
		mean_G = mean_G / 81;
		mean_R = mean_R / 81;
		
		std::cout << "Mean of the channels are: \n" 
		<< "B: " << mean_B << "\n"
		<< "G: " << mean_G << "\n"
		<< "R: " << mean_R << "\n"
		<< std::endl;	
		
		cv::Scalar T(30,30,30);
	    cv::Scalar treshold(mean_B, mean_G, mean_R);
        cv::Mat mask; 
        cv::inRange( (*imgColor), treshold - T, treshold + T, mask); // returns a binary mask where values
															 // of 1 indicate values within the range
	    cv::namedWindow("Mask", cv::WINDOW_AUTOSIZE);
	    cv::imshow("Mask", mask);
        
        
        // std::cout << mask.at<uchar>(y, x) << std::endl;
        std::cout << "x: " << x << std::endl;
        
        cv::Mat newImage((*imgColor).rows, (*imgColor).cols, CV_8UC3, cv::Scalar(0,0,0));
        if (x >= 600) { // 0 = black
            newImage = (*imgColor);
        } else {
            newImage = cv::Scalar(92, 37, 201);
        }
        
        cv::namedWindow("New Image", cv::WINDOW_AUTOSIZE);
	    cv::imshow("New Image", newImage);
	    
	}
	
}

int main(int argc, char** argv)
{
    cv::Mat img = cv::imread("../images/robocup.jpg", cv::IMREAD_COLOR); 
    
    
    if (img.empty())
    {
        std::cout << "!!! Failed imread(): image not found" << std::endl;
    // don't let the execution continue, else imshow() will crash.
    }
    
    cv::namedWindow("Original image");
    cv::imshow("Original image", img);
    cv::waitKey(0); 
    
    //####################(  Resize images by Scaling factor - preserve aspect ratio  )##########################

    
    /*
    while(true) {
        cv::setMouseCallback("My Window", CallBackFunc, &img);    //set the callback function for any mouse event
        cv::imshow("My Window", img);     //show the image
        
        int c = cv::waitKey(1);
	    if (c == 27) // Esc key to stop
            break;
    }
    */
    
    //####################(  Create a window and bind the callback function to that window )##########################
    cv::namedWindow("My Window", 1); 	    //Create a window
    cv::setMouseCallback("My Window", CallBackFunc, &img);    //set the callback function for any mouse event
    cv::imshow("My Window", img);     //show the image
    cv::destroyWindow("Mask");
    // Wait until user press some key
    cv::waitKey(0);
     
    return 0;
}

