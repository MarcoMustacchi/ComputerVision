/**
 * @file pixel_accuracy.cpp
 *
 * @brief  Pixel Accuracy
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
#include "write_to_file.h"


//_____________________________________________ Functions _____________________________________________//

float compute_pixel_accuracy(cv::Mat mask_true, cv::Mat mask_predict, std::string image_number)
{
	
    int rows = mask_true.rows;
    int cols = mask_true.cols;
    
	int totCorrect = 0;
	
	for (int i = 1; i <= mask_true.rows; i++) {
        for (int j = 1; j <= mask_true.cols; j++) {
	        if ( mask_true.at<uchar>(i,j) == mask_predict.at<uchar>(i,j) ) 
	            totCorrect += 1;
	    }
    }
    
    int totArea = rows * cols;
    
    float pixelAccuracy = (float) totCorrect / totArea;
	
	write_performance_Segmentation(pixelAccuracy, image_number);
	
	return pixelAccuracy;
	
}




void pixel_accuracy()
{

    std::string image_number;
    
    std::cout << "Insert image number from 01 to 30" << std::endl;
    std::cin >> image_number;   
        
	//___________________________ Load Dataset mask ___________________________ //
	
    cv::Mat mask_true = cv::imread("../Dataset/mask/" + image_number + ".png", cv::IMREAD_GRAYSCALE);
    cv::namedWindow("Ground truth mask");
    cv::imshow("Ground truth mask", mask_true);
    cv::waitKey(0); 
    
    //___________________________ Load Predicted mask ___________________________ //
	    
    cv::Mat mask_predict = cv::imread("../results/mask/" + image_number + ".png", cv::IMREAD_GRAYSCALE);
    cv::namedWindow("Predicted mask");
    cv::imshow("Predicted mask", mask_predict);
    cv::waitKey(0);
        
    int rows = mask_true.rows;
    int cols = mask_true.cols;
    
    std::cout << "Dim original " << rows << " and " << cols << std::endl;

	
	// ________________ Final pixel accuracy ________________ //
	// unique for all the mask, different wrt iou which is evaluated for each bouding box ?
	
	float pixelAccuracy;
		
    pixelAccuracy = compute_pixel_accuracy(mask_true, mask_predict, image_number);
    std::cout << "Pixel accuracy is: " << pixelAccuracy << std::endl;
    
  
}


