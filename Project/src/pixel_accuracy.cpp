/**
 * @file iou.cpp
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

std::vector<int> read_numbers(std::string file_name)
{
    std::ifstream infile;
    infile.open(file_name);
    std::vector<int> numbers;

    if (infile.is_open())
    {
        int num; 
        while(infile >> num)
        {
            numbers.push_back(num);
        }
    }

    return numbers;
}

void otsu_Thresholding(cv::Mat& img_roi, cv::Mat& img_roi_thr) 
{

	//___________________________ ROI segmentation Otsu thresholding ___________________________//	
	
    cv::Mat img_roi_gray;
	cv::cvtColor(img_roi, img_roi_gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat roi_blur_gray;
    cv::GaussianBlur(img_roi_gray,roi_blur_gray,cv::Size(3,3),0);
    
	double th = cv::threshold(roi_blur_gray,img_roi_thr,0,255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	
    std::cout << "Otsu method threshold after Gaussian Filter: " << th << std::endl;
    
    cv::namedWindow("Otsu's thresholding after Gaussian filtering");
	cv::imshow("Otsu's thresholding after Gaussian filtering", img_roi_thr);
	cv::waitKey(0);

}



float pixel_accuracy(cv::Mat mask_true, cv::Mat mask_predict, int x, int y, int width, int height)
{
	
    int rows = mask_true.rows;
    int cols = mask_true.cols;
    
	int totCorrect = 0;
	
	for (int i = 1; i <= mask_true.rows; i++) 
    {
	    for (int j = 1; j <= mask_true.cols; j++)
	    {
	        if ( mask_true.at<uchar>(i,j) == mask_predict.at<uchar>(i,j) ) 
	            totCorrect += 1;
	    }
    }
    
    int totArea = rows * cols;
    
    float pixelAccuracy = (float) totCorrect / totArea;
	
	write_results_Segmentation(pixelAccuracy);
	
	return pixelAccuracy;
	
}




int main(int argc, char* argv[])
{

	//___________________________ Load Dataset image ___________________________ //
		
	cv::Mat img = cv::imread("../Dataset/rgb/02.jpg", cv::IMREAD_COLOR);
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", img);
	cv::waitKey(0);
	
	//___________________________ Load Dataset mask ___________________________ //
		
	cv::Mat mask_true = cv::imread("../Dataset/mask/02.png", cv::IMREAD_COLOR);
	cv::namedWindow("Ground truth mask");
	cv::imshow("Ground truth mask", mask_true);
	cv::waitKey(0);
    
    int rows = img.rows;
    int cols = img.cols;
    
    std::cout << "Dim original " << rows << " and " << cols << std::endl;
    
    
	//___________________________ Load Dataset bounding box coordinates ___________________________ //
	
	std::vector<int> coordinates_bb;
	
	coordinates_bb = read_numbers("../Dataset/det/02.txt");
	
	for (int i=0; i<coordinates_bb.size(); ++i)
    	std::cout << coordinates_bb[i] << ' ';
    std::cout << std::endl;
    
	int n_hands = coordinates_bb.size() / 4;
	std::cout << "Number of hands detected are " << n_hands << std::endl;
    
    
    std::vector<cv::Mat> img_roi(n_hands);
    std::vector<cv::Mat> img_roi_thr(n_hands);
    
	//___________________________ Important parameters declaration ___________________________//
    
	int x, y, width, height;
	// cv::Point pt1(0,0), pt2(0,0);
	int a,b,c,d;
	// cv::Point pt3(0,0), pt4(0,0);
	float pixelAccuracy;
	
	int temp = 0; // in order to get right index in vector of coordinates
	
	cv::Mat prediction_mask(rows, cols, CV_8UC1, cv::Scalar::all(0));
	
	for (int i=0; i<n_hands; i++) 
	{
	    
	    //_________ ROI extraction _________//
    	x = coordinates_bb[i+temp];
	    y = coordinates_bb[i+temp+1];
	    width = coordinates_bb[i+temp+2];
	    height = coordinates_bb[i+temp+3];
	
		cv::Range colonna(x, x+width);
        cv::Range riga(y, y+height);
	    img_roi[i] = img(riga, colonna);
	    
      	cv::namedWindow("ROI");
	    cv::imshow("ROI", img_roi[i]);
	    cv::waitKey(0);
	    
	    
        //_________ Otsu thresholding _________//
        otsu_Thresholding(img_roi[i], img_roi_thr[i]); 
        
      	cv::namedWindow("ROI Otsu");
	    cv::imshow("ROI Otsu", img_roi_thr[i]);
	    cv::waitKey(0);
	    
	    //_________ Insert ROI mask in image original dimension _________//
	    
	    img_roi_thr[i].copyTo(prediction_mask(cv::Rect(x, y, img_roi_thr[i].cols, img_roi_thr[i].rows)));
	    
        cv::namedWindow("Prediction mask");
        cv::imshow("Prediction mask", prediction_mask);
        cv::waitKey(0);
        
        temp = temp + 3;
	
	}
	
	// ________________ Final pixel accuracy ________________ //
	// unique for all the mask, different wrt iou which is evaluated for each bouding box
	
    pixelAccuracy = pixel_accuracy(mask_true, prediction_mask, x, y, width, height);
    std::cout << "Pixel accuracy is: " << pixelAccuracy << std::endl;
    
  
	return 0;
  
}


