/**
 * @file iou.cpp
 *
 * @brief  Intersection over Union
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
#include <algorithm>
#include "read_numbers.h"
#include "write_to_file.h"



//_____________________________________________ Functions _____________________________________________//

float bb_intersection_over_union(int x_truth, int y_truth, int width_truth, int height_truth, int x_predict, int y_predict, int width_predict, int height_predict)
{

    int xA = std::max(x_truth, x_predict);
    int yA = std::max(y_truth, y_predict);
    int xB = std::min(x_truth+width_truth, x_predict+width_predict);
    int yB = std::min(y_truth+height_truth, y_predict+height_predict);
    
    std::cout << xA << std::endl;
    std::cout << yA << std::endl;
    std::cout << xB << std::endl;
    std::cout << yB << std::endl;
    
    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    
    std::cout << interArea << std::endl;
    
    int area_box_truth = width_truth * height_truth;
    int area_box_predict = width_predict * height_predict;
    
    std::cout << area_box_truth << " and " << area_box_predict << std::endl;
    
    float iou = (float) interArea / (area_box_truth + area_box_predict - interArea);
    
    // write_results_Detection(iou);
    
    return iou;
    
}



int main(int argc, char* argv[])
{
	
	//___________________________ Load Dataset bounding box coordinates ___________________________ //
	
	std::vector<int> coordinates_bb;
	
	coordinates_bb = read_numbers("../Dataset/det/02.txt");
	
	int n_hands = coordinates_bb.size() / 4;
	std::cout << "Number of hands detected are " << n_hands << std::endl;
	
	for (int i=0; i<coordinates_bb.size(); ++i)
    	std::cout << coordinates_bb[i] << ' ';
    std::cout << std::endl;
	
	//___________________________ Load Dataset image ___________________________ //
		
	cv::Mat img = cv::imread("../Dataset/rgb/02.jpg", cv::IMREAD_COLOR);
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", img);
	cv::waitKey(0);
	
	//___________________________ Draw Ground Thruth Bounding Boxes ___________________________ //
	
	int x, y, width, height;
	// cv::Point pt1(0,0), pt2(0,0);
	int a,b,c,d;
	// cv::Point pt3(0,0), pt4(0,0);
	float iou;
	
	int temp = 0; // in order to get right index in vector of coordinates
	
	std::ofstream myfile;
	myfile.open("../results/performanceDetection.txt");
	
	for (int i=0; i<n_hands; i++) 
	{
	    
	    //_________ Draw Ground Thruth Bounding Boxes _________//
    	x = coordinates_bb[i+temp];
	    y = coordinates_bb[i+temp+1];
	    width = coordinates_bb[i+temp+2];
	    height = coordinates_bb[i+temp+3];
	
    	cv::Point pt1(x, y);
        cv::Point pt2(x + width, y + height);
        cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0));
	
	    // _________ Draw Detected Bounding Boxes _________//
    	a = x + 15;
	    b = y + 3;
	    c = width - 5;
	    d = height + 20;
	    
	    cv::Point pt3(a, b);
        cv::Point pt4(a + c, b + d);
        cv::rectangle(img, pt3, pt4, cv::Scalar(0, 0, 255));
        
        // _________ Compute IoU measurements _________//
    	iou = bb_intersection_over_union(x, y, width, height, a, b, c, d);
    	myfile << iou << std::endl;
    	std::cout << "IoU is " << iou << std::endl;
    	
        temp = temp + 3;
	
	}
	
	myfile.close();

	cv::namedWindow("New Image");
	cv::imshow("New Image", img);
	cv::waitKey(0);
	
	
	//_________________ Disegno rettangolo blu nell'intersezione bounding box _________________//
	cv::Point pt5(656, 325);
    cv::Point pt6(774, 433);
    cv::rectangle(img, pt5, pt6, cv::Scalar(255, 0, 0));
	
	cv::namedWindow("New Image");
	cv::imshow("New Image", img);
	cv::waitKey(0);

  
	return 0;
  
}


