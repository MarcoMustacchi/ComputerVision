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



float bb_intersection_over_union(int x, int y, int width, int height, int a, int b, int c, int d)
{
    int xA = std::max(x, a);
    int yA = std::max(y, b);
    int xB = std::min(x+width, a+c);
    int yB = std::min(y+height, b+d);
    
    std::cout << xA << std::endl;
    std::cout << yA << std::endl;
    std::cout << xB << std::endl;
    std::cout << yB << std::endl;
    
    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    
    std::cout << interArea << std::endl;
    
    int area_boxA = width * height;
    int area_boxB = c * d;
    
    std::cout << area_boxA << " and " << area_boxB << std::endl;
    
    float iou = (float) interArea / (area_boxA + area_boxB - interArea);
    
    return iou;
}



int main(int argc, char* argv[])
{
	
	//___________________________ Load Dataset bounding box coordinates ___________________________ //
	
	std::vector<int> coordinates_bb;
	
	coordinates_bb = read_numbers("../Dataset/det/02.txt");
	
	for (int i=0; i<coordinates_bb.size(); ++i)
    	std::cout << coordinates_bb[i] << ' ';
    std::cout << std::endl;
	
	//___________________________ Load Dataset image ___________________________ //
		
	cv::Mat img = cv::imread("../Dataset/rgb/02.jpg", cv::IMREAD_COLOR);
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", img);
	cv::waitKey(0);
	
	//___________________________ Draw Ground Thruth Bounding Boxes ___________________________ //
	int x = coordinates_bb[0];
	int y = coordinates_bb[1];
	int width = coordinates_bb[2];
	int height = coordinates_bb[3];
	
	cv::Point pt1(x, y);
    cv::Point pt2(x + width, y + height);
    cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0));
	
	//___________________________ Draw Detected Bounding Boxes ___________________________ //
	int a = x + 15;
	int b = y + 3;
	int c = width - 5;
	int d = height + 20;
	
	cv::Point pt3(a, b);
    cv::Point pt4(a + c, b + d);
    cv::rectangle(img, pt3, pt4, cv::Scalar(0, 0, 255));
	
	
	cv::namedWindow("New Image");
	cv::imshow("New Image", img);
	cv::waitKey(0);
	
	
	float iou = bb_intersection_over_union(x, y, width, height, a, b, c, d);
	
	cv::Point pt5(656, 325);
    cv::Point pt6(774, 433);
    cv::rectangle(img, pt5, pt6, cv::Scalar(255, 0, 0));
	
	cv::namedWindow("New Image");
	cv::imshow("New Image", img);
	cv::waitKey(0);
	
	std::cout << "IoU is " << iou << std::endl;
	
	/*
  
	//___________________________ ROI extraction ___________________________//
	
	cv::Range rows(x, x+width);
    cv::Range cols(y, y+height);
	cv::Mat img_roi = img(cols, rows);
	
  	cv::namedWindow("ROI");
	cv::imshow("ROI", img_roi);
	cv::waitKey(0);
	*/


  
	return 0;
  
}


