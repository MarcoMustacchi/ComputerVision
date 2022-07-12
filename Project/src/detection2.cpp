
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
#include <tuple>
#include <map>
#include "read_sort_BB_matrix.h"
#include "removeOutliers.h"
#include "fillMaskHoles.h"
#include "insertMask.h"  
#include "randomColorMask.h" 
#include <opencv2/dnn/dnn.hpp>
#include <typeinfo>


using namespace cv;
using namespace std;


//Window size
const std::tuple<int, int> INITIAL_WINDOW_SIZE = std::make_tuple(168, 168);

//Strides
const float STRIDE_ROWS_FACTOR = 0.5f;
const float STRIDE_COLS_FACTOR = 0.5f;

//Input CNN
const int WIDTH_INPUT_CNN = 224;
const int HEIGHT_INPUT_CNN = 224;

//Threshold used to understand if a blob is an image or not
const float THRESHOLD_DETECTION = 0.25f;

//Threshold used to understand how much two overlapping regions overlap each other
const float THRESHOLD_OVERLAPPING = 0.70f;

//Image width and heigth
const int IMAGE_WIDTH = 1280;
const int IMAGE_HEIGTH = 720;

//Factor resizer for images from 21-30
const float FACTOR_RESIZER = 0.6f;

//Threshold Occlusion
const float THRESHOLD_OCCLUSION = 0.25f;


std::tuple<int, int> convertCoordinates(const std::tuple<int, int>& coordinatesToConvert, const std::tuple<int, int>& orginalDimensions, const std::tuple<int, int>& currentDimensions)
{
	//Convert x coordinate
	int newX = (std::get<0>(coordinatesToConvert) * std::get<1>(orginalDimensions)) / (std::get<1>(currentDimensions));
	if (newX > std::get<1>(orginalDimensions))
		newX = std::get<1>(orginalDimensions);

	//Convert y coordinate
	int newY = (std::get<1>(coordinatesToConvert) * std::get<0>(orginalDimensions)) / (std::get<0>(currentDimensions));
	if (newY > std::get<0>(orginalDimensions))
		newY = std::get<0>(orginalDimensions);

	return std::tuple<int, int>(newX, newY);
}

void detection(cv::Mat& img)
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
		
		
	std::vector<cv::Rect> boundingBoxesHands;
	std::vector<float> probabilities;	
	std::tuple<int, int> orginalDimensions = std::make_tuple(img.rows, img.cols);
	
	// cout << "orginalDimensions: " << orginalDimensions << endl;
	
    int windowSizeWidth = 480;
    int windowSizeHeigth = 480;
    
    //Compute the Stride for the Rows and Cols
	int strideRows = 240;
	int strideCols = 240;
	
	int i = 0;

	for (int row = 0; row < img.rows - windowSizeHeigth; row += strideRows)
	{
		//Range of rows coordinates
		cv::Range rowRange(row, row + windowSizeHeigth);
		
		for (int col = 0; col < img.cols - windowSizeWidth; col += strideCols)
		{			
			//Range of cols coordinates
			cv::Range colRange(col, col + windowSizeWidth);

			cv::Mat roi = img(rowRange, colRange);
			
			cv::namedWindow("roi resize");
            cv::imshow("roi resize", roi);
            cv::waitKey(0);

            cv::Mat resized, outputImage;

	        cv::resize(roi, resized, cv::Size(WIDTH_INPUT_CNN, HEIGHT_INPUT_CNN), cv::INTER_CUBIC);

	        resized.convertTo(outputImage, CV_32FC3); // Convert to CV_32
	        
	        cv::dnn::Net network = cv::dnn::readNetFromTensorflow("../model/model.pb");
	        
	        network.setInput(cv::dnn::blobFromImage(roi, 1.0 / 255.0, cv::Size(WIDTH_INPUT_CNN, HEIGHT_INPUT_CNN), true, false));

	        cv::Mat outputCNN = network.forward();
	        
	        cout << outputCNN.at<float>(0, 0) << "and" << i << endl;
	        
	        bool hand_Detected;
	        
	        i = i+1;
	        
	        /*
	        //Check if it is an hand
	        if (outputCNN.at<float>(0, 0) > THRESHOLD_DETECTION) {

                hand_Detected  = true;
				//Need to convert bounding box coordinates to original image size
				//(x1,y1) 
				std::tuple<int, int> x1y1 = convertCoordinates(std::tuple<int, int>(col, row),
					orginalDimensions,
					std::tuple<int, int>(img.rows, img.cols));

				//(x2,y2)
				std::tuple<int, int> x2y2 = convertCoordinates(std::tuple<int, int>(col + windowSizeWidth, row + windowSizeHeigth),
					orginalDimensions,
					std::tuple<int, int>(img.rows, img.cols));
				

				//Add Bounding Boxes
				boundingBoxesHands.push_back(cv::Rect(cv::Point(std::get<0>(x1y1), std::get<1>(x1y1)), 
													cv::Point(std::get<0>(x2y2), std::get<1>(x2y2))));

				std::cout << "X1 " << std::get<0>(x1y1) << " Y1 " << std::get<1>(x1y1)
					<< " X2 " << std::get<0>(x2y2) << " Y2 " << std::get<1>(x2y2) << std::endl;
					
	        } else {
	        
	   			hand_Detected  = true;
	   			
	   	    }					  
            */
            
            
            if (outputCNN.at<float>(0, 0) < THRESHOLD_DETECTION) {
                cout << "found hand" << endl;

            }
            
        }
        
    }
	
}

