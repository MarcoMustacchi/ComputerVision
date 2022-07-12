
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



bool imgGrayscale(const cv::Mat& img)
{
	
	cv::Mat temp = img.clone();
    cv::Mat Bands_BGR[3];
    cv::split(temp, Bands_BGR);
    
	for (int r = 0; r < temp.rows; r++)	
		for (int c = 0; c < temp.cols; c++)		
			if ( (Bands_BGR[0].at<uchar>(r, c) != Bands_BGR[1].at<uchar>(r, c)) // Grayscale if all the channel equal for all the pixel 
			    || (Bands_BGR[0].at<uchar>(r, c) != Bands_BGR[2].at<uchar>(r, c)) // so just find a pixel different in one channel
			    || (Bands_BGR[1].at<uchar>(r, c) != Bands_BGR[2].at<uchar>(r, c)) )
				return false;
				
	return true;
	
}

void removeDetectionOutliers(cv::Mat& input) {
	
    cv::Mat labelImage, stats, centroids;
    
    int nLabels =  cv::connectedComponentsWithStats(input, labelImage, stats, centroids, 8);
    
    std::cout << "nLabels = " << nLabels << std::endl;
    std::cout << "stats.size() = " << stats.size() << std::endl;
	
	std::cout << stats.col(4) << std::endl;
	std::cout << "test2 colonna" << cv::CC_STAT_AREA << std::endl;
	
	// eseguo soltanto se numero di label e' > 4, altrimenti esco perche' non devo rimuovere nnt 
    
    if (nLabels <= 4) {
        return;
    } 
    else {
        int max1, max2, max3, max4;
        max1 = max2 = max3 = max4 = 0;
        
        for (int i = 1; i < nLabels; i++) { //label  0 is the background
            if ((stats.at<int>(i, cv::CC_STAT_AREA)) > max1) {
                max4 = max3;
                max3 = max2;
                max2 = max1;
                max1 = stats.at<int>(i, cv::CC_STAT_AREA);
            }
            else if ((stats.at<int>(i, cv::CC_STAT_AREA)) > max2) {
                max4 = max3;
                max3 = max2;
                max2 = stats.at<int>(i, cv::CC_STAT_AREA);
            } 
            else if ((stats.at<int>(i, cv::CC_STAT_AREA)) > max3) {
                max4 = max3;
                max3 = stats.at<int>(i, cv::CC_STAT_AREA);
            }
            else if ((stats.at<int>(i, cv::CC_STAT_AREA)) > max4) {
                max4 = stats.at<int>(i, cv::CC_STAT_AREA);
            }
        }
        
        std::cout << "Biggest area in order: " << max1 << " " << max2 << " " << max3 << " " << max4 << std::endl;
        
        Mat surfSup = stats.col(4) >= max4;

        Mat mask(labelImage.size(), CV_8UC1, Scalar(0));
        
        for (int i = 1; i < nLabels; i++)
        {
            if (surfSup.at<uchar>(i, 0))
            {
                mask = mask | (labelImage==i);
            }
        }
        
        input = mask.clone();
        
        imshow("mask", input);
        cv::waitKey(0);
    
        return;     
    }
    
}


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
	
	
	if (imgGrayscale(img) == false) 
	{
	    cout << "Color image" << endl;
	}
	else {
	    cout << "Grayscale image" << endl;
	}
	
    cvtColor(img, img, COLOR_BGR2YCrCb);
    // Detect the object based on HSV Range Values
    
    cv::Mat img_threshold;
    Scalar min_YCrCb(0,150,100);  // 0,150,100
    Scalar max_YCrCb(255,200,150); // 235,173,127
    inRange(img, min_YCrCb, max_YCrCb, img_threshold);
    
	cv::namedWindow("Thresh Image");
	cv::imshow("Thresh Image", img_threshold);
	cv::waitKey(0);
	
	// remove outliers
	removeDetectionOutliers(img_threshold);
	
    cvtColor(img, img, COLOR_YCrCb2BGR);
		
	std::vector<std::vector<int>> coordinates_bb;
	std::vector<float> probabilities;	
	std::tuple<int, int> orginalDimensions = std::make_tuple(img.rows, img.cols);
	
	// cout << "orginalDimensions: " << orginalDimensions << endl;
	
	int windows_n_rows;
	int windows_n_cols;
	
	// La sliding window size deve dipendere dalla dimensione dell'immagine originale
	// get if cols or row is min
	if (img.rows < img.cols) {
        windows_n_rows = img.rows * 0.25;
        windows_n_cols = img.rows * 0.25;
	}
	else
	{
        windows_n_rows = img.cols * 0.25;
        windows_n_cols = img.cols * 0.25;
	}
    
    cout << "windows_n_rows " << windows_n_rows << endl;
    cout << "windows_n_cols " << windows_n_cols << endl;
    
    //Compute the Stride for the Rows and Cols
    int stepSlideRow = windows_n_rows * 0.5;
	int stepSlideCols = windows_n_cols * 0.5;
	
	cout << "stepSlideRow " << stepSlideRow << endl;
	cout << "stepSlideCols " << stepSlideCols << endl;
	
	
	int n_hands = 0;

	for (int r = 0; r < img.rows - windows_n_rows; r += stepSlideRow)
	{
		//Range of rows coordinates
		cv::Range rowRange(r, r + windows_n_rows);
		
		for (int c = 0; c < img.cols - windows_n_cols; c += stepSlideCols)
		{			
			//Range of cols coordinates
			cv::Range colRange(c, c + windows_n_cols);

			cv::Mat roi = img(rowRange, colRange);
			cv::Mat roi_threshold = img_threshold(rowRange, colRange);
            
            if ( cv::countNonZero(roi_threshold) == 0 ) // could be an hand
                continue;

			// cv::namedWindow("Thresh ROI");
	        // cv::imshow("Thresh ROI", roi_threshold);
	        // cv::waitKey(0);
            
            cv::Mat resized, outputImage;

	        cv::resize(roi, resized, cv::Size(WIDTH_INPUT_CNN, HEIGHT_INPUT_CNN), cv::INTER_CUBIC);
	        
			// cv::namedWindow("roi resize");
            // cv::imshow("roi resize", resized);
            // cv::waitKey(0);

	        resized.convertTo(outputImage, CV_32FC3); // Convert to CV_32
	        
	        cv::dnn::Net network = cv::dnn::readNetFromTensorflow("../model/model.pb");
	        
	        network.setInput(cv::dnn::blobFromImage(roi, 1.0 / 255.0, cv::Size(WIDTH_INPUT_CNN, HEIGHT_INPUT_CNN), true, false));

	        cv::Mat outputCNN = network.forward();
	        
	        cout << "Output CNN: " << outputCNN.at<float>(0, 0) << endl;
	        
	        bool hand_Detected;
	        
            if (outputCNN.at<float>(0, 0) < THRESHOLD_DETECTION) {
                cout << "found hand" << endl;
    			cv::namedWindow("Hand");
                cv::imshow("Hand", roi);
                cv::waitKey(1000);
                
				//Need to convert bounding box coordinates to original image size
				std::tuple<int, int> x1y1 = convertCoordinates(std::tuple<int, int>(c, r), orginalDimensions, std::tuple<int, int>(img.rows, img.cols));

				std::tuple<int, int> x2y2 = convertCoordinates(std::tuple<int, int>(c + windows_n_cols, r + windows_n_rows), orginalDimensions, std::tuple<int, int>(img.rows, img.cols));
				
				//Add Bounding Boxes
                vector<int> inVect; // Define the inner vector
                inVect.push_back(std::get<0>(x1y1));
                inVect.push_back(std::get<1>(x1y1));
                inVect.push_back(std::get<0>(x2y2));
                inVect.push_back(std::get<1>(x2y2));
                
                //Insert the inner vector to outer vector
                coordinates_bb.push_back(inVect);
				
			    n_hands = n_hands+1;
            }
            
            // A questo punto posso trovare anche piu' mani, quindi detectOverlap per tutte le mani trovate
            
        }
        
    }
    
    cv::RNG rng(12345); // warning, it's a class
    
    for (int i=0; i<n_hands; i++) 
	{
	    std::cout << " X1 " << coordinates_bb[i][0] << " Y1 " << coordinates_bb[i][1]
	          << " X2 " << coordinates_bb[i][2] << " Y2 " << coordinates_bb[i][3] << std::endl;
      
        cv::Scalar random_color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        
        int x1 = coordinates_bb[i][0];
        int y1 = coordinates_bb[i][1];
        int x2 = coordinates_bb[i][2];
        int y2 = coordinates_bb[i][3];
        
        cv::Point p1(x1, y1);
        cv::Point p2(x2, y2);
        
        rectangle(img, p1, p2, random_color, 2, cv::LINE_8);
	}
	
	cv::namedWindow("Final Image");
	cv::imshow("Final Image", img);
	cv::waitKey(0);
	
	
}

