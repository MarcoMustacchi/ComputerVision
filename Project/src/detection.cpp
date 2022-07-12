
/**
 * @file detection.cpp
 *
 * @brief  Detection algorithm, sliding Window approach
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
#include "write_to_file.h"


using namespace cv;
using namespace std;


//Input CNN
const int WIDTH_INPUT_CNN = 224;
const int HEIGHT_INPUT_CNN = 224;

//Threshold used to understand if a blob is an image or not
const float THRESHOLD_DETECTION = 0.25f;


float detectionIoU(std::vector<int> loop_coordinates_bb_old, std::vector<int> loop_coordinates_bb_new)
{

    // last element [4] is the confidence
    int x_truth = loop_coordinates_bb_old[0];
    int y_truth = loop_coordinates_bb_old[1];
    int width_truth = loop_coordinates_bb_old[2];
    int height_truth = loop_coordinates_bb_old[3];
    int x_predict = loop_coordinates_bb_new[0];
    int y_predict = loop_coordinates_bb_new[1];
    int width_predict = loop_coordinates_bb_new[2];
    int height_predict = loop_coordinates_bb_new[3];
    
    // coordinates of the intersection Area
    int xA = std::max(x_truth, x_predict);
    int yA = std::max(y_truth, y_predict);
    int xB = std::min(x_truth+width_truth, x_predict+width_predict);
    int yB = std::min(y_truth+height_truth, y_predict+height_predict);
    
    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    
    int area_box_truth = width_truth * height_truth;
    int area_box_predict = width_predict * height_predict;
    
    float iou = (float) interArea / (area_box_truth + area_box_predict - interArea);
    
    return iou;

}


bool sortcolDetection(const std::vector<int>& v1, const std::vector<int>& v2)
{
    return v1[4] < v2[4];
}


void nonMaximumSuppression(std::vector<std::vector<int>>& old_coordinates_bb, std::vector<std::vector<int>>& new_coordinates_bb) 
{

    //____________________________________________________ Sort rows by last column ____________________________________________________//
    // Use of "sort()" for sorting on basis of 5th (last) column
    sort(old_coordinates_bb.begin(), old_coordinates_bb.end(), sortcolDetection);
    
    //lets check out the elements of the 2D vector are sorted correctly
    std::cout << "Ordered coordinates by confidence" << std::endl;
    
    for(std::vector<int> &newvec: old_coordinates_bb)
    {
        for(const int &elem: newvec)
        {
            std::cout<<elem<<" ";
        }
        std::cout<<std::endl;
    }
    
    float threshIoU = 0.5f;
    
    //______________________________________ Take last column in a loop, move and compare IoU ______________________________________//
    
    while ( old_coordinates_bb.size() > 0 )
    {
        int i = 0;
        
        new_coordinates_bb.push_back(old_coordinates_bb.back());
        old_coordinates_bb.erase(old_coordinates_bb.end());
        
        for (int j=0; j<old_coordinates_bb.size(); j++) 
        {
            float iou = detectionIoU( old_coordinates_bb.at(j), old_coordinates_bb.at(i));
            
            if (iou < threshIoU)
            {
                old_coordinates_bb.erase(old_coordinates_bb.begin()+j);
            }
        }
        
        i = i+1;
    
    }
    
}


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


void convert2originalXYWH(const std::tuple<int, int, int, int>& resizedCoordinates, const std::tuple<int, int>& resizedDimension, 
                            std::tuple<int, int, int, int>& originalCoordinates, const std::tuple<int, int>& orginalDimensions) 
{
    
    int x = ( std::get<0>(resizedCoordinates) * std::get<1>(orginalDimensions) ) / (std::get<1>(resizedDimension));
    int y = ( std::get<1>(resizedCoordinates) * std::get<0>(orginalDimensions) ) / (std::get<0>(resizedDimension));
    int x2 = ( std::get<2>(resizedCoordinates) * std::get<1>(orginalDimensions) ) / (std::get<1>(resizedDimension));
    int y2 = ( std::get<3>(resizedCoordinates) * std::get<0>(orginalDimensions) ) / (std::get<0>(resizedDimension));
    	
    int width = x2-x;
    int height = y2-y; 
    
    originalCoordinates = std::make_tuple(x, y, width, height);

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
		
	int windows_n_rows_BIG;
	int windows_n_cols_BIG;
	
	int windows_n_rows_SMALL;
	int windows_n_cols_SMALL;
	
	// La sliding window size deve dipendere dalla dimensione dell'immagine originale
	// get if cols or row is min
	if (img.rows < img.cols) {
	    windows_n_rows_BIG = img.rows * 0.5;
	    windows_n_cols_BIG = img.rows * 0.5;
        windows_n_rows_SMALL = img.rows * 0.25;
        windows_n_cols_SMALL = img.rows * 0.25;
	}
	else
	{
	    windows_n_rows_BIG = img.cols * 0.5;
	    windows_n_cols_BIG = img.cols * 0.5;
        windows_n_rows_SMALL = img.cols * 0.25;
        windows_n_cols_SMALL = img.cols * 0.25;
	}
    
    //Compute the Stride for the Rows and Cols
    int stepSlideRow_BIG = windows_n_rows_BIG * 0.25;
	int stepSlideCols_BIG = windows_n_cols_BIG * 0.25;
    int stepSlideRow_SMALL = windows_n_rows_SMALL * 0.5;
	int stepSlideCols_SMALL = windows_n_cols_SMALL * 0.5;
	
	
	int n_hands = 0;
	
	
	
	for (int i=0; i<2; i++) 
	{
	    switch (i)
	    {
	        case 0:
		    {    
		        cout << "Starting Case 0" << endl;
		        
		        for (int r = 0; r < img.rows - windows_n_rows_BIG; r += stepSlideRow_BIG)
	            {
		            //Range of rows coordinates
		            cv::Range rowRange(r, r + windows_n_rows_BIG);
		            
		            for (int c = 0; c < img.cols - windows_n_cols_BIG; c += stepSlideCols_BIG)
		            {			
			            //Range of cols coordinates
			            cv::Range colRange(c, c + windows_n_cols_BIG);

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
                                                       
                            std::tuple<int, int, int, int> resizedCoordinates = std::make_tuple(c, r, c + windows_n_cols_BIG, r + windows_n_rows_BIG);
                            std::tuple<int, int, int, int> originalCoordinates;
                            std::tuple<int, int> resizedDimension = std::make_tuple(img.rows, img.cols);

				            //Need to convert bounding box coordinates to original image size
                            convert2originalXYWH(resizedCoordinates, resizedDimension, originalCoordinates, orginalDimensions);
                                                        
                            vector<int> inVect; // Define the inner vector
                            inVect.push_back(std::get<0>(originalCoordinates));
                            inVect.push_back(std::get<1>(originalCoordinates));
                            inVect.push_back(std::get<2>(originalCoordinates));
                            inVect.push_back(std::get<3>(originalCoordinates));
                            inVect.push_back(outputCNN.at<float>(0, 0) * 1000000);  // last column confidence in percentage, because int
                            
                            //Insert the inner vector to outer vector
                            coordinates_bb.push_back(inVect);
				            
			                n_hands = n_hands+1;
                        }
                        
                        // A questo punto posso trovare anche piu' mani, quindi detectOverlap per tutte le mani trovate
                        
                    }
                    
                }
                
                break;     
            }
            
            case 1:
            {
            
                cout << "Starting Case 1" << endl;
                
                for (int r = 0; r < img.rows - windows_n_rows_SMALL; r += stepSlideRow_SMALL)
	            {
		            //Range of rows coordinates
		            cv::Range rowRange(r, r + windows_n_rows_SMALL);
		            
		            for (int c = 0; c < img.cols - windows_n_cols_SMALL; c += stepSlideCols_SMALL)
		            {			
			            //Range of cols coordinates
			            cv::Range colRange(c, c + windows_n_cols_SMALL);

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
                            
                            
                            std::tuple<int, int, int, int> resizedCoordinates = std::make_tuple(c, r, c + windows_n_cols_SMALL, r + windows_n_rows_SMALL);
                            std::tuple<int, int, int, int> originalCoordinates;
                            std::tuple<int, int> resizedDimension = std::make_tuple(img.rows, img.cols);

				            //Need to convert bounding box coordinates to original image size
                            convert2originalXYWH(resizedCoordinates, resizedDimension, originalCoordinates, orginalDimensions);
                            
                            vector<int> inVect; // Define the inner vector
                            inVect.push_back(std::get<0>(originalCoordinates));
                            inVect.push_back(std::get<1>(originalCoordinates));
                            inVect.push_back(std::get<2>(originalCoordinates));
                            inVect.push_back(std::get<3>(originalCoordinates));
                            inVect.push_back(outputCNN.at<float>(0, 0) * 1000000);  // last column confidence in percentage, because int
                            
                            //Insert the inner vector to outer vector
                            coordinates_bb.push_back(inVect);
				            
			                n_hands = n_hands+1;
                        }
                        
                        // A questo punto posso trovare anche piu' mani, quindi detectOverlap per tutte le mani trovate
                        
                    }
                    
                }
                
                break;
            }
	    }
    }


  	// Displaying the 2D vector
    for (int i = 0; i < coordinates_bb.size(); i++) 
    {
        for (int j = 0; j < coordinates_bb[i].size(); j++)
            cout << coordinates_bb[i][j] << " ";
        cout << endl;
    }
    
    
    //__________________________________________ non Maximum Suppression __________________________________________//
    std::vector<std::vector<int>> new_coordinates_bb;
    nonMaximumSuppression(coordinates_bb, new_coordinates_bb);
    
    
    cout << "remaining coordinates" << endl;
    
  	// Displaying the 2D vector
    for (int i = 0; i < new_coordinates_bb.size(); i++) 
    {
        for (int j = 0; j < new_coordinates_bb[i].size(); j++)
            cout << new_coordinates_bb[i][j] << " ";
        cout << endl;
    }
    
    
    //__________________________________________ Draw random color Bounding Boxes __________________________________________//
    	
	n_hands = new_coordinates_bb.size();
	
    cv::RNG rng(12345); // warning, it's a class
    
    for (int i=0; i<n_hands; i++) 
	{
	    std::cout << " X1 " << new_coordinates_bb[i][0] << " Y1 " << new_coordinates_bb[i][1]
	          << " X2 " << new_coordinates_bb[i][2] << " Y2 " << new_coordinates_bb[i][3] << std::endl;
      
        cv::Scalar random_color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        
        int x1 = new_coordinates_bb[i][0];
        int y1 = new_coordinates_bb[i][1];
        int x2 = new_coordinates_bb[i][0] + new_coordinates_bb[i][2];
        int y2 = new_coordinates_bb[i][1] + new_coordinates_bb[i][3];
        
        cv::Point p1(x1, y1);
        cv::Point p2(x2, y2);
        
        rectangle(img, p1, p2, random_color, 2, cv::LINE_8);
	}
	
		
	// Delete last column for consistency and Write results
	int columnToDelete = 4;

    for (int i=0; i < new_coordinates_bb.size(); ++i)
    {
        if (new_coordinates_bb[i].size() > columnToDelete)
        {
            new_coordinates_bb[i].erase(new_coordinates_bb[i].begin() + columnToDelete);
        }
    }
    
  	// Displaying the 2D vector
    for (int i = 0; i < new_coordinates_bb.size(); i++) 
    {
        for (int j = 0; j < new_coordinates_bb[i].size(); j++)
            cout << new_coordinates_bb[i][j] << " ";
        cout << endl;
    }
	
	
	// Write results to file
    write_results_Detection(new_coordinates_bb, image_number);
	
	return;
	
}

