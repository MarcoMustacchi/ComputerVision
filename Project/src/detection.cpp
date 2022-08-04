
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
#include <opencv2/dnn/dnn.hpp>
#include "write_to_file.h"
#include "fillMaskHoles.h" 
#include "Detector.h"


using namespace cv;
using namespace std;



void detection(cv::Mat& img)
{

    Detector detect;
    
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
	
	cv::Mat img_threshold;
	
	int maxArea;
	
	if (detect.imgGrayscale(img) == false) 
	{
	    cout << "Color image" << endl;
	    cvtColor(img, img, COLOR_BGR2YCrCb);
	    maxArea = detect.skinDetectionColored(img, img_threshold);
	    cvtColor(img, img, COLOR_YCrCb2BGR);
	    // GaussianBlur(img, img, Size(3, 3), 0, 0, BORDER_DEFAULT);
	}
	else {
		
	    cout << "Grayscale image" << endl;
	    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	    maxArea = detect.skinDetectionGrayscale(img, img_threshold);   
	    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
	    // GaussianBlur(img, img, Size(3, 3), 0, 0, BORDER_DEFAULT);
	}   
			
		
	int windows_n_rows_BIG;
	int windows_n_cols_BIG;
	
	int windows_n_rows_SMALL;
	int windows_n_cols_SMALL;
	
	// La sliding window size deve dipendere dalla dimensione dell'immagine originale
	// get if cols or row is min
	if (img.rows < img.cols) {
	    windows_n_rows_BIG = img.rows * 0.4;
	    windows_n_cols_BIG = img.rows * 0.5;
        windows_n_rows_SMALL = img.rows * 0.25;
        windows_n_cols_SMALL = img.rows * 0.25;
	}
	else
	{
	    windows_n_rows_BIG = img.cols * 0.4;
	    windows_n_cols_BIG = img.cols * 0.5;
        windows_n_rows_SMALL = img.cols * 0.25;
        windows_n_cols_SMALL = img.cols * 0.25;
    }
    
    //Compute the Stride for the Rows and Cols
    int stepSlideRow_BIG = windows_n_rows_BIG * 0.25;
	int stepSlideCols_BIG = windows_n_cols_BIG * 0.25;
    int stepSlideRow_SMALL = windows_n_rows_SMALL * 0.5;
	int stepSlideCols_SMALL = windows_n_cols_SMALL * 0.5;
	
	std::vector<std::vector<int>> coordinates_bb;
	std::vector<float> probabilities;
	
	for (int i=0; i<2; i++) 
	{
	    switch (i)
	    {
	        case 0:
		    {    
		        
		        if ( (img.cols>=1280 && img.rows>=720 && (maxArea>75000 || maxArea<8000)) || (img.cols<1280 && img.rows<720 && maxArea>1000 && maxArea<6000) )
		        {
		           break;
		        }
		        else
		        {
	                cout << "Starting Case 0" << endl;
		        
		            //__________________________ Sliding window approach __________________________//
		            detect.slidingWindow(img, img_threshold, windows_n_rows_BIG, windows_n_cols_BIG, stepSlideRow_BIG, stepSlideCols_BIG, coordinates_bb, maxArea);
		            
	                break;  
                }
                   
            }
            
            case 1:
            {
            
		        if ( img.cols<1280 && img.rows<720 && (maxArea<1000 || maxArea>6000) )
		        {
		           break;
		        }
		        else
		        {
                    cout << "Starting Case 1" << endl;
                    
                    //__________________________ Sliding window approach __________________________//
                    detect.slidingWindow(img, img_threshold, windows_n_rows_SMALL, windows_n_cols_SMALL, stepSlideRow_SMALL, stepSlideCols_SMALL, coordinates_bb, maxArea);
                    
                    break;
                }
            
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
    detect.nonMaximumSuppression(coordinates_bb, new_coordinates_bb);
    
    cout << "Remaining coordinates after nonMaximumSuppression" << endl;
    
  	// Displaying the 2D vector
    for (int i = 0; i < new_coordinates_bb.size(); i++) 
    {
        for (int j = 0; j < new_coordinates_bb[i].size(); j++)
            cout << new_coordinates_bb[i][j] << " ";
        cout << endl;
    }
    
    //__________________________________________ Handle BoundingBoxes Inside since Non Maximum Suppression can not __________________________________________//
       
    detect.handleBoundingBoxesInside(new_coordinates_bb);
        
    // detect.joinBoundingBoxesHorizontally(new_coordinates_bb);
    // detect.joinBoundingBoxesVertically(new_coordinates_bb);
    
    cout << "Remaining coordinates after joining Bounding Boxes" << endl;
    
  	// Displaying the 2D vector
    for (int i = 0; i < new_coordinates_bb.size(); i++) 
    {
        for (int j = 0; j < new_coordinates_bb[i].size(); j++)
            cout << new_coordinates_bb[i][j] << " ";
        cout << endl;
    }
    
    detect.deleteRedundantBB(img, new_coordinates_bb);
    
    cout << "Remaining coordinates after removing redundant Bounding Boxes" << endl;
    
  	// Displaying the 2D vector
    for (int i = 0; i < new_coordinates_bb.size(); i++) 
    {
        for (int j = 0; j < new_coordinates_bb[i].size(); j++)
            cout << new_coordinates_bb[i][j] << " ";
        cout << endl;
    }
    
    detect.joinBoundingBoxesHorizontally(new_coordinates_bb);
    
    detect.deleteAlignBB(new_coordinates_bb);
    
    cout << "Remaining coordinates after removing aligned Bounding Boxes" << endl;
    
  	// Displaying the 2D vector
    for (int i = 0; i < new_coordinates_bb.size(); i++) 
    {
        for (int j = 0; j < new_coordinates_bb[i].size(); j++)
            cout << new_coordinates_bb[i][j] << " ";
        cout << endl;
    }
    
    //__________________________________________ Draw random color Bounding Boxes __________________________________________//
	int n_hands = new_coordinates_bb.size();
	
    cv::RNG rng(12345); // warning, it's a class
    
    for (int i=0; i<n_hands; i++) 
	{ 
        cv::Scalar random_color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        
        int x1 = new_coordinates_bb[i][0];
        int y1 = new_coordinates_bb[i][1];
        int x2 = new_coordinates_bb[i][0] + new_coordinates_bb[i][2];
        int y2 = new_coordinates_bb[i][1] + new_coordinates_bb[i][3];
        
        cv::Point p1(x1, y1);
        cv::Point p2(x2, y2);
        
        rectangle(img, p1, p2, random_color, 2, cv::LINE_8);
	}
	
	// save images with bounding boxes
	cv::imwrite("../results/resultsDetection/Color/" + image_number + ".jpg", img);
	
		
	// Delete last column (confidence) for consistency with ground truth and Write results
	int columnToDelete = 4;

    for (int i=0; i < new_coordinates_bb.size(); ++i)
    {
        if (new_coordinates_bb[i].size() > columnToDelete)
        {
            new_coordinates_bb[i].erase(new_coordinates_bb[i].begin() + columnToDelete);
        }
    }
    
    cout << "Remaining coordinates after removing confidence column" << endl;
    
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

