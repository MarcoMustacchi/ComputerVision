
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
	
	if (detect.imgGrayscale(img) == false) 
	{
	    cout << "Color image" << endl;
	    cvtColor(img, img, COLOR_BGR2YCrCb);
	    detect.skinDetectionColored(img, img_threshold);
	    cvtColor(img, img, COLOR_YCrCb2BGR);
	}
	else {
		
	    cout << "Grayscale image" << endl;
	    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	    detect.skinDetectionGrayscale(img, img_threshold);   
	    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
	    
    	cv::namedWindow("mage");
	    cv::imshow("mage", img);
	    cv::waitKey(0);
	}   
			
		
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
	
	std::vector<std::vector<int>> coordinates_bb;
	std::vector<float> probabilities;
	
	for (int i=0; i<2; i++) 
	{
	    switch (i)
	    {
	        case 0:
		    {    
		        cout << "Starting Case 0" << endl;
		        
		        //__________________________ Sliding window approach __________________________//
		        detect.slidingWindow(img, img_threshold, windows_n_rows_BIG, windows_n_cols_BIG, stepSlideRow_BIG, stepSlideCols_BIG, coordinates_bb);

                break;     
            }
            
            case 1:
            {
            
                cout << "Starting Case 1" << endl;
                
                //__________________________ Sliding window approach __________________________//
                detect.slidingWindow(img, img_threshold, windows_n_rows_SMALL, windows_n_cols_SMALL, stepSlideRow_SMALL, stepSlideCols_SMALL, coordinates_bb);
                
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
    detect.nonMaximumSuppression(coordinates_bb, new_coordinates_bb);
    
        
    cout << "remaining coordinates" << endl;
    
  	// Displaying the 2D vector
    for (int i = 0; i < new_coordinates_bb.size(); i++) 
    {
        for (int j = 0; j < new_coordinates_bb[i].size(); j++)
            cout << new_coordinates_bb[i][j] << " ";
        cout << endl;
    }
    
    //_____________ remove Bounding Box if is fully inside another one (non Maximum Suppression doesn't remove it) _____________//
    std::vector<int> indices;
    
    for (int i=0; i<new_coordinates_bb.size(); i++) 
    {
        for (int j=0; j<new_coordinates_bb.size(); j++) 
        {
            if (new_coordinates_bb[i] == new_coordinates_bb[j]) // altrimenti stesso rettangolo confrontato con se stesso risulta inside
            {
                continue;
            }
            
            cv::Rect a(new_coordinates_bb[i][0], new_coordinates_bb[i][1], new_coordinates_bb[i][2], new_coordinates_bb[i][3]);
            cv::Rect b(new_coordinates_bb[j][0], new_coordinates_bb[j][1], new_coordinates_bb[j][2], new_coordinates_bb[j][3]);
                        
            if ((a & b) == a) // means that b is inside a
            {
                cout << "Inside " << i << endl;               
                indices.push_back(i);   // sembra funzionare, ma a logica non dovrebbe essere j? visto che a contiene b?
            }
        }
    }
    
    // attenzione, perche' in questo modo se piu' rettangoli contengono lo stesso rettangolo, devo rimuovere tutti gli indici uguali,
    // altrimenti rimuovo piu' volte
    
    sort( indices.begin(), indices.end() );
    indices.erase( unique( indices.begin(), indices.end() ), indices.end() );
    
    for (int j = 0; j < indices.size(); j++)
        cout << indices[j] << " ";

    
    for (int i=0; i<indices.size(); i++) 
    {
        new_coordinates_bb.erase(new_coordinates_bb.begin()+indices[i]);
    }
    
    
    //__________________________________________ Draw random color Bounding Boxes __________________________________________//
	int n_hands = new_coordinates_bb.size();
	
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
	
		
	// Delete last column (confidence) for consistency with ground truth and Write results
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

