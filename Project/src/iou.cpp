
/**
 * @file iou.cpp
 *
 * @brief  Performance evaluation: Intersection over Union
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
#include "read_sort_BB_matrix.h"


//_____________________________________________ Functions _____________________________________________//

float bb_intersection_over_union(int x_truth, int y_truth, int width_truth, int height_truth, int x_predict, int y_predict, int width_predict, int height_predict)
{
    
    // coordinates of the intersection Area
    int xA = std::max(x_truth, x_predict);
    int yA = std::max(y_truth, y_predict);
    int xB = std::min(x_truth+width_truth, x_predict+width_predict);
    int yB = std::min(y_truth+height_truth, y_predict+height_predict);
    
    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    
    int area_box_truth = width_truth * height_truth;
    int area_box_predict = width_predict * height_predict;
    
    float iou = (float) interArea / (area_box_truth + area_box_predict - interArea);
    
    // write_results_Detection(iou);
    
    return iou;
    
}


void iou(cv::Mat& img)
{
            
    std::string image_number;

    std::cout << "Insert image number from 01 to 30" << std::endl;
    std::cin >> image_number; 
    
    //___________________________ Load Dataset image ___________________________ //
        
    img = cv::imread("../Dataset/rgb/" + image_number + ".jpg", cv::IMREAD_COLOR);
    cv::namedWindow("Original Image");
    cv::imshow("Original Image", img);
    cv::waitKey(0);
    
    //___________________________ Load Dataset bounding box coordinates ___________________________ //
    
    std::string filename_dataset = "../Dataset/det/" + image_number + ".txt";

    std::vector<std::vector<int> > coord_bb_truth;
    
    coord_bb_truth = read_sort_BB_matrix(filename_dataset);
    
    int n_hands = coord_bb_truth.size(); // return number of rows
    std::cout << "Number of hands ground truth are " << n_hands << std::endl;
    
    //___________________________ Load Predicted bounding box coordinates ___________________________ //
    
    std::string filename_predict = "../results/resultsDetection/BoundingBoxes/" + image_number + ".txt";

    std::vector<std::vector<int> > coord_bb_predict;
    
    coord_bb_predict = read_sort_BB_matrix(filename_predict);
    
    
    int n_hands_predict = coord_bb_predict.size(); // return number of rows
    std::cout << "Number of hands predicted are " << n_hands_predict << std::endl;
    
    //___________________________ Draw Ground Thruth Bounding Boxes ___________________________ //
    
    int x_truth, y_truth, width_truth, height_truth;
    int x_predict, y_predict, width_predict, height_predict;
    
    float iou;
    
    // std::ofstream myfile;
    // myfile.open("../results/Performance/performanceDetection.txt");
    
    std::ofstream myfile("../results/performanceDetection/" + image_number + ".txt", std::ofstream::trunc); // to OverWrite text file
    
    int mode = 0;
    
    if (n_hands >= n_hands_predict)
		mode = 1;
	else 
		mode = 2;
		
			
	switch (mode)
	{
		case 1:
			for (int i=0; i<n_hands_predict; i++) 
   			{
		    	float max = 0.0f;
		    	int index = 0;
		    			    	
		    	for (int j=0; j<n_hands; j++) 
		        {
		        	// _________ Draw Detected Bounding Boxes _________//
			    	x_predict = coord_bb_predict[i][0];
			        y_predict = coord_bb_predict[i][1];
			        width_predict = coord_bb_predict[i][2];
			        height_predict = coord_bb_predict[i][3];
			        
		        	//_________ Draw Ground Thruth Bounding Boxes _________//
			    	x_truth = coord_bb_truth[j][0];
			        y_truth = coord_bb_truth[j][1];
			        width_truth = coord_bb_truth[j][2];
			        height_truth = coord_bb_truth[j][3];

			        // _________ Compute IoU measurements _________//
			    	iou = bb_intersection_over_union(x_truth, y_truth, width_truth, height_truth, x_predict, y_predict, width_predict, height_predict);
			    	
			    	if ( iou > max )
			    	{
			   			index = j;
			   			max = iou;
					}
		        }
		        
			    myfile << max << std::endl;
				std::cout << "IoU bounding box " << i+1 << " is: " << max << std::endl;   
		        coord_bb_truth.erase(coord_bb_truth.begin()+index);
		        n_hands = n_hands - 1;
		        
	    	}
	    	
	    	break;
			
		case 2:
			for (int i=0; i<n_hands; i++) 
   			{
		    	float max = 0.0f;
		    	int index = 0;
		    			    	
		    	for (int j=0; j<n_hands_predict; j++) 
		        {
		        	//_________ Draw Ground Thruth Bounding Boxes _________//
			    	x_truth = coord_bb_truth[i][0];
			        y_truth = coord_bb_truth[i][1];
			        width_truth = coord_bb_truth[i][2];
			        height_truth = coord_bb_truth[i][3];
			    
			        // _________ Draw Detected Bounding Boxes _________//
			    	x_predict = coord_bb_predict[j][0];
			        y_predict = coord_bb_predict[j][1];
			        width_predict = coord_bb_predict[j][2];
			        height_predict = coord_bb_predict[j][3];
			        
			        // _________ Compute IoU measurements _________//
			    	iou = bb_intersection_over_union(x_truth, y_truth, width_truth, height_truth, x_predict, y_predict, width_predict, height_predict);
			    	
			    	if ( iou > max )
			    	{
			   			index = j; 
			   			max = iou;
					}
		        }
		        
		        myfile << max << std::endl;
				std::cout << "IoU bounding box " << i+1 << " is: " << max << std::endl;  
		        coord_bb_predict.erase(coord_bb_predict.begin()+index);
		        n_hands_predict = n_hands_predict - 1;
	    	}
	    	
	    	break;
			
	}
	
	myfile.close();
  
}


