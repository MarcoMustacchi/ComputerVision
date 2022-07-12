/**
 * @file drawBB.cpp
 *
 * @brief  Draw detected bounding boxes in the Dataset images
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
#include <opencv2/core/utils/filesystem.hpp> //include per utils::fs::glob, anche se ho gia' core.hpp
#include "write_to_file.h"
#include "read_BB_matrix.h"


int main(int argc, char* argv[])
{
	
    //___________________________ Load path input Dataset images ___________________________ //
    
    std::vector<cv::String> input_image_path;

    cv::utils::fs::glob ("../Dataset/rgb/",
                    "*.jpg",
                    input_image_path,
                    false,
                    false 
                    );	
      
    //___________________________ Load path output bounding boxes images ___________________________ //      
    // warning, need to create before copy of the images file with the correct names in the destination folder, so overwrite
    
    std::vector<cv::String> output_image_path;

    cv::utils::fs::glob ("../results/bb/",
                    "*.jpg",
                    output_image_path,
                    false,
                    false 
                    );	
    
    //___________________________ Load path output coordinates Dataset images ___________________________ //      
    std::vector<cv::String> coordinates_number;
    cv::utils::fs::glob ("../Dataset/det/",
                    "*.txt",
                    coordinates_number,
                    false,
                    false 
                    );	
     
    //___________________________ Boh ___________________________ //          
    int tot_images = input_image_path.size();
	      
    for (int i=0; i<tot_images; i++) { 
        std::cout << input_image_path[i] << std::endl; // in image number I have all the path
    }                 

	std::vector<cv::Mat> images(tot_images);
	
	for (int i = 0; i < tot_images; i++) {
		images[i] = cv::imread(input_image_path[i]);
		if (images[i].empty()) {
			std::cout << "Immagine " + input_image_path[i] + " not found" << std::endl;
			cv::waitKey(0);
			return -1;
		};
	};
	
	
    for (int i=0; i<tot_images; i++) 
    {
	    //___________________________ Load current image bounding box coordinates ___________________________ //
	    
	    std::string filename_dataset = coordinates_number[i];

	    std::vector<std::vector<int>> coord_bb_truth;
	    coord_bb_truth = read_sort_BB_matrix(filename_dataset);
	    
	    int n_hands = coord_bb_truth.size(); // return number of rows
	    std::cout << "Number of hands detected are " << n_hands << std::endl;
	
	    
	    //_____________________________ generate random color and draw bounding box _____________________________//
	    
	    cv::RNG rng(12345); // warning, it's a class
	    
	    for (int n=0; n<n_hands; n++) 
	    {
            cv::Scalar random_color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
            
            int x1 = int(coord_bb_truth[n][0]);
            int y1 = int(coord_bb_truth[n][1]);
            int x2 = int(coord_bb_truth[n][0]+coord_bb_truth[n][2]);
            int y2 = int(coord_bb_truth[n][1]+coord_bb_truth[n][3]);
            
            cv::Point p1(x1, y1);
            cv::Point p2(x2, y2);
            
            rectangle(images[i], p1, p2, random_color, 2, cv::LINE_8);
	    }
        
        cv::imwrite(output_image_path[i], images[i]);
	    
	}
	
	return 0;
	
}
