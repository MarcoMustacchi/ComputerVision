
/**
 * @file write_to_file.cpp
 *
 * @brief  Write results to file
 *
 * @author Marco Mustacchi
 *
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include "write_to_file.h"


void write_results_Detection(const std::vector<std::vector<int>>& new_coordinates_bb, std::string image_number)
{	
          
    std::ofstream myfile("../results/resultsDetection/BoundingBoxes/" + image_number + ".txt", std::ofstream::trunc); // to OverWrite text file
    // std::ofstream myfile;
    // myfile.open("../results/resultsDetection/" + image_number + ".txt");
    std::ostream_iterator<int> output_iterator(myfile, "\t");
    
    for (int i=0; i<new_coordinates_bb.size(); i++) 
    {
        copy(new_coordinates_bb.at(i).begin(), new_coordinates_bb.at(i).end(), output_iterator);
        myfile << '\n';
    }
    
}


void write_performance_Detection(float value, std::string image_number)
{	
    
    std::ofstream myfile("../results/performanceDetection/" + image_number + ".txt", std::ofstream::trunc); // to OverWrite text file
    // std::ofstream myfile;
    // myfile.open("../results/Performance/performanceDetection.txt");
    myfile << value << std::endl;
    myfile.close();
    
}


void write_performance_Segmentation(float value, std::string image_number)
{	
    
    std::ofstream myfile("../results/performanceSegmentation/" + image_number + ".txt", std::ofstream::trunc); // to OverWrite text file
    // std::ofstream myfile;
    // myfile.open("../results/Performance/performanceSegmentation.txt");
    myfile << value << std::endl;
    myfile.close();
    
}

