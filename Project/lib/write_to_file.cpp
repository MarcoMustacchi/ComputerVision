/**
 * @file write_to_file.cpp
 *
 * @brief  Write results to file
 *
 * @author Marco Mustacchi
 *
 */

#include "write_to_file.h"
#include <iostream>
#include <fstream>


void write_results_Detection(float value)
{	
    
  std::ofstream myfile;
  myfile.open("../results/Performance/performanceDetection.txt");
  myfile << value << std::endl;
  myfile.close();
    
}

void write_results_Segmentation(float value)
{	
    
  std::ofstream myfile;
  myfile.open("../results/Performance/performanceSegmentation.txt");
  myfile << value << std::endl;
  myfile.close();
    
}

