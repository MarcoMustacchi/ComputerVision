/**
 * @file iou.cpp
 *
 * @brief  Pixel Accuracy
 *
 * @author Marco Mustacchi
 *
 */

#include "write_to_file.h"
#include <iostream>
#include <fstream>


void write_to_file(float value)
{	
    
  std::ofstream myfile;
  myfile.open("../results/results.txt");
  myfile << value << std::endl;
  myfile.close();
    
}




