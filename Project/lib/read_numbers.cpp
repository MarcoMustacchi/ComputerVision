/**
 * @file read_numbers.cpp
 *
 * @brief  Read coordinates of the bounding boxes
 *
 * @author Marco Mustacchi
 *
 */

#include <iostream>
#include <fstream>
#include <vector>


std::vector<int> read_numbers(std::string file_name)
{
    std::ifstream infile;
    infile.open(file_name);
    std::vector<int> numbers;

    if (infile.is_open())
    {
        int num; 
        while(infile >> num)
        {
            numbers.push_back(num);
        }
    }

    return numbers;
}

