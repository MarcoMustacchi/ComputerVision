/**
 * @file read_BB_matrix.cpp
 *
 * @brief  Read coordinates Bounding Box and sort them
 *
 * @author Marco Mustacchi
 *
 */

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "read_BB_matrix.h"

// Driver function to sort the 2D vector on basis of a particular column
bool sortcol(const std::vector<int>& v1, const std::vector<int>& v2)
{
    return v1[0] < v2[0];
}


std::vector<std::vector<int>> read_sort_BB_matrix(std::string file_name)
{
    std::ifstream inFile;
    inFile.open(file_name);

    std::string line;
    int word;
    
    //create a 2D vector that will store the read information
    std::vector<std::vector<int>> vec;
    
    if(inFile)
    {   //read line by line
        while(getline(inFile, line, '\n'))        
        {
            //create a temporary vector that will contain all the columns
            std::vector<int> tempVec;
            
            std::istringstream ss(line);
            
            //read word by word(or int by int) 
            while(ss >> word)
            {
                //add the word to the temporary vector 
                tempVec.push_back(word);
            }      
            //now all the words from the current line has been added to the temporary vector 
            vec.emplace_back(tempVec);
        }    
    }
    else 
    {
        std::cout<<"file cannot be opened"<<std::endl;
    }
    
    inFile.close();
    
    //lets check out the elements of the 2D vector so the we can confirm if it contains all the right elements(rows and columns)
    std::cout << "Original coordinates" << std::endl;
        
    for(std::vector<int> &newvec: vec)
    {
        for(const int &elem: newvec)
        {
            std::cout<<elem<<" ";
        }
        std::cout<<std::endl;
    }
    
    
    //____________________________ Sort rows by first column ____________________________//
    // Use of "sort()" for sorting on basis of 1st column
    sort(vec.begin(), vec.end(), sortcol);
    
    //lets check out the elements of the 2D vector are sorted correctly
    std::cout << "Ordered coordinates" << std::endl;
    
    for(std::vector<int> &newvec: vec)
    {
        for(const int &elem: newvec)
        {
            std::cout<<elem<<" ";
        }
        std::cout<<std::endl;
    }
    
    return vec;
    
}


