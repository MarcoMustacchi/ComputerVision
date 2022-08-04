
/**
 * @file Detector.h
 *
 * @brief  Detector Header file
 *
 * @author Marco Mustacchi
 *
 */
 
#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <map>
#include "read_sort_BB_matrix.h"
#include "removeOutliers.h"
#include "fillMaskHoles.h"
#include "write_to_file.h"
#include "fillMaskHoles.h" 


using namespace cv;
using namespace std;

class Detector
{
    
    public:
    
        bool imgGrayscale(const cv::Mat& img);
    
        int skinDetectionColored(const cv::Mat& img, cv::Mat& img_threshold);
    
        int skinDetectionGrayscale(const cv::Mat& img, cv::Mat& img_threshold);
        
        int removeDetectionOutliers(cv::Mat& input);
        
        void slidingWindow(cv::Mat& img, cv::Mat& img_threshold, int windows_n_rows, int windows_n_cols, int stepSlideRow, int stepSlideCols, std::vector<std::vector<int>>& coordinates_bb, int maxArea);
        
        float detectionIoU(std::vector<int> loop_coordinates_bb_old, std::vector<int> loop_coordinates_bb_new);
        
        void convert2originalXYWH(const std::tuple<int, int, int, int>& resizedCoordinates, const std::tuple<int, int>& resizedDimension, 
                            std::tuple<int, int, int, int>& originalCoordinates, const std::tuple<int, int>& orginalDimensions); 
                            
        void nonMaximumSuppression(std::vector<std::vector<int>>& old_coordinates_bb, std::vector<std::vector<int>>& new_coordinates_bb);
                            
        void handleBoundingBoxesInside(std::vector<std::vector<int>>& new_coordinates_bb);   
        
        void joinBoundingBoxesHorizontally(std::vector<std::vector<int>>& new_coordinates_bb);  
        
        void joinBoundingBoxesVertically(std::vector<std::vector<int>>& new_coordinates_bb);                

        void deleteRedundantBB(const cv::Mat img, std::vector<std::vector<int>>& new_coordinates_bb);
        
        void deleteAlignBB(std::vector<std::vector<int>>& new_coordinates_bb);
        
};

#endif

