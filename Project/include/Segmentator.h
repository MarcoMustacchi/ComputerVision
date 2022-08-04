
/**
 * @file Segmentator.h
 *
 * @brief  Segmentator Header file
 *
 * @author Marco Mustacchi
 *
 */
 
#ifndef SEGMENTATOR_H
#define SEGMENTATOR_H

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
#include "write_to_file.h"
#include "fillMaskHoles.h" 


using namespace cv;
using namespace std;

class Segmentator
{
    
    public:
    
        void randomColorMask(const std::vector<cv::Mat>& mask_final_ROI, std::vector<cv::Mat>& img_ROI_color, std::vector<cv::Vec3b>& random_color, int n_hands);

        void insertBinaryMask(const cv::Mat& img, const std::vector<cv::Mat>& mask_final_ROI, cv::Mat& mask_final, const std::vector<std::vector<int>> coordinates_bb, int n_hands);

        void insertColorMask(const cv::Mat& img, const std::vector<cv::Mat>& img_ROI_color, cv::Mat& mask_color_final, const std::vector<std::vector<int>> coordinates_bb, int n_hands, 
            std::vector<cv::Vec3b>& random_color, cv::Mat& mask_Watershed, bool overlap);

        void displayMultiple(const cv::Mat& img, const cv::Mat& mask1, const cv::Mat& mask2, const cv::Mat& mask3, const cv::Mat& mask4);
        
        std::vector<cv::Mat> chooseBestMask(cv::Mat& mask_ground_truth, cv::Mat& mask_inRange, cv::Mat& mask_Otsu, cv::Mat& mask_Canny, cv::Mat& mask_RegionGrowing, 
            std::vector<cv::Mat> mask_inRange_ROI, std::vector<cv::Mat> mask_Otsu_ROI, std::vector<cv::Mat> mask_Canny_ROI, std::vector<cv::Mat> mask_RegionGrowing_ROI); 
            
        void insertBinaryROI(const cv::Mat& img, const cv::Mat& mask_final_ROI, cv::Mat& mask_final_ROI_original_dim, const std::vector<int> coordinates_bb, int n_hands); 
            
        std::vector<cv::Mat> chooseBestROIMask(const cv::Mat& img, std::vector<cv::Mat> mask_inRange_ROI, std::vector<cv::Mat> mask_Otsu_ROI, std::vector<cv::Mat> mask_Canny_ROI, 
            std::vector<cv::Mat> mask_RegionGrowing_ROI, const std::vector<std::vector<int>>& coordinates_bb, std::string image_number, int n_hands, std::vector<cv::Mat>& mask_final_ROI); 
            
        bool detectOverlap(std::vector<std::vector<int>> coordinates_bb); 
        
        void handleOverlap(cv::Mat& mask_final, cv::Mat& mask_Watershed, std::vector<std::vector<int>> coordinates_bb, std::vector<cv::Vec3b>& random_color, int n_hands);             
};

#endif
