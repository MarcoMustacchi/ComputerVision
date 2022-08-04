
/**
 * @file segmentationAlgorithm.h
 *
 * @brief  segmentationAlgorithm Header file
 *
 * @author Marco Mustacchi
 *
 */
 
#ifndef SEGMENTATIONALGORITHM_H
#define SEGMENTATIONALGORITHM_H

// void inRangeSegmentation(const cv::Mat& img, std::vector<cv::Mat>& img_roi_BGR, std::vector<cv::Mat>& mask_inRange_ROI);
void inRangeSegmentation(const std::vector<cv::Mat>& img_roi_BGR, std::vector<cv::Mat>& mask_inRange_ROI, std::vector<std::vector<int>> coordinates_bb, int n_hands);
void OtsuSegmentation(const cv::Mat& img, std::vector<cv::Mat>& img_roi_BGR, std::vector<cv::Mat>& mask_Otsu_ROI, std::vector<std::vector<int>> coordinates_bb, int n_hands);
void CannySegmentation(const cv::Mat& img, std::vector<cv::Mat>& img_roi_BGR, std::vector<cv::Mat>& mask_Canny_ROI, std::vector<std::vector<int>> coordinates_bb, int n_hands);
void RegionGrowingSegmentation(cv::Mat& img, std::vector<cv::Mat> img_roi, std::vector<cv::Mat>& mask_RegionGrowing_ROI, std::vector<std::vector<int>> coordinates_bb, int n_hands);

#endif
