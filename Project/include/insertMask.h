#ifndef INSERTMASK_H
#define INSERTMASK_H

void insertBinaryMask(const cv::Mat& img, const std::vector<cv::Mat>& mask_final_ROI, cv::Mat& mask_final, const std::vector<std::vector<int>> coordinates_bb, int n_hands);	

void insertColorMask(const cv::Mat& img, const std::vector<cv::Mat>& img_ROI_color, cv::Mat& mask_color_final, const std::vector<std::vector<int>> coordinates_bb, int n_hands);

#endif
