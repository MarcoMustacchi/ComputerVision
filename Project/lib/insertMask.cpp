
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include "insertMask.h" 


void insertBinaryMask(const cv::Mat& img, const std::vector<cv::Mat>& mask_final_ROI, cv::Mat& mask_final, const std::vector<std::vector<int>> coordinates_bb, int n_hands) {	
		
	// metto ROI in immagine nera stesse dimensioni originale
	std::vector<cv::Mat> mask_OriginalDim(n_hands);

    for (int i=0; i<n_hands; i++) 
    {
        mask_OriginalDim[i] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
        mask_final_ROI[i].copyTo(mask_OriginalDim[i](cv::Rect(coordinates_bb[i][0], coordinates_bb[i][1], mask_final_ROI[i].cols, mask_final_ROI[i].rows)));
        cv::namedWindow("Hand mask");
        cv::imshow("Hand mask", mask_OriginalDim[i]);
	    cv::waitKey(0);
    } 
	
    for (int i=0; i<n_hands; i++) 
    {
        cv::bitwise_or(mask_OriginalDim[i], mask_final, mask_final);
    }
    
	cv::namedWindow("Mask final");
    cv::imshow("Mask final", mask_final);
	cv::waitKey(0);	
}


void insertColorMask(const cv::Mat& img, const std::vector<cv::Mat>& img_ROI_color, cv::Mat& mask_color_final, const std::vector<std::vector<int>> coordinates_bb, int n_hands) {
	
	// metto ROI colorata in immagine nera stesse dimensioni originale	
	std::vector<cv::Mat> mask_color_OriginalDim(n_hands);
		
    for (int i=0; i<n_hands; i++) 
    {
        mask_color_OriginalDim[i] = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
        img_ROI_color[i].copyTo(mask_color_OriginalDim[i](cv::Rect(coordinates_bb[i][0], coordinates_bb[i][1], img_ROI_color[i].cols, img_ROI_color[i].rows)));
        cv::namedWindow("Hand color mask");
        cv::imshow("Hand color mask", mask_color_OriginalDim[i]);
	    cv::waitKey(0);
    }
	
    for (int i=0; i<n_hands; i++) 
    {
        cv::bitwise_or(mask_color_OriginalDim[i], mask_color_final, mask_color_final);
    }

	cv::namedWindow("Mask color final");
    cv::imshow("Mask color final", mask_color_final);
	cv::waitKey(0);
}
