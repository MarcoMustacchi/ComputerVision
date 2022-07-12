
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include "randomColorMask.h" 	
	
void randomColorMask(const std::vector<cv::Mat>& mask_final_ROI, std::vector<cv::Mat>& img_ROI_color, std::vector<cv::Scalar>& random_color, int n_hands) {
	
	//_____________________________ generate random color  _____________________________//
	cv::RNG rng(12345); // warning, it's a class
	
	for (int i=0; i<n_hands; i++) 
	{
        random_color[i] = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        std::cout << "Random color " << random_color[i] << std::endl;
        
        cv::cvtColor(mask_final_ROI[i], img_ROI_color[i], cv::COLOR_GRAY2RGB);
	}
	
	
	//______________ color the mask cambiando ogni singolo canale con rispettivo colore ________________//
	
	for (int k=0; k<n_hands; k++) {
	    for(int i=0; i<img_ROI_color[k].rows; i++) {
            for(int j=0; j<img_ROI_color[k].cols; j++) {
                if(mask_final_ROI[k].at<uchar>(i,j) == 255) {
                    img_ROI_color[k].at<cv::Vec3b>(i,j)[0] = random_color[k][0];
                    img_ROI_color[k].at<cv::Vec3b>(i,j)[1] = random_color[k][1];
                    img_ROI_color[k].at<cv::Vec3b>(i,j)[2] = random_color[k][2];
                }
            }
        }
    cv::namedWindow("Final random");
	cv::imshow("Final random", img_ROI_color[k]);
	cv::waitKey(0);
	} 
	
}
