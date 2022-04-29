#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

int main(int argc, char** argv)
{
	
	cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR); 
	cv::namedWindow("Example 1");
	cv::imshow("Example 1", img);
	
	cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_COLOR); 
	cv::namedWindow("Example 2");
	cv::imshow("Example 2", img2);
	cv::waitKey(0);
	
	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
	
	//____________________________ Harris Corner detector ____________________________ 
	cv::Mat img_Harris(img.rows, img.cols, CV_32FC1);
    cv::cornerHarris(img_gray, img_Harris, 2, 3, 0.04);
    
    cv::dilate(img_Harris, img_Harris, 5);
    
    cv::Mat dst_norm, dst_norm_scaled;
    cv::normalize( img_Harris, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    cv::convertScaleAbs( dst_norm, dst_norm_scaled );

    double minVal; 
    double maxVal; 
    cv::Point minLoc; 
    cv::Point maxLoc;
    minMaxLoc( img_Harris, &minVal, &maxVal, &minLoc, &maxLoc );
    
    int thresh = 200;
    int nFeatures;
    
    for (int r = 0; r < img.rows; r++) {
        for (int c = 0; c < img.cols; c++) {
            if ( (int) dst_norm.at<float>(r,c) > thresh ) {
                cv::circle( img, cv::Point(c,r), 5,  cv::Scalar(0), 2, 8, 0 ); // warning, point coordinates are different wrt Mat img
                nFeatures++;
            }
        } 
    }
    
    std::cout << "Number of features found: " << nFeatures << std::endl;
    
	cv::namedWindow("Harris Corner"); //added comment test
	cv::imshow("Harris Corner", img);
	cv::waitKey(0);
	
	return 0;
    	
}
