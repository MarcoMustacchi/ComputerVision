#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat minFilter(cv::Mat& img, int sizeKernel);

cv::Mat maxFilter(cv::Mat& img, int sizeKernel);


#endif


