
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "fillMaskHoles.h"     
	
void fillMaskHoles(const cv::Mat& input, cv::Mat& output) {

	// Floodfill from point (0, 0) (background) -> check if it is black (no more since draw rectangle)
    // I have to put all the background connected to point (0,0) -> all extreme pixel contour black
        
    // Top Left Corner
    cv::Point p1(0, 0);
  
    // Bottom Right Corner
    cv::Point p2(input.cols, input.rows); // for rectangle rows and cols inverted
  
    
    int thickness = 2;
    // Drawing the Rectangle
    rectangle(input, p1, p2,
              cv::Scalar(0),
              thickness, cv::LINE_8);
        
    cv::imshow("corrected", input);
    cv::waitKey(0);   
    
    
    cv::Mat im_floodfill = input.clone();
    floodFill(im_floodfill, cv::Point(0,0), cv::Scalar(255));
    
    // Invert floodfilled image (get the holes)
    cv::Mat im_floodfill_inv;
    cv::bitwise_not(im_floodfill, im_floodfill_inv);

    // Combine the two images to get the foreground.
    output = input + im_floodfill_inv;

    // Display images
    cv::imshow("Foreground", output);
    cv::waitKey(0);
    
}
