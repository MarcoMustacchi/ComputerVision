/**
 * @file Project.cpp
 *
 * @brief  Hand Segmentation
 *
 * @author Marco Mustacchi
 *
 */

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <vector>


//_____________________________________________ Functions _____________________________________________//

std::vector<int> read_numbers(std::string file_name)
{
    std::ifstream infile;
    infile.open(file_name);
    std::vector<int> numbers;

    if (infile.is_open())
    {
        int num; 
        while(infile >> num)
        {
            numbers.push_back(num);
        }
    }

    return numbers;
}


int main(int argc, char* argv[])
{
	
	//___________________________ Load Dataset bounding box coordinates ___________________________ //
	
	std::vector<int> coordinates_bb;
	
	coordinates_bb = read_numbers("../Dataset/det/01.txt");
	
	for (int i=0; i<coordinates_bb.size(); ++i)
    	std::cout << coordinates_bb[i] << ' ';
	
	//___________________________ Load Dataset image ___________________________ //
		
	cv::Mat img = cv::imread("../Dataset/rgb/01.jpg", cv::IMREAD_COLOR);
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", img);
	cv::waitKey(0);
	
	//___________________________ Draw Bounding Boxes ___________________________ //
	int x = coordinates_bb[0];
	int y = coordinates_bb[1];
	int width = coordinates_bb[2];
	int height = coordinates_bb[3];
	
	cv::Point pt1(x, y);
    cv::Point pt2(x + width, y + height);
    cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0));
	
	cv::namedWindow("New Image");
	cv::imshow("New Image", img);
	cv::waitKey(0);
  
	//___________________________ ROI extraction ___________________________//
	
	cv::Range rows(x, x+width);
    cv::Range cols(y, y+height);
	cv::Mat img_roi = img(cols, rows);
	
  	cv::namedWindow("ROI");
	cv::imshow("ROI", img_roi);
	cv::waitKey(0);
	
	//___________________________ ROI segmentation Otsu thresholding ___________________________//	
	
    cv::Mat img_roi_gray;
	cv::cvtColor(img_roi, img_roi_gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat blur;
    cv::GaussianBlur(img_roi_gray,blur,cv::Size(5,5),0);
    
    cv:: Mat img_roi_thr;
	double th = cv::threshold(blur,img_roi_thr,0,255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	
    std::cout << "Otsu method threshold after Gaussian Filter: " << th << std::endl;
    
    cv::namedWindow("Otsu's thresholding after Gaussian filtering");
	cv::imshow("Otsu's thresholding after Gaussian filtering", img_roi_thr);
	cv::waitKey(0);
  
    cv::destroyAllWindows();
    
    //___________________________ ROI segmentation CallBack adaptive thresholding ___________________________//
    
    cv::Mat img_roi_HSV;
    cv::cvtColor(img_roi, img_roi_HSV, cv::COLOR_BGR2HSV);
    
    cv::namedWindow("ROI HSV");
	cv::imshow("ROI HSV", img_roi_HSV);
	cv::waitKey(0);
	
	// BGR
    cv::Mat Bands_RGB[3];
    cv::Mat merged;
    cv::split(img_roi, Bands_RGB);
    std::vector<cv::Mat> channels_BGR = {Bands_RGB[0],Bands_RGB[1],Bands_RGB[2]};
    
    // HSV
    cv::Mat Bands_HSV[3];
    // cv::Mat merged;
    cv::split(img_roi_HSV, Bands_HSV);
    std::vector<cv::Mat> channels_HSV = {Bands_HSV[0],Bands_HSV[1],Bands_HSV[2]};
    
    // Results
    cv::imshow("Blue", Bands_RGB[0]);
    cv::imshow("Green", Bands_RGB[1]);
    cv::imshow("Red", Bands_RGB[2]);
    
    cv::imshow("Hue", Bands_HSV[0]);
    cv::imshow("Saturation", Bands_HSV[1]);
    cv::imshow("Intensity", Bands_HSV[2]);
    
	cv::waitKey(0);
    
    cv::merge(channels_HSV, merged);
    cv::imshow("merged",merged);
    
	cv::waitKey(0);
	
	/*
	cv::namedWindow( "Edge Map", cv::WINDOW_AUTOSIZE );
	cv::createTrackbar( "Low Threshold:", "Edge Map", &lowThreshold, max_lowThreshold, MyCallbackForThreshold, &img_roi_HSV ); // puo' avere anche il parametro void* userdata per callback
	MyCallbackForThreshold(0, &img_roi_HSV);
	
    //Create track bar to change maxThreshold
    cv::createTrackbar( "High Threshold:", "Edge Map", &highThreshold, max_highThreshold, MyCallbackForThreshold, &img_roi_HSV );
	*/
	
  
	return 0;
  
}


