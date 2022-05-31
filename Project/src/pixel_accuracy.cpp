/**
 * @file iou.cpp
 *
 * @brief  Pixel Accuracy
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
#include "write_to_file.h"



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


void pixel_accuracy(cv::Mat mask, cv::Mat mask_otsu, int x, int y, int width, int height)
{

    int rows = mask.rows;
    int cols = mask.cols;

    cv::Mat prediction(rows, cols, CV_8UC1, cv::Scalar::all(0)); // must be one channel both, or 3 channel both per sommare / copiare sopra
    
	cv::imshow("Ground truth", mask);
	cv::waitKey(0);
    
    mask_otsu.copyTo(prediction(cv::Rect(x, y, mask_otsu.cols, mask_otsu.rows)));
    
	cv::imshow("Prediction", prediction);
	cv::waitKey(0);
	
	int totCorrect = 0;
	
	for (int i = 1; i <= mask.rows; i++) // in this way is a 9x9
    {
	    for (int j = 1; j <= mask.cols; j++)
	    {
	        if ( mask.at<uchar>(i,j) == mask_otsu.at<uchar>(i,j) ) 
	            totCorrect += 1;
	    }
    }
    
    int totArea = rows * cols;
    
    float pixelAccuracy = (float) totCorrect / totArea;
    
    std::cout << "Pixel accuracy is: " << pixelAccuracy << std::endl;
	
	write_to_file(pixelAccuracy);
	
}




int main(int argc, char* argv[])
{
	
	//___________________________ Load Dataset bounding box coordinates ___________________________ //
	
	std::vector<int> coordinates_bb;
	
	coordinates_bb = read_numbers("../Dataset/det/02.txt");
	
	for (int i=0; i<coordinates_bb.size(); ++i)
    	std::cout << coordinates_bb[i] << ' ';
    std::cout << std::endl;
	
	//___________________________ Load Dataset image ___________________________ //
		
	cv::Mat img = cv::imread("../Dataset/rgb/02.jpg", cv::IMREAD_COLOR);
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", img);
	cv::waitKey(0);
	
	//___________________________ Load Dataset mask ___________________________ //
		
	cv::Mat mask = cv::imread("../Dataset/mask/02.png", cv::IMREAD_COLOR);
	cv::namedWindow("Original mask");
	cv::imshow("Original mask", mask);
	cv::waitKey(0);
    
    int rows = img.rows;
    int cols = img.cols;
    
    std::cout << "Dim original " << rows << " and " << cols << std::endl;
    
    
	//___________________________ ROI extraction ___________________________//
	
	int x = coordinates_bb[0];
	int y = coordinates_bb[1];
	int width = coordinates_bb[2];
	int height = coordinates_bb[3];
	
	cv::Range colonna(x, x+width);
    cv::Range riga(y, y+height);
	cv::Mat img_roi = img(riga, colonna);
	
  	cv::namedWindow("ROI");
	cv::imshow("ROI", img_roi);
	cv::waitKey(0);
	
	//___________________________ ROI segmentation Otsu thresholding ___________________________//	
	
    cv::Mat img_roi_gray;
	cv::cvtColor(img_roi, img_roi_gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat roi_blur_gray;
    cv::GaussianBlur(img_roi_gray,roi_blur_gray,cv::Size(3,3),0);
    
    cv:: Mat img_roi_thr;
	double th = cv::threshold(roi_blur_gray,img_roi_thr,0,255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	
    std::cout << "Otsu method threshold after Gaussian Filter: " << th << std::endl;
    
    cv::namedWindow("Otsu's thresholding after Gaussian filtering");
	cv::imshow("Otsu's thresholding after Gaussian filtering", img_roi_thr);
	cv::waitKey(0);
  
    cv::destroyAllWindows();
    
    
    pixel_accuracy(mask, img_roi_thr, x, y, width, height);
    
    
    // generate random color and color the mask moltiplicando ogni singolo canale con rispettivo colore
    cv::Mat mask_otsu_color;
    
    cv::bitwise_and(img_roi, img_roi, mask_otsu_color, img_roi_thr);
    
    cv::namedWindow("Final");
	cv::imshow("Final", mask_otsu_color);
	cv::waitKey(0);
	
	cv::RNG rng(12345); // warning, it's a class
	cv::Scalar random_color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
	cv::Scalar random_color2 = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
	
	std::cout << "Random color " << random_color << std::endl;
	std::cout << "Random color " << random_color2 << std::endl;
	
    cv::Mat Bands_BGR[3];
    cv::Mat merged;
    cv::split(mask_otsu_color, Bands_BGR);
    
    Bands_BGR[0] = Bands_BGR[0] * random_color[0];
    Bands_BGR[1] = Bands_BGR[1] * random_color[1];
    Bands_BGR[2] = Bands_BGR[2] * random_color[2];
    
    std::vector<cv::Mat> channels_BGR;
	channels_BGR.push_back(Bands_BGR[0]);
	channels_BGR.push_back(Bands_BGR[1]);
	channels_BGR.push_back(Bands_BGR[2]);
    
    cv::merge(channels_BGR, merged);
    
    cv::namedWindow("Final random");
	cv::imshow("Final random", merged);
	cv::waitKey(0);
	
	
	// inserisci maschera in immagine nera stessa dimensione originale e somma con immagine originale
    cv::Mat prediction(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0)); 
    merged.copyTo(prediction(cv::Rect(x, y, merged.cols, merged.rows)));
	cv::imshow("Prediction", prediction);
	cv::waitKey(0);
	
	// back to original image
    cv::Mat ultima;
    ultima = img + prediction;
    
	cv::imshow("Boh", ultima);
	cv::waitKey(0);
  
	return 0;
  
}


