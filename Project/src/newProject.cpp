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
#include "read_numbers.h"



//_____________________________________________ Functions _____________________________________________//

void otsuSegmentation(const cv::Mat& input, cv::Mat& mask, const int ksize, int color_space) 
{

    cv::Mat gray, temp;
    
    if (color_space == 1) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else if (color_space == 2) {
        cv::Mat hsv_channels[3];
        cv::split( input, hsv_channels );
        gray = hsv_channels[2]; // 3 channel of HSV is gray
    } else {
        cv::Mat ycbcr_channels[3];
        cv::split( input, ycbcr_channels );
        gray = ycbcr_channels[0]; // 3 channel of HSV is gray
    }
    
    cv::imshow("Before preprocessing", input);
	cv::waitKey(0);
	
    cv::blur(gray, temp, cv::Size(ksize, ksize));
    cv::equalizeHist(gray, gray);
    
    cv::imshow("After preprocessing", gray);
	cv::waitKey(0);

    double th = cv::threshold(temp, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    std::cout << "Otsu method threshold after Gaussian Filter: " << th << std::endl;
    
    cv::namedWindow("Otsu's thresholding after Gaussian filtering");
	cv::imshow("Otsu's thresholding after Gaussian filtering", mask);
	cv::waitKey(0);
    
}


bool detectOverlapSegmentation(int x, int y, int width, int height, int a, int b, int c, int d)
{

    // intersection region
    int xA = std::max(x, a);
    int yA = std::max(y, b);
    int xB = std::min(x+width, a+c);
    int yB = std::min(y+height, b+d);
    
    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    
    std::cout << "Intersection area is " << interArea << std::endl;
    
    bool overlap = 0;
    
    if (interArea != 0)
        overlap = 1;
    
    return overlap;
    
}



int main(int argc, char* argv[])
{
	
	//___________________________ Load Dataset bounding box coordinates ___________________________ //
	
	std::vector<int> coordinates_bb;
	
	coordinates_bb = read_numbers("../Dataset/det/04.txt");
	
	for (int i=0; i<coordinates_bb.size(); ++i)
    	std::cout << coordinates_bb[i] << ' ';
    	
	int n_hands = coordinates_bb.size() / 4;
	std::cout << "Number of hands detected are " << n_hands << std::endl;
	
	//___________________________ Load Dataset image ___________________________ //
		
	cv::Mat img = cv::imread("../Dataset/rgb/04.jpg", cv::IMREAD_COLOR);
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", img);
	cv::waitKey(0);

    
	//___________________________ Important parameters declaration ___________________________//
    std::vector<cv::Mat> img_roi_BGR(n_hands);
    std::vector<cv::Mat> img_roi_thr(n_hands);
    
	int x, y, width, height;	
	int temp = 0; // in order to get right index in vector of coordinates
		
	for (int i=0; i<n_hands; i++) 
	{
	    
	    //_________ ROI extraction _________//
    	x = coordinates_bb[i+temp];
	    y = coordinates_bb[i+temp+1];
	    width = coordinates_bb[i+temp+2];
	    height = coordinates_bb[i+temp+3];
	
		cv::Range colonna(x, x+width);
        cv::Range riga(y, y+height);
	    img_roi_BGR[i] = img(riga, colonna);
	    
      	cv::namedWindow("ROI");
	    cv::imshow("ROI", img_roi_BGR[i]);
	    cv::waitKey(0);
        
        temp = temp + 3;
	
	}
	
	//__________________________ Change image color space __________________________//
	std::vector<cv::Mat> img_roi_HSV(n_hands);    
    std::vector<cv::Mat> img_roi_YCrCb(n_hands);

    for (int i=0; i<n_hands; i++) 
	{
		cv::cvtColor(img_roi_BGR[i], img_roi_HSV[i], cv::COLOR_BGR2HSV);
		cv::cvtColor(img_roi_BGR[i], img_roi_YCrCb[i], cv::COLOR_BGR2YCrCb);
	}
	
	//___________________________ ROI segmentation Otsu thresholding ___________________________//	
	
    std::vector<cv::Mat> mask_otsu_BGR(n_hands);    
    std::vector<cv::Mat> mask_otsu_HSV(n_hands);
    std::vector<cv::Mat> mask_otsu_YCrCb(n_hands);
	
	for (int i=0; i<n_hands; i++) 
	{
	    otsuSegmentation(img_roi_BGR[i], mask_otsu_BGR[i], 5, 1);
	    otsuSegmentation(img_roi_HSV[i], mask_otsu_HSV[i], 5, 2); 
	    otsuSegmentation(img_roi_YCrCb[i], mask_otsu_YCrCb[i], 5, 3); 
	}
	
	cv::destroyAllWindows();
    
    //_____________________________ Multiplying ROI mask to get final result _____________________________//
    
    std::vector<cv::Mat> mask_final_ROI(n_hands);
	
	for (int i=0; i<n_hands; i++) 
	{
		mask_final_ROI[i].create(mask_otsu_BGR[i].rows, mask_otsu_BGR[i].cols, CV_8UC1);
		mask_final_ROI[i] = mask_otsu_BGR[i].mul(mask_otsu_HSV[i].mul(mask_otsu_YCrCb[i]));
	    cv::namedWindow("Otsu's thresholding final");
		cv::imshow("Otsu's thresholding final", mask_final_ROI[i]);
		cv::waitKey(0);
	}
	
	
	//______________ inserisci maschera in immagine nera stessa dimensione originale ______________//
	
	cv::Mat mask_final(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0)); 
	
	for (int i=0; i<n_hands; i++) 
	{
		mask_final_ROI[i].copyTo(mask_final(cv::Rect(x, y, mask_final_ROI[i].cols, mask_final_ROI[i].rows)));
		cv::imshow("Mask final", mask_final);
		cv::waitKey(0);
	}
	
    cv::imshow("Mask final", mask_final);
	cv::waitKey(0);
	
	cv::destroyAllWindows();
	
	//___________________________________ Save ____________________________________//
	
	cv::imwrite("../results/mask_predict.png", mask_final);
	
	//__________________________ Detect if overlap _________________________________// 
	
	bool overlap = 0;
	
	int x1, y1, width1, height1;
	int x2, y2, width2, height2;
	
	int temp2 = 3;
	
	// attenzione, questo ciclo mi fa solo un controllo se aggiungo i+4
	for (int i = 0; i < n_hands; i+=4) // ciclo for per controllo tutte le combinazioni di bounding box
	{
	    x1 = coordinates_bb[0+i];
	    y1 = coordinates_bb[1+i];
	    width1 = coordinates_bb[2+i];
	    height1 = coordinates_bb[3+i];
	    x2 = coordinates_bb[1+i+temp2];  // attenzione, potrebbe essere sbagliato
	    y2 = coordinates_bb[2+i+temp2];
	    width2 = coordinates_bb[3+i+temp2];
	    height2 = coordinates_bb[4+i+temp2];
	    
	    overlap = detectOverlapSegmentation(x1, y1, width1, height1, x2, y2, width2, height2);  
	    
	    temp2 = temp2 + 3;
	}
	
	std::cout << "Overlap " << overlap << std::endl;
	
	//______________________________ Handle Overlap between masks __________________________//
		
	cv::Mat mask_Overlap1(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
	cv::Mat mask_Overlap2(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
	
	mask_final_ROI[0].copyTo(mask_Overlap1(cv::Rect(coordinates_bb[0], coordinates_bb[1], mask_final_ROI[0].cols, mask_final_ROI[0].rows)));
	mask_final_ROI[1].copyTo(mask_Overlap2(cv::Rect(coordinates_bb[4], coordinates_bb[5], mask_final_ROI[1].cols, mask_final_ROI[1].rows)));

	cv::namedWindow("mask_final_ROI 1");
	cv::imshow("mask_final_ROI 1", mask_final_ROI[0]);
	cv::namedWindow("mask_final_ROI 2");
	cv::imshow("mask_final_ROI 2", mask_final_ROI[1]);
	cv::waitKey(0);
	
	cv::Mat mask_Intersection(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
	cv::bitwise_and(mask_Overlap1, mask_Overlap2, mask_Intersection);
	cv::namedWindow("mask_Intersection");
	cv::imshow("mask_Intersection", mask_Intersection);
	cv::waitKey(0);
	
	cv::destroyAllWindows();

	
	if (overlap == 1) 
	{
	    int smaller = 1;
    	if (smaller == 1) // piu piccola la prima ROI
		    mask_Overlap1 = mask_Overlap1 - mask_Intersection;
	    else 
		    mask_Overlap2 = mask_Overlap2 - mask_Intersection;
		    
		cv::namedWindow("mask_Overlap1");
	    cv::imshow("mask_Overlap1", mask_Overlap1);
	    cv::namedWindow("mask_Overlap2");
	    cv::imshow("mask_Overlap2", mask_Overlap2);
	    cv::waitKey(0);
	}
		
	cv::Mat mask_final_Overlap(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
	cv::bitwise_or(mask_Overlap1, mask_Overlap2, mask_final_Overlap);
	
	cv::namedWindow("mask_final_Overlap");
	cv::imshow("mask_final_Overlap", mask_final_Overlap);
	cv::waitKey(0);
	
	
	/*
	//_____________________________ generate random color  _____________________________//

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
	
	
	//______________ color the mask moltiplicando ogni singolo canale con rispettivo colore ________________//
	
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
	
	
	//____________________ Inserisci maschera immagine colorata in immagine nera stessa dimensione originale _____________________//
    cv::Mat prediction(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0)); 
    merged.copyTo(prediction(cv::Rect(x, y, merged.cols, merged.rows)));
	cv::imshow("Prediction", prediction);
	cv::waitKey(0);
	
	// back to original image
    cv::Mat ultima;
    ultima = img + prediction;
    
	cv::imshow("Boh", ultima);
	cv::waitKey(0);
	
	*/
    
    
	return 0;
  
}


