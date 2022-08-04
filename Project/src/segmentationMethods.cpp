
/**
 * @file segmentationMethods.cpp
 *
 * @brief  main segmentation function calling different Segmentation's Algorithms
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
#include "read_sort_BB_matrix.h"
#include "removeOutliers.h"
#include "fillMaskHoles.h"
#include "segmentationMethods.h"  
#include "segmentationAlgorithm.h" 
#include "Segmentator.h" 

using namespace cv;
using namespace std;


void segmentationMethods(cv::Mat& img)
{
	
	Segmentator segment;
	
    std::string image_number;
    
    std::cout << "Insert image number from 01 to 30" << std::endl;
    std::cin >> image_number; 
	
	//___________________________ Load Dataset image ___________________________ //

	img = cv::imread("../Dataset/rgb/" + image_number + ".jpg", cv::IMREAD_COLOR);
	
	std::cout << img.rows << std::endl;
	std::cout << img.cols << std::endl;
	
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", img);
	cv::waitKey(0);
	
	//___________________________ Load Dataset bounding box coordinates ___________________________ //

	std::vector<std::vector<int>> coordinates_bb;
	
	coordinates_bb = read_sort_BB_matrix("../Dataset/det/" + image_number + ".txt");
	
	int n_hands = coordinates_bb.size(); // return number of rows
	std::cout << "Number of hands detected are " << n_hands << std::endl;

    
	//___________________________ Important parameters declaration ___________________________//
    std::vector<cv::Mat> img_roi_BGR(n_hands);
    std::vector<cv::Mat> img_roi_thr(n_hands);
    cv::Mat tempROI;
    
	int x, y, width, height;	
		
	for (int i=0; i<n_hands; i++) 
	{
	    //_________ ROI extraction _________//
    	x = coordinates_bb[i][0];
	    y = coordinates_bb[i][1];
	    width = coordinates_bb[i][2];
	    height = coordinates_bb[i][3];
	
		cv::Range colonna(x, x+width);
        cv::Range riga(y, y+height);
	    tempROI = img(riga, colonna);
	    img_roi_BGR[i] = tempROI.clone(); // otherwise matrix will not be continuos
	    
      	cv::namedWindow("ROI BGR");
	    cv::imshow("ROI BGR", img_roi_BGR[i]);
	    cv::waitKey(0);
	}
	
	cv::destroyAllWindows();
	
	std::vector<cv::Mat> mask_RegionGrowing_ROI(n_hands);
	std::vector<cv::Mat> mask_inRange_ROI(n_hands);
	std::vector<cv::Mat> mask_Otsu_ROI(n_hands);
	std::vector<cv::Mat> mask_Canny_ROI(n_hands);
	
	int mode = 1;
	
	switch (mode) // no break in order to do all of them
	{
		case 1:
			std::cout << "Starting inRange segmentation" << std::endl;
			inRangeSegmentation(img_roi_BGR, mask_inRange_ROI, coordinates_bb, n_hands);
		case 2:
			std::cout << "Starting Otsu segmentation" << std::endl;
			OtsuSegmentation(img, img_roi_BGR, mask_Otsu_ROI, coordinates_bb, n_hands);
		case 3:
			std::cout << "Starting Canny segmentation" << std::endl;
			CannySegmentation(img, img_roi_BGR, mask_Canny_ROI, coordinates_bb, n_hands);	
		case 4:	
			std::cout << "Starting Region Growing segmentation" << std::endl;
			RegionGrowingSegmentation(img, img_roi_BGR, mask_RegionGrowing_ROI, coordinates_bb, n_hands);
	}
	
	//______________ insert all the binary masks in a image same dimension of the original ______________//
	cv::Mat mask_RegionGrowing(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
    segment.insertBinaryMask(img, mask_RegionGrowing_ROI, mask_RegionGrowing, coordinates_bb, n_hands);
	
	cv::Mat mask_inRange(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
    segment.insertBinaryMask(img, mask_inRange_ROI, mask_inRange, coordinates_bb, n_hands);
    
	cv::Mat mask_Otsu(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
    segment.insertBinaryMask(img, mask_Otsu_ROI, mask_Otsu, coordinates_bb, n_hands);
    
	cv::Mat mask_Canny(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
    segment.insertBinaryMask(img, mask_Canny_ROI, mask_Canny, coordinates_bb, n_hands);
	
	// Display all result 
	segment.displayMultiple(img, mask_inRange, mask_Otsu, mask_Canny, mask_RegionGrowing);
	
	cv::destroyAllWindows();
	    
	    
    //_______________________________________________ Choose best one based on pixel accuracy _______________________________________________//
    
    cv::Mat mask_ground_truth = cv::imread("../Dataset/mask/" + image_number + ".png", cv::IMREAD_GRAYSCALE);
    
	std::vector<cv::Mat> mask_final_ROI(n_hands);
    
    for (int i=0; i<n_hands; i++) 
    {
	    mask_final_ROI[i].create(img_roi_BGR[i].rows, img_roi_BGR[i].cols, CV_8UC1);
    }
	    
    // mask_final_ROI = segment.chooseBestMask(mask_ground_truth, mask_inRange, mask_Otsu, mask_Canny, mask_inRange_ROI, mask_Otsu_ROI, mask_Canny_ROI);
    
    mask_final_ROI = segment.chooseBestROIMask(img, mask_inRange_ROI, mask_Otsu_ROI, mask_Canny_ROI, mask_RegionGrowing_ROI, coordinates_bb, image_number, n_hands, mask_final_ROI);
	    
	
	cv::Mat mask_final(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
    segment.insertBinaryMask(img, mask_final_ROI, mask_final, coordinates_bb, n_hands);
    
  	cv::namedWindow("Mask final Binary");
    cv::imshow("Mask final Binary", mask_final);
    cv::waitKey(0);
	
	// save binary mask
	cv::imwrite("../results/resultsSegmentation/Binary/" + image_number + ".png", mask_final);
	
	cv::destroyAllWindows();
    
	//_____________________________ generate random color and color each mask _____________________________//
	std::vector<cv::Mat> img_ROI_color(n_hands);
	// std::vector<cv::Scalar> randColor(n_hands);
	std::vector<cv::Vec3b> randColor;
	
	segment.randomColorMask(mask_final_ROI, img_ROI_color, randColor, n_hands);
	
	//_________________________________________ Detect Overlap _________________________________________//
	bool overlap = segment.detectOverlap(coordinates_bb);
	
	//_________________________________________ Handle Overlap _________________________________________//
	cv::Mat mask_Watershed;
	if (overlap == true)
	    segment.handleOverlap(mask_final, mask_Watershed, coordinates_bb, randColor, n_hands);

	//____________________ Inserisci maschera immagine colorata in immagine colorata nera stessa dimensione originale _____________________//
	cv::Mat mask_color_final(img.rows, img.cols, CV_8UC3, cv::Scalar::all(0)); 
	segment.insertColorMask(img, img_ROI_color, mask_color_final, coordinates_bb, n_hands, randColor, mask_Watershed, overlap);
	
	cv::namedWindow("Mask final Color");
    cv::imshow("Mask final Color", mask_color_final);
	cv::waitKey(0);	
		
	//____________________ Unisci maschera con immagine di partenza _____________________//
	// quando pixel diverso da zero, vuol dire che ho maschera mano, quindi sostituisco pixel
	for(int i=0; i<img.rows; i++) {
        for(int j=0; j<img.cols; j++) {
            if(mask_color_final.at<cv::Vec3b>(i,j)[0] != 0 && mask_color_final.at<cv::Vec3b>(i,j)[1] != 0 && mask_color_final.at<cv::Vec3b>(i,j)[2] != 0) {
                img.at<cv::Vec3b>(i,j)[0] = mask_color_final.at<cv::Vec3b>(i,j)[0];
                img.at<cv::Vec3b>(i,j)[1] = mask_color_final.at<cv::Vec3b>(i,j)[1];
                img.at<cv::Vec3b>(i,j)[2] = mask_color_final.at<cv::Vec3b>(i,j)[2];
            }
        }
	}
	
	cv::imwrite("../results/resultsSegmentation/Color/" + image_number + ".jpg", img);
	

}
	
