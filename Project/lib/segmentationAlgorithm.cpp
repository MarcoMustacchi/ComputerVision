
/**
 * @file segmentationAlgorithm.cpp
 *
 * @brief inRange Segmentation, Otsu Segmentation and Edges Segmentation
 *
 * @author Marco Mustacchi
 *
 */

#include <opencv2/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "read_sort_BB_matrix.h"
#include "removeOutliers.h"
#include "fillMaskHoles.h"
#include "segmentationAlgorithm.h" 
#include "Segmentator.h" 


using namespace cv;
using namespace std;


//__________________________________________________________________ inRange Segmentation __________________________________________________________________//
/*
void inRangeSegmentation(std::vector<cv::Mat>& img_roi_BGR, std::vector<cv::Mat>& mask_inRange_ROI, int n_hands)
{
	
    for (int i=0; i<n_hands; i++) 
	{
		cvtColor(img_roi_BGR[i], img_roi_BGR[i], COLOR_BGR2YCrCb);
	    
	    Scalar min_YCrCb(0,133,80);  // 0,150,100  
	    Scalar max_YCrCb(255,173,120); // 235,173,127
	    inRange(img_roi_BGR[i], min_YCrCb, max_YCrCb, mask_inRange_ROI[i]);
	    
	    // remove Outliers
        removeOutliers(mask_inRange_ROI[i]);
	    
	    cvtColor(img_roi_BGR[i], img_roi_BGR[i], COLOR_YCrCb2BGR);
	    
 	    cv::namedWindow("mask_inRange_ROI");
	    cv::imshow("mask_inRange_ROI", mask_inRange_ROI[i]);
	    cv::waitKey(0);
	}	
	
}
*/


void inRangeSegmentation(const std::vector<cv::Mat>& img_roi_BGR, std::vector<cv::Mat>& mask_inRange_ROI, std::vector<std::vector<int>> coordinates_bb, int n_hands)
{

    Segmentator segment;
    
    for (int m=0; m<n_hands; m++) 
	{
        
        cv::Mat tempROI(img_roi_BGR[m].rows, img_roi_BGR[m].cols, CV_8UC1, cv::Scalar::all(0));
    
        for (int i = 0; i < img_roi_BGR[m].rows; i++) {
            for (int j = 0; j < img_roi_BGR[m].cols; j++) {          

                int B = img_roi_BGR[m].at<cv::Vec3b>(i, j)[0];
                int G = img_roi_BGR[m].at<cv::Vec3b>(i, j)[1];
                int R = img_roi_BGR[m].at<cv::Vec3b>(i, j)[2];   
                
                // find max and mix value among B, G and R
                int max_BGR;
                int min_BGR;

                if (R >= G && R >= B) 
                    max_BGR = R;
                else if (G >= R && G >= B) 
                    max_BGR = G;
                else 
                    max_BGR = B;

                if (R <= G && R <= B)
                    min_BGR = R;
                else if (G <= R && G <= B)
                    min_BGR = G;
                else
                    min_BGR = B;           

                if ( B > 20 && G > 40 && R > 75 && R > G && R > B && abs(R-G) > 15 && (max_BGR-min_BGR > 15) ) 
                {    
                    tempROI.at<uchar>(i, j) = 255;
                }
                else 
                {
                    tempROI.at<uchar>(i, j) = 0;
                }
            }
        }               
        
        bool overlap = segment.detectOverlap(coordinates_bb);
        
        if (overlap == false)
        {
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
            cv::morphologyEx(tempROI, mask_inRange_ROI[m], cv::MORPH_OPEN, kernel);
        }
        else
        {
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9));
            cv::morphologyEx(tempROI, mask_inRange_ROI[m], cv::MORPH_OPEN, kernel);
        }
        
	    // remove Outliers
        removeOutliers(mask_inRange_ROI[m]);
        
        if (n_hands >= 3)
            fillMaskHoles(mask_inRange_ROI[m], mask_inRange_ROI[m]);
	    
 	    cv::namedWindow("mask_inRange_ROI");
	    cv::imshow("mask_inRange_ROI", mask_inRange_ROI[m]);
	    cv::waitKey(0);
	}
	
    cv::destroyAllWindows();
	 
}



//__________________________________________________________________ Otsu Segmentation __________________________________________________________________//

void OtsuSegmentation(const cv::Mat& img, std::vector<cv::Mat>& img_roi_BGR, std::vector<cv::Mat>& mask_Otsu_ROI, std::vector<std::vector<int>> coordinates_bb, int n_hands)
{
	
	//__________________________ Change image color space __________________________//
	
    int ksize = 5;  

    for (int i=0; i<n_hands; i++) 
	{
		cv::cvtColor(img_roi_BGR[i], mask_Otsu_ROI[i], cv::COLOR_BGR2HSV);
		
	    cv::Mat gray, temp;
    
        cv::Mat hsv_channels[3];
        cv::split( mask_Otsu_ROI[i], hsv_channels );
        gray = hsv_channels[2]; // 3 channel of HSV is gray
	    
        cv::blur(gray, temp, cv::Size(ksize, ksize));

        double th = cv::threshold(temp, mask_Otsu_ROI[i], 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        
	}
	
	
	//_____________________________________________ Remove outliers from segmentation _____________________________________________
    for (int i=0; i<n_hands; i++) {
        removeOutliers(mask_Otsu_ROI[i]);
         
 	    cv::namedWindow("img_roi_HSV");
	    cv::imshow("img_roi_HSV", mask_Otsu_ROI[i]);
	    cv::waitKey(0);
             
    }
        
    cv::destroyAllWindows();
     
}



//__________________________________________________________________ Canny Segmentation __________________________________________________________________//

void CannySegmentation(const cv::Mat& img, std::vector<cv::Mat>& img_roi_BGR, std::vector<cv::Mat>& mask_Canny_ROI, std::vector<std::vector<int>> coordinates_bb, int n_hands)
{
		
	std::vector<cv::Mat> dilate(n_hands);
	std::vector<cv::Mat> mask_final_ROI(n_hands);
	
	std::vector<cv::Mat> final_ROI(n_hands);
	
	cv::Mat markerMask;
	
    for (int k=0; k<n_hands; k++) 
	{

    	cv::Mat imgGray;
	
        cvtColor(img_roi_BGR[k], markerMask, COLOR_BGR2GRAY);
        markerMask = Scalar::all(0);
        
        //___________________________ canny often double edges -> bad result ___________________________//
        //___________________________ This method one edges -> better ___________________________//
	    Mat mask;
	    Mat edges;
	    
        /*
        Mat sharp_kernel = (Mat_<float>(3,3) << 
        -1,  -1, -1,
        -1,  9,  -1,
        -1,  -1, -1); 

        cv::filter2D(img_roi_BGR[k], img_roi_BGR[k], -1, sharp_kernel);
        
        cv::imshow("sharpened", img_roi_BGR[k]);
        cv::waitKey(0); 
        */
        
	    Mat gray;
        cv::cvtColor(img_roi_BGR[k], gray, cv::COLOR_BGR2GRAY);
        
        Mat filtered;
        cv::bilateralFilter(gray, filtered, 11, 75, 90, cv::BORDER_DEFAULT); // important for threshold
        
        /*
        Mat sharpened;
        cv::GaussianBlur(gray, filtered, cv::Size(0, 0), 5);
        cv::addWeighted(gray, 1.5, filtered, -0.5, 0, sharpened);
        
        cv::imshow("sharpened", sharpened);
        cv::waitKey(0); 
        */
        
        double th = cv::threshold(filtered, gray, 150, 200, cv::THRESH_BINARY | cv::THRESH_OTSU);
        
        std::cout << "threshold is " << th << std::endl;

        cv::Mat thresh;
        cv::threshold(gray, thresh, th, 255, cv::THRESH_BINARY);
        
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7));
             
        cv::morphologyEx(thresh, dilate[k], cv::MORPH_DILATE, kernel);
        
        cv::Mat diff;
        cv::absdiff(dilate[k], thresh, diff);

        edges = 255 - diff;

        // cv::imshow("thresh", thresh);
        // cv::imshow("dilate", dilate[k]);
        // cv::imshow("diff", diff);
        // cv::imshow("edges", edges);
        // cv::waitKey(0);  
        
        mask_Canny_ROI[k] = dilate[k].clone();
        
        cv::destroyAllWindows(); 
	}
	
	//_____________________________________________ Remove outliers from segmentation _____________________________________________
    for (int i=0; i<n_hands; i++) {
         removeOutliers(mask_Canny_ROI[i]);
         
 	    cv::namedWindow("mask_Canny_ROI");
	    cv::imshow("mask_Canny_ROI", mask_Canny_ROI[i]);
	    cv::waitKey(0);
             
    }
    
    cv::destroyAllWindows();
	
}


/**
 * @file segmentationAlgorithm.cpp
 *
 * @brief Region Growing Segmentation
 *
 * @author Nicola Rizzetto and Marco Mustacchi
 *
 */

Mat regionGrowing(Mat anImage,vector<pair<int, int>> SeedSet,Vec3b mask_color,float tolerance);
vector<pair<int, int>> seed_finder(Mat img, vector<pair<int, int>> seed_set,int value, int thres);

void RegionGrowingSegmentation(cv::Mat& img, std::vector<cv::Mat> img_roi_BGR, std::vector<cv::Mat>& mask_RegionGrowing_ROI, std::vector<std::vector<int>> coordinates_bb, int n_hands)
{
    vector<Vec3b> mask_colors = {Vec3b(36,85,141),Vec3b(66,134,198),Vec3b(105,172,224),Vec3b(241,194,125),Vec3b(172,219,255),Vec3b(189,224,255),
        Vec3b(148,205,255),Vec3b(134,192,234),Vec3b(96,173,255),Vec3b(159,227,255)}; //obtained from the internet
    uchar grayskin_color = 160; //empirically chosen
	
	bool grayscale = true;
    for(int i = 0; i < img.rows; i++) {  //check if the image is grayscale
        cv::Vec3b* ptr = img.ptr<cv::Vec3b>(i);
        for(int j = 0; j < img.cols; j++) {
            if(!(ptr[j][0] == ptr[j][1] && ptr[j][0] == ptr[j][2])) {
                grayscale = false;
                break;
            }
        }
    if(!grayscale)
        break;
    }
	
    for (int m=0; m<n_hands; m++) 
	{
		Mat gray, filtered, edges, smoothed, canny_edges;
		cv::cvtColor(img_roi_BGR[m], gray, cv::COLOR_BGR2GRAY);
		bilateralFilter(gray, filtered, 11, 10, 90, cv::BORDER_DEFAULT);
		adaptiveThreshold(filtered, edges, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);
		equalizeHist(gray,gray); //to have more or less the same grayscale intensity in every hand
		bilateralFilter(gray, filtered, 11, 10, 90, cv::BORDER_DEFAULT);
		bilateralFilter(img_roi_BGR[m], smoothed, 11, 10, 90, cv::BORDER_DEFAULT);

		Mat gray_masked(gray.size(), CV_8U);
		int mask_thres = 50;
		for(int i = 0; i < img_roi_BGR[m].rows; i++){ //detect skin in smoothed and apply the mask on the grayscale
	        for(int j = 0; j <img_roi_BGR[m].cols; j++){
		        gray_masked.at<uchar>(i,j) = 0;
		        for(int k = 0; k < mask_colors.size(); k++) {
		            if(grayscale && (abs(smoothed.at<Vec3b>(i,j)[0] - grayskin_color) < mask_thres))
		                gray_masked.at<uchar>(i,j) = filtered.at<uchar>(i,j);
	                else if((abs(smoothed.at<Vec3b>(i,j)[0] - mask_colors[k][0]) < mask_thres) && (abs(smoothed.at<Vec3b>(i,j)[1] - mask_colors[k][1]) < mask_thres) 
	                    && (abs(smoothed.at<Vec3b>(i,j)[2] - mask_colors[k][2]) < mask_thres))
	                    gray_masked.at<uchar>(i,j) = filtered.at<uchar>(i,j);
		        }
		    }
        }
        
        Canny(filtered, canny_edges, 50, 100, 3);

		// namedWindow("Mask",WINDOW_NORMAL);
		// imshow("Mask",gray_masked);
		// namedWindow("Threshold",WINDOW_NORMAL);
		// imshow("Threshold",edges);
		// namedWindow("Canny",WINDOW_NORMAL);
		// imshow("Canny",canny_edges);
		
		for(int i=0; i < edges.rows; i++){
		  for(int j=0; j < edges.cols; j++){
		    if(edges.at<uchar>(i,j) == 0){ //if there is an edge in the adaptive threshold draw it in the smoothed
		      smoothed.at<Vec3b>(i,j) = Vec3b(0,0,0); //black is most unlikely to merge with the skin region growing
		    }
		    if(canny_edges.at<uchar>(i,j) == 255){ //if there is an edge in the canny image draw it in the smoothed
		      smoothed.at<Vec3b>(i,j) = Vec3b(0,0,0);
		    }
		  }
		}
		vector<pair<int, int>> seeds_set;
		seeds_set = seed_finder(gray_masked, seeds_set, 170, 80); 
		Mat segmented_image = regionGrowing(smoothed, seeds_set, Vec3b(255, 255, 255), 1);
		
		cv::Mat tempROI;
		cvtColor(segmented_image, tempROI, COLOR_BGR2GRAY);
		threshold(tempROI, mask_RegionGrowing_ROI[m], 0, 255, THRESH_BINARY); //get a binary segmentation image
		
		removeOutliers(mask_RegionGrowing_ROI[m]);
		
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
        cv::morphologyEx(mask_RegionGrowing_ROI[m], mask_RegionGrowing_ROI[m], cv::MORPH_DILATE, kernel);
		
		fillMaskHoles(mask_RegionGrowing_ROI[m], mask_RegionGrowing_ROI[m]);
		
		namedWindow("mask_RegionGrowing_ROI",WINDOW_NORMAL);
		imshow("mask_RegionGrowing_ROI", mask_RegionGrowing_ROI[m]);
	    cv::waitKey(0);
	}
	
	cv::destroyAllWindows();
		
}



vector<pair<int, int>> seed_finder(Mat img, vector<pair<int, int>> seed_set,int value, int thres){  
  Mat seed_mat(img.size(), CV_8U);
  for(int i = 0; i < img.rows; i++){
    for(int j = 0; j < img.cols; j++){
    seed_mat.at<uchar>(i,j) = 0;
      if(img.at<uchar>(i,j) <= value + thres && img.at<uchar>(i,j) >= value - thres){ //check if the pixel can be a seed based on intensity
        seed_set.push_back(pair<int,int>(i,j));
        seed_mat.at<uchar>(i,j) = 255;
      }
    }
  }
  for(int i = 2; i < seed_mat.rows-2; i++){  //5x5 erosion kernel
    for(int j = 2; j < seed_mat.cols-2; j++){
      if(seed_mat.at<uchar>(i,j) == 255){
        for(int k = i-2; k <= i + 2; k++){
          for(int l = j - 2; l <= j + 2; l++){
            if(seed_mat.at<uchar>(k,l) == 255 && (i != k || j != l)){
              seed_mat.at<uchar>(k,l) = 0;
            }
          }
        }
      }
    }
  }
  
  for(int i = 0; i < seed_mat.rows; i++){ //push the remaining seeds
    for(int j = 0; j < seed_mat.cols; j++){
      if(seed_mat.at<uchar>(i,j) == 255){
        seed_set.push_back(pair<int,int>(i,j));
      }
    }
  }
  // namedWindow("Seeds", WINDOW_NORMAL); 
  // imshow("Seeds", seed_mat); 

  return seed_set;
}


Mat regionGrowing(Mat anImage, vector<pair<int, int>> SeedSet, Vec3b mask_color, float tolerance)
{

    Mat visited_matrix = Mat::zeros(anImage.rows, anImage.cols, anImage.type());

    vector<pair<int, int>> starting_points = SeedSet;

    while (!starting_points.empty())
    {
    
        pair<int, int> this_point = starting_points.back();
        starting_points.pop_back();
        
        int x = this_point.first;
        int y = this_point.second;
        Vec3b pixel_value = anImage.at<Vec3b>(x,y);
        if(pixel_value != Vec3b(0,0,0)){ //if it's black the seed ended up on an edge, and we skip it
                                                                                                                            
        visited_matrix.at<Vec3b>(x, y) = mask_color;

        for (int i = x - 1; i <= x + 1; i++)
            {
            if (i >= 0 && i < anImage.rows)
                {
                for (int j = y - 1; j <= y + 1; j++)
                   {
                    if (j >= 0 && j < anImage.cols)
                        {
                        Vec3b neighbour_value = anImage.at<Vec3b>(i, j);
                        Vec3b neighbour_visited = visited_matrix.at<Vec3b>(i, j);
                       
                        
                        if (neighbour_visited == Vec3b(0,0,0) && (abs(neighbour_value[0] - pixel_value[0]) <= (tolerance / 100.0 * 255.0))&&
                        (abs(neighbour_value[1] - pixel_value[1]) <= (tolerance / 100.0 * 255.0))&&(abs(neighbour_value[2] - pixel_value[2]) <= (tolerance / 100.0 * 255.0))) 
                        {
                            starting_points.push_back(pair<int, int>(i, j)); //pixel added to the region
                        }
                    }
                }
            }
        }
      }
     }
    return visited_matrix;
}

