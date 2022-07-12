/**
 * @file Project.cpp
 *
 * @brief Hand Segmentation
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
#include "read_BB_matrix.h"
#include "removeOutliers.h"
#include "fillMaskHoles.h"
#include "insertMask.h"  
#include "randomColorMask.h" 



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
	
    cv::blur(gray, temp, cv::Size(ksize, ksize));
    // cv::equalizeHist(gray, gray);

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

void kmeansSegmentation(const cv::Mat& input, cv::Mat& output, const int k) {
    //data array for kmeans function, input image need to be converted to array like
    cv::Mat data = input.reshape(1, input.rows * input.cols);
    //convert to 32 float
    data.convertTo(data, CV_32F);
    
    //structures for kmeans function
    std::vector<int> labels;
    cv::Mat1f centers;
    //apply kmeans
    double compactness = cv::kmeans(data, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.1), 10, cv::KMEANS_PP_CENTERS, centers);
    std::printf("Compactness: %f\n", compactness);

    //update data array with clusters colors
    for(int i=0; i<data.rows; ++i) {
        data.at<float>(i, 0) = centers(labels[i], 0);
        data.at<float>(i, 1) = centers(labels[i], 1);
        data.at<float>(i, 2) = centers(labels[i], 2);
    }

    //reshape into output image
    output = data.reshape(3, input.rows);
    output.convertTo(output, CV_8UC3);
}


int main(int argc, char* argv[])
{
	
    std::string image_number;
    
    std::cout << "Insert image number from 01 to 30" << std::endl;
    std::cin >> image_number; 
	
	//___________________________ Load Dataset image ___________________________ //

	cv::Mat img = cv::imread("../Dataset/rgb/" + image_number + ".jpg", cv::IMREAD_COLOR);
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
	    
      	cv::namedWindow("ROI");
	    cv::imshow("ROI", img_roi_BGR[i]);
	    cv::waitKey(0);
	
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
		// mask_final_ROI[i] = mask_otsu_BGR[i].mul(mask_otsu_HSV[i].mul(mask_otsu_YCrCb[i]));
		
		mask_final_ROI[i] = mask_otsu_YCrCb[i];
	    cv::namedWindow("Otsu's thresholding final");
		cv::imshow("Otsu's thresholding final", mask_final_ROI[i]);
		cv::waitKey(0);
	}

	
	
	//_____________________________________________ Remove outliers from segmentation _____________________________________________
    for (int i=0; i<n_hands; i++) {
         removeOutliers(mask_final_ROI[i]);
    }
    
    cv::destroyAllWindows();
    
	
	//______________ insert binary mask in a image same dimension of the original ______________//
	cv::Mat mask_final(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
    insertBinaryMask(img, mask_final_ROI, mask_final, coordinates_bb, n_hands);
	
	
	//____________________________________ Distance transform and watershed ____________________________________//
	
	cv::Mat distTransf;
	cv::distanceTransform(mask_final, distTransf, cv::DIST_L2, 3);
	cv::normalize(distTransf, distTransf, 0, 1.0, cv::NORM_MINMAX);
		
	cv::namedWindow("Mask transform");
    cv::imshow("Mask transform", distTransf);
	cv::waitKey(0);
	
	cv::Mat dist;
	cv::threshold(distTransf, dist, 0.3, 1.0, cv::THRESH_BINARY);
	
	cv::namedWindow("Mask transform");
    cv::imshow("Mask transform", dist);
	cv::waitKey(0);
	
    //from each blob create a seed for watershed algorithm
    cv::Mat dist8u, markers8u;
    cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32SC1);
    dist.convertTo(dist8u, CV_8U);
    
    //find total markers
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dist8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    //number of contours
    int ncomp = static_cast<int>(contours.size());
    std::printf("Contours: %d\n", ncomp);

    //draw foreground markers
    for(int i=0; i<ncomp; ++i) {
        cv::drawContours(markers, contours, i, cv::Scalar(i+1), -1);
    }
    
    markers.convertTo(markers8u, CV_8U, 10);
    cv::imshow("Markers", markers8u);
    cv::waitKey(0);
    
    //draw background markers
    cv::circle(markers, cv::Point(5, 5), 3, cv::Scalar(255), -1);
    markers.convertTo(markers8u, CV_8U, 10);
    cv::imshow("Markers", markers8u);
    cv::waitKey(0);
    
	
	cv::imwrite("../results/resultsSegmentation/DistanceTransform" + image_number + ".png", distTransf);
	
	cv::destroyAllWindows();
	
	//___________________________________ Save ____________________________________//
	
	cv::imwrite("../results/resultsSegmentation/" + image_number + ".png", mask_final);
	
	
	/*
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
	
	*/
	
	/*
	bool overlap = 1;
	
	//______________________________ Handle Overlap between masks __________________________//
		
	cv::Mat mask_Overlap1(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
	cv::Mat mask_Overlap2(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
	
	int i = 0;
	mask_final_ROI[i].copyTo(mask_Overlap1(cv::Rect(coordinates_bb[i][0], coordinates_bb[i][1], mask_final_ROI[i].cols, mask_final_ROI[i].rows)));
	
	i = 1;
	mask_final_ROI[i].copyTo(mask_Overlap2(cv::Rect(coordinates_bb[i][0], coordinates_bb[i][1], mask_final_ROI[i].cols, mask_final_ROI[i].rows)));
		

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
	
	cv::imwrite("../results/mask_Overlap.png", mask_final_Overlap);
	
	cv::Mat mask_Opening;
	
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10,10));
	
	cv::morphologyEx(mask_final_Overlap, mask_Opening, cv::MORPH_OPEN, kernel);
		
	cv::namedWindow("mask_final_Opening");
	cv::imshow("mask_final_Opening", mask_final_Overlap);
	cv::waitKey(0);
	*/
	
	
	/*
	//_____________________________________ Kmeans ____________________________________//
	
    std::vector<cv::Mat> mask_Kmeans(n_hands); 
    
    std::vector<cv::Mat> img_roi_YCrCb(n_hands);
    
    std::vector<cv::Mat> mask_otsu_YCrCb(n_hands);
	
	for (int i=0; i<n_hands; i++) 
	{
	    cv::cvtColor(img_roi_BGR[i], img_roi_YCrCb[i], cv::COLOR_BGR2YCrCb);
	    otsuSegmentation(img_roi_YCrCb[i], mask_otsu_YCrCb[i], 5, 3); 
	}
	for (int i=0; i<n_hands; i++) 
	{
	    mask_Kmeans[i] = cv::Mat::zeros(mask_otsu_YCrCb[i].rows, mask_otsu_YCrCb[i].cols, CV_8UC1);
        kmeansSegmentation(mask_otsu_YCrCb[i], mask_Kmeans[i], 3);  
      	cv::namedWindow("Kmeans");
	    cv::imshow("Kmeans", mask_Kmeans[i]);
	    cv::waitKey(0);
	}

	
	// cv::imwrite("../results/mask_Kmeans1.png", mask_Kmeans[0]);
	// cv::imwrite("../results/mask_Kmeans2.png", mask_Kmeans[1]);
	*/
	
    
	//_____________________________ generate random color and color mask _____________________________//
	std::vector<cv::Mat> img_ROI_color(n_hands);
	
	std::vector<cv::Scalar> randColor(n_hands);
	
	randomColorMask(mask_final_ROI, img_ROI_color, randColor, n_hands);
	
	//____________________ Inserisci maschera immagine colorata in immagine nera stessa dimensione originale _____________________//
	cv::Mat mask_color_final(img.rows, img.cols, CV_8UC3, cv::Scalar::all(0)); 
	insertColorMask(img, img_ROI_color, mask_color_final, coordinates_bb, n_hands);
	
	
	//_________________________________________ parte 2 watershed _________________________________________//
	
	
	cv::cvtColor(mask_final, mask_final, cv::COLOR_GRAY2BGR);
	
	//apply the watershed algorithm
    cv::Mat result = mask_final.clone();
    cv::watershed(result, markers);
    cv::Mat mark;
    markers.convertTo(mark, CV_8U);
    cv::bitwise_not(mark, mark);

    //generate random colors
    cv::RNG rng (12345);
    std::vector<cv::Vec3b> colors;
    for(int i=0; i<ncomp; ++i) {
        uchar b = static_cast<uchar>(rng.uniform(0, 255));
        uchar g = static_cast<uchar>(rng.uniform(0, 255));
        uchar r = static_cast<uchar>(rng.uniform(0, 255));
        //insert new color
        colors.push_back(cv::Vec3b(b, g, r));
    }
    
    
    
    cv::Mat output = mask_final.clone();
    
	cv::namedWindow("Mask tram");
    cv::imshow("Mask tram", output);
	cv::waitKey(0);

    //create output image
    for(int i=0; i<markers.rows; ++i) {
        for(int j=0; j<markers.cols; ++j) {
            int index = markers.at<int>(i, j);
            if(index > 0 && index <= ncomp) {
                output.at<cv::Vec3b>(i, j) = colors[index-1];
            }
        }
    }
    
	cv::namedWindow("transform");
    cv::imshow("transform", output);
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
	
	cv::namedWindow("Image final");
    cv::imshow("Image final", img);
	cv::waitKey(0);
	
	cv::imwrite("../results/resultsSegmentation/colored" + image_number + ".png", img);
	
	/*
	
	// change transparency
	cv::Mat imageT(img_ROI_color[1].rows, img_ROI_color[1].cols, CV_8UC4, cv::Scalar(0, 0, 0));
	cv::cvtColor(img_ROI_color[1], imageT, cv::COLOR_BGR2RGBA, 4);
	
    cv::Mat Bands_BGRA[4];
    cv::split(imageT, Bands_BGRA);
    
    Bands_BGRA[3] = Bands_BGRA[3] * 0.5;

    std::vector<cv::Mat> channels_BGRA;
	channels_BGRA.push_back(Bands_BGRA[0]);
	channels_BGRA.push_back(Bands_BGRA[1]);
	channels_BGRA.push_back(Bands_BGRA[2]);
	channels_BGRA.push_back(Bands_BGRA[3]);
    
    cv::Mat mergedL(img_ROI_color[1].rows, img_ROI_color[1].cols, CV_8UC4, cv::Scalar(0, 0, 0));
    cv::merge(channels_BGRA, mergedL);
	
    cv::namedWindow("Final random");
	cv::imshow("Final random", mergedL);
	cv::waitKey(0);
	
	*/
	
	/*
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


