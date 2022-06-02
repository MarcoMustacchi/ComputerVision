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
    
    cv::blur(gray, temp, cv::Size(ksize, ksize));

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



//_____________________________________________ Classes _____________________________________________//

class Histogram {
public:
	cv::Mat calc_histogram(cv::Mat scr) {
		cv::Mat hist;
		hist = cv::Mat::zeros(256, 1, CV_32F);
		scr.convertTo(scr, CV_32F);
		double value = 0;
		for (int i = 0; i < scr.rows; i++)
		{
			for (int j = 0; j < scr.cols; j++)
			{
				value = scr.at<float>(i, j);
				hist.at<float>(value) = hist.at<float>(value) + 1;
			}
		}
		return hist;
	}

	cv::Mat plot_histogram(cv::Mat histogram) {
		cv::Mat histogram_image(400, 512, CV_8UC3, cv::Scalar(0, 0, 0));
		cv::Mat normalized_histogram;
		cv::normalize(histogram, normalized_histogram, 0, 400, cv::NORM_MINMAX, -1, cv::Mat());

		for (int i = 0; i < 256; i++)
		{
			cv::rectangle(histogram_image, cv::Point(2 * i, histogram_image.rows - normalized_histogram.at<float>(i)), cv::Point(2 * (i + 1), histogram_image.rows), cv::Scalar(255, 0, 0));
		}


		return histogram_image;
	}
	
};



int main(int argc, char* argv[])
{
	
	//___________________________ Load Dataset bounding box coordinates ___________________________ //
	
	std::vector<int> coordinates_bb;
	
	coordinates_bb = read_numbers("../Dataset/det/02.txt");
	
	for (int i=0; i<coordinates_bb.size(); ++i)
    	std::cout << coordinates_bb[i] << ' ';
    	
	int n_hands = coordinates_bb.size() / 4;
	std::cout << "Number of hands detected are " << n_hands << std::endl;
	
	//___________________________ Load Dataset image ___________________________ //
		
	cv::Mat img = cv::imread("../Dataset/rgb/02.jpg", cv::IMREAD_COLOR);
	cv::namedWindow("Original Image");
	cv::imshow("Original Image", img);
	cv::waitKey(0);

    
	//___________________________ Important parameters declaration ___________________________//
    std::vector<cv::Mat> img_roi(n_hands);
    std::vector<cv::Mat> img_roi_thr(n_hands);
    
	int x, y, width, height;	
	int temp = 0; // in order to get right index in vector of coordinates
	
	cv::Mat prediction_mask(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
	
	for (int i=0; i<n_hands; i++) 
	{
	    
	    //_________ ROI extraction _________//
    	x = coordinates_bb[i+temp];
	    y = coordinates_bb[i+temp+1];
	    width = coordinates_bb[i+temp+2];
	    height = coordinates_bb[i+temp+3];
	
		cv::Range colonna(x, x+width);
        cv::Range riga(y, y+height);
	    img_roi[i] = img(riga, colonna);
	    
      	cv::namedWindow("ROI");
	    cv::imshow("ROI", img_roi[i]);
	    cv::waitKey(0);
	    
	    //_________ Draw Detected Bounding Boxes _________//
    	cv::Point pt1(x, y);
        cv::Point pt2(x + width, y + height);
        cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0));
        
        temp = temp + 3;
	
	}
	
	cv::namedWindow("New Image");
	cv::imshow("New Image", img);
	cv::waitKey(0);
	

	
	/*
	
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
	*/
  
	//___________________________ ROI extraction ___________________________//
	
	/*
	cv::Range rows(x, x+width);
    cv::Range cols(y, y+height);
	cv::Mat img_roi_BGR = img(cols, rows);
	
  	cv::namedWindow("ROI");
	cv::imshow("ROI", img_roi_BGR);
	cv::waitKey(0);
	*/
	
	
	
	
	/*
	//__________________________ Change image color space __________________________//

	cv::Mat img_roi_HSV;
    cv::cvtColor(img_roi_BGR, img_roi_HSV, cv::COLOR_BGR2HSV);
    
    cv::Mat img_roi_YCrCb;
    cv::cvtColor(img_roi_BGR, img_roi_YCrCb, cv::COLOR_BGR2YCrCb);


    //__________________________ Calculate and Plot Histogram __________________________//
    
	Histogram h1;
	cv::Mat hist_roi_BGR = h1.calc_histogram(img_roi_BGR);
	cv::Mat hist_BGR = h1.plot_histogram(hist_roi_BGR);
	
    Histogram h2;
	cv::Mat hist_roi_HSV = h2.calc_histogram(img_roi_HSV);
	cv::Mat hist_HSV = h2.plot_histogram(hist_roi_HSV);
	
    Histogram h3;
	cv::Mat hist_roi_YCrCb = h3.calc_histogram(img_roi_YCrCb);
	cv::Mat hist_YCrCb = h3.plot_histogram(hist_roi_YCrCb);
    
    cv::namedWindow("Histogram BGR", cv::WINDOW_NORMAL);
	cv::imshow("Histogram BGR", hist_BGR);
	cv::namedWindow("Histogram HSV", cv::WINDOW_NORMAL);
	cv::imshow("Histogram HSV", hist_HSV);
	cv::namedWindow("Histogram YCrCb", cv::WINDOW_NORMAL);
	cv::imshow("Histogram YCrCb", hist_YCrCb);
	cv::waitKey(0);
    cv::destroyAllWindows();
	
	
	//___________________________ ROI segmentation Otsu thresholding ___________________________//	
	
    cv::Mat mask_otsu_BGR, mask_otsu_HSV, mask_otsu_YCrCb;    
    
    otsuSegmentation(img_roi_BGR, mask_otsu_BGR, 5, 1);
    otsuSegmentation(img_roi_HSV, mask_otsu_HSV, 5, 2); 
    otsuSegmentation(img_roi_YCrCb, mask_otsu_YCrCb, 5, 3); 
    
    
    //_____________________________ Multiplying mask to get final result _____________________________//
    
    cv::Mat mask_final_ROI(mask_otsu_BGR.rows, mask_otsu_BGR.cols, CV_8UC1, cv::Scalar::all(0)); // must be one channel
    
    cv::namedWindow("Otsu's thresholding final");
	cv::imshow("Otsu's thresholding final", mask_final_ROI);
	cv::waitKey(0);
    
    mask_final_ROI =  mask_otsu_BGR.mul(mask_otsu_HSV);
    cv::namedWindow("Otsu's thresholding final");
	cv::imshow("Otsu's thresholding final", mask_final_ROI);
	cv::waitKey(0);
	
	//______________ inserisci maschera in immagine nera stessa dimensione originale ______________//
	
    cv::Mat mask_final(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0)); 
    mask_final_ROI.copyTo(mask_final(cv::Rect(x, y, mask_final_ROI.cols, mask_final_ROI.rows)));
	cv::imshow("Mask final", mask_final);
	cv::waitKey(0);
	
	
	//___________________________________ Save ____________________________________//
	
	cv::imwrite("../results/mask_predict.png", mask_final);
	
	*/
	
	//__________________________ Detect if overlap _________________________________// 
	
	bool overlap = 0;
	
	int temp2 = 4;
	
	for (int i = 0; i < n_hands; i+=4) // ciclo for per controllo tutte le combinazioni di bounding box
	{
	      
	    temp2 = temp2 + 4;
	}
	
	int a = x + 300;
    int b = y + 300;
    int c = width + 300;
    int d = height + 300;
	    
	overlap = detectOverlapSegmentation(coordinates_bb[0], coordinates_bb[1], coordinates_bb[2], coordinates_bb[3], a, b, c, d);
	
	std::cout << overlap << std::endl;
	
	
	
	
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
    

    /*
    cv::Scalar lower_color(190, 130, 90); // 0 15 0 // 0 58 50
    cv::Scalar upper_color(210, 140, 115); // 17 170 255 // 30 255 255
    cv::Mat mask; 
    cv::inRange(img_roi, lower_color, upper_color, mask); 
    cv::imshow("nemo mask orange", mask);
    cv::waitKey(0);
    
    cv::Mat result;
    cv::bitwise_and(img_roi, img_roi, result, mask);
    cv::imshow("nemo result orange", result);
    cv::waitKey(0);
    
    
    
    // take middle pixel intensity value
    std::cout << "Middle pixel is " << (x+width)/2 << " and " << (y+height)/2 << std::endl;
    cv::Vec3b bgrPixel = img_roi.at<cv::Vec3b>((x+width)/2, (y+height)/2);
    std::cout << "Middle pixel value is " << bgrPixel << std::endl;
    
    
    // merge hand  detection (YCbCr and hsv)
    
    // HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))


    cv::namedWindow("img_YCrCb");
    cv::imshow("img_YCrCb", img_roi_YCrCb);
    cv::waitKey(0);
    
    

    //___________________________________ in Range segmentation ________________________//
    
    cv::Scalar lower_color(140, 92, 8); // 0 15 0 // 0 58 50
    cv::Scalar upper_color(168, 141, 11); // 17 170 255 // 30 255 255
    cv::Mat mask; 
    cv::inRange(img_roi_YCrCb, lower_color, upper_color, mask); 
    cv::imshow("nemo mask orange", mask);
    cv::waitKey(0);
    
    cv::Mat result;
    cv::bitwise_and(img_roi_YCrCb, img_roi_YCrCb, result, mask);
    cv::imshow("nemo result orange", result);
    cv::waitKey(0);
    
    
    /*
    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))


    /*

    global_mask = cv2::bitwise_and(YCrCb_mask,HSV_mask)
    global_mask = cv2::medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result=cv2.bitwise_not(global_mask)  
    */
    
    /*
    
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
    
	

	
	cv::Mat fin_img;
	std::vector<cv::Mat> channels_BR;
	channels_BR.push_back(Bands_RGB[0]);
	channels_BR.push_back(Bands_RGB[2]);
	cv::merge(channels_BR, fin_img);
    cv::imshow("merged BR", fin_img);
    
    cv::merge(channels_HSV, merged);
    cv::imshow("merged", merged);
    
	cv::waitKey(0);
	
	*/
	

  
	return 0;
  
}


