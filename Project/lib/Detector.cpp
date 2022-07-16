
/**
 * @file detection.cpp
 *
 * @brief  Detection algorithm, sliding Window approach
 *
 * @author Marco Mustacchi
 *
 */

#include "Detector.h"

using namespace cv;
using namespace std;


void Detector::skinDetectionColored(const cv::Mat& img, cv::Mat& img_threshold)
{

	    //______________________ Skin detection Colored ______________________//
	    // Detect the object based on HSV Range Values
	    
	    Scalar min_YCrCb(0,150,100);  // 0,150,100
	    Scalar max_YCrCb(255,200,150); // 235,173,127
	    inRange(img, min_YCrCb, max_YCrCb, img_threshold);
	    
		cv::namedWindow("Thresh Image");
		cv::imshow("Thresh Image", img_threshold);
		cv::waitKey(0);
		
		std::cout<< "Number of channels" << img_threshold.channels() << endl;
		
		// remove outliers
		removeDetectionOutliers(img_threshold);

}


void Detector::skinDetectionGrayscale(const cv::Mat& img, cv::Mat& img_threshold)
{

	    //______________________ Skin detection Grayscale ______________________//	    	    
	    cv::adaptiveThreshold(img, img_threshold, 255, cv::ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 1);
	    
    	cv::namedWindow(" Image");
	    cv::imshow(" Image", img_threshold);
	    
	    cv::Mat img_threshold_inv = ~img_threshold;
	    
    	cv::namedWindow("Thresh ");
	    cv::imshow("Thresh ", img_threshold_inv);
	    
	    cv::Mat img_threshold_inv_filled;
	    fillMaskHoles(img_threshold_inv, img_threshold_inv_filled);
	    
    	cv::namedWindow("teso ");
	    cv::imshow("teso ", img_threshold_inv_filled);
	    
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7));
        cv::morphologyEx(img_threshold_inv_filled, img_threshold_inv_filled, cv::MORPH_ERODE, kernel);
        
    	cv::namedWindow("erode ");
	    cv::imshow("erode ", img_threshold_inv_filled);
	    
	    std::cout<< "Number of channels" << img_threshold_inv_filled.channels() << endl;
	    
	    removeDetectionOutliers(img_threshold_inv_filled);
	    
	    img_threshold = img_threshold_inv_filled.clone();
	    
    	cv::namedWindow("img threshold");
	    cv::imshow("img threshold", img_threshold);
	    	    
}


void Detector::removeDetectionOutliers(cv::Mat& input) {
	
    cv::Mat labelImage, stats, centroids;
    
    int nLabels =  cv::connectedComponentsWithStats(input, labelImage, stats, centroids, 8);
    
    std::cout << "nLabels = " << nLabels << std::endl;
    std::cout << "stats.size() = " << stats.size() << std::endl;
	
	std::cout << stats.col(4) << std::endl;
	std::cout << "test2 colonna" << cv::CC_STAT_AREA << std::endl;
	
	// eseguo soltanto se numero di label e' > 4, altrimenti esco perche' non devo rimuovere nnt 
    
    if (nLabels <= 4) {
        return;
    } 
    else {
        int max1, max2, max3, max4;
        max1 = max2 = max3 = max4 = 0;
        
        for (int i = 1; i < nLabels; i++) { //label  0 is the background
            if ((stats.at<int>(i, cv::CC_STAT_AREA)) > max1) {
                max4 = max3;
                max3 = max2;
                max2 = max1;
                max1 = stats.at<int>(i, cv::CC_STAT_AREA);
            }
            else if ((stats.at<int>(i, cv::CC_STAT_AREA)) > max2) {
                max4 = max3;
                max3 = max2;
                max2 = stats.at<int>(i, cv::CC_STAT_AREA);
            } 
            else if ((stats.at<int>(i, cv::CC_STAT_AREA)) > max3) {
                max4 = max3;
                max3 = stats.at<int>(i, cv::CC_STAT_AREA);
            }
            else if ((stats.at<int>(i, cv::CC_STAT_AREA)) > max4) {
                max4 = stats.at<int>(i, cv::CC_STAT_AREA);
            }
        }
        
        std::cout << "Biggest area in order: " << max1 << " " << max2 << " " << max3 << " " << max4 << std::endl;
        
        Mat surfSup = stats.col(4) >= max4;

        Mat mask(labelImage.size(), CV_8UC1, Scalar(0));
        
        for (int i = 1; i < nLabels; i++)
        {
            if (surfSup.at<uchar>(i, 0))
            {
                mask = mask | (labelImage==i);
            }
        }
        
        input = mask.clone();
        
        std::cout<< "Number of channels" << input.channels() << endl;
        
        cv::namedWindow("mask");
        imshow("mask", input);
        cv::waitKey(1000);
    
    }
    
}


void Detector::slidingWindow(cv::Mat& img, cv::Mat& img_threshold, int windows_n_rows, int windows_n_cols, int stepSlideRow, int stepSlideCols, std::vector<std::vector<int>>& coordinates_bb)
{

    //Input CNN
    const int WIDTH_INPUT_CNN = 224;
    const int HEIGHT_INPUT_CNN = 224;

    //Threshold used to understand if a blob is an image or not
    const float THRESHOLD_DETECTION = 0.01f;
    
    std::tuple<int, int> orginalDimensions = std::make_tuple(img.rows, img.cols);
    
    for (int r = 0; r < img.rows - windows_n_rows; r += stepSlideRow)
    {
        //Range of rows coordinates
        cv::Range rowRange(r, r + windows_n_rows);
        
        for (int c = 0; c < img.cols - windows_n_cols; c += stepSlideCols)
        {			
            //Range of cols coordinates
            cv::Range colRange(c, c + windows_n_cols);

            cv::Mat roi = img(rowRange, colRange);
            cv::Mat roi_threshold = img_threshold(rowRange, colRange);
            
            if ( cv::countNonZero(roi_threshold) == 0 ) // could be an hand
                continue;

            // cv::namedWindow("Thresh ROI");
            // cv::imshow("Thresh ROI", roi_threshold);
            // cv::waitKey(0);
            
            cv::Mat resized, outputImage;

            cv::resize(roi, resized, cv::Size(WIDTH_INPUT_CNN, HEIGHT_INPUT_CNN), cv::INTER_CUBIC);
            
            // cv::namedWindow("roi resize");
            // cv::imshow("roi resize", resized);
            // cv::waitKey(0);

            resized.convertTo(outputImage, CV_32FC3); // Convert to CV_32
            
            cv::dnn::Net network = cv::dnn::readNetFromTensorflow("../model/model.pb");
            
            network.setInput(cv::dnn::blobFromImage(roi, 1.0 / 255.0, cv::Size(WIDTH_INPUT_CNN, HEIGHT_INPUT_CNN), true, false));

            cv::Mat outputCNN = network.forward();
            
            cout << "Output CNN: " << outputCNN.at<float>(0, 0) << endl;
            
            bool hand_Detected;
            
            if (outputCNN.at<float>(0, 0) < THRESHOLD_DETECTION) {
                cout << "found hand" << endl;
    			cv::namedWindow("Hand");
                cv::imshow("Hand", roi);
                cv::waitKey(1000);
                                           
                std::tuple<int, int, int, int> resizedCoordinates = std::make_tuple(c, r, c + windows_n_cols, r + windows_n_rows);
                std::tuple<int, int, int, int> originalCoordinates;
                std::tuple<int, int> resizedDimension = std::make_tuple(img.rows, img.cols);

	            //Need to convert bounding box coordinates to original image size
                convert2originalXYWH(resizedCoordinates, resizedDimension, originalCoordinates, orginalDimensions); // no need to use detect.convert because inside class
                                            
                vector<int> inVect; // Define the inner vector
                inVect.push_back(std::get<0>(originalCoordinates));
                inVect.push_back(std::get<1>(originalCoordinates));
                inVect.push_back(std::get<2>(originalCoordinates));
                inVect.push_back(std::get<3>(originalCoordinates));
                inVect.push_back(outputCNN.at<float>(0, 0) * 1000000);  // last column confidence in percentage, because int
                
                //Insert the inner vector to outer vector
                coordinates_bb.push_back(inVect);
	            
            }
            
            // A questo punto posso trovare anche piu' mani, quindi detectOverlap per tutte le mani trovate 
        }
    }
    
}

float Detector::detectionIoU(std::vector<int> loop_coordinates_bb_old, std::vector<int> loop_coordinates_bb_new)
{

    // last element [4] is the confidence
    int x_truth = loop_coordinates_bb_old[0];
    int y_truth = loop_coordinates_bb_old[1];
    int width_truth = loop_coordinates_bb_old[2];
    int height_truth = loop_coordinates_bb_old[3];
    int x_predict = loop_coordinates_bb_new[0];
    int y_predict = loop_coordinates_bb_new[1];
    int width_predict = loop_coordinates_bb_new[2];
    int height_predict = loop_coordinates_bb_new[3];
    
    // coordinates of the intersection Area
    int xA = std::max(x_truth, x_predict);
    int yA = std::max(y_truth, y_predict);
    int xB = std::min(x_truth+width_truth, x_predict+width_predict);
    int yB = std::min(y_truth+height_truth, y_predict+height_predict);
    
    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    
    int area_box_truth = width_truth * height_truth;
    int area_box_predict = width_predict * height_predict;
    
    float iou = (float) interArea / (area_box_truth + area_box_predict - interArea);
    
    return iou;

}

bool sortcolDetection(const std::vector<int>& v1, const std::vector<int>& v2)
{
    return v1[4] < v2[4];
}


void Detector::nonMaximumSuppression(std::vector<std::vector<int>>& old_coordinates_bb, std::vector<std::vector<int>>& new_coordinates_bb) 
{

    //____________________________________________________ Sort rows by last column ____________________________________________________//
    // Use of "sort()" for sorting on basis of 5th (last) column
    sort(old_coordinates_bb.begin(), old_coordinates_bb.end(), sortcolDetection);
    
    //lets check out the elements of the 2D vector are sorted correctly
    std::cout << "Ordered coordinates by confidence" << std::endl;
    
    for(std::vector<int> &newvec: old_coordinates_bb)
    {
        for(const int &elem: newvec)
        {
            std::cout<<elem<<" ";
        }
        std::cout<<std::endl;
    }
    
    float threshIoU = 0.5f;
    
    //______________________________________ Take last column in a loop, move and compare IoU ______________________________________//
    
    while ( old_coordinates_bb.size() > 0 )
    {
        int i = 0;
        
        new_coordinates_bb.push_back(old_coordinates_bb.back());
        old_coordinates_bb.erase(old_coordinates_bb.end());
        
        for (int j=0; j<old_coordinates_bb.size(); j++) 
        {
            float iou = detectionIoU( old_coordinates_bb.at(j), old_coordinates_bb.at(i));
            
            if (iou < threshIoU)
            {
                old_coordinates_bb.erase(old_coordinates_bb.begin()+j);
            }
        }
        
        i = i+1;
    
    }
    
}


bool Detector::imgGrayscale(const cv::Mat& img)
{
	
	cv::Mat temp = img.clone();
    cv::Mat Bands_BGR[3];
    cv::split(temp, Bands_BGR);
    
	for (int r = 0; r < temp.rows; r++)	
		for (int c = 0; c < temp.cols; c++)		
			if ( (Bands_BGR[0].at<uchar>(r, c) != Bands_BGR[1].at<uchar>(r, c)) // Grayscale if all the channel equal for all the pixel 
			    || (Bands_BGR[0].at<uchar>(r, c) != Bands_BGR[2].at<uchar>(r, c)) // so just find a pixel different in one channel
			    || (Bands_BGR[1].at<uchar>(r, c) != Bands_BGR[2].at<uchar>(r, c)) )
				return false;
				
	return true;
	
}


void Detector::convert2originalXYWH(const std::tuple<int, int, int, int>& resizedCoordinates, const std::tuple<int, int>& resizedDimension, 
                            std::tuple<int, int, int, int>& originalCoordinates, const std::tuple<int, int>& orginalDimensions) 
{
    
    int x = (std::get<0>(resizedCoordinates) * std::get<1>(orginalDimensions) ) / (std::get<1>(resizedDimension));
    int y = (std::get<1>(resizedCoordinates) * std::get<0>(orginalDimensions) ) / (std::get<0>(resizedDimension));
    int x2 = (std::get<2>(resizedCoordinates) * std::get<1>(orginalDimensions) ) / (std::get<1>(resizedDimension));
    int y2 = (std::get<3>(resizedCoordinates) * std::get<0>(orginalDimensions) ) / (std::get<0>(resizedDimension));
    	
    int width = x2-x;
    int height = y2-y; 
    
    originalCoordinates = std::make_tuple(x, y, width, height);

}
