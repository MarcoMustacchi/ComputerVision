
/**
 * @file Detector.cpp
 *
 * @brief  Detector class functions
 *
 * @author Marco Mustacchi
 *
 */

#include "Detector.h"

using namespace cv;
using namespace std;


int Detector::skinDetectionColored(const cv::Mat& img, cv::Mat& img_threshold)
{

	    //______________________ Skin detection Colored ______________________//
	    // Detect the object based on HSV Range Values
	    
	    Scalar min_YCrCb(0,150,100);  // 0,150,100
	    Scalar max_YCrCb(255,200,150); // 235,173,127
	    inRange(img, min_YCrCb, max_YCrCb, img_threshold);
	    
		cv::namedWindow("Thresh Image");
		cv::imshow("Thresh Image", img_threshold);
		cv::waitKey(0);
		
		// remove outliers
		int maxArea = removeDetectionOutliers(img_threshold);
		
		cout << "Max area is " << maxArea << endl;
		
		return maxArea;

}


int Detector::skinDetectionGrayscale(const cv::Mat& img, cv::Mat& img_threshold)
{

	    //______________________ Skin detection Grayscale ______________________//	    	    
	    cv::adaptiveThreshold(img, img_threshold, 255, cv::ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 1);
	    
    	cv::namedWindow("Image");
	    cv::imshow("Image", img_threshold);
	    
	    cv::Mat img_threshold_inv = ~img_threshold;
	    
    	cv::namedWindow("Thresh");
	    cv::imshow("Thresh", img_threshold_inv);
	    
	    cv::Mat img_threshold_inv_filled;
	    fillMaskHoles(img_threshold_inv, img_threshold_inv_filled);
	    
    	cv::namedWindow("inverse threshold");
	    cv::imshow("inverse threshold", img_threshold_inv_filled);
	    
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7));
        cv::morphologyEx(img_threshold_inv_filled, img_threshold_inv_filled, cv::MORPH_ERODE, kernel);
        
    	cv::namedWindow("inverse threshold erode");
	    cv::imshow("inverse threshold erode", img_threshold_inv_filled);
	    
	    std::cout<< "Number of channels" << img_threshold_inv_filled.channels() << endl;
	    
	    int maxArea = removeDetectionOutliers(img_threshold_inv_filled);
	    
	    cout << "Max area is " << maxArea << endl;
	    
	    img_threshold = img_threshold_inv_filled.clone();
	    
	    return maxArea;
	    	    
}


int Detector::removeDetectionOutliers(cv::Mat& input) {
	
    cv::Mat labelImage, stats, centroids;
    
    int nLabels =  cv::connectedComponentsWithStats(input, labelImage, stats, centroids, 8);
    
    // std::cout << "nLabels = " << nLabels << std::endl;
    // std::cout << "stats.size() = " << stats.size() << std::endl;
	
	// std::cout << stats.col(4) << std::endl;
	// std::cout << "test2 colonna" << cv::CC_STAT_AREA << std::endl;
	
	// eseguo soltanto se numero di label e' > 4, altrimenti esco perche' non devo rimuovere nnt 
    
    if (nLabels <= 4) {
        int max_stats = 0;
    
        for (int label = 1; label < nLabels; label++) { //label  0 is the background
            if ((stats.at<int>(label, cv::CC_STAT_AREA)) > max_stats) { // in order to get row i and column CC_STAT_AREA (=4)
                max_stats = stats.at<int>(label, cv::CC_STAT_AREA);
            }
        }
        return max_stats;
    } 
    else 
    {
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
        
    	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
        cv::morphologyEx(input, input, cv::MORPH_DILATE, kernel);
        
        cv::namedWindow("mask");
        imshow("mask", input);
        cv::waitKey(1000);
        
        return max1;

    }
    
}


int getNextIndex(int currentIndex, int frameSize, int roi, int step)
{
    int temp = min(currentIndex + step, frameSize - roi - 1);
    return currentIndex != temp ? temp : frameSize;
}


void Detector::slidingWindow(cv::Mat& img, cv::Mat& img_threshold, int windows_n_rows, int windows_n_cols, int stepSlideRow, int stepSlideCols, std::vector<std::vector<int>>& coordinates_bb, int maxArea)
{

    const int WIDTH_INPUT_CNN = 224;
    const int HEIGHT_INPUT_CNN = 224;
    const float THRESHOLD_DETECTION = 0.025f;
    
    std::tuple<int, int> initDimension = std::make_tuple(img.rows, img.cols);
    
    // check if maxArea of skin detection is large enough, otherwise probably no hand, so I have to check the whole image
    if ( (img.cols >= 1280 && img.rows >= 720 && maxArea >= 11000) || (img.cols < 1280 && img.rows < 720 && maxArea >= 1500) ) 
        cout << "Check only skin" << endl;
    
    for (int r = 0; r < img.rows - windows_n_rows; r = getNextIndex(r, img.rows, windows_n_rows, stepSlideRow))
    {
        //Range of rows coordinates
        cv::Range rowRange(r, r + windows_n_rows);
        
        for (int c = 0; c < img.cols - windows_n_cols; c = getNextIndex(c, img.cols, windows_n_cols, stepSlideCols))
        {			
            //Range of cols coordinates
            cv::Range colRange(c, c + windows_n_cols);

            cv::Mat roi = img(rowRange, colRange);
            cv::Mat roi_threshold = img_threshold(rowRange, colRange);
            
            // check if maxArea of skin detection is large enough, otherwise probably no hand, so I have to check the whole image
            if ( (img.cols >= 1280 && img.rows >= 720 && maxArea >= 11000) || (img.cols < 1280 && img.rows < 720 && maxArea >= 1500) ) 
                if ( cv::countNonZero(roi_threshold) == 0 ) // could be an hand
                {
                    continue;
                }
            
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
                std::tuple<int, int, int, int> initCoordinates;
                std::tuple<int, int> resizedDimension = std::make_tuple(img.rows, img.cols);

	            //Need to convert bounding box coordinates to original image size
                convert2originalXYWH(resizedCoordinates, resizedDimension, initCoordinates, initDimension); // no need to use detect.convert because inside class
                                            
                vector<int> inVect; // Define the inner vector
                inVect.push_back(std::get<0>(initCoordinates));
                inVect.push_back(std::get<1>(initCoordinates));
                inVect.push_back(std::get<2>(initCoordinates));
                inVect.push_back(std::get<3>(initCoordinates));
                inVect.push_back(outputCNN.at<float>(0, 0) * 1000000);  // last column confidence in percentage, because int
                
                //Insert the inner vector to outer vector
                coordinates_bb.push_back(inVect);
	            
            }
            
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
    
    float interArea = (float) std::max(0, xB - xA) * std::max(0, yB - yA);
    
    float area_box_truth = width_truth * height_truth;
    float area_box_predict = width_predict * height_predict;
    
    float iou = interArea / (area_box_truth + area_box_predict - interArea);
    
    return iou;

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
                            std::tuple<int, int, int, int>& initCoordinates, const std::tuple<int, int>& initDimension) 
{
    
    int x = (std::get<0>(resizedCoordinates) * std::get<1>(initDimension) ) / (std::get<1>(resizedDimension));
    int y = (std::get<1>(resizedCoordinates) * std::get<0>(initDimension) ) / (std::get<0>(resizedDimension));
    int x2 = (std::get<2>(resizedCoordinates) * std::get<1>(initDimension) ) / (std::get<1>(resizedDimension));
    int y2 = (std::get<3>(resizedCoordinates) * std::get<0>(initDimension) ) / (std::get<0>(resizedDimension));
    	
    int width = x2-x;
    int height = y2-y; 
    
    initCoordinates = std::make_tuple(x, y, width, height);

}



bool sortcolDetection(const std::vector<int>& v1, const std::vector<int>& v2)
{
    return v1[4] > v2[4];
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
    
    float threshIoU = 0.05f;
    
    //______________________________________ Take last row in a loop, move and compare IoU ______________________________________//
    
    while ( old_coordinates_bb.size() > 0 )
    {
        int i = 0;
        
        new_coordinates_bb.push_back(old_coordinates_bb.back());
        old_coordinates_bb.pop_back();
        
        for (int j=0; j<old_coordinates_bb.size(); j++) 
        {
            float iou = detectionIoU(old_coordinates_bb.at(j), new_coordinates_bb.at(i));
                   
            if (iou >= threshIoU)
            {
                old_coordinates_bb.erase(old_coordinates_bb.begin()+j);
            }
        }
        
        i = i+1;
    }
    
}



void Detector::handleBoundingBoxesInside(std::vector<std::vector<int>>& new_coordinates_bb)
{
    //_____________ remove Bounding Box if is fully inside another one (non Maximum Suppression doesn't remove it) _____________//
    std::vector<int> indices;
    
    for (int i=0; i<new_coordinates_bb.size(); i++) 
    {
        for (int j=0; j<new_coordinates_bb.size(); j++) 
        {
            if (new_coordinates_bb[i] == new_coordinates_bb[j]) // altrimenti stesso rettangolo confrontato con se stesso risulta inside
            {
                continue;
            }
            
            cv::Rect a(new_coordinates_bb[i][0], new_coordinates_bb[i][1], new_coordinates_bb[i][2], new_coordinates_bb[i][3]);
            cv::Rect b(new_coordinates_bb[j][0], new_coordinates_bb[j][1], new_coordinates_bb[j][2], new_coordinates_bb[j][3]);
            
            // float iou = detectionIoU(new_coordinates_bb.at(j), new_coordinates_bb.at(i));
            
            // if ( iou != 0 && new_coordinates_bb[i][3] != new_coordinates_bb[j][] )
            
            if ((a & b) == a) // means that b is inside a
            {
                if (new_coordinates_bb[i][4] < new_coordinates_bb[j][4])
                    indices.push_back(j);
                else 
                    indices.push_back(i);
                    break; // break will exit only the innermost loop containing it
            }
        }
    }
    
    // attenzione, perche' in questo modo se piu' rettangoli contengono lo stesso rettangolo, devo rimuovere tutti gli indici uguali,
    // altrimenti rimuovo piu' volte
    
    sort( indices.begin(), indices.end() );
    indices.erase( unique( indices.begin(), indices.end() ), indices.end() );
    
    for (int j = 0; j < indices.size(); j++)
        cout << indices[j] << " ";

    
    for (int i=0; i<indices.size(); i++) 
    {
        new_coordinates_bb.erase(new_coordinates_bb.begin()+indices[i]);
    }
    
}


void Detector::joinBoundingBoxesHorizontally(std::vector<std::vector<int>>& new_coordinates_bb)
{
    
    for (int i=0; i<new_coordinates_bb.size(); i++) 
    {
        for (int j=0; j<new_coordinates_bb.size(); j++) 
        {
            if (new_coordinates_bb[i] == new_coordinates_bb[j]) // altrimenti stesso rettangolo confrontato con se stesso risulta inside
            {
                continue;
            }
            
            float iou = detectionIoU(new_coordinates_bb.at(j), new_coordinates_bb.at(i));
                        
            if ( iou!=0 && new_coordinates_bb[i][1]==new_coordinates_bb[j][1] ) // y is the same
            {
                new_coordinates_bb[i][2] = (new_coordinates_bb[j][0]+new_coordinates_bb[j][2]) - new_coordinates_bb[i][0]; // modify x length
                new_coordinates_bb[i][4] = min(new_coordinates_bb[i][4], new_coordinates_bb[j][4]); // keep lower confidence
                new_coordinates_bb.erase(new_coordinates_bb.begin()+j); 
            }
        }
    }

}


void Detector::joinBoundingBoxesVertically(std::vector<std::vector<int>>& new_coordinates_bb)
{
    
    for (int i=0; i<new_coordinates_bb.size(); i++) 
    {
        for (int j=0; j<new_coordinates_bb.size(); j++) 
        {
            if (new_coordinates_bb[i] == new_coordinates_bb[j]) // altrimenti stesso rettangolo confrontato con se stesso risulta inside
            {
                continue;
            }
            
            float iou = detectionIoU(new_coordinates_bb.at(j), new_coordinates_bb.at(i));
                        
            if ( iou!=0 && new_coordinates_bb[i][0]==new_coordinates_bb[j][0] ) // x is the same
            {
                new_coordinates_bb[i][3] = (new_coordinates_bb[j][1]+new_coordinates_bb[j][3]) - new_coordinates_bb[i][1]; // modify y length
                new_coordinates_bb[i][4] = min(new_coordinates_bb[i][4], new_coordinates_bb[j][4]); // keep lower confidence
                new_coordinates_bb.erase(new_coordinates_bb.begin()+j); 
            }
        }
    }
    
}


void Detector::deleteRedundantBB(const cv::Mat img, std::vector<std::vector<int>>& new_coordinates_bb)
{
    
    if ( img.cols >= 1280 && img.rows >= 720 && new_coordinates_bb.size() > 4 )
    {
        int num_BB = new_coordinates_bb.size();
        int diff = num_BB - 4;
        
        for (int i=0; i<diff; i++) 
            new_coordinates_bb.pop_back();   
    }
    else if ( img.cols < 1280 && img.rows < 720 && new_coordinates_bb.size() >= 2 )
    {
        int num_BB = new_coordinates_bb.size();
        int diff = num_BB - 2;
        
        for (int i=0; i<diff; i++) 
        {
            new_coordinates_bb.pop_back();
        }
        
        float iou = detectionIoU(new_coordinates_bb[0], new_coordinates_bb[1]);
        
        if ( (iou != 0) || (new_coordinates_bb[0][0] == new_coordinates_bb[1][0]) || (new_coordinates_bb[0][1] == new_coordinates_bb[1][1])) 
        {
                new_coordinates_bb.pop_back();
        }   
    }
}

void Detector::deleteAlignBB(std::vector<std::vector<int>>& new_coordinates_bb)
{
   
    for (int i=0; i<new_coordinates_bb.size(); i++) 
    {
        for (int j=0; j<new_coordinates_bb.size(); j++) 
        {
            if (new_coordinates_bb[i] == new_coordinates_bb[j]) // altrimenti stesso rettangolo confrontato con se stesso risulta inside
            {
                continue;
            }
                        
            if ( new_coordinates_bb[i][0] == new_coordinates_bb[j][0] )
            {
                new_coordinates_bb.erase(new_coordinates_bb.begin()+j); 
            }
        }
    } 

}


