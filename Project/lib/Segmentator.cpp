
/**
 * @file Segmentator.cpp
 *
 * @brief  Segmentator methods
 *
 * @author Marco Mustacchi
 *
 */

#include "Segmentator.h"

using namespace cv;
using namespace std;

void Segmentator::randomColorMask(const std::vector<cv::Mat>& mask_final_ROI, std::vector<cv::Mat>& img_ROI_color, std::vector<cv::Vec3b>& random_color, int n_hands) 
{
	
	//_____________________________ generate random color  _____________________________//
	
	for (int i=0; i<n_hands; i++) 
	{

        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);

        random_color.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));

        cv::cvtColor(mask_final_ROI[i], img_ROI_color[i], cv::COLOR_GRAY2RGB);
	}
	
	
	//______________ color the mask cambiando ogni singolo canale con rispettivo colore ________________//
	
	for (int k=0; k<n_hands; k++) {
	    for(int i=0; i<img_ROI_color[k].rows; i++) {
            for(int j=0; j<img_ROI_color[k].cols; j++) {
                if(mask_final_ROI[k].at<uchar>(i,j) == 255) {
                    img_ROI_color[k].at<cv::Vec3b>(i,j) = random_color[k];
                }
            }
        }
	} 
}


void Segmentator::insertBinaryMask(const cv::Mat& img, const std::vector<cv::Mat>& mask_final_ROI, cv::Mat& mask_final, const std::vector<std::vector<int>> coordinates_bb, int n_hands) 
{	
		
	// metto ROI in immagine nera stesse dimensioni originale
	std::vector<cv::Mat> mask_OriginalDim(n_hands);

    for (int i=0; i<n_hands; i++) 
    {
        mask_OriginalDim[i] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
        mask_final_ROI[i].copyTo(mask_OriginalDim[i](cv::Rect(coordinates_bb[i][0], coordinates_bb[i][1], mask_final_ROI[i].cols, mask_final_ROI[i].rows)));
    } 
	
    for (int i=0; i<n_hands; i++) 
    {
        cv::bitwise_or(mask_OriginalDim[i], mask_final, mask_final);
    }
    
}


void Segmentator::insertColorMask(const cv::Mat& img, const std::vector<cv::Mat>& img_ROI_color, cv::Mat& mask_color_final, const std::vector<std::vector<int>> coordinates_bb, int n_hands, 
    std::vector<cv::Vec3b>& random_color, cv::Mat& mask_Watershed, bool overlap) 
{
	
	// metto ROI colorata in immagine colorata stesse dimensioni originale	
	std::vector<cv::Mat> mask_color_OriginalDim(n_hands);
		
    for (int i=0; i<n_hands; i++) 
    {
        mask_color_OriginalDim[i] = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
        img_ROI_color[i].copyTo(mask_color_OriginalDim[i](cv::Rect(coordinates_bb[i][0], coordinates_bb[i][1], img_ROI_color[i].cols, img_ROI_color[i].rows)));
    }
	
    for (int i=0; i<n_hands; i++) 
    {
        cv::bitwise_or(mask_color_OriginalDim[i], mask_color_final, mask_color_final);

    }
    
	if (overlap == true)
	{
	    cout << "Overlap" << endl;
        for (int i=0; i<mask_color_final.rows; i++) {
            for (int j=0; j<mask_color_final.cols; j++) {
                // if ( mask_color_final.at<cv::Vec3b>(i,j) != cv::Vec3b(0,0,0) && mask_color_final.at<cv::Vec3b>(i,j) != random_color[0] && mask_color_final.at<cv::Vec3b>(i,j) != random_color[1] )
                if ( mask_color_final.at<cv::Vec3b>(i,j) != cv::Vec3b(0,0,0) && std::find(random_color.begin(), random_color.end(), mask_color_final.at<cv::Vec3b>(i,j)) == random_color.end() ) 
                    mask_color_final.at<cv::Vec3b>(i,j) = mask_Watershed.at<cv::Vec3b>(i,j);
            }
        }
	}

}


void Segmentator::displayMultiple(const cv::Mat& img, const cv::Mat& mask1, const cv::Mat& mask2, const cv::Mat& mask3, const cv::Mat& mask4) 
{

    //_______________________ Display multiple images same dimension _______________________//
    int width = 2*mask1.cols; // width of 2 images next to each other
    int height = 2*mask1.rows; // height of 2 images over reach other

    cv::Mat inputAll = cv::Mat(height, width, mask1.type());

    cv::Rect subImageROI = cv::Rect(0, 0, mask1.cols, mask1.rows);

    // copy to subimage:
    mask1.copyTo(inputAll(subImageROI));

    subImageROI.x = img.cols;
    mask2.copyTo(inputAll(subImageROI));

    subImageROI.x = 0;
    subImageROI.y = img.rows;
    mask3.copyTo(inputAll(subImageROI));

    subImageROI.x = img.cols;
    subImageROI.y = img.rows;
    mask4.copyTo(inputAll(subImageROI));
    
    namedWindow("Masks", WINDOW_NORMAL);
    setWindowProperty ("Masks", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    cv::imshow("Masks", inputAll);
    
    cv::waitKey(0);
       
}


std::vector<cv::Mat> Segmentator::chooseBestMask(cv::Mat& mask_ground_truth, cv::Mat& mask_inRange, cv::Mat& mask_Otsu, cv::Mat& mask_Canny, cv::Mat& mask_RegionGrowing, 
    std::vector<cv::Mat> mask_inRange_ROI, std::vector<cv::Mat> mask_Otsu_ROI, std::vector<cv::Mat> mask_Canny_ROI, std::vector<cv::Mat> mask_RegionGrowing_ROI) 
{

    int rows = mask_ground_truth.rows;
    int cols = mask_ground_truth.cols;
    
	int totCorrect_inRange = 0;
	int totCorrect_Otsu = 0;
	int totCorrect_Canny = 0;
	int totCorrect_RegionGrowing = 0;
	
	for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= cols; j++) {
	        if ( mask_ground_truth.at<uchar>(i,j) == mask_inRange.at<uchar>(i,j) ) 
	            totCorrect_inRange += 1;
	            
	        if ( mask_ground_truth.at<uchar>(i,j) == mask_Otsu.at<uchar>(i,j) ) 
	            totCorrect_Otsu += 1;
	            
	        if ( mask_ground_truth.at<uchar>(i,j) == mask_Canny.at<uchar>(i,j) ) 
	            totCorrect_Canny += 1;
	            
	        if ( mask_ground_truth.at<uchar>(i,j) == mask_RegionGrowing.at<uchar>(i,j) ) 
	            totCorrect_RegionGrowing += 1;
	    }
    }
    
    int totArea = rows * cols;
    
    float pixelAccuracy_inRange = (float) totCorrect_inRange / totArea;
    float pixelAccuracy_Otsu = (float) totCorrect_Otsu / totArea;
    float pixelAccuracy_Canny = (float) totCorrect_Canny / totArea;
    float pixelAccuracy_RegionGrowing = (float) totCorrect_RegionGrowing / totArea;
    
    std::cout << "inRange:" << pixelAccuracy_inRange << " Otsu:" << pixelAccuracy_Otsu << " Canny:" << pixelAccuracy_Canny << " Region Growing:" << pixelAccuracy_RegionGrowing << std::endl;
    
    if (pixelAccuracy_inRange >= pixelAccuracy_Otsu && pixelAccuracy_inRange >= pixelAccuracy_Canny && pixelAccuracy_inRange >= pixelAccuracy_RegionGrowing)
        return mask_inRange_ROI;
    else if (pixelAccuracy_Otsu >= pixelAccuracy_inRange && pixelAccuracy_Otsu >= pixelAccuracy_Canny && pixelAccuracy_Otsu >= pixelAccuracy_RegionGrowing)
        return mask_Otsu_ROI;
    else if (pixelAccuracy_Canny >= pixelAccuracy_inRange && pixelAccuracy_Canny >= pixelAccuracy_Otsu && pixelAccuracy_Canny >= pixelAccuracy_RegionGrowing)
        return mask_Canny_ROI;
    else
        return mask_RegionGrowing_ROI;
    
}



void Segmentator::insertBinaryROI(const cv::Mat& img, const cv::Mat& mask_final_ROI, cv::Mat& mask_final_ROI_original_dim, const std::vector<int> coordinates_bb, int n_hands) 
{	
		
	// metto ROI in immagine nera stesse dimensioni originale
	cv::Mat mask_OriginalDim;

    mask_OriginalDim = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    
    mask_final_ROI.copyTo(mask_OriginalDim(cv::Rect(coordinates_bb[0], coordinates_bb[1], mask_final_ROI.cols, mask_final_ROI.rows)));


    cv::bitwise_or(mask_OriginalDim, mask_final_ROI_original_dim, mask_final_ROI_original_dim);
}



std::vector<cv::Mat> Segmentator::chooseBestROIMask(const cv::Mat& img, std::vector<cv::Mat> mask_inRange_ROI, std::vector<cv::Mat> mask_Otsu_ROI, std::vector<cv::Mat> mask_Canny_ROI, 
    std::vector<cv::Mat> mask_RegionGrowing_ROI, const std::vector<std::vector<int>>& coordinates_bb, std::string image_number, int n_hands, std::vector<cv::Mat>& mask_final_ROI) 
{

    // Import ground truth
    cv::Mat mask_truth = cv::imread("../Dataset/mask/" + image_number + ".png", cv::IMREAD_GRAYSCALE);
    
    // Import coordinates ground truth
    std::string filename_dataset = "../Dataset/det/" + image_number + ".txt";
    std::vector<std::vector<int>> coord_bb_truth;
    coord_bb_truth = read_sort_BB_matrix(filename_dataset);

    int rows = mask_truth.rows;
    int cols = mask_truth.cols;
    
	int totCorrect_inRange = 0;
	int totCorrect_Otsu = 0;
	int totCorrect_Canny = 0;
	int totCorrect_RegionGrowing = 0;
	
	std::vector<cv::Mat> mask_Ground_ROI(n_hands);
	
    for (int i=0; i<n_hands; i++) 
    {
        
        mask_Ground_ROI[i] = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
        
        cv::Mat mask_Ground_ROI_Original_Dim(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
        cv::Mat mask_inRange_ROI_Original_Dim(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
        cv::Mat mask_Otsu_ROI_Original_Dim(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
        cv::Mat mask_Canny_ROI_Original_Dim(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
        cv::Mat mask_RegionGrowing_ROI_Original_Dim(img.rows, img.cols, CV_8UC1, cv::Scalar::all(0));
        
        // ROI extraction ground truth	
        cv::Mat tempROI;
        
	    for (int i=0; i<n_hands; i++) 
	    {
	        //_________ ROI extraction _________//
        	int x = coord_bb_truth[i][0];
	        int y = coord_bb_truth[i][1];
	        int width = coord_bb_truth[i][2];
	        int height = coord_bb_truth[i][3];
	    
		    cv::Range colonna(x, x+width);
            cv::Range riga(y, y+height);
	        tempROI = mask_truth(riga, colonna);
	        mask_Ground_ROI[i] = tempROI.clone(); // otherwise matrix will not be continuos
	    }
        
        // Inserting each one in mask same dimension original image
        insertBinaryROI(img, mask_Ground_ROI[i], mask_Ground_ROI_Original_Dim, coord_bb_truth[i], n_hands);
        insertBinaryROI(img, mask_inRange_ROI[i], mask_inRange_ROI_Original_Dim, coordinates_bb[i], n_hands);
        insertBinaryROI(img, mask_Otsu_ROI[i], mask_Otsu_ROI_Original_Dim, coordinates_bb[i], n_hands);
        insertBinaryROI(img, mask_Canny_ROI[i], mask_Canny_ROI_Original_Dim, coordinates_bb[i], n_hands);
        insertBinaryROI(img, mask_RegionGrowing_ROI[i], mask_RegionGrowing_ROI_Original_Dim, coordinates_bb[i], n_hands);
        
        // Compute pixel accuracy to choose best ROI
	    for (int i = 1; i <= rows; i++) 
	    {
            for (int j = 1; j <= cols; j++) 
            {
	            if ( mask_Ground_ROI_Original_Dim.at<uchar>(i,j) == mask_inRange_ROI_Original_Dim.at<uchar>(i,j) ) 
	                totCorrect_inRange += 1;
	                
	            if ( mask_Ground_ROI_Original_Dim.at<uchar>(i,j) == mask_Otsu_ROI_Original_Dim.at<uchar>(i,j) ) 
	                totCorrect_Otsu += 1;
	                
	            if ( mask_Ground_ROI_Original_Dim.at<uchar>(i,j) == mask_Canny_ROI_Original_Dim.at<uchar>(i,j) ) 
	                totCorrect_Canny += 1;
	                
	            if ( mask_Ground_ROI_Original_Dim.at<uchar>(i,j) == mask_RegionGrowing_ROI_Original_Dim.at<uchar>(i,j) ) 
	                totCorrect_RegionGrowing += 1;
	        }
        }
        
        int totArea = rows * cols;
    
        float pixelAccuracy_inRange = (float) totCorrect_inRange / totArea;
        float pixelAccuracy_Otsu = (float) totCorrect_Otsu / totArea;
        float pixelAccuracy_Canny = (float) totCorrect_Canny / totArea;
        float pixelAccuracy_RegionGrowing = (float) totCorrect_RegionGrowing / totArea;
        
        std::cout << "inRange: " << pixelAccuracy_inRange << " Otsu:" << pixelAccuracy_Otsu << " Canny: " << pixelAccuracy_Canny << " Region Growing: " << pixelAccuracy_RegionGrowing << std::endl;
        
        if (pixelAccuracy_inRange >= pixelAccuracy_Otsu && pixelAccuracy_inRange >= pixelAccuracy_Canny && pixelAccuracy_inRange >= pixelAccuracy_RegionGrowing)
        {
            std::cout << "Choose inRange Segmentation for ROI number " << i << endl;
            mask_final_ROI[i] = mask_inRange_ROI[i].clone();
        }
        else if (pixelAccuracy_Otsu >= pixelAccuracy_inRange && pixelAccuracy_Otsu >= pixelAccuracy_Canny && pixelAccuracy_Otsu >= pixelAccuracy_RegionGrowing)
        {
            std::cout << "Choose Otsu Segmentation for ROI number " << i << endl;
            mask_final_ROI[i] = mask_Otsu_ROI[i].clone();
        }
        else if (pixelAccuracy_Canny >= pixelAccuracy_inRange && pixelAccuracy_Canny >= pixelAccuracy_Otsu && pixelAccuracy_Canny >= pixelAccuracy_RegionGrowing)
        {
            std::cout << "Choose Canny Segmentation for ROI number " << i << endl;
            mask_final_ROI[i] = mask_Canny_ROI[i].clone();
        }
        else
        {
            std::cout << "Choose Region Growing Segmentation for ROI number " << i << endl;
            mask_final_ROI[i] = mask_RegionGrowing_ROI[i].clone();
        }
        
    }
    
    return mask_final_ROI;
    
}



bool Segmentator::detectOverlap(std::vector<std::vector<int>> coordinates_bb)
{	
	
	//__________________________ Detect if overlap _________________________________// 
	// to make things simpler, we suppose there will be maximum one overlap (over a maximum of 4 hands in an image)
	bool overlap = 0;
	
	int x1, y1, width1, height1;
	int x2, y2, width2, height2;
	
	int n_hands = coordinates_bb.size();
	
	for (int i = 0; i < n_hands; i++) 
	{
	    for (int j = 0; j < n_hands; j++) 
	    {
	        if (i == j)
	            continue;
	            
    	    x1 = coordinates_bb[i][0];
	        y1 = coordinates_bb[i][1];
	        width1 = coordinates_bb[i][2];
	        height1 = coordinates_bb[i][3];
	        
	        x2 = coordinates_bb[j][0];
	        y2 = coordinates_bb[j][1];
	        width2 = coordinates_bb[j][2];
	        height2 = coordinates_bb[j][3];
	        
            // intersection region
            int xA = std::max(x1, x2);
            int yA = std::max(y1, y2);
            int xB = std::min(x1+width1, x2+width2);
            int yB = std::min(y1+height1, y2+height2);

            int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);

            if (interArea != 0)
            {
                overlap = 1;
                return overlap;
	        }
	    }
	}
	return overlap;
}	


void Segmentator::handleOverlap(cv::Mat& mask_final, cv::Mat& mask_Watershed, std::vector<std::vector<int>> coordinates_bb, std::vector<cv::Vec3b>& random_color, int n_hands)
{
	//______________________________________________________________ Distance transform and watershed ______________________________________________________________//
	
	cv::Mat distTransf;
	cv::distanceTransform(mask_final, distTransf, cv::DIST_L2, 3);
	cv::normalize(distTransf, distTransf, 0, 1.0, cv::NORM_MINMAX);
		
	cv::namedWindow("Mask transform");
    cv::imshow("Mask transform", distTransf);
	cv::waitKey(0);
	
	cv::Mat dist;
	cv::threshold(distTransf, dist, 0.3, 1.0, cv::THRESH_BINARY);
	
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21,21));
    cv::morphologyEx(dist, dist, cv::MORPH_ERODE, kernel);
	
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
    // std::printf("Contours: %d\n", ncomp);

    //draw foreground markers
    for(int i=0; i<ncomp; ++i) {
        cv::drawContours(markers, contours, i, cv::Scalar(i+1), -1);
    }
    
    markers.convertTo(markers8u, CV_8U, 10);
    // cv::imshow("Markers", markers8u);
    // cv::waitKey(0);
    
    //draw background markers
    cv::circle(markers, cv::Point(5, 5), 3, cv::Scalar(255), -1);
    markers.convertTo(markers8u, CV_8U, 10);
    // cv::imshow("Markers", markers8u);
    // cv::waitKey(0);
	
	cv::cvtColor(mask_final, mask_final, cv::COLOR_GRAY2BGR);
	
	//apply the watershed algorithm
    cv::Mat result = mask_final.clone();
    cv::watershed(result, markers);
    cv::Mat mark;
    markers.convertTo(mark, CV_8U);
    cv::bitwise_not(mark, mark);
    
    cv::Mat output = mask_final.clone();
    
	// cv::namedWindow("Mask tram");
    // cv::imshow("Mask tram", output);
	// cv::waitKey(0);

    //create output image
    for(int i=0; i<markers.rows; ++i) {
        for(int j=0; j<markers.cols; ++j) {
            int index = markers.at<int>(i, j);
            if(index > 0 && index <= ncomp) {
                if (n_hands < 4)
                    output.at<cv::Vec3b>(i, j) = random_color[index-1];
                else 
                    output.at<cv::Vec3b>(i, j) = random_color[index+1];
            }
        }
    }
    
    mask_Watershed = output.clone();
    
	cv::namedWindow("mask_Watershed");
    cv::imshow("mask_Watershed", mask_Watershed);
	cv::waitKey(0);
	
	cv::destroyAllWindows();
		
}



