
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include "removeOutliers.h" 


//_____________________________________________ Remove outliers from segmentation _____________________________________________

void removeOutliers(cv::Mat& input) {
	
    cv::Mat labelImage, stats, centroids;
    
    int nLabels =  cv::connectedComponentsWithStats(input, labelImage, stats, centroids, 8);
    
    std::cout << "nLabels = " << nLabels << std::endl;
    std::cout << "stats.size() = " << stats.size() << std::endl;
	
	std::cout << stats.col(4) << std::endl;
	std::cout << "test2 colonna" << cv::CC_STAT_AREA << std::endl;
	
	
    std::vector<cv::Vec3b> colors(nLabels);
    std::vector<int> labels_finals;
    colors[0] = cv::Vec3b(0, 0, 0); //background
    
    int max_stats = 0;
    
    for (int label = 1; label < nLabels; ++label) { //label  0 is the background
        if ((stats.at<int>(label, cv::CC_STAT_AREA)) > max_stats) { // in order to get row i and column CC_STAT_AREA (=4)
            max_stats = stats.at<int>(label, cv::CC_STAT_AREA);
        }
    }
    
    std::cout << "Max component area : " << max_stats << std::endl;
    
    for (int label = 1; label < nLabels; ++label) { //label  0 is the background
        if ((stats.at<int>(label, cv::CC_STAT_AREA)) >= max_stats) { // in order to get row i and column CC_STAT_AREA (=4)
            labels_finals.push_back(label);
        }
        std::cout << "area del component: " << label << "-> " << stats.at<int>(label, cv::CC_STAT_AREA) << std::endl;
        // colors[label] = cv::Vec3b(0, 255, 0);
    }
    
    cv::Mat dst(input.size(), CV_8UC3);
    for (int r = 0; r < dst.rows; ++r) {
        for (int c = 0; c < dst.cols; ++c) {
            int label = labelImage.at<int>(r, c);
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(r, c); //acceso all'elemento 
            pixel = colors[label];
        }
    }

    cv::Mat dst2(input.size(), CV_8UC1);

    for (int i = 0; i < labels_finals.size(); ++i) {
        std::cout << "path i:  " << labels_finals[i] << ' ' << std::endl;
        compare(labelImage, labels_finals[i], input, cv::CMP_EQ);   ////???????????????
    }
    
    cv::imshow("compare imagem ", input);
    cv::waitKey(0);
    
}
