#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "PanoramicImage.h"
#include "PanoramicUtils.h"


PanoramicImage::PanoramicImage(std::vector<cv::Mat> input_set, double camera_fov, int nfeatures1, double ratio1) {

		assert(ratio1 >= 1);  //stop if ratio is smaller than 1

		FoV = camera_fov;

		num_img = input_set.size();

		image_set.resize(num_img);
		for (int i = 0; i < num_img; i++)  //project the images on a cilider
			image_set[i] = PanoramicUtils::cylindricalProj(input_set[i], FoV / 2);

		nfeatures = nfeatures1;
		
		ratio = ratio1;
};


std::vector<cv::Mat> PanoramicImage::getCylindricalProj() {
	return image_set;
};


std::vector<std::vector<cv::KeyPoint>> PanoramicImage::getKeyPoints() {

	//if the keypoints are already calulated, return
	if ( !keypoints.empty())
		return keypoints;

	//if the keypoints aren't calulated, calculate them
	float  	scaleFactor = 1.2f;
	int  	nlevels = 8;
	int  	edgeThreshold = 31;
	int  	firstLevel = 0;
	int  	WTA_K = 2;
	int  	patchSize = 31;
	int  	fastThreshold = 20;

	descriptors.resize(num_img);

    // Create smart pointer for ORB feature detector
	cv::Ptr<cv::Feature2D> orb = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
		firstLevel, WTA_K, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);

	//detect keypoints
	for (int i = 0; i < num_img; i++) {
		std::vector<cv::KeyPoint> keyp;
		orb->detectAndCompute(image_set[i], cv::Mat(), keyp, descriptors[i]);
		keypoints.push_back(keyp);
	};

	return keypoints;
};


std::vector<cv::Mat> PanoramicImage::getDescriptors() {
	getKeyPoints();		//getKeyPoints() claculate also the decriptors of the keypoints
	return descriptors;

};


std::vector<std::vector<cv::KeyPoint>> PanoramicImage::getSortedKeypoints() {

	//if the matches are already calulated, return
	if (!sortedKeypoints.empty())
		return sortedKeypoints;

	// getKeyPoints();		//we need keypoints to calculate matches

	//create the brute force matcher
	cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING, false);

	for (int j = 0; j < num_img - 1; j++) {		//do for each couple of consecutive images
		std::vector<std::vector<cv::DMatch>> matches;
		matcher.knnMatch(descriptors[j], descriptors[j + 1], matches, 1);

		//trasform from vector<vector<DMatch>> to vector<DMatch>
		std::vector<cv::DMatch> match;
		for (int i = 0; i < matches.size(); i++)
			match.push_back(matches[i][0]);

		//find the minimum distance between two features
		int min = match[0].distance;
		for (int i = 1; i < match.size(); i++)
			if (match[i].distance < min)
				min = match[i].distance;

		std::vector<cv::KeyPoint> key1, key2;
		std::vector<cv::DMatch> matchr;
		for (int i = 0; i < matches.size(); i++) {
			if (match[i].distance < (min * ratio)) {	
				matchr.push_back(match[i]);				//copy only "good" matches
				key1.push_back(keypoints[j][match[i].queryIdx]);		//copy only "good" features
				key2.push_back(keypoints[j+1][match[i].trainIdx]);	//sort the good features of the second image of each couple
			}

		};
		sortedKeypoints.push_back(key1);
		sortedKeypoints.push_back(key2);
		matchRefined.push_back(matchr);
	}

	return sortedKeypoints;

};

//___________________________________ Next call in main ___________________________________//   
std::vector<cv::Mat> PanoramicImage::drawMatches() {

	// getSortedKeypoints();	//for draw the matches we need keypoints and matches

	std::vector<cv::Mat> output_img(num_img - 1);	//vector of output images
	for (int i = 0; i < num_img - 1; i++) 
		cv::drawMatches(image_set[i], keypoints[i], image_set[i + 1], keypoints[i + 1], matchRefined[i], output_img[i]);
	return output_img;
};


std::vector<std::vector<cv::Point2f>> PanoramicImage::getSortedPoints() {

	getSortedKeypoints();	//get matches and keypoints

	std::vector<std::vector<cv::Point2f>> sortedPoints(sortedKeypoints.size());
	
	//extract the coordinates from cv::KeyPoint and create vectors of cv::Point2f
	for (int i = 0; i < sortedKeypoints.size(); i++) {
		std::vector<cv::Point2f> points(sortedKeypoints[i].size());
		for (int j = 0; j < sortedKeypoints[i].size(); j++)
			points[j] = sortedKeypoints[i][j].pt;
		sortedPoints[i] = points;
	};

	return sortedPoints;

};


void PanoramicImage::computeTranslations(std::vector<std::vector<cv::Point2f>> sortedPoints) {

	std::vector<cv::Mat> mask(num_img - 1);
	//run ransac
	for (int i = 0; i < num_img - 1; i++) {
		findHomography(sortedPoints[2 * i], sortedPoints[2 * i + 1], cv::RANSAC, 5, mask[i]);
	};

	//compute translations
	translations_x.resize(num_img - 1);
	translations_y.resize(num_img - 1);

	for (int i = 0; i < num_img - 1; i++) {
		int sum_x = 0;
		int sum_y = 0;
		int k = 0;
		//compute the average of the differences of the coordinates of the features
		for (int j = 0; j < mask[i].rows; j++) {
			if (mask[i].at<uchar>(j, 0) != 0) {		//consider only the inliers
				sum_x = sum_x + sortedPoints[2 * i][j].x - sortedPoints[2 * i + 1][j].x;
				sum_y = sum_y + sortedPoints[2 * i][j].y - sortedPoints[2 * i + 1][j].y;
				k++;
			};
		};

		translations_x[i] = sum_x / k;
		translations_y[i] = sum_y / k;
	};

};


std::vector<cv::Mat> PanoramicImage::cutY() {

	if (translations_y.empty()) { //we need translation to cut the images in y direction
		computeTranslations(getSortedPoints());
	};

	int min = 0;
	int max = 0;
	std::vector<int> sum_y(num_img);
	sum_y[0] = 0;
	for (int i = 1; i < num_img; i++) {
		sum_y[i] = sum_y[i - 1] + translations_y[i-1];
		if (sum_y[i] < min)
			min = sum_y[i];
		if (sum_y[i] > max)
			max = sum_y[i];
		std::cout << "Test " << sum_y[i] << std::endl;
	};

	std::cout << min << std::endl;
	std::cout << max << std::endl;
	std::vector<cv::Mat> image_cut(num_img);

	int height = image_set[0].rows + min - max;
	for (int i = 0; i < num_img; i++) {		//cut in y directions the images
		image_cut[i] = image_set[i](cv::Rect(0, -sum_y[i]+max, image_set[i].cols, height));
		std::cout << "Prova " << sum_y[i] << std::endl;
	};

	return image_cut;
}


void PanoramicImage::computePanoramic() {

	std::vector<std::vector<cv::Point2f>> sortedPoints = getSortedPoints();		//get the matches
	
	computeTranslations(sortedPoints);	//we need know the translations
	
	std::vector<cv::Mat> image_cut = cutY();

	for (int i = 0; i < num_img - 1; i++) {		//cut in x directions the images
		image_cut[i] = image_cut[i](cv::Rect(0, 0, translations_x[i], image_cut[i].rows));
	};
	
	image_cut[num_img - 1] = image_cut[num_img - 1];		//the last image isn't cut
	
    const float RESIZE_RATIO = 1.5;		//some images are too big or too small for visualization
	
	for (int i=0; i<image_cut.size(); i++) { //show all image cutted
		cv::resize(image_cut[i], image_cut[i], cv::Size(), RESIZE_RATIO * 0.6, RESIZE_RATIO * 0.6);  //resize for showing
        cv::imshow("Cut" + std::to_string(i), image_cut[i]);
    }
    
    cv::waitKey(0);
    
	output_image = image_cut[0];	//inizialization of the loop
	for (int i = 1; i < num_img; i++)		//union of the cutted images
		cv::hconcat(output_image, image_cut[i], output_image);
};


cv::Mat PanoramicImage::getResult() {
	return output_image;
};
