#ifndef PanoramicImage_H
#define PanoramicImage_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


class PanoramicImage {

	// Methods

public:

	// constructor
	PanoramicImage(std::vector<cv::Mat> input_set, double camera_fov, int nfeatures, double ratio);

	// get the result of computations
	cv::Mat getResult();

	// get the image set after the cylindrical projection
	std::vector < cv::Mat > getCylindricalProj();

	//get the keypoints found in all the images
	std::vector<std::vector<cv::KeyPoint>> getKeyPoints();

	//get descriptors of the image keypoint
	std::vector<cv::Mat> getDescriptors();

	//calculate the matches
	std::vector<std::vector<cv::KeyPoint>> getSortedKeypoints();

	//return a vector of images that show the matches found, useful for debug
	std::vector<cv::Mat> drawMatches();

	//calculate the panoramic image
	void computePanoramic();

	//extract the coordinates from cv::KeyPoint and create vectors of cv::Point2f
	std::vector<std::vector<cv::Point2f>> getSortedPoints();

	std::vector<cv::Mat> cutY();

	//compute traslations necessary to paste all images
	void computeTranslations(std::vector<std::vector<cv::Point2f>> sortedPoints);

	//Data
protected:

	//set of imput images
	std::vector<cv::Mat> image_set;

	//number of images
	int num_img;

	//Fov of the camera
	double FoV;

	//number of feature to search in each image
	int nfeatures;

	//distace ratio for matching
	double ratio; 
	
	//keypoints found in all the images
	std::vector<std::vector<cv::KeyPoint>> keypoints;

	//descriptors of the image keypoint
	std::vector<cv::Mat> descriptors;

	//keypoint sorted: sortedKeypoints[2i][j] is the j-th keypoint in the i-th image that correspond
	//to sortedKeypoints[2i+1][j] which is the j-th keypoint in the (i+1)-th image
	std::vector<std::vector<cv::KeyPoint>> sortedKeypoints;

	//all matches found between the keypoints of all consecutive images which distance is smaller a certain threshold
	std::vector<std::vector<cv::DMatch>> matchRefined;

	//traslations in x necessary to paste all images
	std::vector<int> translations_x;

	//traslations in y necessary to paste all images
	std::vector<int> translations_y;

	//output image
	cv::Mat output_image;

};

#endif
