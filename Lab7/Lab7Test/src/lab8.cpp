#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/utils/filesystem.hpp> //include per utils::fs::glob, anche se ho gia' core.hpp
#include <iostream> 
#include "PanoramicImage.h"

using namespace cv;
using namespace std;

const float RESIZE_RATIO = 1.5;		//some images are too big or too small for visualization


int main()
{
    
	double camera_fov = 54;			//fov of the camera(degrees)
	int nfeatures = 1500;			//number of the features in each image
	double ratio = 4;				//distace ratio for matching


    vector<String> result;

    utils::fs::glob ("../images/dataset_dolomites/",
                    "i*.png",
                    result,
                    false,
                    false 
                    );	
                    
    for (int i=0; i<result.size(); i++) { 
        cout << result[i] << endl;
    }                 

	int num_img = result.size();				//number of the images
	vector<Mat> images(num_img);
	
	for (int i = 0; i < num_img; i++) {
		images[i] = imread(result[i]);
		if ((images[i].cols == 0) || (images[i].rows == 0)) {		//if the program can't load an image, stop
			cout << "Immagine " + result[i] + " non trovata" << endl << endl;
			waitKey(0);
			return -1;
		};
	};
	
	
	/*
	vector<Mat> image_set;
	
	image_set.resize(num_img);
	for (int i = 0; i < num_img; i++)  //project the images on a cilider
		image_set[i] = cylindricalProj(images[i], camera_fov / 2);
	
	
	cout << image_set.size() << endl;
	
	for (int i=0; i<image_set.size(); i++){
        imshow("Projected", image_set[i]);
        waitKey(0);
    }
    */	
    
	PanoramicImage panoramic = PanoramicImage(images, camera_fov, nfeatures, ratio);	//create PanoramicImage object
	
    std::vector<std::vector<cv::KeyPoint>> keypoints = panoramic.getKeyPoints();
    
    vector<vector<KeyPoint>> sortedKeypoints = panoramic.getSortedKeypoints();
	
	vector<Mat> match = panoramic.drawMatches();		//show the good matches

	for (int i = 0; i < match.size(); i++) {
		resize(match[i], match[i], Size(), RESIZE_RATIO * 0.6, RESIZE_RATIO * 0.6);  //resize for showing
		imshow("match" + to_string(i), match[i]);
	};
    
	waitKey(0);
	destroyAllWindows();
    	
	panoramic.computePanoramic();		//compute the panoramic image

	Mat r = panoramic.getResult();		//get the result
	imwrite("panoramic.png", r);		//save the result for a better visualization
	resize(r, r, Size(), RESIZE_RATIO * 0.4, RESIZE_RATIO * 0.4);  //resize for showing
	imshow("Final result", r);
    
	waitKey(0);

}
