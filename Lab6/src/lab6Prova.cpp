#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
const float inlier_threshold = 2.5f; // Distance threshold to identify inliers with homography check
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio


int main(int argc, char* argv[])
{
	Mat img1 = cv::imread("../images/all_souls_000002.jpg");
	Mat img2 = cv::imread("../images/all_souls_000006.jpg");

if (img1.empty()) 
{
    cerr << "Could not open or find image 1" << endl;
    return -1;
}

if (img2.empty()) 
{
    cerr << "Could not open or find image 2" << endl;
    return -1;
}
	 
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
    akaze->detectAndCompute(img2, noArray(), kpts2, desc2);
    
    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);
    
    // Take only good matches wrt distance. The less distance, the better.
    vector<KeyPoint> matched1, matched2;
    vector <Point2f> points1, points2;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
            points1.push_back(kpts1[first.queryIdx].pt);
            points2.push_back(kpts2[first.trainIdx].pt);
        }
    }
    
    cout << matched1[0].pt.x << endl;
    cout << matched2[0].pt.x << endl;
        
    // Take only matches which are true positive. Get rid of the outliers.
    vector<DMatch> good_matches;
    vector<KeyPoint> inliers1, inliers2;
    
    // To calculate the homography between two images, we must know at least four corresponding points between them.
    Mat homography = findHomography(points1, points2, RANSAC); // 3×3 matrix that maps the points between the two images
    cout << "Homography matrix: " << endl << homography << endl;
    
    for(size_t i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F); // i need the last row of 1 in homogeneous transformation
        col.at<double>(0) = matched1[i].pt.x; // col is a column vector, so in this way we put in first line
        col.at<double>(1) = matched1[i].pt.y; // col is a column vector, so in this way we put in second 
        col = homography * col; // in this way each keypoint is mapped from the first image to the second image
        col /= col.at<double>(2); // remove the last fow of 1
        
        double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                            pow(col.at<double>(1) - matched2[i].pt.y, 2));
                            
         cout << "col matrix: " << endl << col.at<double>(0) << endl;                   
         cout << "macthed matrix: " << endl << matched2[i].pt.x << endl; 
                           
        if(dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }

    Mat result;
    drawMatches(img1, inliers1, // vector keypoints 1
                img2, inliers2, // vector keypoints 2
                good_matches, 
                result); // want keypoints1[i] = keypoints2[matches[i]]
    imwrite("akaze_result.png", result);
    cout << endl;
    imshow("result", result);
    waitKey(0);
    
    return 0;
    
}
