#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp> //include per utils::fs::glob, anche se ho gia' core.hpp
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace cv;

int main() {

	//load image
    vector<String> nomi;

    utils::fs::glob ("../images/",
                    "img*.jpg",
                    nomi,
                    false,
                    false 
                    );	
                    
    for (int i=0; i<nomi.size(); i++) { 
        cout << nomi[i] << endl;
    }                 

	int num_img = nomi.size();      //number of the images
	vector<Mat> images(num_img);
	
	for (int i = 0; i < num_img; i++) {
		images[i] = imread(nomi[i]);
		if ((images[i].cols == 0) || (images[i].rows == 0)) {		//if the program can't load an image, stop
			cout << "Immagine " + nomi[i] + " non trovata" << endl << endl;
			waitKey();
			return -1;
		};
	};
	//imshow("img", img[7]);
	//waitKey();


	//find corners
	Size patternSize(12,8);
	vector<vector<Point2f>> imagePoints;
	for (int i = 0;i < num_img; i++) {
		vector<Point2f> corners;
		bool patternWasFound = findChessboardCorners(images[i], patternSize, corners, CALIB_CB_FAST_CHECK);
		imagePoints.push_back(corners);
		//cout << corners.size();
	};
	/*drawChessboardCorners(img[0], patternSize, corners, patternWasFound);
	Mat dst;
	resize(img[0], dst, Size(1008,754));
	imshow("img", dst);*/
	

	//create 3d coordinates
	vector<vector<Vec3f>> objectPoints;
	for (int i = 0; i < num_img; i++) {
		vector<Vec3f> points;
		for (int y = 0;y < 8;y++)
			for (int x = 0;x < 12; x++) {
				Vec3f point = Vec3f(2*x, 2*y, 0);
				points.push_back(point);
			};
		objectPoints.push_back(points);
	};

	//find camera parameters
	Mat cameraMatrix;
	Mat distCoeffs;
	vector<Mat> rvecs;
	vector<Mat> tvecs;
	int flags = 0;
	calibrateCamera(objectPoints,imagePoints,patternSize,cameraMatrix,distCoeffs,rvecs,tvecs,flags);

	//project 3d points in 2d plane
	vector<vector<Point2f>> projection;
	for (int i = 0; i < num_img; i++) {
		vector<Point2f> projectedPoints;
		//Mat jacobian;
		projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, projectedPoints/*, jacobian, (double)flags*/);
		projection.push_back(projectedPoints);
	};


	//compute error
	double error = 0;
	int count = 0;
	vector<double> err;
		for (int i = 0; i < num_img; i++) {
		vector<Point2f> projectedPoints = projection[i];
		vector<Point2f> imgPoints = imagePoints[i];
		err.push_back(0);
		for (int j = 0;j < projectedPoints.size();j++) {
			err[i] += norm(projectedPoints[j] - imgPoints[j]);
			count++;
		};
		
		error += err[i];
	};

	//sort image by error
	cout << "best image :img" << ((distance(err.begin(), min_element(err.begin(), err.end()))) + 1)
	 << " with e=" << *min_element(err.begin(), err.end())<< endl;

	cout << "worst image :img" << ((distance(err.begin(), max_element(err.begin(), err.end()))) + 1)
		<< " with e=" << *max_element(err.begin(), err.end()) << endl;

	error = error / count;
	cout << error << endl;
	
	imshow("Best image", images[1]);
	imshow("Worst image", images[5]);
	waitKey(0);

	destroyAllWindows();
	
	return 0;
	
}
