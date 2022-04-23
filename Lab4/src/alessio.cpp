//Stefan Luca 1206186

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

const float RESIZE_RATIO = 0.6;

using namespace cv;
using namespace std;

int main()
{
	Mat img = imread("../images/street_scene.png");
	Mat image = img.clone();
	Mat img_out;
	
	Mat edges;
	double threshold1 = 300;
	double threshold2 = 700;
	int apertureSize = 3;
	bool L2gradient = false;
	Canny(img, edges, threshold1, threshold2, apertureSize, L2gradient);

	resize(edges, img_out, Size(), RESIZE_RATIO, RESIZE_RATIO);  //resize for showing
	imshow("edges", img_out);
	
	
	vector<Vec2f> lines;		// will hold the results of the detection
	double rho_h = 2;
	double theta_h = CV_PI / 80;
	int threshold = 100;
	double srn = 0, stn = 0;
	double min_theta = -CV_PI *7/16 ;
	double max_theta = CV_PI *7/16;
	HoughLines(edges, lines, rho_h, theta_h, threshold, srn, stn, min_theta, max_theta); // runs the actual detection
	
	// PlotLines
	for (size_t i = 0; i < lines.size(); i++) 
	{ 
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double ct = cos(theta), st = sin(theta);
		double x0 = ct * rho, y0 = st * rho, num=3000;
		pt1.x = cvRound(x0 + num * (-st));
		pt1.y = cvRound(y0 + num * (ct));
		pt2.x = cvRound(x0 - num * (-st));
		pt2.y = cvRound(y0 - num * (ct));
		line(image, pt1, pt2, Scalar(0, 0, 255),3);
	}
	
	//PaintStreet
	for (size_t x = 0; x < img.cols; x++) {
		for (size_t y = 0; y < img.rows; y++) {
			bool flag = true;
			for (size_t i = 0; i < lines.size() && flag; i++) {
				if (y < - 1 / tan(lines[i][1])*x + lines[i][0] / sin(lines[i][1]))
					flag = false;
			}
			if(flag)
				img.at<Vec3b>(y,x) = Vec3b(0, 0, 255);
		}
	}
	
	Mat img_gray;
	cvtColor(img, img_gray, COLOR_BGR2GRAY);
	//cout << corners[0][0] << "   " << corners[0][1];
	vector<Vec3f> circles;
	double dp = 1;
	double minDist = 3;
	double param1 = threshold2 /2;
	double param2 = 30;
	int minRadius = 0;
	int maxRadius = 65;
	//GaussianBlur(img_gray, img_gray, Size(9, 9), 3, 3);
	HoughCircles(img_gray, circles, HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius);
	
	// PaintCircle
	for (size_t i = 0; i < circles.size(); i++)
	{
		cout << circles[i][0]<<"  "<< circles[i][1];
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		//circle(img, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(img, center, radius, Scalar(0, 255, 0), -radius, 8, 0);
	}

	
	resize(image, img_out, Size(), RESIZE_RATIO, RESIZE_RATIO);  //resize for showing
	imshow("Original image", img_out);
	resize(img, img_out, Size(), RESIZE_RATIO, RESIZE_RATIO);  //resize for showing
	imshow("Final result", img_out);


	waitKey(0);

}

