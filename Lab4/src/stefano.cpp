//Stefan Luca 1206186

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

const float RESIZE_RATIO = 0.8;		//some images are too big or too small for visualization

using namespace cv;
using namespace std;


void PlotLines(Mat image_line, vector<Vec2f> lines) {

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double ct = cos(theta), st = sin(theta);
		double x0 = ct * rho, y0 = st * rho, num = 3000;
		pt1.x = cvRound(x0 + num * (-st));
		pt1.y = cvRound(y0 + num * (ct));
		pt2.x = cvRound(x0 - num * (-st));
		pt2.y = cvRound(y0 - num * (ct));
		line(image_line, pt1, pt2, Scalar(0, 0, 255), 2);
	}

	return;
}


void PaintStreet(Mat image_road, vector<Vec2f> lines) {
	for (size_t x = 0; x < image_road.cols; x++) {
		for (size_t y = 0; y < image_road.rows; y++) {
			bool flag = true;
			for (size_t i = 0; i < lines.size() && flag; i++) {
				if (y < -1 / tan(lines[i][1])*x + lines[i][0] / sin(lines[i][1]))
					flag = false;
			}
			if (flag)
				image_road.at<Vec3b>(y, x) = Vec3b(0, 0, 255);	//paint in red all pixel under all the lines
		}
	}

	return;
}


void PaintCircle(Mat image_circle, vector<Vec3f> circles) {
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle
		circle(image_circle, center, radius, Scalar(0, 255, 0), -radius, 8, 0);
	}

	return;

}



int main()
{
	Mat img = imread("../images/street_scene.png");			//import the image
	Mat image_line = img.clone();
	Mat image_road = img.clone();
	Mat image_circle = img.clone();
	Mat img_out, img_gray, edges;
	

	resize(img, img_out, Size(), RESIZE_RATIO, RESIZE_RATIO);  //resize for showing
	imshow("Original image", img_out);

	//all the parameters is tuned for the image "road2.png", if we want to run the code with an other image
	//it is necessary to change some parameters


	double threshold1 = 300;	//canny lower threshold
	double threshold2 = 700;	//canny higher threshold
	int apertureSize = 3;
	bool L2gradient = false;
	Canny(img, edges, threshold1, threshold2, apertureSize, L2gradient);	//find edges with canny

	resize(edges, img_out, Size(), RESIZE_RATIO, RESIZE_RATIO);  //resize for showing
	imshow("Edges", img_out);
	
	
	vector<Vec2f> lines;		// will hold the results of the detection
	double rho_h = 2;
	double theta_h = CV_PI / 80;
	int threshold = 150;
	double srn = 0, stn = 0;
	double min_theta = -CV_PI *3/8 ;
	double max_theta = CV_PI *3/8;
	HoughLines(edges, lines, rho_h, theta_h, threshold, srn, stn, min_theta, max_theta); // runs the lines detection

	PlotLines(image_line, lines); //plot the lines found in the image

	resize(image_line, img_out, Size(), RESIZE_RATIO, RESIZE_RATIO);  //resize for showing
	imshow("Lines found", img_out);
	
	PaintStreet(image_road, lines);	//paint in red all pixel of the street

	resize(image_road, img_out, Size(), RESIZE_RATIO, RESIZE_RATIO);  //resize for showing
	imshow("Streed colored", img_out);
	
	cvtColor(img, img_gray, COLOR_BGR2GRAY);	//gray scale conversion
	vector<Vec3f> circles;		//will hold the results of detection
	double dp = 1;
	double minDist = 2;		//minimum distance between two center
	double param1 = threshold2 /2;	
	double param2 = 20;
	int minRadius = 0;
	int maxRadius = 30;
	HoughCircles(img_gray, circles, HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius);

	PaintCircle(image_circle, circles);	//paint in green all circle

	resize(image_circle, img_out, Size(), RESIZE_RATIO, RESIZE_RATIO);  //resize for showing
	imshow("Circle found", img_out);


	waitKey(0);

}
