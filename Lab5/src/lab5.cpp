#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>

class Histogram {
public:
	cv::Mat calc_histogram(cv::Mat scr) {
		cv::Mat hist;
		hist = cv::Mat::zeros(256, 1, CV_32F);
		scr.convertTo(scr, CV_32F);
		double value = 0;
		for (int i = 0; i < scr.rows; i++)
		{
			for (int j = 0; j < scr.cols; j++)
			{
				value = scr.at<float>(i, j);
				hist.at<float>(value) = hist.at<float>(value) + 1;
			}
		}
		return hist;
	}

	cv::Mat plot_histogram(cv::Mat histogram) {
		cv::Mat histogram_image(400, 512, CV_8UC3, cv::Scalar(0, 0, 0));
		cv::Mat normalized_histogram;
		cv::normalize(histogram, normalized_histogram, 0, 400, cv::NORM_MINMAX, -1, cv::Mat());

		for (int i = 0; i < 256; i++)
		{
			cv::rectangle(histogram_image, cv::Point(2 * i, histogram_image.rows - normalized_histogram.at<float>(i)), cv::Point(2 * (i + 1), histogram_image.rows), cv::Scalar(255, 0, 0));
		}


		return histogram_image;
	}
	
};

int main(int argc, char** argv)
{
    cv::Mat img1 = cv::imread("../images/Asphalt-1.png", cv::IMREAD_COLOR); 
    cv::Mat img2 = cv::imread("../images/Asphalt-2.png", cv::IMREAD_COLOR);
    cv::Mat img3 = cv::imread("../images/Asphalt-3.png", cv::IMREAD_COLOR);
    
    if (img1.empty() || img2.empty() || img3.empty())
    {
        std::cout << "!!! Failed imread(): image not found" << std::endl;
    // don't let the execution continue, else imshow() will crash.
    }
    
    cv::namedWindow("Asphalt 1");
    cv::imshow("Asphalt 1", img1);
    cv::imshow("Asphalt 2", img2);
    cv::imshow("Asphalt 3", img3);
	cv::waitKey(0); 
	
	cv::Mat img_gray1;
    cv::Mat img_gray2;
    cv::Mat img_gray3;
    
	cv::cvtColor(img1, img_gray1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(img2, img_gray2, cv::COLOR_BGR2GRAY);
	cv::cvtColor(img3, img_gray3, cv::COLOR_BGR2GRAY);

    cv::namedWindow("Grayscale image");
    cv::imshow("Grayscale image", img_gray1);
    cv::waitKey(0); 
    
    cv::Mat blur;
    cv::GaussianBlur(img_gray1,blur,cv::Size(5,5),0);

    
    //__________________________ Calculate and Plot Histogram __________________________
	Histogram h1;
	cv::Mat histImage1 = h1.calc_histogram(img_gray1);
	cv::Mat histSrc = h1.plot_histogram(histImage1);
	
    Histogram h2;
	cv::Mat histImage2 = h2.calc_histogram(blur);
	cv::Mat histBlur = h2.plot_histogram(histImage2);
    
    cv::namedWindow("Histogram Source", cv::WINDOW_NORMAL);
	cv::imshow("Histogram Source", histSrc);
	cv::namedWindow("Histogram Gaussian Filter", cv::WINDOW_NORMAL);
	cv::imshow("Histogram Gaussian Filter", histBlur);
	cv::waitKey(0);
    cv::destroyAllWindows();
    
    
    //__________________________ Segmentation by Thresholding __________________________
    // global thresholding
    cv:: Mat img_thr1, img_thr2, img_thr3;
	cv::threshold(img_gray1,img_thr1,50,255,cv::THRESH_BINARY); 
	// Otsu's thresholding
	double th2 = cv::threshold(img_gray1,img_thr2,0,255,cv::THRESH_BINARY | cv::THRESH_OTSU);
	// Otsu's thresholding after Gaussian filtering
	double th3 = cv::threshold(blur,img_thr3,0,255,cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    std::cout << "Otsu method threshold: " << th2 << std::endl;
    std::cout << "Otsu method threshold after Gaussian Filter: " << th3 << std::endl;
    
    cv::imshow("Global thresholding", img_thr1);
    cv::imshow("Otsu's thresholding", img_thr2);
	cv::imshow("Otsu's thresholding after Gaussian filtering", img_thr3);
	cv::waitKey(0);
    
    cv::imwrite("../images/results/global_thresholding.jpg", img_thr1);
    cv::imwrite("../images/results/otsu_thresholding.jpg", img_thr2);
    cv::imwrite("../images/results/otsu_thresholding_with_gaussian_filter.jpg", img_thr3);
    	
    return 0;
}
