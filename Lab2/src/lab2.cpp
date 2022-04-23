#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include "filters.h" //declaration of this two functions

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
    cv::Mat img = cv::imread("../images/image.jpg", cv::IMREAD_COLOR); 
    
    
    if (img.empty())
    {
        std::cout << "!!! Failed imread(): image not found" << std::endl;
    // don't let the execution continue, else imshow() will crash.
    }
    
    //####################(  Resize images by Scaling factor - preserve aspect ratio  )##########################
    // Scaling Up the image 1.2 times by specifying both scaling factors
    // double scale_up_x = 1.2;
    // double scale_up_y = 1.2;
    // Scaling Down the image 0.3 times specifying a single scale factor.
    double scale_down = 0.3;
    // cv::Mat scaled_f_up;
    // cv::Mat scaled_f_down;
    
    cv::resize(img, img, cv::Size(), scale_down, scale_down, cv::INTER_LINEAR); // resize
    // cv::resize(image, scaled_f_up, Size(), scale_up_x, scale_up_y, INTER_LINEAR); // resize
    
    cv::namedWindow("Original image");
    cv::imshow("Original image", img);
	cv::waitKey(0); 
	
	cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    cv::namedWindow("Grayscale image");
    cv::imshow("Grayscale image", img);
    cv::waitKey(0); 
    
    cv::imwrite("/home/local/mustmar19581/Documents/Project/lab2/images/image_grayscale.jpg", img);
    
    //####################(  Min/Max Filter with Function  )##########################
    int ksize = 3;
    
    cv::Mat testMax = maxFilter(img, ksize);
    cv::imshow("Max filter Image" , testMax);
    
    cv::Mat testMin = minFilter(img, ksize);
    cv::imshow("Min filter Image" , testMin);
    
    cv::Mat medianFilter;
    cv::medianBlur(img, medianFilter, 15);
    cv::imshow("Median filter Image" , medianFilter);
    
    cv::Mat gaussianFilter;
    cv::GaussianBlur(img, gaussianFilter, cv::Size(5, 5), 0);
    cv::imshow("Gaussian smoothing Image" , gaussianFilter);
    
    cv::destroyAllWindows();
    
    //####################(  Display multiple images  )##########################
    int width = 3*img.cols; // width of 2 images next to each other
    int height = 2*img.rows; // height of 2 images over reach other

    cv::Mat inputAll = cv::Mat(height, width, img.type());

    cv::Rect subImageROI = cv::Rect(0, 0, img.cols, img.rows);

    // copy to subimage:
    img.copyTo(inputAll(subImageROI));

    subImageROI.x = img.cols;
    testMax.copyTo(inputAll(subImageROI));

    subImageROI.x = 2*img.cols;
    testMin.copyTo(inputAll(subImageROI));
    
    // subImageROI.x = 3*img.cols;
    // medianFilter.copyTo(inputAll(subImageROI));
    
    subImageROI.x = 0;
    subImageROI.y = img.rows;
    medianFilter.copyTo(inputAll(subImageROI));

    subImageROI.x = img.cols;
    subImageROI.y = img.rows;
    gaussianFilter.copyTo(inputAll(subImageROI));

    //subImageROI.x = 4*img.cols;
    //gaussianFilter.copyTo(inputAll(subImageROI));

    cv::imshow("Trasformations", inputAll);
    
    cv::waitKey(0);
    
    //####################(  Erosion and Dilation  )##########################
    cv::Mat im_gray;
    cv::Mat img_bw;
    cv::Mat img_dilate;
    cv::Mat img_erode;

    cv::Mat im_rgb  = cv::imread("../images/image.jpg", cv::IMREAD_COLOR);
    cv::cvtColor(im_rgb, im_gray, cv::COLOR_BGR2GRAY);
    
    scale_down = 0.3;
    cv::resize(im_gray, im_gray, cv::Size(), scale_down, scale_down, cv::INTER_LINEAR); // resize
    
	// Adaptive thresholding is the method where the threshold value is calculated for smaller regions and therefore, there will be different threshold values for different regions.
    cv::adaptiveThreshold(im_gray, img_bw, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 105, 1); 

	cv::Mat kernel = cv::Mat(3, 3, CV_8U, 1); // OpenCV replaces `1` with `Scalar(1,0,0)`
    cv::dilate(im_gray, img_dilate, kernel, cv::Point(-1, -1), 2, 1, 1); //cv::Mat(), a default 3x3 kernel will be used (all 1)
	
    cv::erode(im_gray, img_erode, cv::Mat(), cv::Point(-1, -1), 2, 1, 1); //cv::Mat(), a default 3x3 kernel will be used (all 1)
    
    cv::namedWindow( "gray image", cv::WINDOW_AUTOSIZE );
    cv::namedWindow( "threshold image", cv::WINDOW_AUTOSIZE );
    cv::namedWindow( "dilate image", cv::WINDOW_AUTOSIZE );
    cv::namedWindow( "erode image", cv::WINDOW_AUTOSIZE );
    
    cv::imshow( "gray image", im_gray );
    cv::imshow( "threshold image", img_bw );
    cv::imshow( "dilate image", img_dilate );
    cv::imshow( "erode image", img_erode );

    cv::waitKey(0); 
    cv::destroyAllWindows();
    
    //__________________________ Calculate and Plot Histogram __________________________
	Histogram h1;
	cv::Mat histImage1 = h1.calc_histogram(img);
	cv::Mat histSrc = h1.plot_histogram(histImage1);
	
    Histogram h2;
	cv::Mat histImage2 = h2.calc_histogram(testMax);
	cv::Mat histMax = h2.plot_histogram(histImage2);
	
	Histogram h3;
	cv::Mat histImage3 = h3.calc_histogram(testMin);
	cv::Mat histMin = h3.plot_histogram(histImage3);
    
    cv::namedWindow("Histogram Source", cv::WINDOW_NORMAL);
	cv::imshow("Histogram Source", histSrc);
	cv::namedWindow("Histogram testMax", cv::WINDOW_NORMAL);
	cv::imshow("Histogram testMax", histMax);
	cv::namedWindow("Histogram testMin", cv::WINDOW_NORMAL);
	cv::imshow("Histogram testMin", histMin);
	cv::waitKey(0);
    cv::destroyAllWindows();
    
    //__________________________ Equalize Image __________________________
    cv::Mat img_equalized;
    cv::equalizeHist(img, img_equalized);
    cv::imshow( "Source image", img);
    cv::imshow( "Equalized Image", img_equalized);
    cv::waitKey(0);
     
    return 0;
}

