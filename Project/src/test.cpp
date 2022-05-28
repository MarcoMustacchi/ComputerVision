
cv::Mat img_roi_HSV;  // definita come variabile globale 
cv::Mat mask;
int lowThreshold = 0;
const int max_lowThreshold = 100;
int highThreshold = 101;
const int max_highThreshold = 255;

static void MyCallbackForThreshold(int, void* param)
{

    cv::Mat test = *(cv::Mat *)param;
    
    cv::inRange(test, lowThreshold, highThreshold, mask);  
    
    // cv::namedWindow("imgCanny", cv::WINDOW_AUTOSIZE);
	cv::imshow("Edge Map", mask);   
}


//_____________________________________________ Classes _____________________________________________//

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






main() {

    //__________________________ Calculate and Plot Histogram __________________________
	Histogram h1;
	cv::Mat hist_roi_BGR = h1.calc_histogram(img_roi);
	cv::Mat histSrc = h1.plot_histogram(hist_roi_BGR);
	
    Histogram h2;
	cv::Mat hist_roi_HSV = h2.calc_histogram(img_roi_HSV);
	cv::Mat histBlur = h2.plot_histogram(hist_roi_HSV);
    
    cv::namedWindow("Histogram BGR", cv::WINDOW_NORMAL);
	cv::imshow("Histogram BGR", histSrc);
	cv::namedWindow("Histogram HSV", cv::WINDOW_NORMAL);
	cv::imshow("Histogram HSV", histBlur);
	cv::waitKey(0);
    cv::destroyAllWindows();

    Histogram h3;
	cv::Mat hist_roi_HSV_Intensity = h3.calc_histogram(Bands_HSV[2]);
	cv::Mat hist = h3.plot_histogram(hist_roi_HSV_Intensity);
	cv::namedWindow("Histogram HSV Intensity", cv::WINDOW_NORMAL);
	cv::imshow("Histogram HSV Intensity", hist);
	cv::waitKey(0);
	
	//___________________________ Trackbar ___________________________
	cv::namedWindow( "Edge Map", cv::WINDOW_AUTOSIZE );
    //Create track bar to change maxThreshold
	cv::createTrackbar( "Low Threshold:", "Edge Map", &lowThreshold, max_lowThreshold, MyCallbackForThreshold, &img_roi_HSV ); // puo' avere anche il parametro void* userdata per callback
	MyCallbackForThreshold(0, &img_roi_HSV);
	
    //Create track bar to change maxThreshold
    cv::createTrackbar( "High Threshold:", "Edge Map", &highThreshold, max_highThreshold, MyCallbackForThreshold, &img_roi_HSV );
    cv::waitKey(0);  
    
    
	
	cv::namedWindow( "Edge Map", cv::WINDOW_AUTOSIZE );
	cv::createTrackbar( "Low Threshold:", "Edge Map", &lowThreshold, max_lowThreshold, MyCallbackForThreshold, &img_roi_HSV ); // puo' avere anche il parametro void* userdata per callback
	MyCallbackForThreshold(0, &img_roi_HSV);
	
    //Create track bar to change maxThreshold
    cv::createTrackbar( "High Threshold:", "Edge Map", &highThreshold, max_highThreshold, MyCallbackForThreshold, &img_roi_HSV );


}


